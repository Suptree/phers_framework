#!/usr/bin/env python3

import time
from collections import deque
import random
import numpy as np
from agent import PPOAgent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import gymnasium as gym
import pandas as pd
import gazebo_env

import signal
import sys


# GPUが使える場合はGPUを使う
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


total_iterations = 5000
plot_interval = 10  # 10イテレーションごとにグラフを保存
save_model_interval = 100  # 100イテレーションごとにモデルを保存

env = gazebo_env.GazeboEnvironment("hero_0")

# env = gym.make('InvertedPendulum-v4')
# env = gym.make('Pendulum-v1', render_mode="rgb_array")
# env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array")
    
set_seeds(1023)

agent = PPOAgent(env_name="gazebo_env_avoidance",
                 n_iteration=total_iterations, 
                 n_states=13, 
                 action_bounds=[-1,1], 
                 n_actions=2,
                 actor_lr=3e-4, 
                 critic_lr=3e-4, 
                 gamma=0.99, 
                 gae_lambda=0.95, 
                 clip_epsilon=0.2, 
                 buffer_size=10000, 
                 batch_size=512,
                 entropy_coefficient=0.01, 
                 device=device)
agent.save_hyperparameters()
signal.signal(signal.SIGINT, exit)
for iteration in range(total_iterations):
    print("+++++++++++++++++++  iteration: {}++++++++++++++".format(iteration))

    step_count = 0

    #1回学習を行うあたりに必要な 2048ステップ分のトラジェクトリーを収集
    while True:
        state = env.reset(seed=random.randint(0,100000))
        done = False
        episode_data = []
        total_reward = 0
        # 1エピソード分のデータを収集

        while not done:
            step_count += 1
            action, log_prob_old = agent.get_action(state)

            next_state, reward, terminated, baseline_reward = env.step([(action[0]+1.0)*0.1,action[1]])

            total_reward += baseline_reward            
            if terminated:
                done = 1

            episode_data.append((state, action, log_prob_old, reward, next_state, done))
            state = next_state

        print("total_reward", total_reward)
        agent.logger.reward_history.append(total_reward)
        agent.trajectory_buffer.add_trajectory(episode_data)
        
        if step_count >= 4096:
            break
    agent.compute_advantages_and_add_to_buffer()
        
    # パラメータの更新
    epochs = 3
    for epoch in range(epochs):
        # ミニバッチを取得
        state, action, log_prob_old, reward, next_state, done, advantage = agent.replay_memory_buffer.get_minibatch()

        actor_loss = agent.compute_ppo_loss(state, action, log_prob_old, advantage)
        critic_loss = agent.compute_value_loss(state, reward, next_state, done)
        agent.optimize(actor_loss, critic_loss)

        # LOGGER
        agent.logger.losses_actors.append(actor_loss.item())
        agent.logger.losses_critics.append(critic_loss.item())

    agent.schedule_lr()

    agent.trajectory_buffer.reset()
    agent.replay_memory_buffer.reset()
    
    # LOGGER
    agent.logger.clear()
    if iteration % save_model_interval == 0:
        agent.save_weights(iteration)
    if iteration % plot_interval == 0:
        agent.logger.plot_graph(iteration)    
        agent.logger.save_csv()

def exit(signal, frame):
    print("\nCtrl+C detected. Saving training data and exiting...")    
    sys.exit(0)