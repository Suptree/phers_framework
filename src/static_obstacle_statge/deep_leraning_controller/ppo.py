#!/usr/bin/env python

import time
from collections import deque
import random
import numpy as np
import gazebo_env
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

import pandas as pd

import signal
import sys


# GPUが使える場合はGPUを使う
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


class TrajectoryBuffer:
    # 初期化
    def __init__(self):
        self.buffer = []

    # バッファにトラジェクトリを追加
    def add_trajectory(self, trajectory):
        self.buffer.append(trajectory)

    def __len__(self):
        return len(self.buffer)

    # 与えられたトラジェクトリの要素を分解して返す
    def get_per_trajectory(self, trajectory):
        state = torch.tensor(
            np.stack([x[0] for x in trajectory]), device=device, dtype=torch.float32
        )
        linear_action = torch.tensor(
            np.array([x[1] for x in trajectory]), device=device, dtype=torch.float32
        )
        angular_action = torch.tensor(
            np.array([x[2] for x in trajectory]), device=device, dtype=torch.float32
        )
        linear_log_prob_old = torch.tensor(
            np.array([x[3] for x in trajectory]), device=device, dtype=torch.float32
        )
        angular_log_prob_old = torch.tensor(
            np.array([x[4] for x in trajectory]), device=device, dtype=torch.float32
        )
        reward = torch.tensor(
            np.array([x[5] for x in trajectory]), device=device, dtype=torch.float32
        )
        next_state = torch.tensor(
            np.stack([x[6] for x in trajectory]), device=device, dtype=torch.float32
        )
        done = torch.tensor(
            np.array([x[7] for x in trajectory]), device=device, dtype=torch.int32
        )

        return state, linear_action, angular_action, linear_log_prob_old, angular_log_prob_old, reward, next_state, done
        
    def reset(self):
        self.buffer.clear()


class ReplayMemoryBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size


    def add_replay_memory(
        self, state, linear_action, angular_action, linear_log_prob_old, angular_log_prob_old, reward, next_state, done, advantage
    ):
        data = (state, linear_action, angular_action, linear_log_prob_old, angular_log_prob_old, reward, next_state, done, advantage)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    # 情報の確認必須
    def get_minibatch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = torch.stack([x[0] for x in data]).to(device)
        linear_action = torch.cat([x[1].unsqueeze(0) for x in data]).to(device)
        angular_action = torch.cat([x[2].unsqueeze(0) for x in data]).to(device)
        
        linear_log_prob_old = torch.cat([x[3].unsqueeze(0) for x in data]).to(device)
        angular_log_prob_old = torch.cat([x[4].unsqueeze(0) for x in data]).to(device)
        reward = torch.cat([x[5].unsqueeze(0) for x in data]).to(device)
        next_state = torch.stack([x[6] for x in data]).to(device)
        done = torch.cat([x[7].unsqueeze(0) for x in data]).to(device)
        advantage = torch.cat([x[8].unsqueeze(0) for x in data]).to(device)

        return state, linear_action, angular_action, linear_log_prob_old, angular_log_prob_old, reward, next_state, done, advantage
    def reset(self):
        self.buffer.clear()

class ActorNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3_linear_mean = nn.Linear(64, 1)
        self.fc3_angular_mean = nn.Linear(64, 1)

        self.log_std_linear = nn.Parameter(torch.zeros(1,1))
        self.log_std_angular = nn.Parameter(torch.zeros(1,1))

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        linear_mean = 0.2 * torch.sigmoid(self.fc3_linear_mean(x))
        angular_mean = torch.tanh(self.fc3_angular_mean(x))
        linear_std = self.log_std_linear.exp()
        angular_std = self.log_std_angular.exp()

        return linear_mean, linear_std, angular_mean, angular_std

class CriticNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
    
class PPOAgent:
    def __init__(self):
        # hyperparameters
        self.lr_actor = 5.5e-4
        self.lr_critic = 5.5e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.buffer_size = 1000
        self.batch_size = 64
        self.action_size = 2
        self.entropy_coefficient = 0.01  # エントロピー係数の定義

        self.actor = ActorNet(13).to(device)
        self.critic = CriticNet(13).to(device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.trajectory_buffer = TrajectoryBuffer()
        self.replay_memory_buffer = ReplayMemoryBuffer(self.buffer_size, self.batch_size)

        # log
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.reward_history = []
        self.losses_actors = []
        self.losses_critics = []
        self.advantage_history = []
        self.advantage_mean_history = []
        self.advantage_variance_history = []
        self.entropy_history = []
        self.linear_action_history = []
        self.linear_action_mean_history = []
        self.linear_action_variance_history = []
        self.linear_entropy_history = []
        self.angular_action_history = []
        self.angular_action_mean_history = []
        self.angular_action_variance_history = []
        self.angular_entropy_history = []
        self.mean_entropy_per_iteration = []
        self.linear_mean_entropy_per_iteration = []
        self.angular_mean_entropy_per_iteration = [] 
        self.clipped_ratio_history = []
        self.clipped_ratios_per_iteration = []


    def get_action(self, state):
        state = torch.tensor(state, device=device, dtype=torch.float32)
        linear_mean, linear_std, angular_mean, angular_std = self.actor(state)
        self.linear_action_mean_history.append(linear_mean.item())
        self.linear_action_variance_history.append(linear_std.item())
        self.angular_action_mean_history.append(angular_mean.item())
        self.angular_action_variance_history.append(angular_std.item())

        
        linear_distribution = torch.distributions.Normal(linear_mean, linear_std)
        angular_distribution = torch.distributions.Normal(angular_mean, angular_std)
        
        self.linear_entropy_history.append(linear_distribution.entropy().item())
        self.angular_entropy_history.append(angular_distribution.entropy().item())
       

        # ガウス分布から行動をサンプリング
        linear_action = linear_distribution.sample()
        angular_action = angular_distribution.sample()
        
        self.linear_action_history.append(linear_action.item())
        self.angular_action_history.append(angular_action.item())

        linear_log_prob_old = linear_distribution.log_prob(linear_action)
        angular_log_prob_old = angular_distribution.log_prob(angular_action)

        # ガウス分布でサンプリングした値はアクション範囲で収まる保障はないので、クリッピングする
        # linear_action_clipped = torch.clamp(linear_action, 0.0, 0.2)
        # angular_action_clipped = torch.clamp(angular_action, -1.0, 1.0)
        
        return linear_action.item(), angular_action.item(), linear_log_prob_old.item(), angular_log_prob_old.item()

    def compute_advantages_and_add_to_buffer(self):
        for trajectory in self.trajectory_buffer.buffer:
            states, linear_actions,angular_actions, linear_log_prob_old, angular_log_prob_old,rewards, next_states, dones = self.trajectory_buffer.get_per_trajectory(trajectory)
            state_values = self.critic(states).squeeze()
            next_state_values = self.critic(next_states).squeeze()

            advantages = []
            advantage = 0
            for t in reversed(range(0, len(rewards))):
                # エピソードの終了した場合の価値は0にする
                non_terminal = 1 - dones[t]

                # TD誤差を計算
                delta = rewards[t] + self.gamma * next_state_values[t] * non_terminal - state_values[t]

                # Advantage関数を計算
                advantage = (delta + self.gamma * self.gae_lambda * non_terminal * advantage).detach()
                advantages.append(advantage)
            advantages = advantages[::-1]

            for idx, state in enumerate(states):
                self.replay_memory_buffer.add_replay_memory(state, linear_actions[idx], angular_actions[idx], linear_log_prob_old[idx], angular_log_prob_old[idx],rewards[idx], next_states[idx], dones[idx], advantages[idx])
                self.advantage_history.append(advantages[idx].item())


    def compute_ppo_loss(self, state, linear_action, angular_action, linear_log_prob_old, angular_log_prob_old, advantage):
        linear_mean, linear_std, angular_mean, angular_std = self.actor(state)

        linear_distribution = torch.distributions.Normal(linear_mean, linear_std)
        angular_distribution = torch.distributions.Normal(angular_mean, angular_std)

        linear_log_prob_new = linear_distribution.log_prob(linear_action)
        angular_log_prob_new = angular_distribution.log_prob(angular_action)


        linear_ratio = torch.exp(linear_log_prob_new - linear_log_prob_old)
        angular_ratio = torch.exp(angular_log_prob_new - angular_log_prob_old)

        ratio = linear_ratio * angular_ratio

        clip = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        loss_clip = torch.min(ratio * advantage, clip * advantage)        
        loss_clip = -loss_clip.mean()

        clipped_and_min_selected_count = torch.sum((ratio * advantage) > (clip * advantage)).item()
        if clipped_and_min_selected_count > 0:
            self.clipped_ratio_history.append((clipped_and_min_selected_count) / self.batch_size)


        

        # linear_entropy = linear_distribution.entropy().mean()
        # angular_entropy = angular_distribution.entropy().mean()
        # entropy = linear_entropy + angular_entropy

        # エントロピー項を追加したもの
        # loss = loss_clip - self.entropy_coefficient * entropy
        # エントロピー項を追加しないもの
        loss = loss_clip
        return loss

    def compute_value_loss(self, state, reward, next_state, done) -> torch.Tensor:
        state_value = self.critic(state).squeeze()
        next_state_value = self.critic(next_state).squeeze()

        target = reward + (1 - done) * self.gamma * next_state_value

        loss_fn = nn.MSELoss()
        loss = loss_fn(state_value, target)
        return loss
    def compute_and_reset_advantage_stats(self):
        # アドバンテージの平均と分散を計算
        adv_mean = np.mean(self.advantage_history)
        adv_variance = np.var(self.advantage_history)

        # ヒストリーリストに追加
        self.advantage_mean_history.append(adv_mean)
        self.advantage_variance_history.append(adv_variance)

        # アドバンテージヒストリーをリセット
        self.advantage_history.clear()

    # 移動平均を計算
    def compute_moving_average(self, history, window_size=10):
        moving_average = []
        for i in range(len(history)):
            start_idx = max(0, i - window_size + 1)
            moving_average.append(np.mean(history[start_idx:i+1]))
        return moving_average    
    
    def plot_and_save(self, iteration):
        filename = f"./ppo_avoid/{self.start_time}_training_plot_{iteration}.png"

        plt.figure(figsize=(32, 18))

        # 累積報酬のプロット
        plt.subplot(3, 4, 1)
        reward_window_size = 10
        plt.plot(self.reward_history,label="reward", color='green')
        plt.plot(self.compute_moving_average(self.reward_history, window_size=reward_window_size),label="moving reward", color='red')
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True)

        # アクターの損失のプロット
        plt.subplot(3, 4, 2)
        plt.plot(self.losses_actors)
        plt.title("Actor Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid(True)

        # クリティックの損失のプロット
        plt.subplot(3, 4, 3)
        plt.plot(self.losses_critics)
        plt.title("Critic Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid(True)

        # アドバンテージの平均のプロット
        plt.subplot(3, 4, 4)
        plt.plot(self.advantage_mean_history)
        plt.title("Average Advantage")
        plt.xlabel("Iteration")
        plt.ylabel("Average Advantage")
        plt.grid(True)

        # アドバンテージの分散のプロット
        plt.subplot(3, 4, 5)
        plt.plot(self.advantage_variance_history)
        plt.title("Variance of Advantage")
        plt.xlabel("Iteration")
        plt.ylabel("Variance")
        plt.grid(True)

        # # 状態価値のプロット
        # plt.subplot(3, 3, 7)  # Adjust the subplot position
        # plt.plot(self.value_function_history)
        # plt.title("Value Function")
        # plt.xlabel("Step")
        # plt.ylabel("Value")
        # plt.grid(True)

        # イテレーションごとのエントロピーの平均のプロット
        plt.subplot(3, 4, 6)
        plt.plot(self.linear_mean_entropy_per_iteration, label="linear", color='lime')
        plt.plot(self.angular_mean_entropy_per_iteration, label="angular", color='red')
        plt.title("Mean Entropy per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Mean Entropy")
        plt.legend()
        plt.grid(True)

        #イテレーションごとのクリップされた個数のプロット
        plt.subplot(3, 4, 7)
        plt.plot(self.clipped_ratios_per_iteration)
        plt.title("Clipped Count per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Clipped Count")
        plt.grid(True)

        # linear_actionの平均とサンプリングされた値のプロット
        plt.subplot(3, 4, 8)
        plt.scatter(range(len(self.linear_action_history)),self.linear_action_history, label="sampled", color='lime')
        plt.plot(self.linear_action_mean_history, label="mean", color='red')
        plt.title("Linear Action")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        # liner_actionの分散のプロット
        plt.subplot(3, 4, 9)
        plt.plot(self.linear_action_variance_history)
        plt.title("Linear Action Variance")
        plt.xlabel("Step")
        plt.ylabel("Variance")
        plt.grid(True)

        # angular_actionの平均とサンプリングされた値のプロット
        plt.subplot(3, 4, 10)
        plt.scatter(range(len(self.angular_action_history)),self.angular_action_history, label="sampled", color='lime')
        plt.plot(self.angular_action_mean_history, label="mean", color='red')
        plt.title("Angular Action")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)


        # angular_actionの分散のプロット
        plt.subplot(3, 4, 11)
        plt.plot(self.angular_action_variance_history)
        plt.title("Angular Action Variance")
        plt.xlabel("Step")
        plt.ylabel("Variance")
        plt.grid(True)

        # ハイパーパラメータのプロット
        # plt.subplot(3, 3, 9)  # Adjust the subplot position to the bottom right
        # # plt.plot(self.clip_epsilon_history, label="Clip Epsilon", color='blue')
        # plt.plot(self.lr_actor_history, label="Actor Learning Rate", color='green')
        # plt.plot(self.lr_critic_history, label="Critic Learning Rate", color='red')
        # plt.legend()
        # plt.title("Hyperparameters over Iterations")
        # plt.xlabel("Iteration")
        # plt.ylabel("Value")
        # plt.grid(True)


        # 保存
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # def track_hyperparameters(self):
    #     self.clip_epsilon_history.append(self.clip_epsilon)
        # self.lr_actor_history.append(self.optimizer_actor.param_groups[0]['lr'])
        # self.lr_critic_history.append(self.optimizer_critic.param_groups[0]['lr'])

    def save_to_csv(self):
        # 最大のリストの長さを取得
        max_length = max(
            len(self.reward_history),
            # len(self.compute_moving_average(self.reward_history, window_size=10)),
            len(self.losses_actors),
            len(self.losses_critics),
            len(self.advantage_mean_history),
            len(self.advantage_variance_history),
            # len(self.value_function_history),
            len(self.linear_mean_entropy_per_iteration),
            len(self.angular_mean_entropy_per_iteration),
            len(self.clipped_ratios_per_iteration),
            len(self.linear_action_mean_history),
            len(self.linear_action_variance_history),
            len(self.linear_action_history),
            len(self.angular_action_mean_history),
            len(self.angular_action_variance_history),
            len(self.angular_action_history),
            
            # len(self.clip_epsilon_history),
            # len(self.lr_actor_history),
            # len(self.lr_critic_history)
        )

        # 各リストの長さをmax_lengthに合わせる
        reward_history = self.reward_history + [None] * (max_length - len(self.reward_history))
        # reward_moving_avg = self.compute_moving_average(self.reward_history, window_size=10) + [None] * (max_length - len(self.compute_moving_average(self.reward_history, window_size=10)))
        losses_actors = self.losses_actors + [None] * (max_length - len(self.losses_actors))
        losses_critics = self.losses_critics + [None] * (max_length - len(self.losses_critics))
        advantage_mean_history = self.advantage_mean_history + [None] * (max_length - len(self.advantage_mean_history))
        advantage_variance_history = self.advantage_variance_history + [None] * (max_length - len(self.advantage_variance_history))
        
        # value_function_history = self.value_function_history + [None] * (max_length - len(self.value_function_history))
        # mean_entropy_per_iteration = self.mean_entropy_per_iteration + [None] * (max_length - len(self.mean_entropy_per_iteration))
        linear_mean_entropy_per_iteration = self.linear_mean_entropy_per_iteration + [None] * (max_length - len(self.linear_mean_entropy_per_iteration))
        angular_mean_entropy_per_iteration = self.angular_mean_entropy_per_iteration + [None] * (max_length - len(self.angular_mean_entropy_per_iteration))
        clipped_ratios_per_iteration = self.clipped_ratios_per_iteration + [None] * (max_length - len(self.clipped_ratios_per_iteration))
        # clip_epsilon_history = self.clip_epsilon_history + [None] * (max_length - len(self.clip_epsilon_history))
        # lr_actor_history = self.lr_actor_history + [None] * (max_length - len(self.lr_actor_history))
        # lr_critic_history = self.lr_critic_history + [None] * (max_length - len(self.lr_critic_history))
        linear_action_mean_history = self.linear_action_mean_history + [None] * (max_length - len(self.linear_action_mean_history))
        linear_action_variance_history = self.linear_action_variance_history + [None] * (max_length - len(self.linear_action_variance_history))
        linear_action_history = self.linear_action_history + [None] * (max_length - len(self.linear_action_history))
        angular_action_mean_history = self.angular_action_mean_history + [None] * (max_length - len(self.angular_action_mean_history))
        angular_action_variance_history = self.angular_action_variance_history + [None] * (max_length - len(self.angular_action_variance_history))
        angular_action_history = self.angular_action_history + [None] * (max_length - len(self.angular_action_history))

        # データフレームの作成
        data = {
            'Episode Rewards': reward_history,
            # 'Reward Moving Average': reward_moving_avg,
            'Actor Loss': losses_actors,
            'Critic Loss': losses_critics,
            'Average Advantage': advantage_mean_history,
            'Variance of Advantage': advantage_variance_history,
            # 'Value Function': value_function_history,
            # 'Mean Entropy per Iteration': mean_entropy_per_iteration,
            'Linear Mean Entropy per Iteration': linear_mean_entropy_per_iteration,
            'Angular Mean Entropy per Iteration': angular_mean_entropy_per_iteration,
            'Clipped Count per Iteration': clipped_ratios_per_iteration,
            'Linear Action Mean': linear_action_mean_history,
            'Linear Action Variance': linear_action_variance_history,
            'Linear Action': linear_action_history,
            'Angular Action Mean': angular_action_mean_history,
            'Angular Action Variance': angular_action_variance_history,
            'Angular Action': angular_action_history,
            # 'Clip Epsilon': clip_epsilon_history,
            # 'Actor Learning Rate': lr_actor_history,
            # 'Critic Learning Rate': lr_critic_history,
        }
        df = pd.DataFrame(data)

        # CSVファイルに保存
        filename = f"./ppo_avoid/{self.start_time}_ppo_training_data.csv"
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    
    
    def save_and_exit(self,signal, frame):
        print("\nCtrl+C detected. Saving training data and exiting...")
        # ここで学習データをCSVに保存するメソッドを呼び出します。
        self.save_to_csv()
        
        sys.exit(0)

total_iterations = 3000000

env = gazebo_env.GazeboEnvironment("hero_0")
    
set_seeds(1023)

agent = PPOAgent()
save_interval = 10  # 10イテレーションごとにグラフを保存
save_model_interval = 50  # 100イテレーションごとにモデルを保存
signal.signal(signal.SIGINT, agent.save_and_exit)

for iteration in range(total_iterations):
    print("+++++++++++++++++++  iteration: {}++++++++++++++".format(iteration))

    step_count = 0

    #1回学習を行うあたりに必要な 2048ステップ分のトラジェクトリーを収集
    while True:
        state = env.reset(seed=random.randint(0,100000))
        # print("state: ", state)
        # state = env.reset(seed=1023)[0]
        done = False
        episode_data = []
        total_reward = 0

        # 1エピソード分のデータを収集
        while not done:
            step_count += 1
            linear_action, angular_action, linear_log_prob_old, angular_log_prob_old = agent.get_action(state)

            next_state, reward, done, baseline_reward = env.step([linear_action,angular_action])

            total_reward += baseline_reward

            # 報酬の正規化
            # reward = (reward + 30.0) / (30.0 + 1e-8)
            
            
            episode_data.append((state, linear_action,angular_action, linear_log_prob_old,angular_log_prob_old, reward, next_state, done))
            
            state = next_state
        print("total_reward: ", total_reward)
        if iteration != 0:
            agent.reward_history.append(total_reward)
        agent.trajectory_buffer.add_trajectory(episode_data)
        print("step_count: ", step_count)
        if step_count >= 128:
            break
        
    # 2048ステップ分のトラジェクトリーのアドバンテージを計算してバッファに追加
    agent.compute_advantages_and_add_to_buffer()
    
    # パラメータの更新
    epochs = 5
    for epoch in range(epochs):
        # ミニバッチを取得
        state, linear_action, angular_action, linear_log_prob_old, angular_log_prob_old, reward, next_state, done, advantage = agent.replay_memory_buffer.get_minibatch()

        agent.optimizer_actor.zero_grad()
        agent.optimizer_critic.zero_grad()
        actor_loss = agent.compute_ppo_loss(state, linear_action, angular_action, linear_log_prob_old, angular_log_prob_old, advantage)
        critic_loss = agent.compute_value_loss(state, reward, next_state, done)
        
        # LOG
        agent.losses_actors.append(actor_loss.item())
        agent.losses_critics.append(critic_loss.item())
        
        critic_loss.backward()
        actor_loss.backward()
                
        agent.optimizer_actor.step()
        agent.optimizer_critic.step()

    agent.trajectory_buffer.reset()
    agent.replay_memory_buffer.reset()
    agent.linear_mean_entropy_per_iteration.append(np.mean(agent.linear_entropy_history))
    agent.angular_mean_entropy_per_iteration.append(np.mean(agent.angular_entropy_history))
    if len(agent.clipped_ratio_history) > 0:
        agent.clipped_ratios_per_iteration.append(np.mean(agent.clipped_ratio_history))
    else:
        agent.clipped_ratios_per_iteration.append(0)
    agent.entropy_history.clear()
    agent.linear_entropy_history.clear()
    agent.angular_entropy_history.clear()
    agent.clipped_ratio_history.clear()

    agent.compute_and_reset_advantage_stats()
    if iteration % save_interval == 0:
        agent.plot_and_save(iteration)
        # agent.scheduler_actor.step()
        # agent.scheduler_critic.step()
    # agent.track_hyperparameters()
    if iteration % save_model_interval == 0:
        save_path = f"./ppo_avoid/{agent.start_time}_{iteration}_model.pt"

        torch.save({'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'actor_optimizer' : agent.optimizer_actor.state_dict(),
                    'critic_optimizer' : agent.optimizer_critic.state_dict(),
                    },save_path)



agent.save_to_csv()

save_path = f"./ppo_avoid/{agent.start_time}__complete_model.pt"

torch.save({'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'actor_optimizer' : agent.optimizer_actor.state_dict(),
            'critic_optimizer' : agent.optimizer_critic.state_dict(),
            },save_path)
