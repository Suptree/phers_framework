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

# GPUが使える場合はGPUを使う
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed_value, env):
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
        action = torch.tensor(
            np.array([x[1] for x in trajectory]), device=device, dtype=torch.float32
        )
        log_prob = torch.tensor(
            np.array([x[2] for x in trajectory]), device=device, dtype=torch.float32
        )
        reward = torch.tensor(
            np.array([x[3] for x in trajectory]), device=device, dtype=torch.float32
        )
        next_state = torch.tensor(
            np.stack([x[4] for x in trajectory]), device=device, dtype=torch.float32
        )
        done = torch.tensor(
            np.stack([x[5] for x in trajectory]), device=device, dtype=torch.int32
        )

        return state, action, log_prob, reward, next_state, done

    def reset(self):
        self.buffer.clear()


class ReplayMemoryBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size


    def add_replay_memory(
        self, state, action, log_prob_old, reward, next_state, done, advantage
    ):
        data = (state, action, log_prob_old, reward, next_state, done, advantage)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    # 情報の確認必須
    def get_minibatch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = torch.stack([x[0] for x in data]).to(device)
        action = torch.cat([x[1].unsqueeze(0) for x in data]).to(device)
        log_prob = torch.cat([x[2].unsqueeze(0) for x in data]).to(device)
        reward = torch.cat([x[3].unsqueeze(0) for x in data]).to(device)
        next_state = torch.stack([x[4] for x in data]).to(device)
        done = torch.cat([x[5].unsqueeze(0) for x in data]).to(device)
        advantage = torch.cat([x[6].unsqueeze(0) for x in data]).to(device)

        return state, action, log_prob, reward, next_state, done, advantage
    def reset(self):
        self.buffer.clear()

class ActorNet(nn.Module):
    def __init__(self, input_dim, action_size):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3_mean = nn.Linear(128, action_size)
        self.fc3_std = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = 2.0 * torch.tanh(self.fc3_mean(x))  # tanhの範囲[-1,1]を[-2,2]にスケール
        std = F.softplus(self.fc3_std(x)) + 1e-7      # stdが正になるようにsoftplusを適用
        return mean, std

class CriticNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class PPOAgent:
    def __init__(self):
        # hyperparameters
        self.lr_actor = 1e-4
        self.lr_critic = 1e-3
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.buffer_size = 10000
        self.batch_size = 128
        self.action_size = 1
        self.entropy_coefficient = 0.01  # エントロピー係数の定義

        self.actor = ActorNet(3, self.action_size).to(device)
        self.critic = CriticNet(3).to(device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.trajectory_buffer = TrajectoryBuffer()
        self.replay_memory_buffer = ReplayMemoryBuffer(self.buffer_size, self.batch_size)

    
    def get_action(self, state):
        state = torch.tensor(state, device=device, dtype=torch.float32)
        mean, std = self.actor(state)
        distribution = torch.distributions.Normal(mean, std)

        # ガウス分布から行動をサンプリング
        action = distribution.sample()
        log_prob_old = distribution.log_prob(action)
        # ガウス分布でサンプリングした値はアクション範囲で収まる保障はないので、クリッピングする
        action_clipped = torch.clamp(action, -2.0, 2.0)

        return action_clipped.item(), log_prob_old.item()


    def compute_advantages_and_add_to_buffer(self):
        for trajectory in self.trajectory_buffer.buffer:
            states, actions, log_probs_old, rewards, next_states, dones = self.trajectory_buffer.get_per_trajectory(trajectory)
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
                self.replay_memory_buffer.add_replay_memory(state, actions[idx], log_probs_old[idx], rewards[idx], next_states[idx], dones[idx], advantages[idx])


    def compute_ppo_loss(self, state, action, log_prob_old, advantage):
        mean, std = self.actor(state)

        distribution = torch.distributions.Normal(mean, std)
        log_prob_new = distribution.log_prob(action)

        ratio = torch.exp(log_prob_new - log_prob_old)
        clip = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        loss_clip = torch.min(ratio * advantage, clip * advantage)        
        loss_clip = -loss_clip.mean()

        entropy = distribution.entropy().mean()
        loss = loss_clip - self.entropy_coefficient * entropy
        return loss

    def compute_value_loss(self, state, reward, next_state, done) -> torch.Tensor:
        state_value = self.critic(state).squeeze()
        next_state_value = self.critic(next_state).squeeze()

        target = reward + (1 - done) * self.gamma * next_state_value

        loss_fn = nn.MSELoss()
        loss = loss_fn(state_value, target)
        return loss
    

total_iterations = 100

# env = gym.make('Pendulum-v1', render_mode="rgb_array")
env = gazebo_env.GazeboEnvironment("hero_0")

while(True):
    env.step([0.01,0])
    
# set_seeds(1023, env)

# agent = PPOAgent()

# for iteration in range(total_iterations):
#     print("+++++++++++++++++++  iteration: {}++++++++++++++".format(iteration))

#     step_count = 0
#     first_while = True

#     #1回学習を行うあたりに必要な 2048ステップ分のトラジェクトリーを収集
#     while True:
#         state = env.reset(seed=random.randint(0,100000))[0]
#         # state = env.reset(seed=1023)[0]
#         done = False
#         episode_data = []
#         total_reward = 0
#         frames = []

#         # 1エピソード分のデータを収集
#         while not done:
#             step_count += 1
#             action, log_prob_old = agent.get_action(state)
            
#             next_state, reward, done, done2, info = env.step([action])

#             total_reward += reward
#             reward = (reward + 16.2736044) / 16.2736044
            
#             if done or done2:
#                 done = 1
#             episode_data.append((state, action, log_prob_old, reward, next_state, done))
            
#             state = next_state
#         print("total_reward: ", total_reward)
#         agent.trajectory_buffer.add_trajectory(episode_data)

#         if step_count >= 900:
#             break
        
#     # 2048ステップ分のトラジェクトリーのアドバンテージを計算してバッファに追加
#     agent.compute_advantages_and_add_to_buffer()
    
#     # パラメータの更新
#     epochs = 5
#     for epoch in range(epochs):
#         # ミニバッチを取得
#         state, action, log_prob_old, reward, next_state, done, advantage = agent.replay_memory_buffer.get_minibatch()

#         agent.optimizer_actor.zero_grad()
#         agent.optimizer_critic.zero_grad()
#         actor_loss = agent.compute_ppo_loss(state, action, log_prob_old, advantage)
#         critic_loss = agent.compute_value_loss(state, reward, next_state, done)
        
#         critic_loss.backward()
#         actor_loss.backward()
                
#         agent.optimizer_actor.step()
#         agent.optimizer_critic.step()

#     agent.trajectory_buffer.reset()
#     agent.replay_memory_buffer.reset()
