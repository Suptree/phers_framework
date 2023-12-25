from model import Actor, Critic
from buffer import TrajectoryBuffer, ReplayMemoryBuffer
from logger import Logger
from datetime import datetime

import torch
import torch.nn as nn

import torch.optim as optim

from torch.optim.lr_scheduler import LambdaLR
import os
import json


class PPOAgent:
    def __init__(self, env_name, n_iteration, n_states, action_bounds, n_actions, actor_lr, critic_lr, gamma, gae_lambda, clip_epsilon, buffer_size, batch_size, entropy_coefficient,device):
        self.env_name = env_name
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir_name = f"./ppo_{self.env_name}/{self.start_time}"
        # ディレクトリを作成
        os.makedirs(self.dir_name, exist_ok=True)

        # hyperparameters
        self.n_iteration = n_iteration # イテレーション数
        self.actor_lr = actor_lr # アクターの学習率
        self.critic_lr = critic_lr # クリティックの学習率
        self.gamma = gamma # 割引率
        self.gae_lambda = gae_lambda # GAEのλ
        self.clip_epsilon = clip_epsilon # クリッピングするときのε
        self.buffer_size = buffer_size # バッファサイズ
        self.batch_size = batch_size # バッチサイズ
        self.n_states = n_states # 状態の次元数
        print("n_states: {}".format(self.n_states))
        self.action_bounds = action_bounds # 行動の最大値と最小値
        self.n_actions = n_actions # 行動の次元数
        self.entropy_coefficient = entropy_coefficient # エントロピー項の係数
        self.device = device # CPUかGPUか

        self.actor = Actor(n_states=self.n_states,n_actions=self.n_actions).to(self.device)
        print("actor: {}".format(self.actor))
        self.critic = Critic(n_states=self.n_states).to(self.device)
        print("critic: {}".format(self.critic))

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=1e-5)

        self.trajectory_buffer = TrajectoryBuffer(device=self.device)
        self.replay_memory_buffer = ReplayMemoryBuffer(self.buffer_size, self.batch_size, self.device)

        self.scheduler = lambda step: max(1.0 - float(step / self.n_iteration), 0)

        self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=self.scheduler)
        self.critic_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=self.scheduler)
        
        self.logger = Logger(self.dir_name)

    def get_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            action_mean, action_std = self.actor(state)

        # ガウス分布を作成
        action_distribution = torch.distributions.Normal(action_mean, action_std)
        self.logger.entropy_history.append(action_distribution.entropy().mean().item())
       
        # ガウス分布から行動をサンプリング
        action = action_distribution.sample()
        
        # ガウス分布でサンプリングときの確率密度の対数を計算
        log_prob_old = action_distribution.log_prob(action)

        # action = torch.tanh(action)

        # log_prob_old -= torch.log(1 - action.pow(2) + 1e-6)

        sample_action = min(max(action[0].item(), 0.0), 0.2)
        sample_angular_action = min(max(action[1].item(), -1.0), 1.0)
        
        # LOGGER
        self.logger.linear_action_means_history.append(action_mean[0].cpu().numpy())
        self.logger.linear_action_stds_history.append(action_std[0].cpu().numpy())
        self.logger.limear_action_samples_history.append(sample_action)
        self.logger.angular_action_means_history.append(action_mean[1].cpu().numpy())
        self.logger.angular_action_stds_history.append(action_std[1].cpu().numpy())
        self.logger.angular_action_samples_history.append(sample_angular_action)

        return action.cpu().numpy(), log_prob_old.cpu().numpy()

    def compute_advantages_and_add_to_buffer(self):
        for trajectory in self.trajectory_buffer.buffer:
            
            states, actions, log_prob_olds, rewards, next_states, dones = self.trajectory_buffer.get_per_trajectory(trajectory)
            with torch.no_grad():
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
                self.replay_memory_buffer.add_replay_memory(state, actions[idx], log_prob_olds[idx],rewards[idx], next_states[idx], dones[idx], advantages[idx])
                self.logger.advantage_history.append(advantages[idx].item())

    def compute_ppo_loss(self, state, action, log_prob_old, advantage):
        advantage.unsqueeze_(-1)
        action_mean, action_std = self.actor(state)
        action_distribution = torch.distributions.Normal(action_mean, action_std)
        log_prob_new = action_distribution.log_prob(action)
        ratio = torch.exp(log_prob_new - log_prob_old)
        unprocessed_loss = ratio * advantage
        clipped_loss = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
        loss =  -torch.min(unprocessed_loss, clipped_loss).mean() - self.entropy_coefficient * action_distribution.entropy().mean()
        return loss

    def compute_value_loss(self, state, reward, next_state, done):
        state_value = self.critic(state)
        next_state_value = self.critic(next_state)

        target = reward + (1 - done) * self.gamma * next_state_value
        loss_fn = nn.MSELoss()
        loss = loss_fn(state_value, target)
        return loss
    
    def optimize(self, actor_loss, critic_loss):
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def schedule_lr(self):
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        self.logger.lr_actor_history.append(self.actor_scheduler.get_last_lr()[0])
        self.logger.lr_critic_history.append(self.critic_scheduler.get_last_lr()[0])

    def save_weights(self, iteration):
        filepath = os.path.join(self.dir_name, f'{iteration}_weights.pth')

        torch.save({"current_policy_state_dict": self.actor.state_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                    "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                    "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                    "actor_scheduler_state_dict": self.actor_scheduler.state_dict(),
                    "critic_scheduler_state_dict": self.critic_scheduler.state_dict(),
                    "iteration": iteration,
                    # "state_rms_mean": state_rms.mean,
                    # "state_rms_var": state_rms.var,
                    # "state_rms_count": state_rms.count,
                    },filepath)
        
    def save_hyperparameters(self):
        # ハイパーパラメータの辞書を作成
        
        hyperparameters = {
            'n_iteration': self.n_iteration,
            'actor_lr': float(self.actor_lr),
            'critic_lr': float(self.critic_lr),
            'gamma': float(self.gamma),
            'gae_lambda': float(self.gae_lambda),
            'clip_epsilon': float(self.clip_epsilon),
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'n_states': self.n_states,
            'action_bounds': [float(x) for x in self.action_bounds],
            'n_actions': self.n_actions,
            'entropy_coefficient': float(self.entropy_coefficient),
            'device': str(self.device)
        }
        # JSON ファイルに保存
        filepath = os.path.join(self.dir_name, 'hyperparameters.json')
        with open(filepath, 'w') as json_file:
            json.dump(hyperparameters, json_file, indent=4)

        print(f"Hyperparameters saved to {filepath}")

    
    def load_weights(self):
        checkpoint = torch.load(self.env_name + "_weights.pth")
        self.actor.load_state_dict(checkpoint["current_policy_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.actor_scheduler.load_state_dict(checkpoint["actor_scheduler_state_dict"])
        self.critic_scheduler.load_state_dict(checkpoint["critic_scheduler_state_dict"])
        iteration = checkpoint["iteration"]
        state_rms_mean = checkpoint["state_rms_mean"]
        state_rms_var = checkpoint["state_rms_var"]
        return iteration, state_rms_mean, state_rms_var

    def set_to_eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def set_to_train_mode(self):
        self.actor.train()
        self.critic.train()