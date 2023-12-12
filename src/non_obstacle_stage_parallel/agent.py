from model import Actor, Critic
from buffer import TrajectoryBuffer, ReplayMemoryBuffer
from logger import Logger
from datetime import datetime
import torch
import torch.nn as nn
import parallel_gazebo_non_obstacle_env as gazebo_env

import numpy as np
import torch.optim as optim
import random
from torch.multiprocessing import Manager
from torch.optim.lr_scheduler import LambdaLR
import os
import json
import time


class PPOAgent:
    def __init__(self, env_name, n_iteration, n_states, action_bounds, n_actions, actor_lr, critic_lr, gamma, gae_lambda, clip_epsilon, buffer_size, batch_size,collect_step, entropy_coefficient,device):
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
        self.collect_step = collect_step # 1回学習を行うあたりに必要なステップ数
        self.n_states = n_states # 状態の次元数
        self.action_bounds = action_bounds # 行動の最大値と最小値
        self.n_actions = n_actions # 行動の次元数
        self.entropy_coefficient = entropy_coefficient # エントロピー項の係数
        self.device = device # CPUかGPUか

        self.actor = Actor(n_states=self.n_states,n_actions=self.n_actions).to(self.device)
        self.critic = Critic(n_states=self.n_states).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=1e-5)

        self.trajectory_buffer = TrajectoryBuffer(device=self.device)
        self.replay_memory_buffer = ReplayMemoryBuffer(self.buffer_size, self.batch_size, self.device)

        self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=self.scheduler)
        self.critic_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=self.scheduler)

        self.logger = Logger(self.dir_name, self.n_actions)

    def get_action(self, id, state, local_actor):
        state = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            action_mean, action_std = local_actor(state)

        # ガウス分布を作成
        action_distribution = torch.distributions.Normal(action_mean, action_std)
        # shared_resources["entropy_history"].append(action_distribution.entropy().mean().item())
        logger_entropy = action_distribution.entropy().mean().item()
       
        # ガウス分布から行動をサンプリング
        action = action_distribution.sample()
        
        # ガウス分布でサンプリングときの確率密度の対数を計算
        log_prob_old = action_distribution.log_prob(action)

        # 線形速度の値を0~1の範囲に変換 : -1~1ではないのは、NNの出力層がReLUなので正の値を出力するため
        action[0] = torch.tanh(action[0])
        # 行動の確率密度の対数の計算
        log_prob_old[0] -= torch.log(1 - action[0].pow(2) + 1e-6)

        # LOGGER - 新しいリスト構造を使用
        # if id == 0:
        logger_action_mean = action_mean.cpu().numpy()
        logger_action_std = action_std.cpu().numpy()
        logger_action = action.cpu().numpy()
            # for i in range(self.n_actions):
            #     shared_resources["action_means_history"][i].append(action_mean[i].cpu().numpy())
            #     shared_resources["action_stds_history"][i].append(action_std[i].cpu().numpy())
            #     shared_resources["action_samples_history"][i].append(action[i].cpu().numpy())

        return action.cpu().numpy(), log_prob_old.cpu().numpy(), logger_entropy, logger_action_mean, logger_action_std, logger_action
    
    def data_collection(self, id ,seed, share_memory_actor):
        # 子プロセスで再度シード値を設定
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # 環境のインスタンスを作成
        env = gazebo_env.GazeboEnvironment(id=id)

        # データ収集のステップ数を初期化
        collect_step_count = 0

        # 一連のエピソードデータを格納するリスト
        trajectory = []

        # 1エピソード分のステップ数
        total_steps = 0

        # ログ用のリストを作成
        logger_reward = []
        logger_entropies = []
        logger_action_means = []
        logger_action_stds = []
        logger_actions = []
        try:
            while True:
                state = env.reset(seed=random.randint(0,100000))
                done = False
                # 1エピソードを格納するリスト
                episode_data = []
                total_reward = 0
                total_steps = 0

                while not done:
                    collect_step_count += 1
                    total_steps += 1
                    action, log_prob_old, logger_entropy, logger_action_mean, logger_action_std, logger_action = self.get_action(id, state, share_memory_actor)

                    next_state, reward, terminated, baseline_reward = env.step([0.2 * action[0],action[1]])

                    total_reward += baseline_reward
                    if terminated:
                        done = 1

                    episode_data.append((state, action, log_prob_old, reward, next_state, done))
                    state = next_state


                    logger_entropies.append(logger_entropy)
                    
                    logger_action_means.append(logger_action_mean)
                    # print("logger_action_mean", logger_action_mean)
                    # print("logger_action_means", logger_action_means)
                    logger_action_stds.append(logger_action_std)
                    logger_actions.append(logger_action)

                print(f"Process {id} Reward : {total_reward}, Step : {total_steps}")
                trajectory.append(episode_data)
                logger_reward.append(total_reward)
                
                if collect_step_count >= self.collect_step:
                    break
            print(f"Finishing : Process {id} finished collecting data")
            env.shutdown()
            print(f"Process {id} is Finished")
            return (trajectory, logger_reward, logger_entropies, logger_action_means, logger_action_stds, logger_actions)
        except Exception as e:
            print(f"Process {id} is Finished with error")
            print(e)
            env.shutdown()
            return (trajectory, logger_reward, logger_entropies, logger_action_means, logger_action_stds, logger_actions)
        
    def compute_advantages_and_add_to_buffer(self):
        for trajectory in self.trajectory_buffer.buffer:
            local_critic = self.create_critics_copy()
            states, actions, log_prob_olds, rewards, next_states, dones = self.trajectory_buffer.get_per_trajectory(trajectory)
            with torch.no_grad():
                state_values = local_critic(states).squeeze()
                next_state_values = local_critic(next_states).squeeze()

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
    def scheduler(self, step):
        return max(1.0 - float(step / self.n_iteration), 0)
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

    def create_actor_copy(self):
        # 新しい Actor インスタンスを作成し、現在のモデルの状態をコピー
        # actor_copy = Actor(n_states=self.n_states, n_actions=self.n_actions)
        actor_copy = Actor(n_states=self.n_states, n_actions=self.n_actions).to("cpu")
        actor_copy.load_state_dict(self.actor.state_dict())
        return actor_copy
    
    def create_critics_copy(self):
        # 新しい Critic インスタンスを作成し、現在のモデルの状態をコピー
        critic_copy = Critic(n_states=self.n_states)
        critic_copy.load_state_dict(self.critic.state_dict())
        return critic_copy