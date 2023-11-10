import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

import pandas as pd

import sys
import os


class Logger:
    def __init__(self, dir_name):
        # log
        self.dir_name = dir_name
        self.reward_history = []

        self.losses_actors = []
        self.losses_critics = []
        
        self.advantage_history = []
        self.advantage_mean_history = []
        self.advantage_variance_history = []
        
        self.entropy_history = []
        self.entropy_mean_history = []

        self.lr_actor_history = []
        self.lr_critic_history = []

        # self.action_means_history = []
        # self.action_stds_history = []
        # self.action_samples_history = []

        self.linear_action_means_history = []
        self.linear_action_stds_history = []
        self.limear_action_samples_history = []
        self.angular_action_means_history = []
        self.angular_action_stds_history = []
        self.angular_action_samples_history = []

    # def exit_save(self, signal, frame):
    #     print("\nCtrl+C detected. Saving training data and exiting...")
    #     # ここで学習データをCSVに保存するメソッドを呼び出す
    #     self.save_csv()
    #     sys.exit(0)

    def plot_graph(self, iteration):
        plt.figure(figsize=(32, 18))
        # 累積報酬のプロット
        plt.subplot(3, 4, 1)
        reward_window_size = 10
        plt.plot(self.reward_history, label="reward", color="green")
        plt.plot(
            self.compute_moving_average(
                self.reward_history, window_size=reward_window_size
            ),
            label="moving reward",
            color="red",
        )
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
        # エントロピーの平均のプロット
        plt.subplot(3, 4, 6)
        plt.plot(self.entropy_mean_history)
        plt.title("Entropy")
        plt.xlabel("Iteration")
        plt.ylabel("Entropy")
        plt.grid(True)
        # 学習率のプロット
        plt.subplot(3, 4, 7)
        plt.plot(self.lr_actor_history, label="actor", color="red")
        plt.plot(self.lr_critic_history, label="critic", color="blue")
        plt.title("Learning Rate")
        plt.xlabel("Iteration")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.grid(True)

        # アクションの平均とサンプリングのプロット
        plt.subplot(3, 4, 8)
        plt.scatter(range(len(self.limear_action_samples_history)),self.limear_action_samples_history, label="sample", color="lime",alpha=0.2)
        plt.plot(self.linear_action_means_history, label="mean", color="red")
        plt.title("Linear Means and Samples")
        plt.xlabel("Step")
        plt.ylabel("Action Values")
        plt.grid(True)
        plt.legend()

        # アクションの標準偏差のプロット
        plt.subplot(3, 4, 9)
        plt.plot(self.linear_action_stds_history, label="std", color="blue")
        plt.title("Linear Standard Deviations")
        plt.xlabel("Step")
        plt.ylabel("Action Standard Deviation")
        plt.grid(True)

        # アクションの平均とサンプリングのプロット
        plt.subplot(3, 4, 10)
        plt.scatter(range(len(self.angular_action_samples_history)),self.angular_action_samples_history, label="sample", color="lime",alpha=0.2)
        plt.plot(self.angular_action_means_history, label="mean", color="red")
        plt.title("Angular Means and Samples")
        plt.xlabel("Step")
        plt.ylabel("Action Values")
        plt.grid(True)
        plt.legend()

        # アクションの標準偏差のプロット
        plt.subplot(3, 4, 11)
        plt.plot(self.angular_action_stds_history, label="std", color="blue")
        plt.title("Angular Standard Deviations")
        plt.xlabel("Step")
        plt.ylabel("Action Standard Deviation")
        plt.grid(True)

        

        

        # 保存
        plt.tight_layout()
        filename = f"{self.dir_name}/reward_{iteration}.png"
        plt.savefig(filename)
        plt.close()


    def save_csv(self):
        filename = f"{self.dir_name}/data_history.csv"


        max_length = max(
            len(self.reward_history),
            # len(self.compute_moving_average(self.reward_history, window_size=10)),
            len(self.losses_actors),
            len(self.losses_critics),
            len(self.advantage_mean_history),
            len(self.advantage_variance_history),
            len(self.entropy_mean_history),
            len(self.lr_actor_history),
            len(self.lr_critic_history),
        )

        # 各リストの長さをmax_lengthに合わせる
        reward_history = self.reward_history + [None] * (max_length - len(self.reward_history))
        losses_actors = self.losses_actors + [None] * (max_length - len(self.losses_actors))
        losses_critics = self.losses_critics + [None] * (max_length - len(self.losses_critics))
        advantage_mean_history = self.advantage_mean_history + [None] * (max_length - len(self.advantage_mean_history))
        advantage_variance_history = self.advantage_variance_history + [None] * (max_length - len(self.advantage_variance_history))        
        lr_actor_history = self.lr_actor_history + [None] * (max_length - len(self.lr_actor_history))
        lr_critic_history = self.lr_critic_history + [None] * (max_length - len(self.lr_critic_history))
        entropy_mean_history = self.entropy_mean_history + [None] * (max_length - len(self.entropy_mean_history))

        # self.reward_history をデータフレームに変換
        new_df = pd.DataFrame({
            "Episode Rewards": reward_history,
            "Actor Loss": losses_actors,
            "Critic Loss": losses_critics,
            "Average Advantage": advantage_mean_history,
            "Variance of Advantage": advantage_variance_history,
            "Entropy": entropy_mean_history,
            "Actor Learning Rate": lr_actor_history,
            "Critic Learning Rate": lr_critic_history,
            })

        # 既存の CSV ファイルがある場合は読み込む
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)

            # 既存のデータと新しいデータを比較
            if not existing_df.equals(new_df):
                # 新しいデータのみを抽出
                new_data = new_df.iloc[len(existing_df):]

                # 既存のデータと新しいデータを結合
                updated_df = pd.concat([existing_df, new_data])
            else:
                updated_df = existing_df
        else:
            updated_df = new_df

        # CSV ファイルに保存
        updated_df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")


    # 移動平均を計算
    def compute_moving_average(self, history, window_size=10):
        moving_average = []
        for i in range(len(history)):
            start_idx = max(0, i - window_size + 1)
            moving_average.append(np.mean(history[start_idx : i + 1]))
        return moving_average
    
    def calucurate_advantage_mean_and_variance(self):
        # アドバンテージの平均と分散を計算
        adv_mean = np.mean(self.advantage_history)
        adv_variance = np.var(self.advantage_history)

        # ヒストリーリストに追加
        self.advantage_mean_history.append(adv_mean)
        self.advantage_variance_history.append(adv_variance)

    def calucurate_entropy_mean(self):
        # エントロピーの平均を計算
        entropy_mean = np.mean(self.entropy_history)

        # ヒストリーリストに追加
        self.entropy_mean_history.append(entropy_mean)


    def clear(self):
        self.calucurate_advantage_mean_and_variance()
        self.calucurate_entropy_mean()
        self.advantage_history = []
        self.entropy_history = []
    
    # def clear_per_episode(self):
