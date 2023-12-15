import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd

import os

class Logger:
    def __init__(self, dir_name, n_actions):
        # log
        self.dir_name = dir_name
        ## 報酬ログ
        self.reward_history = []
        ## アクターとクリティックの損失ログ
        self.losses_actors = []
        self.losses_critics = []
        ## アドバンテージのログ
        self.advantage_history = []
        self.advantage_mean_history = []
        self.advantage_variance_history = []
        ## エントロピーのログ
        self.entropy_history = []
        self.entropy_mean_history = []
        ## 学習率のログ
        self.lr_actor_history = []
        self.lr_critic_history = []
        ## アクションの平均と標準偏差のログ
        self.action_means_history = [[] for _ in range(n_actions)]
        self.action_stds_history = [[] for _ in range(n_actions)]
        self.action_samples_history = [[] for _ in range(n_actions)]


    def plot_graph(self, iteration, n_actions):
        plt.figure(figsize=(30, 18))
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
        plt.legend(loc='upper left')
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
        plt.legend(loc='upper left')
        plt.grid(True)

        # 保存
        plt.tight_layout()
        filename = f"{self.dir_name}/training_data_{iteration}.png"
        plt.savefig(filename)
        plt.close()


        # n_actions の値に基づいて新しい Figure を作成
        n_rows = n_actions
        n_cols = 2  # 平均とサンプルのグラフと標準偏差のグラフのために2列

        plt.figure(figsize=(16, 6 * n_rows))  # Figure のサイズを調整
        for i in range(n_actions):
            # 平均とサンプルのグラフ
            plt.subplot(n_rows, n_cols, 2 * i + 1)
            plt.scatter(range(len(self.action_samples_history[i])), self.action_samples_history[i], label="sample", color="lime", alpha=0.2)
            plt.plot(self.action_means_history[i], label="mean", color="red")
            plt.title(f"Action {i} - Means and Samples")
            plt.xlabel("Step")
            plt.ylabel("Action Values")
            plt.grid(True)
            plt.legend(loc='upper left')

            # 標準偏差のグラフ
            plt.subplot(n_rows, n_cols, 2 * i + 2)
            plt.plot(self.action_stds_history[i], label="std", color="blue")
            plt.title(f"Action {i} - Standard Deviations")
            plt.xlabel("Step")
            plt.ylabel("Action Standard Deviation")
            plt.grid(True)
            plt.legend(loc='upper left')

        # グラフを表示
        plt.tight_layout()

        # 必要に応じてファイルに保存
        filename = f"{self.dir_name}/action_plots_{iteration}.png"
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
