#!/usr/bin/env python3

import random
import numpy as np
from agent import PPOAgent
import torch
import matplotlib
matplotlib.use('Agg')
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
import signal
import sys
import parallel_gazebo_non_obstacle_env as gazebo_env
import csv  # CSVモジュールをインポート
# GPUが使える場合はGPUを使う
import datetime  # datetimeモジュールをインポート
import os

def set_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   #　例外処理
    if len(sys.argv) < 2:
        print("Usage: python3 non_obstacle_evaluation.py [model_path]")
        sys.exit(0) 

    # 引数からモデルのパスを取得
    model_path = sys.argv[1]

    device = torch.device("cpu")

    total_run = 20 # 実験の試行回数
    plot_interval = 10  # 10イテレーションごとにグラフを保存
    save_model_interval = 100  # 100イテレーションごとにモデルを保存
    num_env = 16
    seed_value = 1023
    # 実験結果を格納するリスト
    results = []
    set_seeds(seed_value)

    # 引数からモデルのパスを取得

    agent = PPOAgent(env_name="Parallel-Non-Obstacle",
                    n_iteration=total_run, 
                    n_states=4, 
                    action_bounds=[-1, 1], 
                    n_actions=2, # 線形速度と角速度
                    actor_lr=3e-4, 
                    critic_lr=3e-4, 
                    gamma=0.99, 
                    gae_lambda=0.95, 
                    clip_epsilon=0.2, 
                    buffer_size=100000, 
                    batch_size=1024,
                    collect_step=256,
                    entropy_coefficient=0.01, 
                    device=device)
    
    agent.load_weights(model_path)
    agent.set_to_eval_mode()
    env = gazebo_env.GazeboEnvironment(id=0)
    

    for run in range(total_run):
        print("+++++++++++++++++++  evaluation : {}++++++++++++++".format(run))

        step_count = 0

        state = env.reset(seed=random.randint(0,100000))
        done = False
        # 1エピソードを格納するリスト
        total_reward = 0
        total_steps = 0

        while not done:
            total_steps += 1
            state = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action_mean, _ = agent.actor(state)
            action = action_mean.cpu().numpy()

            next_state, reward, terminated, _ , info = env.step([action[0]*0.2,action[1]])

            total_reward += reward
            if terminated:
                done = 1

            state = next_state

        task_time = info["task_time"]
        done_category = info["done_category"] # 0: reach goal, 1: collision, 2: time up
        string_done_category = "reach goal" if done_category == 0 else "collision" if done_category == 1 else "time up"

        print("total steps: {}, task time: {:.3f}, total reward: {:.3f}, done category: {}".format(total_steps, task_time, total_reward, string_done_category))                
        # 結果をリストに追加
        results.append([run, total_steps, task_time, total_reward, done_category])
    # CSVファイルに結果を書き出す
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"./Evaluate-Non-Obstacle/{start_time}"
    # ディレクトリを作成
    os.makedirs(dir_name, exist_ok=True)
    filename = f'non_obstacle_results.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # ヘッダーを書き込む
        writer.writerow(['Run', 'Total Steps', 'Task Time', 'Total Reward', 'Done Category'])
        # データを書き込む
        writer.writerows(results)

def exit(signal, frame):
    print("\nCtrl+C detected. Saving training data and exiting...")    
    sys.exit(0)


if __name__ == '__main__':
    main()