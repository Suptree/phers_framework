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


# GPUが使える場合はGPUを使う


def set_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    total_iterations = 5000
    plot_interval = 10  # 10イテレーションごとにグラフを保存
    save_model_interval = 10  # 100イテレーションごとにモデルを保存
    num_env = 4
    seed_value = 1023
        
    set_seeds(seed_value)

    agent = PPOAgent(env_name="Parallel-Dynamic-Obstacle",
                    n_iteration=total_iterations, 
                    n_states=13, 
                    action_bounds=[-1, 1], 
                    n_actions=2, # 線形速度と角速度
                    actor_lr=3e-4, 
                    critic_lr=3e-4, 
                    gamma=0.99, 
                    gae_lambda=0.95, 
                    clip_epsilon=0.2, 
                    buffer_size=100000, 
                    batch_size=512,
                    epoch=3,
                    collect_step=256,
                    entropy_coefficient=0.01, 
                    device=device)

    agent.save_setting_config()
    # 途中から始める場合、以下のコメントアウトを外す
    # load_path = \
    # "/home/nishilab/catkin_ws/src/phers_framework/src/static_obstacle_stage_parallel/ppo_Parallel-Static-Obstacle/2024-01-15_18-27-34"
    # load_iteration = 50
    # agent.load_weights(load_path + "/" + f"{load_iteration}" + "_weights.pth")
    # agent.logger.load_and_merge_csv_data(load_path+"/training_history")

    signal.signal(signal.SIGINT, exit)

    for _ in range(agent.iteration, total_iterations):
        
        print("+++++++++++++++++++  iteration: {}++++++++++++++".format(agent.iteration))
        share_memory_actor = agent.create_actor_copy()
        share_memory_actor.share_memory()
        
        with mp.Pool(processes=num_env) as pool:
            # id と seed値 を渡す. idは偶数 (0, 2, 4)
            tasks = [(i, agent.iteration*num_env + i, share_memory_actor) for i in range(num_env)]
            results = pool.starmap(agent.data_collection, tasks)
            pool.close()
            pool.terminate()
        
        print("Parallel data collection finished")
        for result in results: 
            if result[0] is None:
                continue
            episode_data, rewards, baseline_rewards, entoripies, action_means, action_stds, action_samples, angle_to_goals, pheromone_average_value, pheromone_left_value, pheromone_right_value, step_counts = result
            if len(action_means) != 0:
                action_T_means = np.array(action_means).T.tolist()
                action_T_stds = np.array(action_stds).T.tolist()
                action_T_samples = np.array(action_samples).T.tolist()
                for i in range(agent.n_actions):
                    for action_mean in action_T_means[i]:
                        # print(action_mean)
                        agent.logger.action_means_history[i].append(action_mean)
                    for action_std in action_T_stds[i]:
                        agent.logger.action_stds_history[i].append(action_std)
                    for action_sample in action_T_samples[i]:
                        agent.logger.action_samples_history[i].append(action_sample)
                for angle_to_goal in angle_to_goals:
                    agent.logger.angle_to_goal_history.append(angle_to_goal)
                for pheromone_average in pheromone_average_value:
                    agent.logger.pheromone_average_history.append(pheromone_average)
                for pheromone_left in pheromone_left_value:
                    agent.logger.pheromone_left_history.append(pheromone_left)
                for pheromone_right in pheromone_right_value:
                    agent.logger.pheromone_right_history.append(pheromone_right)
            for episode in episode_data:
                agent.trajectory_buffer.add_trajectory(episode)
            for reward in rewards:
                agent.logger.reward_history.append(reward)
            for baseline_reward in baseline_rewards:
                agent.logger.baseline_reward_history.append(baseline_reward)
            for entropy in entoripies:
                agent.logger.entropy_history.append(entropy)
            for step_count in step_counts:
                agent.logger.step_count_history.append(step_count)
            # print("action_means", action_means[0])

        # アドバンテージの計算
        agent.compute_advantages_and_add_to_buffer()
            
        # パラメータの更新
        for epoch in range(agent.epoch):
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
        
        if agent.iteration % save_model_interval == 0:
            agent.save_weights(agent.iteration)
            agent.logger.save_csv()

        if agent.iteration % plot_interval == 0:
            agent.logger.plot_graph(agent.iteration, agent.n_actions)
            agent.logger.plot_action_graph(agent.iteration, agent.n_actions)
        # Agentのログをクリア
        agent.logger.clear_action_logs()

        # LOGGER
        agent.logger.calucurate_advantage_mean_and_variance()
        agent.logger.calucurate_entropy_mean()
        agent.iteration += 1

def exit(signal, frame):
    print("\nCtrl+C detected. Saving training data and exiting...")    
    sys.exit(0)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()