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

    total_iterations = 50000
    plot_interval = 10  # 10イテレーションごとにグラフを保存
    save_model_interval = 50  # 100イテレーションごとにモデルを保存
    num_env = 16
    seed_value = 1023
        
    set_seeds(seed_value)

    agent = PPOAgent(env_name="Parallel-Non-Obstacle",
                    n_iteration=total_iterations, 
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


    agent.save_hyperparameters()
    signal.signal(signal.SIGINT, exit)
    for iteration in range(total_iterations):
        print("+++++++++++++++++++  iteration: {}++++++++++++++".format(iteration))
        share_memory_actor = agent.create_actor_copy()
        share_memory_actor.share_memory()
        
        with mp.Pool(processes=num_env) as pool:
            tasks = [(i, seed_value+i+iteration, share_memory_actor) for i in range(num_env)]
            results = pool.starmap(agent.data_collection, tasks)
            pool.close()
            pool.terminate()
        
        print("Parallel data collection finished")
        for result in results: 
            episode_data, rewards, baseline_rewards,entoripies, action_means, action_stds, action_samples = result

            action_T_means = np.array(action_means).T.tolist()
            action_T_stds = np.array(action_stds).T.tolist()
            action_T_samples = np.array(action_samples).T.tolist()

            for episode in episode_data:
                agent.trajectory_buffer.add_trajectory(episode)
            for reward in rewards:
                agent.logger.reward_history.append(reward)
            for baseline_reward in baseline_rewards:
                agent.logger.baseline_reward_history.append(baseline_reward)
            for entropy in entoripies:
                agent.logger.entropy_history.append(entropy)
            # print("action_means", action_means[0])
            for i in range(agent.n_actions):
                for action_mean in action_T_means[i]:
                    # print(action_mean)
                    agent.logger.action_means_history[i].append(action_mean)
                for action_std in action_T_stds[i]:
                    agent.logger.action_stds_history[i].append(action_std)
                for action_sample in action_T_samples[i]:
                    agent.logger.action_samples_history[i].append(action_sample)

        # アドバンテージの計算
        agent.compute_advantages_and_add_to_buffer()
            
        # パラメータの更新
        epochs =3
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
            agent.logger.plot_graph(iteration, agent.n_actions)    
            agent.logger.save_csv()
def exit(signal, frame):
    print("\nCtrl+C detected. Saving training data and exiting...")    
    sys.exit(0)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()