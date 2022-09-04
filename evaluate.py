

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, A2C
from vizdoomgym import VizDoomGym
import pandas as pd
import time
from fsm import fsm

"""
This script runs each implementation for 100 episodes.
At the end of 100 episodes,
The mean of rewards gained during each of the 100 episodes
is calculated and saved
"""

algorithms = ['ppo', 'a2c', 'fsm']

def evaluate():
    env = VizDoomGym(render = True)
    env = Monitor(env)
    for algo in algorithms:
        model = None

        # if PPO or A2C, load best weights
        if algo == 'ppo':
            model = PPO.load('./train/train_deadly_corridor_ppo_optimized_05/best_model_500000')
        if algo == 'a2c':
            model = A2C.load('./train/train_deadly_corridor_a2c_optimized_03/best_model_500000')
        if model == None:
            pass
        else:
            mean_reward, _ = evaluate_policy(model, env, n_eval_episodes = 10)
            for episode in range(100): 
                obs = env.reset()
                done = False
                total_reward = 0
                while not done: 
                    action, _ = model.predict(obs)
                    obs, rewards, done, info = env.step(action)
                    total_reward += rewards
                # save rewards to CSV
                df = pd.read_csv('./testing_results/ppo_a2c_fsm.csv')
                df.loc[episode, algo] = total_reward
                df.to_csv('./testing_results/ppo_a2c_fsm.csv', index = False) 
                print('Total Reward for episode {} is {}'.format(episode, total_reward))
        if algo == 'fsm':
            for episode in range(100):
                df = pd.read_csv('./testing_results/ppo_a2c_fsm.csv')
                score = fsm()
                if score > 0:
                    df.loc[episode, 'fsm'] = score
                else:
                    df.loc[episode, 'fsm'] = 0
                df.to_csv('./testing_results/ppo_a2c_fsm.csv', index = False) 

if __name__ == "__main__":
    evaluate()
    df = pd.read_csv('./testing_results/ppo_a2c_fsm.csv')
    results_df = pd.read_csv('./testing_results/final_results_mean.csv')
    # calculate mean rewards for ever algorithm
    results_df.loc[0, 'rewards_mean'] = df['fsm'].mean()
    results_df.loc[1, 'rewards_mean'] = df['ppo'].mean()
    results_df.loc[2, 'rewards_mean'] = df['a2c'].mean()
    results_df.to_csv('./testing_results/final_results_mean.csv', index = False) 
