import gym
from stable_baselines3.common.callbacks import EvalCallback
import torch
import os
import sys
import optuna
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import env_checker
from stable_baselines3 import PPO, A2C
from vizdoomgym import VizDoomGym
from params import ppo_params, a2c_params
from utilities import TrainAndLoggingCallback

TOTAL_TIMESTEPS = int(sys.argv[2])
ALGORITHM = sys.argv[1]
TRIAL = 'deadly_corridor_' + ALGORITHM.lower() + '_trial'
LOG_DIR = 'logs/' + TRIAL

def agent(trial):
    params = None
    model = None
    RUN_NAME = 'trial_' + str(trial.number)
    env = VizDoomGym()
    env = Monitor(env)

    # load model
    if ALGORITHM == 'A2C':
        params = a2c_params(trial)
        model = A2C('CnnPolicy', env, tensorboard_log = LOG_DIR, verbose = 0, **params)
    elif ALGORITHM == 'PPO':
        params = ppo_params(trial)
        model = PPO('CnnPolicy', env, tensorboard_log = LOG_DIR, verbose = 0, **params)
    if model == None:
        pass
    else:
        logger = configure(LOG_DIR + '/' + RUN_NAME, ["stdout", "csv", "tensorboard"])
        model.set_logger(logger)
        callback = TrainAndLoggingCallback(100000, LOG_DIR + '/' + RUN_NAME)
        model.learn(total_timesteps = TOTAL_TIMESTEPS, callback = callback, log_interval = 20)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes = 10)
        return mean_reward

def run():
    study = optuna.create_study(direction = 'maximize', storage=f'sqlite:///trials_{ALGORITHM}_02.db', study_name = TRIAL)
    study.optimize(agent, n_trials = 100, gc_after_trial = True)

if __name__ == "__main__":
    run()

