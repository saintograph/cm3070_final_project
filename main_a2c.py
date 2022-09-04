import gym
from stable_baselines3.common.callbacks import EvalCallback
import torch
import random
import time
import matplotlib.pyplot as plt
import os
import sys
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import env_checker
from stable_baselines3 import PPO
from vizdoomgym import VizDoomGym
from utilities import TrainAndLoggingCallback

total_timesteps = 5e5
algorithm = 'A2C'
config = './config/deadly_corridor_1.cfg' # start with difficulty of 1.

torch.cuda.empty_cache()

def agent_train():
    env = VizDoomGym(False, config, 'A2C')
    env = Monitor(env)
    CHECKPOINT_DIR = './train/train_deadly_corridor_a2c'
    LOG_DIR = './logs/log_deadly_corridor_a2c'
    policy_kwargs = dict(net_arch = [dict(pi = [64, 64], vf = [64, 64])], activation_fn = torch.nn.Tanh)

    model = A2C(
        'CnnPolicy',
        env,
        tensorboard_log = LOG_DIR, 
        verbose = 1,
        n_steps = 128,
        gamma = 0.99,
        learning_rate = 0.000001,
        gae_lambda = 0.99,
        max_grad_norm = 0.8,
    )
    """
    Uncomment lines below to enable baseline hyperparameters
    """
    # model = A2C(
    #     'CnnPolicy',
    #     env,
    #     tensorboard_log = LOG_DIR, 
    #     verbose = 1,
    #     n_steps = 128,
    #     learning_rate = 0.001,
    # )


    # model.load('./train/train_deadly_corridor_a2c_optimized_04/best_model_500000') # uncomment to load weights for transfer learnig
    logger = configure(LOG_DIR + '/' 'A2C_train', ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    model.set_env(env)
    callback = TrainAndLoggingCallback(check_freq = 10000, save_path = CHECKPOINT_DIR)
    model.learn(
        total_timesteps = total_timesteps,
        callback = callback,
    )

if __name__ == "__main__":
    agent_train()
