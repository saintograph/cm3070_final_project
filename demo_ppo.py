
import time
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from vizdoomgym import VizDoomGym

# load weights
model = PPO.load('./weights/ppo_end')
env = VizDoomGym(render = True)
env = Monitor(env)
# evaluate model
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes = 10)

# run model for 20 episodes
for episode in range(20): 
    obs = env.reset()
    done = False
    total_reward = 0
    while not done: 
        action, _ = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        time.sleep(0.02)
        total_reward += rewards
    print('Total Reward for episode {} is {}'.format(episode, total_reward))
    time.sleep(2)