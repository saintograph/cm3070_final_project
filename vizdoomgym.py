import numpy as np
import cv2
from vizdoom import *
from gym import Env
from gym.spaces import Discrete, Box

class VizDoomGym(Env):
    def __init__(self, render = False, config = './config/deadly_corridor_1.cfg', algorithm = 'A2C'):
        super().__init__()
        self.game = DoomGame()
        self.game.load_config(config)
        self.algorithm = algorithm
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        self.game.set_available_game_variables([GameVariable.HEALTH, GameVariable.KILLCOUNT, GameVariable.SELECTED_WEAPON_AMMO, GameVariable.HITS_TAKEN, GameVariable.HITCOUNT]) 
        self.game.init()

        #initialize variables to hold game variables
        self.ammo = self.game.get_state().game_variables[0]
        self.kill_count = 0
        self.health = 100
        self.hits_taken = 0
        self.hit_count = 0

        self.observation_space = Box(low = 0, high = 255, shape = (100, 160, 1), dtype = np.uint8)
        
        """
        7 Available actions in the action space:
		MOVE_LEFT 
		MOVE_RIGHT 
		ATTACK 
		MOVE_FORWARD
		MOVE_BACKWARD
		TURN_LEFT
		TURN_RIGHT
        """
        self.action_space = Discrete(7)

    # stepping function between time-steps
    def step(self, action):
        reward = 0
        actions = np.identity(7, dtype = np.uint8)
        
        # calculate rewards for moving
        movement_reward = self.game.make_action(actions[action], 4) / 5
        
        if self.game.get_state(): 
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)

            game_variables = self.game.get_state().game_variables

            # rewards dictionary
            calculated_rewards = {
                'health': 0,
                'ammo': 0,
                'kills': 0,
                'hits_taken': 0,
                'hit_count': 0
            }

            health, kill_count, ammo, hits_taken, hit_count = game_variables

            # get hits taken
            hits_taken_delta = -hits_taken + self.hits_taken
            self.hits_taken = hits_taken
            # get remaining health
            health_delta = health - self.health
            self.health = health
            # get amount of kills
            kill_count_delta = kill_count - self.kill_count
            self.kill_count = kill_count
            # get remaining ammo
            ammo_delta = ammo - self.ammo
            self.ammo = ammo
            # get amount of hits on enemies
            hit_count_delta = hit_count - self.hit_count
            self.hit_count = hit_count

            """
            This is where reward shaping takes place.
            e.g. scores for taking hits can be multiplied to make the agent
            averse to getting hit
            """
            calculated_rewards['health'] = -5 if health_delta < 0 else 0
            calculated_rewards['hits_taken'] = hits_taken_delta * -0.33 if hits_taken_delta > 0 else 0
            calculated_rewards['ammo'] = 0 if ammo_delta == 0 else ammo_delta * 0.5
            calculated_rewards['kills'] = kill_count_delta * 150 if kill_count_delta > 0 else 0 # optimal is 150
            calculated_rewards['hit_count'] = hit_count_delta * 200 if hit_count_delta > 0 else 0

            reward = (calculated_rewards['hits_taken'] + calculated_rewards['ammo'] + calculated_rewards['kills'] + movement_reward) * 0.001
            info = ammo
        else: 
            state = np.zeros(self.observation_space.shape)
            info = 0 

        info = { "info": info }
        done = self.game.is_episode_finished()
        
        return state, reward, done, info 

    def close(self):
        self.game.close()
    
    # convert screen buffer from colour to grayscale
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation = cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))
        return state
    
    # reset game environment
    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state)

