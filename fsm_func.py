"""
This file holds the Finite State Machine class for Kane
"""

class DoomStateMachine():
    def __init__(self, game = None):
        self.start = True
        self.game = game
        # manual actions available to the agent in this scenario
        self.actions = {
            'shoot': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            'turn_left': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, -0.6, 0],
            'turn_right': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.6, 0],
            'move_forward': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0, 0, 0],
            'stop': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
        self.game_variables = self.game.get_state().game_variables
        if self.game == None:
            raise ValueError('Game environment required.')
        
    def detect():
        pass
    # If there's enough ammo, shoot
    def shoot(self):
        ammo = self.game_variables[4]
        if (ammo < 2.0):
            self.abort_mission()
        self.game.make_action(self.actions['shoot'])

    def move(self, action):
        self.game.make_action(self.actions[action])

    # no action or movement
    def stop(self):
        self.game.make_action(self.actions['stop'])
    
    # reorientate self towards goal
    def reorientate(self):
        if round(self.game_variables[3]) <= 90 and round(self.game_variables[3]) >= 1:
            self.move('turn_right')
        elif round(self.game_variables[3]) >= 270 and round(self.game_variables[3]) <= 359:
            self.move('turn_left')
        elif round(self.game_variables[3]) == 360 or round(self.game_variables[3]) == 0:
            self.move('move_forward')
        else:
            self.stop()

    # low ammo, attempt to proceed straight to objective
    def abort_mission(self):  
        self.reorientate()
        self.move('move_forward')
    