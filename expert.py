import numpy as np
from gym_duckietown.simulator import NotInLane
import copy

class Expert:
    def __init__(self):
        self.action_space = np.array([
            [1, 0],
            [1, 0.5],
            [1, -0.5],
            [0, 0.5],
            [0, -0.5],
        ])
        
    def predict(self, env, lookahead=1):
        best_action=None
        best_reward=-1000000
        
        cur_env = copy.copy(env)
        for i in range(len(self.action_space)):
            action = self.action_space[i]
            reward = 0
            for j in range(lookahead):
                try:
                    obs, r, done, info = cur_env.step(action)
                except NotInLane:
                    r = -100
                reward += r
                if done:
                    break
            if reward > best_reward:
                best_reward = reward
                best_action = action
            cur_env = copy.copy(env)
        
        return best_action
        
        
        