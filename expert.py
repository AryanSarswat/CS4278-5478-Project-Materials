import numpy as np
from LaneFollower import LaneFollower


class Expert:
    def __init__(self, est_map, env):
        self.LEFT = [0.6, np.pi]
        self.RIGHT = [0.9, -np.pi]
        self.STRAIGHT = [0.44, 0]
        self.env = env
        
        lane_follower = LaneFollower()

        self.map = map
        
        tile_size = (100, 100)
        
        processed_map = np.zeros((est_map.shape[0] // tile_size[0], est_map.shape[1] // tile_size[1]), dtype=np.uint8)
        
        for i in range(processed_map.shape[0]):
            for j in range(processed_map.shape[1]):
                if np.mean(est_map[i * tile_size[0] : (i + 1) * tile_size[0], j * tile_size[1] : (j + 1) * tile_size[1]]) > 0:
                    processed_map[i, j] = 1
                else:
                    processed_map[i, j] = 0
        
        self.processed_map = processed_map
    
    
    def turn_left(self):
        for _ in range(10):
            self.env.step(self.LEFT)
            
    def turn_right(self):
        for _ in range(5):
            self.env.step(self.RIGHT)
            
    def lane_follow(self):
        for _ in range(10):
            self.env.step(self.STRAIGHT)
    
    def predict_high_level_action(self, curr_coord, goal_coord):
        pass
        
        
        