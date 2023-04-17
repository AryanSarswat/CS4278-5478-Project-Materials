import numpy as np
from LaneFollower import LaneFollower

class Expert:
    def __init__(self, high_level_path):
        self.LEFT = [0.8, 1.5]
        self.RIGHT = [0.65, -2.2]
        self.STRAIGHT = [0.9, 0]
        self.FAST_STRAIGHT = [1, 0]
        
        self.buffer_length = 100
        self.idx = 0
        self.str_buffer = list(range(self.buffer_length))
        
        self.lane_follower = LaneFollower()
        self.reached = False
        self.actions = dict()
        
        # Load actions 
        f = open(high_level_path, 'r')
        temp = f.read().split("\n")
        f.close()
        
        if temp[-1] == "":
            temp.pop(-1)
        
        for i in range(len(temp[:-1])):
            # Find index of second comma
            curr = temp[i]
            next_ = temp[i + 1]
            
            index_c = curr.find(",", curr.find(",") + 1)
            index_n = next_.find(",", next_.find(",") + 1)

            coords = curr[:index_c]
            action = next_[index_n + 1:].strip()
            x = int(coords[1:coords.find(",")])
            y = int(coords[coords.find(",") + 1:-1])
            coord = (x, y)
            self.actions[coord] = action
        
        goal = temp[-1]
        index = goal.find(",", goal.find(",") + 1)
        coords = goal[:index]
        x = int(coords[1:coords.find(",")])
        y = int(coords[coords.find(",") + 1:-1])
        coord = (x, y)
        action = "stay"
        self.actions[coord] = action
        
    def num_lines_detected(self, obs):
        lanes = self.lane_follower.detect_lane(obs)[2]
        return len(lanes)
        
    def turn_left(self, obs):
        return self.LEFT
            
    def turn_right(self, obs):
        return self.RIGHT
            
    def go_straight(self, obs):
        angle = self.lane_follower.steer(obs)
        # print(angle)
        self.str_buffer[self.idx] = angle
        self.idx = (self.idx + 1) % self.buffer_length
        if angle == 0:
            if sum(self.str_buffer) <= 0.3:
                return self.FAST_STRAIGHT
            return self.STRAIGHT
        return np.array([0.3, angle]) # added rotation scale
    
    def predict(self, coord, obs, is_buffer=False):
        try:
            action = self.actions[coord]
        except KeyError:
            action = "forward"
        if action == "left":
            angle = self.lane_follower.steer(obs)
            if abs(angle) > 0.1 or is_buffer:
                return np.array([0.2, angle])
            else:
                return self.turn_left(obs)
        elif action == "right":
            angle = self.lane_follower.steer(obs)
            if abs(angle) > 0.1 or is_buffer:
                return np.array([0.2, angle])
            else:
                return self.turn_right(obs)
        elif action == "forward":
            return self.go_straight(obs)
        elif action == "stay":
            if self.reached:
                return np.array([0, 0])
            self.reached = True
            return np.array([1, 0])
        else:
            print("Error: invalid action")


if __name__ == "__main__":
    t = Expert("testcases/milestone1_paths/map4_0_seed4_start_3,3_goal_10,4.txt")
    
        
        