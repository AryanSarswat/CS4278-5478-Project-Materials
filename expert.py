import numpy as np
from LaneFollower import LaneFollower
import heapq

class PriorityQueue:
    def __init__(self):
        self.queue = []
        self._index = -1
    
    def push(self, item):
        heapq.heappush(self.queue, item)
    
    def pop(self):
        return heapq.heappop(self.queue)
    
    def __repr__(self):
        return self.queue.__repr__()       
    
    def __len__(self):
        return len(self.queue)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self._index += 1
        if self._index >= len(self.queue):
            self._index = -1
            raise StopIteration
        else:
            return self.queue[self._index]

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        
        
        self.g = 0
        self.h = 0
        self.f = 0
        
    def isGoal(self, goal):
        if self.position[0] == goal[0] and self.position[1] == goal[1]:
            return True
        else:
            return False
        
    def __repr__(self):
        return "{" + ",".join(str(x) for x in self.position) + "}"
        
    def __eq__(self, other):
        return self.position == other.position

class Expert:
    def __init__(self, env, high_level_path):
        self.LEFT = [0.6, np.pi]
        self.RIGHT = [0.9, -np.pi]
        self.STRAIGHT = [0.44, 0]
        self.env = env
        
        self.lane_follower = LaneFollower()
        
        self.actions = dict()
        
        # Load actions 
        f = open(high_level_path, 'r')
        temp = f.read().split("\n")
        f.close()
        
        if temp[-1] == "":
            temp.pop(-1)
        
        for i in temp:
            # Find index of second comma
            index = i.find(",", i.find(",") + 1)
            coords = i[:index]
            action = i[index + 1:].strip()
            x = int(coords[1:coords.find(",")])
            y = int(coords[coords.find(",") + 1:-1])
            coord = (x, y)
            self.actions[coord] = action
        
    def turn_left(self):
        for _ in range(10):
            obs, reward, done, info = self.env.step(self.LEFT)
        return info['cur_pos'], obs
            
    def turn_right(self):
        for _ in range(5):
            obs, reward, done, info = self.env.step(self.RIGHT)
        return info['cur_pos'], obs
            
    def straight(self, obs):
        for _ in range(3):
            angle = self.lane_follower.steer(obs)
            obs, reward, done, info = self.env.step([0.3, angle])
        return info['cur_pos'], obs


if __name__ == "__main__":
    t = Expert(None, "testcases/milestone1_paths/map4_0_seed4_start_3,3_goal_10,4.txt")
    
        
        