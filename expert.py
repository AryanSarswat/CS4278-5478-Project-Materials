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
    def __init__(self, est_map, env):
        self.LEFT = [0.6, np.pi]
        self.RIGHT = [0.9, -np.pi]
        self.STRAIGHT = [0.44, 0]
        self.env = env
        
        self.lane_follower = LaneFollower()

        self.map = map
        
        tile_size = (100, 100)
        
        self.processed_map_height = est_map.shape[0] // tile_size[0]
        self.processed_map_width = est_map.shape[1] // tile_size[1]
        
        processed_map = np.zeros((self.processed_map_height,  self.processed_map_width), dtype=np.uint8)
        
        for i in range(processed_map.shape[0]):
            for j in range(processed_map.shape[1]):
                if np.mean(est_map[i * tile_size[0] : (i + 1) * tile_size[0], j * tile_size[1] : (j + 1) * tile_size[1]]) > 0:
                    processed_map[i, j] = 1
                else:
                    processed_map[i, j] = 0
        
        self.processed_map = processed_map
        
    def heuristic(self, start, goal):
        dx = abs(start[0] - goal[0])
        dy = abs(start[1] - goal[1])
        return dx + dy
    
    def turn_left(self):
        for _ in range(10):
            self.env.step(self.LEFT)
            
    def turn_right(self):
        for _ in range(5):
            self.env.step(self.RIGHT)
            
    def lane_follow(self, obs):
        for _ in range(3):
            angle = self.lane_follower.steer(obs)
            obs, reward, done, info = self.env.step([0.3, angle])
        return info['cur_pos'], obs
    
    def predict_high_level_actions(self, curr_coord, goal_coord):
        start_node = Node(None, curr_coord)
        goal_node = Node(None, goal_coord)
        
        openSet = PriorityQueue()
        closedSet = set()
        gScore = dict()
        fScore = dict()
        
        gScore[start_node.position] = 0
        fScore[start_node.position] = self.heuristic(start_node, goal_node)
        
        openSet.push((fScore[start_node.position], start_node))
        numNodesExplored = 0
        path = []
        
        while len(openSet.queue) > 0:
            if numNodesExplored > (self.processed_map_height * self.processed_map_width):
                print("No path found, too many nodes explored")
                break
                
            current = openSet.pop()[1]
            closedSet.add(current.position)
            numNodesExplored += 1
        
            if current.isGoal(goal_node):
                print("Path found")
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                path = path[::-1]
                break
            
            # Check if neighbours are valid
            neighbours = [(current.position[0] + 1, current.position[1]), (current.position[0] - 1, current.position[1]), (current.position[0], current.position[1] + 1), (current.position[0], current.position[1] - 1)]
            
            children = []
            
            for i in range(len(neighbours)):
                if neighbours[i][0] < 0 or neighbours[i][0] >= self.processed_map_height or neighbours[i][1] < 0 or neighbours[i][1] >= self.processed_map_width:
                    continue
                if self.processed_map[neighbours[i][0], neighbours[i][1]] == 0:
                    continue
                else:
                    children.append(Node(current, neighbours[i]))

            for child in children:
                tentative_gScore = gScore[current.position] + 1
                
                if child.position not in gScore or tentative_gScore < gScore[child.position]:
                    gScore[child.position] = tentative_gScore
                    fScore[child.position] = tentative_gScore + self.heuristic(child, goal_node)
                    if child.position not in closedSet:
                        openSet.push((fScore[child.position], child))
        
        print("Number of nodes explored: ", numNodesExplored)
        if len(path) == 0:
            print("No path found")
            return []
        else:
            actions = []
            for i in range(len(path) - 1):
                current = path[i]
                next_node = path[i + 1]
                
                current_neighbours = [(current[0] + 1, current[1]), (current[0] - 1, current[1]), (current[0], current[1] + 1), (current[0], current[1] - 1)]
                num_drivable_neighbours = 0
                for i in current_neighbours:
                    if i[0] < 0 or i[0] >= self.processed_map_height or i[1] < 0 or i[1] >= self.processed_map_width:
                        continue
                    if self.processed_map[i[0], i[1]] == 1:
                        num_drivable_neighbours += 1
                
                if num_drivable_neighbours == 2:
                    actions.append("straight")
                else:
                    if current[2] == next_node[2]:
                        actions.append("straight")
                    elif (current[2] + 1 % 4) == next_node[2]:
                        actions.append("left")
                    elif (current[2] - 1 % 4) == next_node[2]:
                        actions.append("right")
                    else:
                        print("Invalid heading")
                
            return actions
    
    def get_init_state(self):
        map_img, goal, start_pos = self.env.get_task_info()
        while True:
            new_state, map_img = self.lane_follow(map_img)
            if info['cur_pos'] != start_pos:
                break
        
        dx = new_state[0] - start_pos[0]
        dy = new_state[1] - start_pos[1]
        heading = None
        
        if dx > 0:
            heading = 1
        elif dx < 0:
            heading = 2
        elif dy > 0:
            heading = 3
        elif dy < 0:
            heading = 0
            
        return (new_state[0], new_state[1], heading), (goal[0], goal[1], heading), map_img
        
    def run(self):
        start_pos, goal_pos, map_img = self.get_init_state()
        
        actions = self.predict_high_level_actions(start_pos, goal_pos)
        
        while len(actions) > 0:
            cur_action = actions.pop(0)
            
            if cur_action == "straight":
                self.lane_follow(map_img)
            elif cur_action == "left":
                self.turn_left()
            elif cur_action == "right":
                self.turn_right()
        
        print("Reached goal")
            
    
        
        