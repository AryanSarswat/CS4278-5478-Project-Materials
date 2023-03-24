import argparse

import numpy as np
import pyglet
from gym_duckietown.envs import DuckietownEnv
from expert import Expert
from pyglet.window import key
import sys
import cv2
from LaneFollower import LaneFollower
from expert import Expert
import json
  
def str2bool(v):
    """
    Reads boolean value from a string
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def list2str(l):
    """
    Converts a list of strings to a single string
    """
    t = [str(i) for i in l]
    return ",".join(t)

test_cases = json.load(open("./testcases/milestone1.json"))

def run_test(map_name, seed, start_tile, goal_tile, max_steps):
    # simulator instantiation
    env = DuckietownEnv(
        domain_rand=False,
        max_steps=max_steps,
        map_name=map_name,
        seed=seed,
        user_tile_start=start_tile,
        goal_tile=goal_tile,
        randomize_maps_on_reset=False
    )
    
    env.render()
    map_img, goal, start_pos = env.get_task_info()

    high_level_path = map_name + "_seed" + str(seed) + "_start_" + start_tile + "_goal_" + goal_tile + ".txt"
    high_level_path = "./testcases/milestone1_paths/" + high_level_path
    expert = Expert(high_level_path)

    print("start tile:", start_pos, " goal tile:", goal)

    # Show the map image
    # White pixels are drivable and black pixels are not.
    # Blue pixels indicate lan center
    # Each tile has size 100 x 100 pixels
    # Tile (0, 0) locates at left top corner.
    cv2.imshow("map", map_img)
    cv2.waitKey(200)
    
    obs, reward, done, info = env.step([0, 0])
    
    actions_taken = []
    
    step = 0
    max_step = 25
    line_detected = False
    duckieReward = 0
    init_prev_angle = 0
    sweepDir = False
    needToSweep = True
    sweepSteps = 20
    currentSweepSteps = 0
    is_wrong_lane = True
    # Initial lane following
    while step < max_step:
        num_lines = expert.num_lines_detected(obs)
        if num_lines > 0:
            line_detected = True
        angle = expert.predict(info['curr_pos'], obs)[1]
        if angle < 0:
            action = np.array([-0.01, angle * 0.5])
        elif angle > 0:
            action = np.array([-0.01, angle * 0.5])
        else:
            action = np.array([-0.1, 0])

        obs, reward, done, info = env.step(action)
        duckieReward += reward
        actions_taken.append(action)
        env.render()
        
        step += 1
        
        # Give allowance if no lane detected
        if not line_detected:
            step -= 0.5
        if needToSweep or is_wrong_lane: # still cannot find, continue to sweep
            step -= 0.5
        
    # print("Done adjusting")

    while info['curr_pos'] != goal:
        curr_pos = info['curr_pos']
        action = expert.predict(curr_pos, obs)
        action = action if action[1] is not None else np.array([0.1, 0])
        obs, reward, done, info = env.step(action)
        actions_taken.append(action)
        duckieReward += reward
        env.render()
    
    print(f"Finished map {map_name}, reward: {duckieReward}")
    
    if done:
        np.savetxt(f'./controls/{map_name}_seed{seed}_start_{start_pos[0]},{start_pos[1]}_goal_{goal[0]},{goal[1]}.txt', actions_taken, delimiter=',')
        
    env.close()
    return duckieReward


MAP_1 = []
MAP_2 = []
MAP_3 = []
MAP_4 = []
MAP_5 = []

all_maps = [MAP_1, MAP_2, MAP_3, MAP_4, MAP_5]
all_rewards = [[], [], [], [], []]
for test in test_cases:
    map_name = test
    if "map1" in map_name:
        MAP_1.append(test)
    elif "map2" in map_name:
        MAP_2.append(test)
    elif "map3" in map_name:
        MAP_3.append(test)
    elif "map4" in map_name:
        MAP_4.append(test)
    elif "map5" in map_name:
        MAP_5.append(test)


# for test in MAP_5:
#     map_name = test
#     seed = test_cases[test]["seed"][0]
#     start = list2str(test_cases[test]["start"])
#     goal = list2str(test_cases[test]["goal"])
    
#     run_test(map_name, seed, start, goal, 1500)


map_name = "map5_4"
seed = test_cases[map_name]["seed"][0]
start = list2str(test_cases[map_name]["start"])
goal = list2str(test_cases[map_name]["goal"])

run_test(map_name, seed, start, goal, 1500)
    

print(f"Reward is {total}")