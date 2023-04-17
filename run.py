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
from intelligent_robots_project import LaneFollower as IntelligentRobotsLaneFollower  

  
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

def load_high_level_path(path):
    actions = dict()
    
    f = open(path, 'r')
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
        actions[coord] = action
    
    goal = temp[-1]
    index = goal.find(",", goal.find(",") + 1)
    coords = goal[:index]
    x = int(coords[1:coords.find(",")])
    y = int(coords[coords.find(",") + 1:-1])
    coord = (x, y)
    action = "stay"
    actions[coord] = action

    return actions


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
    high_level_path = "./testcases/milestone2_paths/" + high_level_path
    intentions = load_high_level_path(high_level_path)
    expert = Expert(high_level_path)
    
    ELaneFollower = IntelligentRobotsLaneFollower(intentions, map_img, goal)

    print("start tile:", start_pos, " goal tile:", goal)

    # Show the map image
    # White pixels are drivable and black pixels are not.
    # Blue pixels indicate lan center
    # Each tile has size 100 x 100 pixels
    # Tile (0, 0) locates at left top corner.
    map_img[goal[1] * 100: goal[1] * 100 + 100, goal[0] * 100: goal[0] * 100 + 100] = [0, 0, 255]
    map_img[start_pos[1] * 100: start_pos[1] * 100 + 100, start_pos[0] * 100: start_pos[0] * 100 + 100] = [0, 255, 0]
    
    cv2.imshow("map", map_img)
    cv2.imwrite('map5_0.jpg', map_img)
    cv2.waitKey(200)
    
    obs, reward, done, info = env.step([0, 0])
    
    actions_taken = []
    
    step = 0
    max_step = 20
    line_detected = False
    init_prev_angle = 0
    sweepDir = False
    needToSweep = True
    sweepSteps = 20
    currentSweepSteps = 0
    is_wrong_lane = True
    cum_rewards = 0
    episodes_lapsed = 0
    max_init_episode = 100
    # Initial lane following
    # while episodes_lapsed < max_init_episode:
    #     num_lines = expert.num_lines_detected(obs)
    #     if num_lines > 0:
    #         line_detected = True
    #     angle = expert.predict(info['curr_pos'], obs, is_buffer=True)[1]
    #     if angle == None: # action does not exist
    #         action = np.array([0.1,0])
    #     elif angle < 0:
    #         action = np.array([-0.05, angle * 0.8])
    #     elif angle > 0:
    #         action = np.array([-0.05, angle * 0.8])
    #     elif angle == 0 and init_prev_angle != 0:
    #         action = np.array([-0.1, 0.0])
    #     else:
    #         action = np.array([-0.01 if needToSweep else -0.1, -1.2 if sweepDir and needToSweep else 1.2 if needToSweep else 0.5 ])
    #         if needToSweep:
    #             currentSweepSteps += 1
    #             if currentSweepSteps > sweepSteps:
    #                 sweepSteps += 5
    #                 sweepDir = not sweepDir
    #                 currentSweepSteps = 0
        
    #     init_prev_angle = angle
    #     if num_lines > 0:
    #         needToSweep = False

    #     action = action if action[1] is not None else np.array([action[0], 0])
    #     obs, reward, done, info = env.step(action)
    #     # print(f"current pose = {info['curr_pos']}, step count = {env.unwrapped.step_count}, step reward = {reward:.3f}")
    #     cum_rewards += reward
    #     actions_taken.append(action)
    #     env.render()
        
    #     step += 1
    #     episodes_lapsed += 1
        
    #     # Give allowance if no lane detected
    #     if not line_detected:
    #         step -= 0.5
    #     if needToSweep or is_wrong_lane: # still cannot find, continue to sweep
    #         step -= 0.5
    
    # print("Done adjusting")

    print("Init complete")
    action = [0, 0]
    obs, reward, done, info = env.step(action)
    
    
    duckieReward = 0
    while info['curr_pos'] != goal and episodes_lapsed < 1500:
        curr_pos = info['curr_pos']
        action = ELaneFollower(obs, info, action)
        action = action if action[1] is not None else np.array([0.1, 0])
        obs, reward, done, info = env.step(action)
        # print(f"current pose = {info['curr_pos']}, step count = {env.unwrapped.step_count}, step reward = {reward:.3f}")
        cum_rewards += reward
        episodes_lapsed += 1
        actions_taken.append(action)
        duckieReward += reward
        env.render()
    
    print(f"Finished map {map_name}, reward: {duckieReward}")
    
    if done:
        np.savetxt(f'./controls/{map_name}_seed{seed}_start_{start_pos[0]},{start_pos[1]}_goal_{goal[0]},{goal[1]}.txt', actions_taken, delimiter=',')
    print(f"Reward is: {cum_rewards / episodes_lapsed}")
    env.close()
    return duckieReward

test_cases = json.load(open("./testcases/milestone2.json"))

# for test in MAP_4:
#     map_name = test
#     seed = test_cases[test]["seed"][0]
#     start = list2str(test_cases[test]["start"])
#     goal = list2str(test_cases[test]["goal"])
    
#     run_test(map_name, seed, start, goal, 1500)


map_name = "map2_1" #2_1, 2_2, 2_3, 3_1, 4_0, 4_1, 4_2, 5_0, 5_1, 5_3, 5_4
seed = test_cases[map_name]["seed"][0]
start = list2str(test_cases[map_name]["start"])
goal = list2str(test_cases[map_name]["goal"])

run_test(map_name, seed, start, goal, 1500)
    

print(f"Reward is {total}")