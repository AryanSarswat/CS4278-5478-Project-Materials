import argparse

import numpy as np
from gym_duckietown.envs import DuckietownEnv
from pyglet.window import key
from a_star_planner import parse, view_parsed_map, plan, generate_paths
import json

# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

test_cases = json.load(open("./testcases/milestone2.json"))

for map_name in test_cases:
    seed = test_cases[map_name]["seed"][0]
    start = test_cases[map_name]["start"]
    goal = test_cases[map_name]["goal"]


    # simulator instantiation
    env = DuckietownEnv(
        domain_rand=False,
        max_steps=1500,
        map_name=map_name,
        seed=seed,
        user_tile_start=start,
        goal_tile=goal,
        randomize_maps_on_reset=False
    )

    # obs = env.reset() # WARNING: never call this function during testing
    # env.render()

    map_img, goal, start_pos = env.get_task_info()
    print("start tile:", start_pos, " goal tile:", goal)

    # main loop
    # Register a keyboard handler

    map_data = parse(env.map_data["tiles"])
    # print(view_parsed_map(map_data, start_pos, goal))
    planned_path = plan((start_pos[1], start_pos[0]), (goal[1], goal[0]), map_data, 0)
    # print(planned_path)
    filename = f"{map_name}_seed{seed}_start_{start[0]},{start[1]}_goal_{goal[0]},{goal[1]}.txt"
    generate_paths(planned_path, "./testcases/milestone2_paths/" + filename)

    env.close()

