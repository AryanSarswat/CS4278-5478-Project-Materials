import argparse
import os
import sys

import cv2
import numpy as np
import pyglet
import torch
from gym_duckietown.envs import DuckietownEnv
from pyglet.window import key
from gym_duckietown.learning.imitation.iil_dagger.learner.neural_network_policy import NeuralNetworkPolicy
from gym_duckietown.learning.imitation.iil_dagger.model import Squeezenet

from model import Model, Model2

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', '-m', default="map1_0", type=str)
parser.add_argument('--seed', '-s', default=1, type=int)
parser.add_argument('--start-tile', '-st', default="0,1", type=str, help="two numbers separated by a comma")
parser.add_argument('--goal-tile', '-gt', default="15,1", type=str, help="two numbers separated by a comma")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_velocity = 0.7
input_shape = (120,160)
model = Squeezenet(num_outputs=2, max_velocity=max_velocity)
model_path = "iil_baseline/model.pt"

agent = NeuralNetworkPolicy(model=model,
        optimizer= None,
        dataset=None,
        storage_location="",
        input_shape=input_shape,
        max_velocity = max_velocity,
        model_path = model_path
    )

env = DuckietownEnv(
    domain_rand=False,
    max_steps=1500,
    map_name=args.map_name,
    seed=args.seed,
    user_tile_start=args.start_tile,
    goal_tile=args.goal_tile,
    randomize_maps_on_reset=False,
)
obs = env.reset()

NUM_STEPS = 1500

for step in range(NUM_STEPS):
    obs = obs[150:, :]
    obs = cv2.resize(obs, (128, 128))
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2YUV)
    obs = torch.from_numpy(obs).float().to(device)
    obs = obs.permute(2, 0, 1)
    obs = obs.unsqueeze(0)
    obs = obs / 255.0
    
    action = agent.predict(obs)
    obs, reward, done, info = env.step(action)
    
    curr_pos = info['curr_pos']
    
    print(f"current pose = {info['curr_pos']}, step count = {env.unwrapped.step_count}, step reward = {reward:.3f}")
    
    env.render()

env.close()