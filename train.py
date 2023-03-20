import argparse
import os
import sys

import cv2
import numpy as np
import pyglet
import torch
import torch.nn as nn
import torch.optim as optim
from gym_duckietown.envs import DuckietownEnv
from pyglet.window import key
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import Model, Model2

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label

class PurePursuitExpert:
    def __init__(self, env, ref_velocity=0.2, position_threshold=0.1,
                 following_distance=0.4, max_iterations=1000):
        self.env = env.unwrapped
        self.following_distance = following_distance
        self.max_iterations = max_iterations
        self.ref_velocity = ref_velocity
        self.position_threshold = position_threshold

    def predict(self, observation):  # we don't really care about the observation for this implementation
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)

        iterations = 0
        lookup_distance = self.following_distance
        curve_point = None
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, _ = self.env.closest_curve_point(follow_point, self.env.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos
        point_vec /= np.linalg.norm(point_vec)

        dot = np.dot(self.env.get_right_vec(), point_vec)
        steering = 1 * -dot

        return self.ref_velocity, steering

# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', '-m', default="map5_0", type=str)
parser.add_argument('--seed', '-s', default=1, type=int)
parser.add_argument('--start-tile', '-st', default="10,4", type=str, help="two numbers separated by a comma")
parser.add_argument('--goal-tile', '-gt', default="2,9", type=str, help="two numbers separated by a comma")

args = parser.parse_args()

NUM_EPISODES = 100
NUM_STEPS = 500
NUM_EPOCHS = 20
BATCH_SIZE = 256

env = DuckietownEnv(
    domain_rand=True,
    max_steps=150000,
    map_name="zig_zag",
    seed=args.seed,
    user_tile_start=None,
    goal_tile=None,
    randomize_maps_on_reset=True,
    distortion=True,
)

obs = env.reset()
obs_shape = obs.shape

expert = PurePursuitExpert(env) 

observations = []
actions = []

key_handler = key.KeyStateHandler()


for episode in tqdm(range(NUM_EPISODES)):
    obs = env.reset()
    for step in range(NUM_STEPS):
        if np.random.rand() < 0.25:
            actions = [[0.2, 0], [0.2, 1], [0.2, -1], [0.2, -0.5], [0.2, 0.5]]
            action = actions[np.random.randint(0, len(actions))]
            env.step(action)
        else:
            action = expert.predict(obs)
            obs, reward, done, info = env.step(action)
            # Crop top of the image
            obs = obs[150:, :]
            obs = cv2.resize(obs, (128, 128))
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2YUV)
            cv2.imshow("Observation", obs)
            obs = obs / 255
            obs = obs.transpose(2, 0, 1)
            observations.append(obs)
            actions.append(action)
            env.render()
            if done:
                break

print("Number of observations: ", len(observations))
observations = np.array(observations, dtype=np.float32)
actions = np.array(actions, dtype=np.float32)

dataset = CustomDataset(observations, actions)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

max_angle = np.max(np.abs(actions[:, 1]))
min_angle = np.min(np.abs(actions[:, 1]))
abs_angle = round(np.max([max_angle, abs(min_angle)]))
print("Max abs angle: ", abs_angle)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model()
model.train().to(device)

optimizer = optim.AdamW(model.parameters(),lr=0.0001, weight_decay=0.01)
criteria = nn.SmoothL1Loss()

avg_loss = 0

NUM_BATCHES = len(observations) // BATCH_SIZE

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=NUM_BATCHES):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criteria(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch: {epoch}, Epoch Loss: {epoch_loss / NUM_BATCHES}")


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
env.render()

torch.save(model.state_dict(), "imitation/pytorch/models/zig_zag.pt")
    

    
    
    

        