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

class PurePursuitExpert:
    def __init__(self, env, ref_velocity=0.8, position_threshold=0.04,
                 following_distance=0.3, max_iterations=1000):
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
        steering = 10 * -dot

        return self.ref_velocity, steering
    

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

# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', '-m', default="map4_0", type=str)
parser.add_argument('--seed', '-s', default=4, type=int)
parser.add_argument('--start-tile', '-st', default="3,3", type=str, help="two numbers separated by a comma")
parser.add_argument('--goal-tile', '-gt', default="10,4", type=str, help="two numbers separated by a comma")
parser.add_argument('--control_path', default='./map4_0_seed2_start_1,13_goal_3,3.txt', type=str,
                    help="the control file to run")
parser.add_argument('--manual', default=False, type=str2bool, help="whether to manually control the robot")
args = parser.parse_args()

args.manual = True

# simulator instantiation
env = DuckietownEnv(
    domain_rand=False,
    max_steps=3000,
    map_name=args.map_name,
    seed=args.seed,
    user_tile_start=args.start_tile,
    goal_tile=None,
    randomize_maps_on_reset=False
)

# obs = env.reset() # WARNING: never call this function during testing
env.render()

expert = PurePursuitExpert(env)
expert2 = LaneFollower()
map_img, goal, start_pos = env.get_task_info()

high_level_path = args.map_name + "_seed" + str(args.seed) + "_start_" + args.start_tile + "_goal_" + args.goal_tile + ".txt"
high_level_path = "./testcases/milestone1_paths/" + high_level_path
expert_map = Expert(high_level_path)

print("start tile:", start_pos, " goal tile:", goal)

# Show the map image
# White pixels are drivable and black pixels are not.
# Blue pixels indicate lan center
# Each tile has size 100 x 100 pixels
# Tile (0, 0) locates at left top corner.
cv2.imshow("map", map_img)
cv2.waitKey(200)



# save map (example)
# cv2.imwrite(env.map_name + ".png", env.get_occupancy_grid(env.map_data))

obs, reward, done, info = env.step([1, 1])


# cv2.namedWindow("left_speed")
# cv2.namedWindow("right_speed")
# cv2.namedWindow("left_angle")
# cv2.namedWindow("right_angle")

# class Action:
#     def __init__(self):
#         self.left_speed = 0
#         self.right_speed = 0
#         self.left_angle = 0
#         self.right_angle = 0
        
#     def update_left_speed(self, val):
#         self.left_speed = val * 1.2 / 100
    
#     def update_right_speed(self, val):
#         self.right_speed = val * 1.2 / 100
        
#     def update_left_angle(self, val):
#         self.left_angle = val * np.pi / 100
    
#     def update_right_angle(self, val):
#         self.right_angle = val * -np.pi / 100
        
#     def get_left_action(self):
#         return np.array([self.left_speed, self.left_angle])

#     def get_right_action(self):
#         return np.array([self.right_speed, self.right_angle])

# actions_table = Action()

# cv2.createTrackbar("left_speed", "left_speed", 0, 100, actions_table.update_left_speed)
# cv2.createTrackbar("right_speed", "right_speed", 0, 100, actions_table.update_right_speed)
# cv2.createTrackbar("left_angle", "left_angle", 0, 100, actions_table.update_left_angle)
# cv2.createTrackbar("right_angle", "right_angle", 0, 100, actions_table.update_right_angle)

# main loop
if args.manual:
    # Register a keyboard handler
    key_handler = key.KeyStateHandler()
    env.unwrapped.window.push_handlers(key_handler)
    index = 0
    def update(dt):
        obs, reward, done, info = env.step([0, 0])
        if done:
            return
        action = np.array([0.0, 0.0])

        if key_handler[key.UP]:
            action = np.array([0.44, 0.0])
        elif key_handler[key.DOWN]:
            action = np.array([-0.44, 0])
        elif key_handler[key.LEFT]:
            action = np.array([0.44, 0.44])
        elif key_handler[key.RIGHT]:
            action = np.array([0.44, -0.44])
        elif key_handler[key.SPACE]:
            action = np.array([0, 0])
        else:
        #     #action = expert.predict(env)
            coord = info['curr_pos']
            action = expert_map.predict(coord, obs)
        #     #print("expert angle = ", action[1])
        # Speed boost when pressing shift
        if key_handler[key.LSHIFT]:
            action *= 3
        
        if key_handler[key.SPACE]:
            print(f"current pose = {info['curr_pos']}, step count = {env.unwrapped.step_count}, step reward = {reward:.3f}")
        
        
        #preprocess, angle = expert2.steer(obs)
        #print("expert angle = ", angle)
        print("action = ", action)
        
        obs, reward, done, info = env.step(action)

        #cv2.imshow("lane_lines", img_lane_lines)
        #cv2.imshow("preprocess", preprocess)
        
        env.render()

    pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
    pyglet.app.run()

else:
    # load control file
    actions = np.loadtxt(args.control_path, delimiter=',')

    for speed, steering in actions:
        obs, reward, done, info = env.step([speed, steering])
        curr_pos = info['curr_pos']

        print(f"current pose = {info['curr_pos']}, step count = {env.unwrapped.step_count}, step reward = {reward:.3f}")

        env.render()

    # dump the controls using numpy
    # np.savetxt(f'./{args.map_name}_seed{args.seed}_start_{start_pos[0]},{start_pos[1]}_goal_{goal[0]},{goal[1]}.txt',
    #            actions, delimiter=',')

env.close()

