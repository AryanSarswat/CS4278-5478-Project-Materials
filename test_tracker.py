import argparse

import numpy as np
import pyglet
from gym_duckietown.envs import DuckietownEnv
from pyglet.window import key
import sys
import cv2


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

def track_yellow(frame, MIN_SIZE=(20, 20), MARGIN=20):
    # Get frame resolution
    frame_height, frame_width, _ = frame.shape

    # Define region of interest (ROI) coordinates for the top 1/3 of the frame
    roi_x = 0
    roi_y = 0
    roi_width = frame_width
    roi_height = int(frame_height / 8)

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Extract ROI from HSV image
    roi = hsv[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

    # Initialize yellow color range in HSV color space
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold HSV image to get binary mask of yellow color
    mask = cv2.inRange(roi, lower_yellow, upper_yellow)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = None
    largest_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    # Draw bounding box around the largest yellow object
    bbox = None
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        if w > MIN_SIZE[0] and h > MIN_SIZE[1]:
            x = max(0, x - MARGIN)
            y = min(frame_height, max(0, y - MARGIN) + h)
            w = min(frame_width, x + w + 2 * MARGIN) - x
            h = min(frame_height, y + h + 3 * MARGIN) - y
            cv2.rectangle(frame, (x + roi_x, y + roi_y), (x + roi_x + w, y + roi_y + h), (0, 255, 255), 2)
            bbox = (x, y, w, h)

    # Display the frame with yellow color tracked
    cv2.imshow('Yellow Color Tracker + Object Tracking', frame)
    return bbox

prev_bbox = None
def track_roi(frame, tracker, RESIZE=0.5):
    new_frame = cv2.resize(frame, (int(frame.shape[1]*RESIZE), int(frame.shape[0]*RESIZE)))
    # Update the tracker with the current frame
    success, new_bbox = tracker.update(new_frame)

    # Extract the tracked object's position and size
    x, y, w, h = [int(v / RESIZE) for v in new_bbox]

    # Draw a bounding box around the tracked object
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with the tracking result
    cv2.imshow("Yellow Color Tracker + Object Tracking", frame)
    # print(new_bbox)

    return success, [x,y,w,h]

    # pass

# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', '-m', default="map4_0", type=str)
parser.add_argument('--seed', '-s', default=2, type=int)
parser.add_argument('--start-tile', '-st', default="1,13", type=str, help="two numbers separated by a comma")
parser.add_argument('--goal-tile', '-gt', default="3,3", type=str, help="two numbers separated by a comma")
parser.add_argument('--control_path', default='./map4_0_seed2_start_1,13_goal_3,3.txt', type=str,
                    help="the control file to run")
parser.add_argument('--manual', default=False, type=str2bool, help="whether to manually control the robot")
args = parser.parse_args()


# simulator instantiation
env = DuckietownEnv(
    domain_rand=False,
    max_steps=1500,
    map_name=args.map_name,
    seed=args.seed,
    user_tile_start=args.start_tile,
    goal_tile=args.goal_tile,
    randomize_maps_on_reset=False
)

# obs = env.reset() # WARNING: never call this function during testing
env.render()

map_img, goal, start_pos = env.get_task_info()
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

# main loop
if args.manual:
    # Register a keyboard handler
    key_handler = key.KeyStateHandler()
    env.unwrapped.window.push_handlers(key_handler)
    bbox = None
    success = False
    RESIZE=1

    def initialize_tracker(frame, bbox, RESIZE=0.5):
        tracker = cv2.legacy.TrackerTLD_create() # change around

        # Set the default values for the parameters
        new_frame = cv2.resize(frame, (int(frame.shape[1]*RESIZE), int(frame.shape[0]*RESIZE)))
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        small_bbox = [int(v * RESIZE) for v in bbox]
        # ret, small_bbox = cv2.meanShift(new_frame, small_bbox, ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ))
        tracker.init(new_frame, small_bbox)
        return tracker

        # try Lucas Kanade optical flow
        # return dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def update(dt):
        global bbox, success, prev_bbox
        action = np.array([0.0, 0.0])

        if key_handler[key.UP]:
            action = np.array([0.44, 0.0])
        if key_handler[key.DOWN]:
            action = np.array([-0.44, 0])
        if key_handler[key.LEFT]:
            action = np.array([0, +1])
        if key_handler[key.RIGHT]:
            action = np.array([0, -1])
        if key_handler[key.SPACE]:
            action = np.array([0, 0])

        # Speed boost when pressing shift
        if key_handler[key.LSHIFT]:
            action *= 3

        obs, reward, done, info = env.step(action)
        
        first_time_seen = True
        if bbox is not None:
            if first_time_seen:
                # Initialize the tracker with the first frame and bounding box
                tracker = initialize_tracker(obs, bbox, RESIZE=RESIZE)
                # initialize_tracker(obs, bbox, RESIZE=RESIZE) # for meanshift
                first_time_seen = False
            success, new_bbox = track_roi(obs, tracker, RESIZE=RESIZE)
            # success, prev_bbox = track_meanshift(obs, RESIZE=RESIZE)

            if not success:
                bbox = None
                first_time_seen = True
        else:
            bbox = track_yellow(obs)
            prev_bbox = bbox
            first_time_seen = True
        
        print(f"current pose = {info['curr_pos']}, step count = {env.unwrapped.step_count}, step reward = {reward:.3f}")
        if key_handler[key.SPACE]:
            cv2.imwrite("duck.png", obs)

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
    np.savetxt(f'./{args.map_name}_seed{args.seed}_start_{start_pos[0]},{start_pos[1]}_goal_{goal[0]},{goal[1]}.txt',
               actions, delimiter=',')

env.close()

