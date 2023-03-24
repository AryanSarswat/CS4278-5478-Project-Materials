import numpy as np
import cv2
import math

class LaneFollower:
    def __init__(self):
        self.W_SENSITIVITY = 95
        self.LOWER_WHITE = np.array([0, 0, 255 - self.W_SENSITIVITY])
        self.UPPER_WHITE = np.array([255, self.W_SENSITIVITY, 235])
        
        self.LOWER_YELLOW = np.array([20, 85, 80])
        self.UPPER_YELLOW = np.array([30, 255, 255])
        
        self.HEIGHT_CROP_SCALE = 1/2
        
        self.MIN_LINE_LENGTH = 150
        self.MIN_VOTES = 30
        self.MAX_LINE_GAP = 100
        self.prev_angle = 0

    def detect_edges(self, img):
        # Blur first to smoothen
        img = cv2.medianBlur(img, 5)
        img = cv2.GaussianBlur(img, (5, 5), 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        mask_yellow = cv2.inRange(hsv, self.LOWER_YELLOW, self.UPPER_YELLOW)
        mask_white = cv2.inRange(hsv, self.LOWER_WHITE, self.UPPER_WHITE)
        
        edges_white = cv2.Canny(mask_white, 200, 400)
        edges_yellow = cv2.Canny(mask_yellow, 200, 400)
        edges_combined = cv2.bitwise_or(edges_white, edges_yellow)
        
        return edges_white, edges_yellow, edges_combined

    def isolate_roi(self, img):
        h, w = img.shape
        mask = np.zeros_like(img)
        
        polygon = np.array([[
            (0, h * self.HEIGHT_CROP_SCALE),
            (w, h * self.HEIGHT_CROP_SCALE),
            (w, h),
            (0, h),
        ]], np.int32)
        
        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(img, mask)
        return cropped_edges

    def detect_line_segments(self, img):
        rho = 1
        angle = np.pi / 180
        min_threshold = self.MIN_VOTES
        line_segments = cv2.HoughLinesP(img, rho, angle, min_threshold, np.array([]), minLineLength=self.MIN_LINE_LENGTH, maxLineGap=self.MAX_LINE_GAP)
        return line_segments

    def make_points(self, frame, line):
        height, width, _ = frame.shape
        slope, intercept = line
        y1 = height  # bottom of the frame
        y2 = int(y1 * 1 / 4)  # make points from middle of the frame down

        # bound the coordinates within the frame
        x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
        x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
        return [[x1, y1, x2, y2]]
    
    def average_line(self, frame, line_segments, white=False):
        height, width, _ = frame.shape
        if line_segments is None:
            return []
        
        slope_intercepts = []
        boundary = 1/2
        left_region_boundary = width * (1 - boundary)
        
        for line_segment in line_segments:
            for x1, y1, x2, y2 in line_segment:
                if x1 == x2:
                    continue
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if white and slope < 0 and x1 < left_region_boundary:
                    continue
                slope_intercepts.append((slope, intercept))
        
        if len(slope_intercepts) == 0:
            return []
        avg = np.average(slope_intercepts, axis=0)
        lane_lines = self.make_points(frame, avg)
        return lane_lines

    def average_slope_intercept(self, frame, line_segments):
        """
        This function combines line segments into one or two lane lines
        If all line slopes are < 0: then we only have detected left lane
        If all line slopes are > 0: then we only have detected right lane
        """
        lane_lines = []
        if line_segments is None:
            return lane_lines

        height, width, _ = frame.shape
        left_fit = []
        right_fit = []

        boundary = 1/2
        left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
        right_region_boundary = width * boundary # right lane line segment should be on left 1/3 of the screen

        for line_segment in line_segments:
            for x1, y1, x2, y2 in line_segment:
                if x1 == x2:
                    continue
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0:
                    if x1 < left_region_boundary and x2 < left_region_boundary:
                        left_fit.append((slope, intercept))
                else:
                    if x1 > right_region_boundary and x2 > right_region_boundary:
                        right_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit, axis=0)
        if len(left_fit) > 0:
            lane_lines.append(self.make_points(frame, left_fit_average))

        right_fit_average = np.average(right_fit, axis=0)
        if len(right_fit) > 0:
            lane_lines.append(self.make_points(frame, right_fit_average))

        return lane_lines

    def detect_lane(self, frame):
        edges_w, edges_y, edges_combined = self.detect_edges(frame)
        cropped_edges_w, cropped_edges_y, cropped_combined = self.isolate_roi(edges_w), self.isolate_roi(edges_y), self.isolate_roi(edges_combined)
        
        line_segments_w, line_segments_y, line_segments_combined = self.detect_line_segments(cropped_edges_w), self.detect_line_segments(cropped_edges_y), self.detect_line_segments(cropped_combined)
        
        lane_lines_w = self.average_slope_intercept(frame, line_segments_w)
        lane_lines_y = self.average_slope_intercept(frame, line_segments_y)
        lane_lines_comb = self.average_slope_intercept(frame, line_segments_combined)
        
        return lane_lines_w, lane_lines_y, lane_lines_comb
    
    def display_one_line(self, frame, line, line_color=(0, 255, 0), line_width=10):
        line_image = np.zeros_like(frame)
        if line is not None:
            for line in line:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
                line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        return line_image

    def display_lines(self, frame, lines, line_color=(0, 255, 0), line_width=10):
        line_image = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
        line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        return line_image

    def display_heading_line(self, frame, steering_angle, line_color=(0, 0, 255), line_width=15):
        heading_image = np.zeros_like(frame)
        height, width, _ = frame.shape

        # figure out the heading line from steering angle
        # heading line (x1,y1) is always center bottom of the screen
        # (x2, y2) requires a bit of trigonometry

        # Note: the steering angle of:
        # 0-89 degree: turn left
        # 90 degree: going straight
        # 91-180 degree: turn right 
        x1 = int(width / 2)
        y1 = height
        try:
            x2 = int(x1 - height / (2 / math.tan(steering_angle)))
        except ZeroDivisionError:
            x2 = int(x1 - height / (2 / math.tan(0.01)))
        y2 = int(height / 2)

        heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

        return heading_image
    
    def check_if_yellow_is_on_right(self, frame):
        lane_lines_y = self.detect_lane(frame)[1]
        height, width, _ = frame.shape
        if len(lane_lines_y) == 0:
            return False
        for line in lane_lines_y:
            for x1, y1, x2, y2 in line:
                if x1 < width/2 and x2 < width/2:
                    return False
        return True
    
    def check_if_yellow_is_on_left(self, frame):
        height, width, _ = frame.shape
        lane_lines_y = self.detect_lane(frame)[1]
        if len(lane_lines_y) == 0:
            return False
        for line in lane_lines_y:
            for x1, y1, x2, y2 in line:
                if x1 > width/2 and x2 > width/2:
                    return False
        return True
        
    
    def steer(self, frame):
        lane_lines_w, lane_lines_y, lane_lines_combined = self.detect_lane(frame)
        

        lane_lines_combined_img = self.display_lines(frame, lane_lines_combined)
        
        yellow_lane_img = self.display_lines(frame, lane_lines_y)
        
        cv2.imshow("Comb Lines", lane_lines_combined_img)
        cv2.imshow("Yellow Lines", yellow_lane_img)
        
        dist_y = self.check_if_yellow_is_on_right(frame)
        
        new_angle = self.compute_steering_angle_comb(frame, lane_lines_combined)
        
        # If it ends up in the wrong lane
        if dist_y and new_angle > 0:
            new_angle -= np.pi / 2
        
        new_angle = self.stabilize_steering_angle_comb(self.prev_angle, new_angle, len(lane_lines_combined))
        self.prev_angle = new_angle
        return new_angle
    
    def compute_steering_angle_comb(self, frame, lane_lines):
        if len(lane_lines) == 0:
            return 0

        height, width, _ = frame.shape
        if len(lane_lines) == 1:
            x1, _, x2, _ = lane_lines[0][0]
            x_offset = x2 - x1
            
            if x1 > width / 1.5 and x_offset < 0:
                x_offset = -x_offset
                
            if x1 < width / 3.5 and x_offset > 0:
                x_offset = -x_offset
                
        else:
            _, _, left_x2, _ = lane_lines[0][0]
            _, _, right_x2, _ = lane_lines[1][0]
            camera_mid_offset_percent = 0.02 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
            mid = int((width / 2) * (1 + camera_mid_offset_percent))
            x_offset = (left_x2 + right_x2) / 2 - mid

        # find the steering angle, which is angle between navigation direction to end of center line
        y_offset = int(height / 2)

        angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical 

        return angle_to_mid_radian
    
    def stabilize_steering_angle_comb(self, curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=15*np.pi/180, max_angle_deviation_one_lane=15*np.pi/180):
        if num_of_lane_lines == 2 :
            # if both lane lines detected, then we can deviate more
            max_angle_deviation = max_angle_deviation_two_lines
        else :
            # if only one lane detected, don't deviate too much
            max_angle_deviation = max_angle_deviation_one_lane
        
        angle_deviation = new_steering_angle - curr_steering_angle
        if abs(angle_deviation) > max_angle_deviation:
            stabilized_steering_angle = curr_steering_angle + (max_angle_deviation * angle_deviation / abs(angle_deviation))
        else:
            stabilized_steering_angle = new_steering_angle
        return stabilized_steering_angle