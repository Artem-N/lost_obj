import cv2
import numpy as np
import time

from track_math.test_inits import *
import math


video_path = r"/home/qknife/Zir/modelling/GENERIC_RTSP-realmonitor_2023_09_20_15_57_55.avi"
    #r"/home/qknife/Zir/modelling/GENERIC_RTSP-realmonitor_2023_09_20_15_38_16.avi"
screen_width, screen_height = get_screen_size()
frame_center = (screen_width // 2, screen_height // 2)
cap = initialize_video_capture(video_path)

class Config:
    threshold_value = 30  # Threshold value for binary thresholding
    min_contour_area = 10  # Minimum contour area to consider for tracking
    min_contour_distance = 10  # Minimum contour distance to consider for tracking
    morph_kernel_size = (7, 7)  # Kernel size for morphological operations
    dilation_iterations = 3  # Number of dilation iterations
    erosion_iterations = 1  # Number of erosion iterations
    target_memory_frames = 5  # Number of frames to "remember" the target before resetting

class Target:
    def __init__(self, contour):
        self.position = None
        self.frames_to_activate = 0
        self.frames_to_kill = 3
        self.angle = 0
        self.speed = 0
        self.active = False
        # self.kalman = initialize_kalman_filter()
        self.set_contour(contour)
        # self.contour = None
        # self.prev_position = None
        # self.position = None
        # self.bbox = None

    def set_contour(self, contour):
        self.contour = contour
        if self.contour is not None:
            (x, y, w, h) = cv2.boundingRect(contour)  # This will now be a valid call
            self.bbox = (x, y, w, h)
            self.prev_position = self.position
            self.position = (x + w // 2, y + h // 2)
            self.frames_to_activate +=1
            if (self.frames_to_activate > 3):
                self.active = True
            self.frames_to_kill = 3



targets = []

def read_frame():
    is_frame_read, frame = cap.read()
    if not is_frame_read or frame is None:
        raise Exception("Error: Could not read the first frame.")

    # If the frame has 3 channels (e.g., BGR or RGB), convert to grayscale
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If the frame is already single-channel (e.g., thermal data), return as is
    return cv2.resize(frame, (screen_width, screen_height)), cv2.resize(gray_frame, (screen_width, screen_height))

# def resize_frame(frame, width, height):
#     """Resize the frame to the specified width and height."""
#     return cv2.resize(frame, (width, height))
#
#
# def convert_to_grayscale(frame):
#     """Convert the frame to grayscale if necessary."""
#     # If the frame has 3 channels (e.g., BGR or RGB), convert to grayscale
#     if len(frame.shape) == 3 and frame.shape[2] == 3:
#         return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # If the frame is already single-channel (e.g., thermal data), return as is
#     return frame


# def calculate_difference(prev_frame, curr_frame):
#     """Calculate the absolute difference between consecutive frames."""
#     # Compute the absolute difference between frames
#     diff = cv2.absdiff(prev_frame, curr_frame)
#     # Apply Gaussian blur to reduce noise
#     diff = cv2.GaussianBlur(diff, (7, 7), 0)
#     return diff


def thresholdFrame(prev_frame, curr_frame):
    """Calculate the absolute difference between consecutive frames."""
    # Compute the absolute difference between frames
    diff = cv2.absdiff(prev_frame, curr_frame)
    # Apply Gaussian blur to reduce noise
    diff = cv2.GaussianBlur(diff, (7, 7), 0)

    """Apply thresholding and morphological operations to detect moving objects."""
    # Apply binary thresholding
    _, thresh = cv2.threshold(diff, Config.threshold_value, 255, cv2.THRESH_BINARY)

    # Create a kernel for morphological operations
    kernel = np.ones(Config.morph_kernel_size, np.uint8)

    # Apply dilation and erosion to remove noise and fill gaps
    thresh = cv2.dilate(thresh, kernel, iterations=Config.dilation_iterations)
    thresh = cv2.erode(thresh, kernel, iterations=Config.erosion_iterations)

    # Apply closing to close small holes inside the foreground objects
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh


def find_largest_contour(contours):
    """Find and return the largest contour above a certain area threshold."""
    if contours:#!!!!
        # Find the contour with the maximum area
        contour = max(contours, key=cv2.contourArea)
        # Return the largest contour if its area is above the threshold
        if cv2.contourArea(contour) >= Config.min_contour_area:
            return contour
    return None


# def update_position(contour):
#     # """Update the Kalman filter with the position of the detected contour."""
#     if contour is None or len(contour) == 0:
#         return None, None  # Handle cases where the contour is invalid
#
#     # Get the bounding box of the contour
#     (x, y, w, h) = cv2.boundingRect(contour)  # This will now be a valid call
#     position = (x + w // 2, y + h // 2)  # Calculate the center of the bounding box
#
#     # Create a measurement matrix with the position
#     # measurement = np.array([[np.float32(position[0])],
#     #                         [np.float32(position[1])]])
#     #
#     # # Update the Kalman filter with the current measurement
#     # kalman.correct(measurement)
#
#     return position, (x, y, w, h)  # Return position and bounding box



def draw_bounding_box(frame, bbox):
    """Draw a bounding box around the detected object."""
    x, y, w, h = bbox
    # Draw the rectangle with a blue color
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

def find_closest_target(contour, targets):
    # """Find the contour closest to the last known position."""
    # if not targets: # or not contours:
    #     return None  # Return None if last_position or contours are invalid

    (x, y, w, h) = cv2.boundingRect(contour)  # Ensure each contour works with boundingRect
    center = (x + w // 2, y + h // 2)
    closest_target = None
    min_distance = float("inf")

    for target in targets:
        # Calculate the Euclidean distance from the last known position
        distance = np.linalg.norm(np.array(center) - np.array(target.position))
        if distance < min_distance:
            min_distance = distance
            closest_target = target  # Assign the closest contour

    if min_distance <= Config.min_contour_distance: # * Speed!!
        closest_target.set_contour(contour)
        return closest_target
    return None  # Return the closest valid contour




# def track_object_in_frame(cap, kalman, prev_frame, screen_width, screen_height, threshold_value, min_contour_area, morph_kernel_size,
#                           dilation_iterations, erosion_iterations, target_memory_frames):
#     """Track the object across video frames."""



# def handle_object_tracking(contours, tracked_object, last_position, target_lost_frames, target_memory_frames, kalman,
#                            curr_frame, min_contour_area, trajectory_points, frame_center, last_bbox_area, last_speed):
#     """Handle the logic of tracking the object, considering bounding box area and speed check."""
#
#     # Check if we need to initialize tracking
#     if tracked_object is None and contours: #
#         tracked_object, last_position, last_bbox_area, target_lost_frames = initialize_tracking(
#             contours, kalman, min_contour_area, curr_frame, last_bbox_area, target_lost_frames, trajectory_points
#         )
#
#     # Update tracking if the object is already being tracked
#     elif tracked_object is not None:
#         tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed = update_tracking_with_contours_and_speed(
#             contours, tracked_object, last_position, target_lost_frames, target_memory_frames, kalman, curr_frame,
#             trajectory_points, frame_center, last_bbox_area, last_speed
#         )
#
#     # Perform the angle check
#     if len(trajectory_points) >= 2:
#         if check_angle_and_stop_tracking(trajectory_points, curr_frame):
#             tracked_object = None  # Stop tracking if vertical movement is detected
#
#     return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed


# def initialize_tracking(contours, min_contour_area, curr_frame, last_bbox_area, target_lost_frames, trajectory_points):
#     """Initialize object tracking by finding the largest contour and updating Kalman filter."""
#     target.contour = find_largest_contour(contours)
#     if tracked_object is not None:
#         # Update Kalman filter and get the object's position and bounding box
#         last_position, bbox = update_position(tracked_object)
#         current_bbox_area = bbox[2] * bbox[3]  # Calculate area of the new bounding box
#
#         # Only draw the bounding box if the new area is not more than twice the last saved area
#         if last_bbox_area is None or current_bbox_area <= 1 * last_bbox_area:
#             draw_bounding_box(curr_frame, bbox)
#             last_bbox_area = current_bbox_area
#
#         # Add the new position to trajectory points and reset lost frames counter
#         trajectory_points.append(last_position) #single value
#         target_lost_frames = 0
#
#     return tracked_object, last_position, last_bbox_area, target_lost_frames


def check_angle_and_stop_tracking(trajectory_points, curr_frame):
    """Check the angle of movement and stop tracking if vertical movement is detected."""
    dx = trajectory_points[-1][0] - trajectory_points[-2][0]
    dy = trajectory_points[-1][1] - trajectory_points[-2][1]
    angle = math.degrees(math.atan2(dy, dx))

    # Check if the object is moving vertically (between 70 and 110 degrees)
    if 70 <= abs(angle) <= 110:
        print(f"Stopped tracking due to vertical movement (angle: {angle:.2f} degrees)")
        return True  # Indicate that tracking should be stopped

    # Display the angle on the frame
    cv2.putText(curr_frame, f"Angle: {angle:.2f} degrees", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return False


# def update_tracking_with_contours_and_speed(contours, tracked_object, last_position, target_lost_frames,
#                                             target_memory_frames, curr_frame, trajectory_points, frame_center,
#                                             last_bbox_area, last_speed):
#     # """Update the Kalman filter, handle contour matching, and check object speed and distance."""
#     max_distance = 100  # Maximum allowed distance in pixels
#     bbox_size_threshold = 10  # Bounding box size threshold
#     dynamic_speed_threshold = last_speed * 5 if last_speed > 0 else 100  # Dynamic speed threshold
#
#     if contours:
#         # Find the closest contour to the last known position
#         closest_contour = find_closest_contour(contours, last_position)
#
#         if closest_contour is not None:
#             # Update position and bounding box from Kalman filter
#             last_position, bbox = update_position(closest_contour)
#
#             # Perform checks for distance, speed, and angle
#             if not is_within_distance(last_position, last_position, max_distance):
#                 print(f"Ignored object due to distance > {max_distance} pixels")
#                 return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed
#
#             if not is_speed_valid(last_position, last_position, last_speed, dynamic_speed_threshold):
#                 print(f"Ignored object due to high speed")
#                 return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed
#
#             if not is_angle_valid(trajectory_points):
#                 print(f"Ignored object due to vertical movement")
#                 return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed
#
#             # If all checks passed, draw the bounding box and red line
#             if should_draw_bbox(last_bbox_area, bbox, bbox_size_threshold):
#                 draw_bounding_box_and_line(curr_frame, bbox, last_position, frame_center)
#                 last_bbox_area = bbox[2] * bbox[3]  # Update bounding box area
#
#             # Add the current position to the trajectory points for further calculations
#             trajectory_points.append(last_position)
#             target_lost_frames = 0  # Reset lost frames counter
#         else:
#             target_lost_frames += 1  # No contour found
#     else:
#         target_lost_frames += 1
#
#     # Reset tracking if the object is lost for too many frames
#     if target_lost_frames > target_memory_frames:
#         tracked_object = None
#         target_lost_frames = 0
#
#     return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed


# def update_position_and_bbox(kalman, closest_contour):
#     """Update the position and bounding box based on Kalman filter and closest contour."""
#     last_position, bbox = update_position(kalman, closest_contour)
#     return last_position, bbox


def is_within_distance(current_position, previous_position, max_distance):
    """Check if the current position is within the allowed distance from the previous one."""
    distance = calculate_distance(current_position, previous_position)
    return distance <= max_distance


def calculate_distance(position1, position2):
    """Calculate Euclidean distance between two positions."""
    return math.sqrt((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2)


def is_speed_valid(current_position, previous_position, last_speed, dynamic_speed_threshold):
    """Check if the object's speed is within the allowed dynamic threshold."""
    dx = current_position[0] - previous_position[0]
    dy = current_position[1] - previous_position[1]
    current_speed = math.sqrt(dx ** 2 + dy ** 2)
    return current_speed <= dynamic_speed_threshold


def is_angle_valid(trajectory_points):
    """Check if the angle of the object's movement is within the allowed range."""
    if len(trajectory_points) >= 2:
        dx = trajectory_points[-1][0] - trajectory_points[-2][0]
        dy = trajectory_points[-1][1] - trajectory_points[-2][1]
        angle = math.degrees(math.atan2(dy, dx))
        return not (80 <= abs(angle) <= 100)  # Ignore vertical movement
    return True  # If we don't have enough points, assume valid


def should_draw_bbox(last_bbox_area, bbox, bbox_size_threshold):
    """Check if the bounding box should be drawn based on its size."""
    current_bbox_area = bbox[2] * bbox[3]
    return last_bbox_area is None or current_bbox_area <= bbox_size_threshold * last_bbox_area


def draw_bounding_box_and_line(curr_frame, bbox, last_position, frame_center):
    """Draw the bounding box and red line on the frame."""
    draw_bounding_box(curr_frame, bbox)  # Draw bounding box
    bbox_center_x = int(last_position[0])
    bbox_center_y = int(last_position[1])
    cv2.line(curr_frame, frame_center, (bbox_center_x, bbox_center_y), (0, 0, 255), 1)  # Draw the red line

def drawInfo(frame, start_time):
    # Draw crosshair on the frame
    draw_crosshair(frame)

    # Calculate and display FPS
    fps = calculate_fps(start_time)
    display_fps(frame, fps)

    # Show the result
    cv2.imshow('Frame', frame)

def main():
    # # Parameters for fine-tuning the algorithm
    # THRESHOLD_VALUE = 30  # Threshold value for binary thresholding
    # MIN_CONTOUR_AREA = 10  # Minimum contour area to consider for tracking
    # MORPH_KERNEL_SIZE = (7, 7)  # Kernel size for morphological operations
    # DILATION_ITERATIONS = 3  # Number of dilation iterations
    # EROSION_ITERATIONS = 1  # Number of erosion iterations
    # TARGET_MEMORY_FRAMES = 5  # Number of frames to "remember" the target before resetting

    # # Path to the video file
    # video_path = r"/home/qknife/Zir/modelling/GENERIC_RTSP-realmonitor_2023_09_20_15_38_16.avi"
    #
    # # Get screen size
    # screen_width, screen_height = get_screen_size()
    #
    # # Initialize video capture and read the first frame
    # cap = initialize_video_capture(video_path)
    _,prev_frame = read_frame()
    # prev_gray = convert_to_grayscale(first_frame)

    # # Initialize the Kalman filter
    # kalman = initialize_kalman_filter()

    # Set the video window to full screen
    cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # tracked_object = None
    # last_position = None
    # trajectory_points = []
    # target_lost_frames = 0
    # last_bbox_area = None  # Track the area of the last saved bounding box
    # last_speed = 0  # Track the speed of the last tracked object

    while cap.isOpened():
        start_time = time.time()

        frame, gray_frame = read_frame()

        # # Resize current frame
        # frame = resize_frame(frame, screen_width, screen_height)
        # curr_gray = convert_to_grayscale(frame)
        #
        # # Resize previous frame
        # prev_gray = resize_frame(prev_gray, screen_width, screen_height) #????

        # Calculate difference and apply thresholding
        thresh_frame = thresholdFrame(prev_frame, gray_frame)
        # frame = thresholdFrame(diff)

        # Find contours in thresholded frame
        contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        targets_to_kill = []

        for target in targets:
            target.frames_to_kill -= 1
            if target.frames_to_kill <= 0: targets_to_kill.append(target)
        for target in targets_to_kill:
            targets.remove(target)

        if len(contours) + len(targets) <= 3:

            contours_with_no_targets = []

            for contour in contours:
                target = find_closest_target(contour, targets)
                if (target is None):
                    contours_with_no_targets.append(contour)
                # else:
                #     targets.append(
                #         Target(contour)
                #     )

            for contour in contours_with_no_targets:
                targets.append(
                    Target(contour)
                )
            # if (contours_with_no_targets):
            #     targets.append(
            #         Target(find_largest_contour(contours_with_no_targets))
            #     )

            for target in targets:
                if (target.active):
                    draw_bounding_box(frame, target.bbox) #to method
        # for contour in contours:
        #     (x, y, w, h) = cv2.boundingRect(contour)
        #     draw_bounding_box(frame, (x, y, w, h))  # to metho

        # # Handle object tracking and draw bounding boxes
        # tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed = handle_object_tracking(
        #     contours, tracked_object, last_position, target_lost_frames, target_memory_frames, kalman, frame,
        #     min_contour_area, trajectory_points, frame_center, last_bbox_area, last_speed
        # )

        drawInfo(frame, start_time)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        prev_frame = gray_frame

    # # Track the object across frames
    # track_object_in_frame(cap, kalman, prev_gray, screen_width, screen_height, THRESHOLD_VALUE, MIN_CONTOUR_AREA, MORPH_KERNEL_SIZE,
    #                       DILATION_ITERATIONS, EROSION_ITERATIONS, TARGET_MEMORY_FRAMES)

    # Release video capture and close all frames
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
