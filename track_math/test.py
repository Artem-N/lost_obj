import cv2
import numpy as np
import time
from screeninfo import get_monitors
import math


def get_screen_size():
    """Get the screen width and height."""
    monitor = get_monitors()[0]  # Get the primary monitor's information
    return monitor.width, monitor.height


def initialize_video_capture(video_path):
    """Initialize video capture from the provided video file path."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")  # Raise an error if the video cannot be opened
    return cap


def read_first_frame(cap):
    """Read and return the first frame from the video."""
    ret, frame = cap.read()
    if not ret or frame is None:
        raise Exception("Error: Could not read the first frame.")  # Raise an error if the first frame cannot be read
    return frame


def initialize_kalman_filter():
    """Initialize and return a Kalman filter for tracking."""
    # Kalman filter with 4 dynamic parameters (x, y, dx, dy) and 2 measured parameters (x, y)
    kalman = cv2.KalmanFilter(4, 2)

    # Measurement matrix describes how the measurement relates to the state
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)

    # Transition matrix defines the transition between states (x, y, dx, dy)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)

    # Process noise and measurement noise covariance matrices
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1

    return kalman


def resize_frame(frame, width, height):
    """Resize the frame to the specified width and height."""
    return cv2.resize(frame, (width, height))


def convert_to_grayscale(frame):
    """Convert the frame to grayscale if necessary."""
    # If the frame has 3 channels (e.g., BGR or RGB), convert to grayscale
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If the frame is already single-channel (e.g., thermal data), return as is
    return frame


def calculate_difference(prev_frame, curr_frame):
    """Calculate the absolute difference between consecutive frames."""
    # Compute the absolute difference between frames
    diff = cv2.absdiff(prev_frame, curr_frame)
    # Apply Gaussian blur to reduce noise
    diff = cv2.GaussianBlur(diff, (7, 7), 0)
    return diff


def apply_thresholding(diff, threshold_value, kernel_size, dilation_iterations, erosion_iterations):
    """Apply thresholding and morphological operations to detect moving objects."""
    # Apply binary thresholding
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    # Create a kernel for morphological operations
    kernel = np.ones(kernel_size, np.uint8)

    # Apply dilation and erosion to remove noise and fill gaps
    thresh = cv2.dilate(thresh, kernel, iterations=dilation_iterations)
    thresh = cv2.erode(thresh, kernel, iterations=erosion_iterations)

    # Apply closing to close small holes inside the foreground objects
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh


def find_largest_contour(contours, min_contour_area):
    """Find and return the largest contour above a certain area threshold."""
    if contours:#!!!!
        # Find the contour with the maximum area
        contour = max(contours, key=cv2.contourArea)
        # Return the largest contour if its area is above the threshold
        if cv2.contourArea(contour) >= min_contour_area:
            return contour
    return None


def update_kalman_filter(kalman, contour):
    """Update the Kalman filter with the position of the detected contour."""
    if contour is None or len(contour) == 0:
        return None, None  # Handle cases where the contour is invalid

    # Get the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(contour)  # This will now be a valid call
    position = (x + w // 2, y + h // 2)  # Calculate the center of the bounding box

    # Create a measurement matrix with the position
    measurement = np.array([[np.float32(position[0])],
                            [np.float32(position[1])]])

    # Update the Kalman filter with the current measurement
    kalman.correct(measurement)

    return position, (x, y, w, h)  # Return position and bounding box



def draw_bounding_box(frame, bbox):
    """Draw a bounding box around the detected object."""
    x, y, w, h = bbox
    # Draw the rectangle with a blue color
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


def draw_crosshair(frame):
    """Draw a crosshair across the entire video frame."""
    color = (0, 255, 0)  # Green crosshair
    thickness = 1  # Thickness of the crosshair lines

    # Get the center of the frame
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2

    # Draw the horizontal line across the entire width of the frame
    cv2.line(frame, (0, center_y), (frame.shape[1], center_y), color, thickness)
    # Draw the vertical line across the entire height of the frame
    cv2.line(frame, (center_x, 0), (center_x, frame.shape[0]), color, thickness)


def calculate_fps(start_time):
    """Calculate the frames per second (FPS)."""
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    return fps


def display_fps(frame, fps):
    """Display the FPS on the frame."""
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


def find_closest_contour(contours, last_position):
    """Find the contour closest to the last known position."""
    if last_position is None or not contours:
        return None  # Return None if last_position or contours are invalid

    closest_contour = None
    min_distance = float("inf")

    for contour in contours:
        # Get the center of the contour's bounding box
        (x, y, w, h) = cv2.boundingRect(contour)  # Ensure each contour works with boundingRect
        center = (x + w // 2, y + h // 2)

        # Calculate the Euclidean distance from the last known position
        distance = np.linalg.norm(np.array(center) - np.array(last_position))
        if distance < min_distance:
            min_distance = distance
            closest_contour = contour  # Assign the closest contour

    return closest_contour  # Return the closest valid contour




def track_object_in_frame(cap, kalman, prev_gray, screen_width, screen_height, threshold_value, min_contour_area, morph_kernel_size,
                          dilation_iterations, erosion_iterations, target_memory_frames):
    """Track the object across video frames."""
    tracked_object = None
    last_position = None
    trajectory_points = []
    target_lost_frames = 0
    last_bbox_area = None  # Track the area of the last saved bounding box
    last_speed = 0  # Track the speed of the last tracked object
    frame_center = (screen_width // 2, screen_height // 2)

    while cap.isOpened():
        start_time = time.time()

        ret, curr_frame = cap.read()
        if not ret or curr_frame is None:
            print("Error: Could not read the frame.")
            break

        # Resize current frame
        curr_frame = resize_frame(curr_frame, screen_width, screen_height)
        curr_gray = convert_to_grayscale(curr_frame)

        # Resize previous frame
        prev_gray = resize_frame(prev_gray, screen_width, screen_height) #????

        # Calculate difference and apply thresholding
        diff = calculate_difference(prev_gray, curr_gray)
        thresh = apply_thresholding(diff, threshold_value, morph_kernel_size, dilation_iterations, erosion_iterations)

        # Find contours in thresholded frame
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Handle object tracking and draw bounding boxes
        tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed = handle_object_tracking(
            contours, tracked_object, last_position, target_lost_frames, target_memory_frames, kalman, curr_frame, min_contour_area, trajectory_points, frame_center, last_bbox_area, last_speed
        )

        # Draw crosshair on the frame
        draw_crosshair(curr_frame)

        # Calculate and display FPS
        fps = calculate_fps(start_time)
        display_fps(curr_frame, fps)

        # Show the result
        cv2.imshow('Frame', curr_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        prev_gray = curr_gray#.copy()



def handle_object_tracking(contours, tracked_object, last_position, target_lost_frames, target_memory_frames, kalman,
                           curr_frame, min_contour_area, trajectory_points, frame_center, last_bbox_area, last_speed):
    """Handle the logic of tracking the object, considering bounding box area and speed check."""

    # Check if we need to initialize tracking
    if tracked_object is None and contours: #
        tracked_object, last_position, last_bbox_area, target_lost_frames = initialize_tracking(
            contours, kalman, min_contour_area, curr_frame, last_bbox_area, target_lost_frames, trajectory_points
        )

    # Update tracking if the object is already being tracked
    elif tracked_object is not None:
        tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed = update_tracking_with_contours_and_speed(
            contours, tracked_object, last_position, target_lost_frames, target_memory_frames, kalman, curr_frame,
            trajectory_points, frame_center, last_bbox_area, last_speed
        )

    # Perform the angle check
    if len(trajectory_points) >= 2:
        if check_angle_and_stop_tracking(trajectory_points, curr_frame):
            tracked_object = None  # Stop tracking if vertical movement is detected

    return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed


def initialize_tracking(contours, kalman, min_contour_area, curr_frame, last_bbox_area, target_lost_frames, trajectory_points):
    """Initialize object tracking by finding the largest contour and updating Kalman filter."""
    tracked_object = find_largest_contour(contours, min_contour_area)
    if tracked_object is not None:
        # Update Kalman filter and get the object's position and bounding box
        last_position, bbox = update_kalman_filter(kalman, tracked_object)
        current_bbox_area = bbox[2] * bbox[3]  # Calculate area of the new bounding box

        # Only draw the bounding box if the new area is not more than twice the last saved area
        if last_bbox_area is None or current_bbox_area <= 1 * last_bbox_area:
            draw_bounding_box(curr_frame, bbox)
            last_bbox_area = current_bbox_area

        # Add the new position to trajectory points and reset lost frames counter
        trajectory_points.append(last_position) #single value
        target_lost_frames = 0

    return tracked_object, last_position, last_bbox_area, target_lost_frames


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


def update_tracking_with_contours_and_speed(contours, tracked_object, last_position, target_lost_frames,
                                            target_memory_frames, kalman, curr_frame, trajectory_points, frame_center,
                                            last_bbox_area, last_speed):
    """Update the Kalman filter, handle contour matching, and check object speed and distance."""
    max_distance = 100  # Maximum allowed distance in pixels
    bbox_size_threshold = 10  # Bounding box size threshold
    dynamic_speed_threshold = last_speed * 5 if last_speed > 0 else 100  # Dynamic speed threshold

    if contours:
        # Find the closest contour to the last known position
        closest_contour = find_closest_contour(contours, last_position)

        if closest_contour is not None:
            # Update position and bounding box from Kalman filter
            last_position, bbox = update_position_and_bbox(kalman, closest_contour)

            # Perform checks for distance, speed, and angle
            if not is_within_distance(last_position, last_position, max_distance):
                print(f"Ignored object due to distance > {max_distance} pixels")
                return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed

            if not is_speed_valid(last_position, last_position, last_speed, dynamic_speed_threshold):
                print(f"Ignored object due to high speed")
                return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed

            if not is_angle_valid(trajectory_points):
                print(f"Ignored object due to vertical movement")
                return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed

            # If all checks passed, draw the bounding box and red line
            if should_draw_bbox(last_bbox_area, bbox, bbox_size_threshold):
                draw_bounding_box_and_line(curr_frame, bbox, last_position, frame_center)
                last_bbox_area = bbox[2] * bbox[3]  # Update bounding box area

            # Add the current position to the trajectory points for further calculations
            trajectory_points.append(last_position)
            target_lost_frames = 0  # Reset lost frames counter
        else:
            target_lost_frames += 1  # No contour found
    else:
        target_lost_frames += 1

    # Reset tracking if the object is lost for too many frames
    if target_lost_frames > target_memory_frames:
        tracked_object = None
        target_lost_frames = 0

    return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed


def update_position_and_bbox(kalman, closest_contour):
    """Update the position and bounding box based on Kalman filter and closest contour."""
    last_position, bbox = update_kalman_filter(kalman, closest_contour)
    return last_position, bbox


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


def main():
    # Parameters for fine-tuning the algorithm
    THRESHOLD_VALUE = 30  # Threshold value for binary thresholding
    MIN_CONTOUR_AREA = 10  # Minimum contour area to consider for tracking
    MORPH_KERNEL_SIZE = (7, 7)  # Kernel size for morphological operations
    DILATION_ITERATIONS = 3  # Number of dilation iterations
    EROSION_ITERATIONS = 1  # Number of erosion iterations
    TARGET_MEMORY_FRAMES = 5  # Number of frames to "remember" the target before resetting

    # Path to the video file
    video_path = r"C:\Users\User\Desktop\fly\GENERIC_RTSP-realmonitor_2023_09_20_15_35_49.avi"

    # Get screen size
    screen_width, screen_height = get_screen_size()

    # Initialize video capture and read the first frame
    cap = initialize_video_capture(video_path)
    first_frame = read_first_frame(cap)
    prev_gray = convert_to_grayscale(first_frame)

    # Initialize the Kalman filter
    kalman = initialize_kalman_filter()

    # Set the video window to full screen
    cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Track the object across frames
    track_object_in_frame(cap, kalman, prev_gray, screen_width, screen_height, THRESHOLD_VALUE, MIN_CONTOUR_AREA, MORPH_KERNEL_SIZE,
                          DILATION_ITERATIONS, EROSION_ITERATIONS, TARGET_MEMORY_FRAMES)

    # Release video capture and close all frames
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
