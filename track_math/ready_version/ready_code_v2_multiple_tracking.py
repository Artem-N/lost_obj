import cv2
import numpy as np
import time
from screeninfo import get_monitors
from collections import deque


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
    kalman = cv2.KalmanFilter(4, 2)  # 4 dynamic parameters (x, y, dx, dy) and 2 measured parameters (x, y)

    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)

    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)

    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1

    return kalman


def resize_frame(frame, width, height):
    """Resize the frame to the specified width and height."""
    return cv2.resize(frame, (width, height))


def convert_to_grayscale(frame):
    """Convert the frame to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def calculate_difference(prev_frame, curr_frame):
    """Calculate the absolute difference between consecutive frames."""
    diff = cv2.absdiff(prev_frame, curr_frame)  # Compute the absolute difference between frames
    diff = cv2.GaussianBlur(diff, (7, 7), 0)  # Apply Gaussian blur to reduce noise
    return diff


def apply_thresholding(diff, threshold_value, kernel_size, dilation_iterations, erosion_iterations):
    """Apply thresholding and morphological operations to detect moving objects."""
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)  # Apply binary thresholding

    # Create a kernel for morphological operations
    kernel = np.ones(kernel_size, np.uint8)

    # Apply dilation and erosion to remove noise and fill gaps
    thresh = cv2.dilate(thresh, kernel, iterations=dilation_iterations)
    thresh = cv2.erode(thresh, kernel, iterations=erosion_iterations)

    # Apply closing to close small holes inside the foreground objects
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh


def update_kalman_filter(kalman, contour):
    """Update the Kalman filter with the position of the detected contour."""
    (x, y, w, h) = cv2.boundingRect(contour)  # Get the bounding box of the contour

    # Calculate the center position of the bounding box
    position = (x + w // 2, y + h // 2)

    # Create a measurement matrix with the position
    measurement = np.array([[np.float32(position[0])],
                            [np.float32(position[1])]])

    # Update the Kalman filter with the current measurement
    kalman.correct(measurement)

    return position, (x, y, w, h)  # Return the position and bounding box


def draw_bounding_box(frame, bbox, color=(255, 0, 0)):
    """Draw a bounding box around the detected object."""
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # Draw the rectangle


def draw_crosshair(frame):
    """Draw a crosshair across the entire video frame."""
    color = (0, 255, 0)  # Green crosshair
    thickness = 1  # Thickness of the crosshair lines

    # Get the center of the frame
    center_x = frame.shape[1] // 2  # Width of the frame
    center_y = frame.shape[0] // 2  # Height of the frame

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


def find_largest_contour(contours, min_contour_area):
    """Find and return the largest contour above a certain area threshold."""
    if contours:
        # Find the contour with the maximum area
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) >= min_contour_area:
            return contour  # Return the largest contour if its area is above the threshold
    return None


def find_closest_contour(contours, last_position):
    """Find the contour closest to the last known position."""
    if last_position is None:
        return None, float("inf")  # Return None if last_position is None

    closest_contour = None
    min_distance = float("inf")
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)
        distance = np.linalg.norm(np.array(center) - np.array(last_position))
        if distance < min_distance:
            min_distance = distance
            closest_contour = contour
    return closest_contour, min_distance


def update_tracking_with_contours(contours, tracked_object, last_position, target_lost_frames, target_memory_frames,
                                  kalman, curr_frame, trajectory_points, is_primary):
    """Update the Kalman filter and handle contour matching."""
    if contours and last_position is not None:
        closest_contour, _ = find_closest_contour(contours, last_position)
        if closest_contour is not None:
            last_position, bbox = update_kalman_filter(kalman, closest_contour)
            target_lost_frames = 0
            draw_bounding_box(curr_frame, bbox)
            if is_primary:  # Draw the red line only for the primary object
                bbox_center_x = int(last_position[0])
                bbox_center_y = int(last_position[1])
                frame_center = (curr_frame.shape[1] // 2, curr_frame.shape[0] // 2)
                cv2.line(curr_frame, frame_center, (bbox_center_x, bbox_center_y), (0, 0, 255), 1)
        else:
            target_lost_frames += 1
    else:
        target_lost_frames += 1

    if target_lost_frames > target_memory_frames:
        tracked_object = None
        target_lost_frames = 0

    trajectory_points.append(last_position)
    return tracked_object, last_position, target_lost_frames


def track_objects_in_frame(cap, kalman_filters, prev_gray, screen_width, screen_height, threshold_value,
                           min_contour_area, morph_kernel_size, dilation_iterations, erosion_iterations,
                           target_memory_frames):
    """Track multiple objects across video frames."""
    tracked_objects = deque(maxlen=20)  # List of tracked objects, each with a Kalman filter and trajectory
    last_positions = deque(maxlen=20)
    trajectories = deque(maxlen=20)
    lost_frames = deque(maxlen=100)
    primary_object_index = -1  # Index of the primary object to track

    while cap.isOpened():
        start_time = time.time()

        ret, curr_frame = cap.read()
        if not ret or curr_frame is None:
            print("Error: Could not read the frame.")
            break

        curr_frame = resize_frame(curr_frame, screen_width, screen_height)
        curr_gray = convert_to_grayscale(curr_frame)
        prev_gray = resize_frame(prev_gray, screen_width, screen_height)

        diff = calculate_difference(prev_gray, curr_gray)
        thresh = apply_thresholding(diff, threshold_value, morph_kernel_size, dilation_iterations, erosion_iterations)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assign the primary object based on the largest contour in the first frame
        if primary_object_index == -1 and contours:
            largest_contour = find_largest_contour(contours, min_contour_area)
            if largest_contour is not None:
                primary_object_index = len(tracked_objects)
                kalman_filters.append(initialize_kalman_filter())
                tracked_objects.append(largest_contour)
                last_positions.append(None)
                trajectories.append([])
                lost_frames.append(0)

        tracked_objects, last_positions, lost_frames = handle_multiple_objects_tracking(
            contours, tracked_objects, last_positions, lost_frames, target_memory_frames, kalman_filters, curr_frame,
            min_contour_area, trajectories, primary_object_index
        )

        draw_crosshair(curr_frame)

        fps = calculate_fps(start_time)
        display_fps(curr_frame, fps)

        cv2.imshow('Frame', curr_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        prev_gray = curr_gray.copy()


def handle_multiple_objects_tracking(contours, tracked_objects, last_positions, lost_frames, target_memory_frames,
                                     kalman_filters, curr_frame, min_contour_area, trajectories,
                                     primary_object_index):
    """Handle the logic of tracking multiple objects."""
    if len(contours) > len(tracked_objects):
        # Add new objects if more contours are detected than currently tracked objects
        for _ in range(len(contours) - len(tracked_objects)):
            kalman_filters.append(initialize_kalman_filter())
            tracked_objects.append(None)
            last_positions.append(None)
            trajectories.append([])
            lost_frames.append(0)

    for i in range(len(tracked_objects)):
        if tracked_objects[i] is None and contours:
            tracked_objects[i] = find_largest_contour(contours, min_contour_area)
            if tracked_objects[i] is not None:
                last_positions[i], bbox = update_kalman_filter(kalman_filters[i], tracked_objects[i])
                trajectories[i].append(last_positions[i])
                lost_frames[i] = 0
                draw_bounding_box(curr_frame, bbox, color=(255, 0, 0))  # Blue for newly detected objects
        elif tracked_objects[i] is not None:
            is_primary = (i == primary_object_index)
            tracked_objects[i], last_positions[i], lost_frames[i] = update_tracking_with_contours(
                contours, tracked_objects[i], last_positions[i], lost_frames[i], target_memory_frames, kalman_filters[i], curr_frame, trajectories[i], is_primary
            )

    return tracked_objects, last_positions, lost_frames


def main():
    # Parameters for fine-tuning the algorithm
    THRESHOLD_VALUE = 30  # Threshold value for binary thresholding
    MIN_CONTOUR_AREA = 500  # Minimum contour area to consider for tracking
    MORPH_KERNEL_SIZE = (7, 7)  # Kernel size for morphological operations
    DILATION_ITERATIONS = 3  # Number of dilation iterations
    EROSION_ITERATIONS = 1  # Number of erosion iterations
    TARGET_MEMORY_FRAMES = 10  # Number of frames to "remember" the target before resetting

    # Path to the video file
    video_path = r"C:\Users\User\Desktop\test_shuliavka\test_yolo1.avi"

    # Get screen size
    screen_width, screen_height = get_screen_size()

    # Initialize video capture and read the first frame
    cap = initialize_video_capture(video_path)
    first_frame = read_first_frame(cap)
    prev_gray = convert_to_grayscale(first_frame)

    # Initialize the Kalman filter list for multiple objects
    kalman_filters = []

    cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Track the objects across frames
    track_objects_in_frame(cap, kalman_filters, prev_gray, screen_width, screen_height, THRESHOLD_VALUE, MIN_CONTOUR_AREA, MORPH_KERNEL_SIZE,
                          DILATION_ITERATIONS, EROSION_ITERATIONS, TARGET_MEMORY_FRAMES)

    # Release video capture and close all frames
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
