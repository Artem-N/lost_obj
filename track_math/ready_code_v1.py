import cv2
import numpy as np
import time
from screeninfo import get_monitors


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


def find_largest_contour(contours, min_contour_area):
    """Find and return the largest contour above a certain area threshold."""
    if contours:
        # Find the contour with the maximum area
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) >= min_contour_area:
            return contour  # Return the largest contour if its area is above the threshold
    return None


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


# def predict_future_positions(kalman, num_predictions):
#     """Predict and return future positions using the Kalman filter."""
#     future_positions = []
#
#     # Predict future positions for the specified number of frames
#     for _ in range(num_predictions):
#         prediction = kalman.predict()  # Predict the next position
#         future_positions.append((int(prediction[0][0]), int(prediction[1][0])))  # Extract and append the predicted position
#
#     # return future_positions
#     return None


def draw_bounding_box(frame, bbox):
    """Draw a bounding box around the detected object."""
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw the rectangle with a blue color


# def draw_trajectory(frame, trajectory_points, max_trajectory_length):
#     """Draw the trajectory of the tracked object on the frame."""
#     total_length = 0
#
#     # Iterate over the trajectory points in reverse order to calculate the length
#     for i in range(len(trajectory_points) - 1, 0, -1):
#         pt1 = trajectory_points[i]
#         pt2 = trajectory_points[i - 1]
#         total_length += np.linalg.norm(np.array(pt1) - np.array(pt2))
#
#         # Trim the trajectory if it exceeds the maximum length
#         if total_length > max_trajectory_length:
#             trajectory_points = trajectory_points[i:]
#             break
#
#     # Draw the trajectory line between the points
#     for i in range(1, len(trajectory_points)):
#         pt1 = tuple(map(int, trajectory_points[i - 1]))
#         pt2 = tuple(map(int, trajectory_points[i]))
#         cv2.line(frame, pt1, pt2, (0, 255, 255), 2)  # Draw the line with a yellow color


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
    # return None


def track_object_in_frame(cap, kalman, prev_gray, screen_width, screen_height, crosshair_length, threshold_value, min_contour_area, morph_kernel_size,
                          dilation_iterations, erosion_iterations, num_predictions, max_trajectory_length, target_memory_frames):
    """Track the object across video frames."""
    tracked_object = None
    last_position = None
    trajectory_points = []
    target_lost_frames = 0
    frame_center = (screen_width // 2, screen_height // 2)

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

        tracked_object, last_position, target_lost_frames = handle_object_tracking(
            contours, tracked_object, last_position, target_lost_frames, target_memory_frames, kalman, curr_frame, min_contour_area, trajectory_points, frame_center, num_predictions
        )

        draw_crosshair(curr_frame)

        fps = calculate_fps(start_time)
        display_fps(curr_frame, fps)

        cv2.imshow('Frame', curr_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        prev_gray = curr_gray.copy()


def handle_object_tracking(contours, tracked_object, last_position, target_lost_frames, target_memory_frames, kalman, curr_frame, min_contour_area, trajectory_points, frame_center, num_predictions):
    """Handle the logic of tracking the object."""
    if tracked_object is None and contours:
        tracked_object = find_largest_contour(contours, min_contour_area)
        if tracked_object is not None:
            last_position, bbox = update_kalman_filter(kalman, tracked_object)
            trajectory_points.append(last_position)
            target_lost_frames = 0
            draw_bounding_box(curr_frame, bbox)
    elif tracked_object is not None:
        tracked_object, last_position, target_lost_frames = update_tracking_with_contours(
            contours, tracked_object, last_position, target_lost_frames, target_memory_frames, kalman, curr_frame, trajectory_points, frame_center, num_predictions
        )
    return tracked_object, last_position, target_lost_frames


def update_tracking_with_contours(contours, tracked_object, last_position, target_lost_frames, target_memory_frames, kalman, curr_frame, trajectory_points, frame_center, num_predictions):
    """Update the Kalman filter and handle contour matching."""
    if contours:
        closest_contour, min_distance = find_closest_contour(contours, last_position)
        if closest_contour is not None:
            last_position, bbox = update_kalman_filter(kalman, closest_contour)
            # prediction = kalman.predict()
            target_lost_frames = 0
            draw_bounding_box(curr_frame, bbox)
        else:
            target_lost_frames += 1
    else:
        target_lost_frames += 1

    if target_lost_frames > target_memory_frames:
        tracked_object = None
        target_lost_frames = 0
    else:
        bbox_center_x = int(last_position[0])
        bbox_center_y = int(last_position[1])
        cv2.line(curr_frame, frame_center, (bbox_center_x, bbox_center_y), (0, 0, 255), 1)
        trajectory_points.append(last_position)

    return tracked_object, last_position, target_lost_frames


def find_closest_contour(contours, last_position):
    """Find the contour closest to the last known position."""
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


def main():
    # Parameters for fine-tuning the algorithm
    THRESHOLD_VALUE = 40  # Threshold value for binary thresholding
    MIN_CONTOUR_AREA = 500  # Minimum contour area to consider for tracking
    MORPH_KERNEL_SIZE = (7, 7)  # Kernel size for morphological operations
    DILATION_ITERATIONS = 3  # Number of dilation iterations
    EROSION_ITERATIONS = 1  # Number of erosion iterations
    NUM_PREDICTIONS = 10  # Number of future positions to predict
    MAX_TRAJECTORY_LENGTH = 100  # Maximum length of the trajectory in pixels
    TARGET_MEMORY_FRAMES = 5  # Number of frames to "remember" the target before resetting
    CROSSHAIR_LENGTH = 25  # Length of the crosshair lines in pixels

    # Path to the video file
    video_path = r"C:\Users\User\Desktop\fly\IMG_0 (1).MOV"

    # Get screen size
    screen_width, screen_height = get_screen_size()

    # Initialize video capture and read the first frame
    cap = initialize_video_capture(video_path)
    first_frame = read_first_frame(cap)
    prev_gray = convert_to_grayscale(first_frame)

    # Initialize the Kalman filter
    kalman = initialize_kalman_filter()

    cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Track the object across frames
    track_object_in_frame(cap, kalman, prev_gray, screen_width, screen_height, CROSSHAIR_LENGTH, THRESHOLD_VALUE, MIN_CONTOUR_AREA, MORPH_KERNEL_SIZE,
                          DILATION_ITERATIONS, EROSION_ITERATIONS, NUM_PREDICTIONS, MAX_TRAJECTORY_LENGTH, TARGET_MEMORY_FRAMES)

    # Release video capture and close all frames
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
