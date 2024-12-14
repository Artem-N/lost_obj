import cv2
import numpy as np

def initialize_video_capture(video_path):
    """Initialize video capture from the provided video file path."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")
    return cap

def read_first_frame(cap):
    """Read and return the first frame from the video."""
    ret, frame = cap.read()
    if not ret or frame is None:
        raise Exception("Error: Could not read the first frame.")
    return frame

def initialize_kalman_filter():
    """Initialize and return a Kalman filter for tracking."""
    kalman = cv2.KalmanFilter(4, 2)
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

def calculate_difference(prev_frame, curr_frame):
    """Calculate the absolute difference between consecutive frames."""
    diff = cv2.absdiff(prev_frame, curr_frame)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    return diff

def apply_thresholding(diff, threshold_value, kernel_size, dilation_iterations, erosion_iterations):
    """Apply thresholding and morphological operations to detect moving objects."""
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = np.ones(kernel_size, np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=dilation_iterations)
    thresh = cv2.erode(thresh, kernel, iterations=erosion_iterations)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh

def find_largest_contour(contours, min_contour_area):
    """Find and return the largest contour above a certain area threshold."""
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) >= min_contour_area:
            return contour
    return None

def update_kalman_filter(kalman, contour):
    """Update the Kalman filter with the position of the detected contour."""
    (x, y, w, h) = cv2.boundingRect(contour)
    position = (x + w / 2, y + h / 2)
    measurement = np.array([[np.float32(position[0])],
                            [np.float32(position[1])]])
    kalman.correct(measurement)
    return position, (x, y, w, h)

def calculate_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def predict_future_position(kalman, last_position, center_x, center_y, bullet_speed, frame_rate):
    """Predict the future position of the target considering the bullet speed."""
    prediction = kalman.predict()
    velocity_x, velocity_y = prediction[2], prediction[3]
    distance_to_target = calculate_distance((center_x, center_y), last_position)
    time_to_target = distance_to_target / bullet_speed
    future_position_x = int(last_position[0] + velocity_x * time_to_target * frame_rate)
    future_position_y = int(last_position[1] + velocity_y * time_to_target * frame_rate)
    return future_position_x, future_position_y

def draw_crosshair(frame, center_x, center_y, length):
    """Draw a crosshair in the center of the frame."""
    color = (0, 255, 0)  # Green crosshair
    cv2.line(frame, (center_x - length, center_y), (center_x + length, center_y), color, 1)
    cv2.line(frame, (center_x, center_y - length), (center_x, center_y + length), color, 1)

def draw_bounding_box(frame, bbox):
    """Draw a bounding box around the detected object."""
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

def draw_trajectory(frame, trajectory_points, max_trajectory_length):
    """Draw the trajectory of the tracked object."""
    total_length = 0
    for i in range(len(trajectory_points) - 1, 0, -1):
        pt1 = trajectory_points[i]
        pt2 = trajectory_points[i - 1]
        total_length += calculate_distance(pt1, pt2)
        if total_length > max_trajectory_length:
            trajectory_points = trajectory_points[i:]
            break
    for i in range(1, len(trajectory_points)):
        pt1 = tuple(map(int, trajectory_points[i-1]))
        pt2 = tuple(map(int, trajectory_points[i]))
        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

def process_frame(cap, kalman, prev_gray, frame_height, frame_width, crosshair_length,
                  threshold_value, min_contour_area, morph_kernel_size, dilation_iterations,
                  erosion_iterations, num_predictions, max_trajectory_length, bullet_speed, frame_rate):
    """Process each frame of the video to track the object, draw the bounding box, and display the crosshair."""
    tracked_object = None
    last_position = None
    trajectory_points = []
    target_lost_frames = 0
    target_memory_frames = 30
    center_x, center_y = frame_width // 2, frame_height // 2

    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret or curr_frame is None:
            print("Error: Could not read the frame.")
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        diff = calculate_difference(prev_gray, curr_gray)
        thresh = apply_thresholding(diff, threshold_value, morph_kernel_size, dilation_iterations, erosion_iterations)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if tracked_object is None and contours:
            tracked_object = find_largest_contour(contours, min_contour_area)
            if tracked_object is not None:
                last_position, bbox = update_kalman_filter(kalman, tracked_object)
                trajectory_points.append(last_position)
                target_lost_frames = 0

        elif tracked_object is not None:
            closest_contour = None
            min_distance = float("inf")
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                center = (x + w / 2, y + h / 2)
                distance = np.linalg.norm(np.array(center) - np.array(last_position))
                if distance < min_distance:
                    min_distance = distance
                    closest_contour = contour

            if closest_contour is not None:
                last_position, bbox = update_kalman_filter(kalman, closest_contour)
                future_position_x, future_position_y = predict_future_position(kalman, last_position, center_x, center_y, bullet_speed, frame_rate)
                draw_bounding_box(curr_frame, bbox)
                cv2.line(curr_frame, (center_x, center_y), (future_position_x, future_position_y), (0, 255, 255), 2)
                cv2.circle(curr_frame, (future_position_x, future_position_y), 5, (0, 255, 0), -1)
                trajectory_points.append(last_position)
                target_lost_frames = 0
                draw_trajectory(curr_frame, trajectory_points, max_trajectory_length)
            else:
                target_lost_frames += 1
        else:
            target_lost_frames += 1

        if target_lost_frames > target_memory_frames:
            tracked_object = None
            target_lost_frames = 0

        draw_crosshair(curr_frame, center_x, center_y, crosshair_length)
        cv2.imshow('Frame', curr_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        prev_gray = curr_gray.copy()

def main():
    # Path to your video file
    video_path = r"C:\Users\User\Desktop\fly\Тепловізор - гарне х3 (1).avi"

    # Initialize video capture and read the first frame
    cap = initialize_video_capture(video_path)
    first_frame = read_first_frame(cap)
    frame_height, frame_width = first_frame.shape[:2]

    # Convert the first frame to grayscale
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Initialize the Kalman filter
    kalman = initialize_kalman_filter()

    # Parameters for fine-tuning
    THRESHOLD_VALUE = 40
    MIN_CONTOUR_AREA = 100
    MORPH_KERNEL_SIZE = (5, 5)
    DILATION_ITERATIONS = 3
    EROSION_ITERATIONS = 1
    NUM_PREDICTIONS = 10
    MAX_TRAJECTORY_LENGTH = 100
    CROSSHAIR_LENGTH = 50

    # Bullet parameters (for simulation)
    bullet_speed = 1000  # Speed of the bullet in pixels per second
    frame_rate = 30  # Assuming 30 frames per second

    # Process video frames
    process_frame(cap, kalman, prev_gray, frame_height, frame_width, CROSSHAIR_LENGTH,
                  THRESHOLD_VALUE, MIN_CONTOUR_AREA, MORPH_KERNEL_SIZE, DILATION_ITERATIONS,
                  EROSION_ITERATIONS, NUM_PREDICTIONS, MAX_TRAJECTORY_LENGTH, bullet_speed, frame_rate)

    # Release video capture and close all frames
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()