import cv2
import numpy as np

def initialize_video_capture(video_path):
    """Initialize video capture from the provided video file path."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")
    return cap

def read_and_crop_first_frame(cap, ignore_rows):
    """Read the first frame from the video and crop the top rows."""
    ret, frame = cap.read()
    if not ret or frame is None:
        raise Exception("Error: Could not read the first frame.")
    frame = frame[ignore_rows:, :]  # Crop the frame to ignore top rows
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

def predict_future_positions(kalman, num_predictions):
    """Predict and return future positions using the Kalman filter."""
    future_positions = []
    for _ in range(num_predictions):
        prediction = kalman.predict()
        future_positions.append((int(prediction[0]), int(prediction[1])))
    return future_positions

def draw_bounding_box(frame, bbox):
    """Draw a bounding box around the detected object."""
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

def draw_trajectory(frame, trajectory_points, max_trajectory_length):
    """Draw the trajectory of the tracked object on the frame."""
    total_length = 0
    for i in range(len(trajectory_points) - 1, 0, -1):
        pt1 = trajectory_points[i]
        pt2 = trajectory_points[i - 1]
        total_length += np.linalg.norm(np.array(pt1) - np.array(pt2))
        if total_length > max_trajectory_length:
            trajectory_points = trajectory_points[i:]
            break
    for i in range(1, len(trajectory_points)):
        pt1 = tuple(map(int, trajectory_points[i-1]))
        pt2 = tuple(map(int, trajectory_points[i]))
        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

def adaptive_process_noise(speed):
    """Adjust the process noise based on the speed of the target."""
    base_noise = 0.1
    if speed < 5:
        return base_noise
    elif speed < 20:
        return base_noise * 5
    else:
        return base_noise * 10

def jacobian(state):
    """Jacobian of the state transition function for EKF."""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]], np.float32)

def process_frame(cap, kalman, prev_gray, ignore_rows, threshold_value, min_contour_area,
                  morph_kernel_size, dilation_iterations, erosion_iterations, num_predictions,
                  max_trajectory_length, target_memory_frames):
    """Process each frame of the video to track the object, draw the bounding box, and display the enhanced crosshair."""
    tracked_object = None
    last_position = None
    trajectory_points = []
    target_lost_frames = 0

    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret or curr_frame is None:
            print("Error: Could not read the frame.")
            break

        curr_frame = curr_frame[ignore_rows:, :]  # Crop the frame to ignore top rows
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
                prediction = kalman.predict()

                # Estimate speed and adapt process noise
                speed = np.sqrt(prediction[2]**2 + prediction[3]**2)
                kalman.processNoiseCov = np.eye(4, dtype=np.float32) * adaptive_process_noise(speed)

                draw_bounding_box(curr_frame, bbox)
                trajectory_points.append(last_position)
                total_length = 0
                for i in range(len(trajectory_points) - 1, 0, -1):
                    pt1 = trajectory_points[i]
                    pt2 = trajectory_points[i - 1]
                    total_length += np.linalg.norm(np.array(pt1) - np.array(pt2))
                    if total_length > max_trajectory_length:
                        trajectory_points = trajectory_points[i:]
                        break

            target_lost_frames = 0
        else:
            target_lost_frames += 1

        if target_lost_frames > target_memory_frames:
            tracked_object = None
            target_lost_frames = 0

        cv2.imshow('Frame', curr_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        prev_gray = curr_gray.copy()


IGNORE_ROWS = 100
def main():
    # Path to your video file
    video_path = r"C:\Users\User\Desktop\fly\GENERIC_RTSP-realmonitor_2023_09_20_15_57_55.avi"

    # Initialize video capture and read the first frame
    cap = initialize_video_capture(video_path)
    first_frame = read_and_crop_first_frame(cap, IGNORE_ROWS)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Initialize the Kalman filter
    kalman = initialize_kalman_filter()

    # Parameters for fine-tuning

    THRESHOLD_VALUE = 40
    MIN_CONTOUR_AREA = 100
    MORPH_KERNEL_SIZE = (5, 5)
    DILATION_ITERATIONS = 3
    EROSION_ITERATIONS = 1
    NUM_PREDICTIONS = 50
    MAX_TRAJECTORY_LENGTH = 100
    TARGET_MEMORY_FRAMES = 5

    # Process video frames
    process_frame(cap, kalman, prev_gray, IGNORE_ROWS, THRESHOLD_VALUE, MIN_CONTOUR_AREA,
                  MORPH_KERNEL_SIZE, DILATION_ITERATIONS, EROSION_ITERATIONS, NUM_PREDICTIONS,
                  MAX_TRAJECTORY_LENGTH, TARGET_MEMORY_FRAMES)

    # Release video capture and close all frames
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()