import cv2
import numpy as np
import time
from screeninfo import get_monitors


def get_screen_size():
    """Get monitor info h/w"""
    monitor = get_monitors()[0]
    return monitor.width, monitor.height


def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: could not open video")
    return cap


def read_first_frame(cap):
    ret, frame = cap.read()
    if ret or frame is None:
        raise Exception("Errorr: could not read first frame")


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
    return cv2.resize(frame, (width, height))


def convert_to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def calculate_difference(prev_frame, curr_frame):
    diff = cv2.absdiff(prev_frame, curr_frame)
    diff = cv2.GaussianBlur(diff, (7, 7), 0)
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
    (x, y, w, h) = cv2.boundingRect(contour)

    position = (x + w // 2, y + h // 2)

    measurment = np.array([[np.float32(position[0])],
                           [np.float32(position[1])]])

    kalman.correct(measurment)

    return position, (x, y, w, h)


def find_largest_countour(contours, min_contour_area):
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) >= min_contour_area:
            return contour
    return None


def draw_bounding_box(frame, bbox, color=(255, 0, 0)):
    """Draw a bounding box around the detected object."""
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # Draw the rectangle

def find_closest_contour(contours, last_position):
    if last_position is None:
        return None, float("inf")

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


def update_tracking_with_contours(contours, tracked_object, last_position, target_lost_frame, target_memory_frames,
                                  kalman, curr_frame, trajectory_points, num_prediction, is_primary):
    if contours and last_position is not None:
        closest_contour = find_closest_contour(contours, last_position)
        if closest_contour is not None:
            last_position, bbox = update_kalman_filter(kalman, closest_contour)
            target_lost_frame = 0
            draw_bounding_box(curr_frame, bbox)
            if is_primary:
                bbox_center_x = int(last_position[0])
                bbox_center_y = int(last_position[1])
                frame_center = (curr_frame.shape[1] // 2, curr_frame.shape[0] // 2)
                cv2.line(curr_frame, frame_center, (bbox_center_x, bbox_center_y), (0, 0, 255), 1)
        else:
            target_lost_frame += 1
    else:
        target_lost_frame += 1

    if target_lost_frame > target_memory_frames:
        tracked_object = None
        target_lost_frame = 0

    trajectory_points.append(last_position)
    return tracked_object, last_position, target_lost_frame


def track_object_in_frame(cap, kalman_filters, prev_gray, screen_widht, screen_height, threshold_value,
                          min_contour_area, morph_kernel_size, dilation_iteration, erosion_iteration, num_predictions,
                          target_memory_frames):
    tracked_object = []
    last_position = []
    trajectories = []
    lost_frame = []
    primary_object_index = -1

    while cap.isOpened():
        start_time = time.time()

        ret, curr_frame = cap.read()
        if not ret or curr_frame is None:
            print("Error: could not read the frame")
            break

        curr_frame = resize_frame(curr_frame, screen_widht, screen_height)
        curr_gray = convert_to_grayscale(curr_frame)
        prev_gray = resize_frame(curr_gray, screen_widht, screen_height)

        diff = calculate_difference(prev_gray, curr_frame)
        thresh = apply_thresholding(diff, threshold_value)

