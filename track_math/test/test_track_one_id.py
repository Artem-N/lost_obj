import cv2
import numpy as np
import time
from screeninfo import get_monitors

# Global variables to store tracked objects and their positions
tracked_objects = {}
selected_id = None
target_lost_frames = 0
next_id = 1  # To generate unique IDs for new objects

def get_screen_size():
    """Get the screen width and height."""
    monitor = get_monitors()[0]  # Get the primary monitor's information
    return monitor.width, monitor.height

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

def resize_frame(frame, width, height):
    """Resize the frame to the specified width and height."""
    return cv2.resize(frame, (width, height))

def convert_to_grayscale(frame):
    """Convert the frame to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def calculate_difference(prev_gray, curr_gray):
    """Calculate the absolute difference between consecutive frames."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    diff = cv2.GaussianBlur(diff, (7, 7), 0)
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

def update_kalman_filter(kalman, contour):
    """Update the Kalman filter with the position of the detected contour."""
    (x, y, w, h) = cv2.boundingRect(contour)
    position = (x + w // 2, y + h // 2)
    measurement = np.array([[np.float32(position[0])],
                            [np.float32(position[1])]])
    kalman.correct(measurement)
    return position, (x, y, w, h)

def draw_bounding_box(frame, bbox):
    """Draw a bounding box around the detected object."""
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

def draw_crosshair(frame):
    """Draw a crosshair across the entire video frame."""
    color = (0, 255, 0)
    thickness = 1
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    cv2.line(frame, (0, center_y), (frame.shape[1], center_y), color, thickness)
    cv2.line(frame, (center_x, 0), (center_x, frame.shape[0]), color, thickness)

def calculate_fps(start_time):
    """Calculate the frames per second (FPS)."""
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    return fps

def display_fps(frame, fps):
    """Display the FPS on the frame."""
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

def track_object_in_frame(cap, kalman, prev_gray, screen_width, screen_height, threshold_value, min_contour_area, morph_kernel_size,
                          dilation_iterations, erosion_iterations, target_memory_frames):
    """Track the object across video frames."""
    global tracked_objects, selected_id, next_id

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

        handle_object_tracking(contours, kalman, curr_frame, min_contour_area, frame_center, target_memory_frames)

        draw_crosshair(curr_frame)

        fps = calculate_fps(start_time)
        display_fps(curr_frame, fps)

        cv2.imshow('Frame', curr_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        prev_gray = curr_gray.copy()

def handle_object_tracking(contours, kalman, curr_frame, min_contour_area, frame_center, target_memory_frames):
    global tracked_objects, selected_id, target_lost_frames, next_id

    DISTANCE_THRESHOLD = 50  # Set this based on the object's expected movement speed

    if selected_id is not None:
        tracked_object = tracked_objects.get(selected_id)
        if tracked_object is not None:
            closest_contour, distance = find_closest_contour(contours, tracked_object['position'])
            if closest_contour is not None:
                position, bbox = update_kalman_filter(kalman, closest_contour)
                tracked_objects[selected_id]['position'] = position
                draw_bounding_box(curr_frame, bbox)
                target_lost_frames = 0  # Reset lost frames count
                print(f"[DEBUG] Tracking object ID: {selected_id}")
            else:
                target_lost_frames += 1
                print(f"[DEBUG] Lost frames count: {target_lost_frames}")
                if target_lost_frames > target_memory_frames:
                    print(f"[DEBUG] Object with ID: {selected_id} is lost. Storing last known position.")
                    tracked_objects[selected_id]['lost'] = True  # Mark as lost
        else:
            print("[DEBUG] Tracked object is lost or removed from memory.")
            selected_id = None
    else:
        # Check if any lost object is close to a detected contour
        for obj_id, obj_data in tracked_objects.items():
            if obj_data.get('lost'):
                closest_contour, distance = find_closest_contour(contours, obj_data['position'])
                if closest_contour is not None and distance < DISTANCE_THRESHOLD:
                    print(f"[DEBUG] Object reappeared close to its last known position. Reassigning ID: {obj_id}")
                    position, bbox = update_kalman_filter(kalman, closest_contour)
                    tracked_objects[obj_id]['position'] = position
                    tracked_objects[obj_id]['lost'] = False  # Mark as found
                    selected_id = obj_id
                    draw_bounding_box(curr_frame, bbox)
                    return  # Exit early since we've reassigned the ID

        print("[DEBUG] No object is being tracked, attempting to detect a new object.")
        largest_contour = find_largest_contour(contours, min_contour_area)
        if largest_contour is not None:
            position, bbox = update_kalman_filter(kalman, largest_contour)
            tracked_objects[next_id] = {'position': position, 'lost': False}
            selected_id = next_id
            next_id += 1
            draw_bounding_box(curr_frame, bbox)
            print(f"[DEBUG] New object detected and assigned ID: {selected_id}")


def main():
    # Parameters for fine-tuning the algorithm
    THRESHOLD_VALUE = 30
    MIN_CONTOUR_AREA = 100
    MORPH_KERNEL_SIZE = (7, 7)
    DILATION_ITERATIONS = 3
    EROSION_ITERATIONS = 1
    TARGET_MEMORY_FRAMES = 5

    # Path to the video file
    video_path = r"C:\Users\User\Desktop\fly\GENERIC_RTSP-realmonitor_2023_09_20_15_57_55.avi"

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
    track_object_in_frame(cap, kalman, prev_gray, screen_width, screen_height, THRESHOLD_VALUE, MIN_CONTOUR_AREA, MORPH_KERNEL_SIZE,
                          DILATION_ITERATIONS, EROSION_ITERATIONS, TARGET_MEMORY_FRAMES)

    # Release video capture and close all frames
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
