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


def update_tracking_with_contours(contours, tracked_object, last_position, last_bbox_area, kalman, curr_frame, trajectory_points, is_primary, initial_area):
    """Update the Kalman filter and handle contour matching, ensuring the area is consistent."""
    if contours and last_position is not None:
        # Find the contour closest to the last known position of the tracked object
        closest_contour, _ = find_closest_contour(contours, last_position)
        if closest_contour is not None:
            (x, y, w, h) = cv2.boundingRect(closest_contour)
            current_area = w * h

            # Check if the current bounding box area is within acceptable range compared to the last known bounding box area
            if last_bbox_area is None or current_area <= 5 * last_bbox_area:
                # If the current bounding box area is not more than twice the last bounding box area, update tracking
                last_position, bbox = update_kalman_filter(kalman, closest_contour)
                last_bbox_area = current_area  # Update the last bounding box area

                draw_bounding_box(curr_frame, bbox)  # Draw bounding box around the tracked object
                if is_primary:  # Draw the red line only for the primary object
                    bbox_center_x = int(last_position[0])
                    bbox_center_y = int(last_position[1])
                    frame_center = (curr_frame.shape[1] // 2, curr_frame.shape[0] // 2)
                    cv2.line(curr_frame, frame_center, (bbox_center_x, bbox_center_y), (0, 0, 255), 1)
            else:
                # If the current bounding box is more than twice as large, skip drawing it
                print(f"Skipped drawing a bounding box with area {current_area}, which is too large.")
        else:
            tracked_object = None  # Reset tracking if the object is not detected in the current frame
    else:
        tracked_object = None  # Reset tracking if the object is not detected in the current frame

    trajectory_points.append(last_position)
    return tracked_object, last_position, last_bbox_area



def track_objects_in_frame(cap, kalman_filters, prev_gray, screen_width, screen_height, threshold_value, min_contour_area, morph_kernel_size,
                          dilation_iterations, erosion_iterations, num_predictions, target_memory_frames):
    """Track multiple objects across video frames."""
    tracked_objects = []  # List of tracked objects, each with a Kalman filter and trajectory
    last_positions = []
    trajectories = []
    lost_frames = []
    bbox_frame_counter = []  # Counter for how many frames each bounding box has been shown
    last_bbox_areas = []  # Initialize last_bbox_areas to store the area of each bounding box
    primary_object_index = -1  # Index of the primary object to track
    last_primary_bbox_area = None  # Last area of the primary bounding box

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
                bbox_frame_counter.append(0)  # Add frame counter for the primary object
                last_bbox_areas.append(None)  # Initialize last_bbox_areas for the primary object

        # Pass the necessary variables to handle_multiple_objects_tracking
        tracked_objects, last_positions, last_bbox_areas, lost_frames, bbox_frame_counter, last_primary_bbox_area = handle_multiple_objects_tracking(
            contours, tracked_objects, last_positions, last_bbox_areas, lost_frames, bbox_frame_counter, kalman_filters,
            curr_frame, min_contour_area, trajectories, primary_object_index, last_primary_bbox_area
        )

        draw_crosshair(curr_frame)

        fps = calculate_fps(start_time)
        display_fps(curr_frame, fps)

        cv2.imshow('Frame', curr_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        prev_gray = curr_gray.copy()


def handle_multiple_objects_tracking(contours, tracked_objects, last_positions, last_bbox_areas, lost_frames,
                                     bbox_frame_counter, kalman_filters, curr_frame, min_contour_area, trajectories,
                                     primary_object_index, last_primary_bbox_area):
    """Handle the logic of tracking multiple objects, ensuring bbox size consistency and managing lost objects."""

    if len(contours) > len(tracked_objects):
        # Add new Kalman filters and counters for any extra objects
        for _ in range(len(contours) - len(tracked_objects)):
            kalman_filters.append(initialize_kalman_filter())
            tracked_objects.append(None)
            last_positions.append(None)
            trajectories.append([])
            last_bbox_areas.append(None)  # Initialize bbox area list
            lost_frames.append(0)
            bbox_frame_counter.append(0)  # Start counting frames for the new object

    for i in range(len(tracked_objects)):
        if tracked_objects[i] is None and contours:
            # Initialize tracking for newly detected object
            tracked_objects[i] = find_largest_contour(contours, min_contour_area)
            if tracked_objects[i] is not None:
                last_positions[i], bbox = update_kalman_filter(kalman_filters[i], tracked_objects[i])
                last_bbox_areas[i] = bbox[2] * bbox[3]  # Update initial bbox area
                trajectories[i].append(last_positions[i])
                draw_bounding_box(curr_frame, bbox, color=(255, 0, 0))  # Draw bounding box for newly detected objects
                lost_frames[i] = 0  # Reset lost frame counter
                bbox_frame_counter[i] = 0  # Reset frame counter for bounding box
                if i == primary_object_index:
                    last_primary_bbox_area = last_bbox_areas[i]  # Update the last primary bbox area
        elif tracked_objects[i] is not None:
            is_primary = (i == primary_object_index)
            tracked_objects[i], last_positions[i], last_bbox_areas[i] = update_tracking_with_contours(
                contours, tracked_objects[i], last_positions[i], last_bbox_areas[i], kalman_filters[i], curr_frame,
                trajectories[i], is_primary, last_primary_bbox_area
            )

            # If it's not the primary object and bbox size is within limits, draw a green bbox
            if not is_primary and last_primary_bbox_area is not None:
                (x, y, w, h) = cv2.boundingRect(tracked_objects[i])
                current_bbox_area = w * h
                if current_bbox_area <= last_primary_bbox_area:
                    # Increment the frame counter for the bounding box
                    bbox_frame_counter[i] += 1

                    if bbox_frame_counter[i] <= 5:  # Only draw the green bounding box for the first 5 frames
                        draw_bounding_box(curr_frame, (x, y, w, h),
                                          color=(0, 255, 0))  # Green bounding box for non-primary
                    else:
                        # Skip drawing after 5 frames
                        print(f"Deleted green bbox for object {i} after 5 frames.")

    return tracked_objects, last_positions, last_bbox_areas, lost_frames, bbox_frame_counter, last_primary_bbox_area


def main():
    # Parameters for fine-tuning the algorithm
    THRESHOLD_VALUE = 30  # Threshold value for binary thresholding
    MIN_CONTOUR_AREA = 10  # Minimum contour area to consider for tracking
    MORPH_KERNEL_SIZE = (7, 7)  # Kernel size for morphological operations
    DILATION_ITERATIONS = 3  # Number of dilation iterations
    EROSION_ITERATIONS = 1  # Number of erosion iterations
    NUM_PREDICTIONS = 10  # Number of future positions to predict
    TARGET_MEMORY_FRAMES = 5  # Number of frames to "remember" the target before resetting

    # Path to the video file
    video_path = r"C:\Users\User\Desktop\fly\GENERIC_RTSP-realmonitor_2023_09_20_15_35_49.avi"

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
                          DILATION_ITERATIONS, EROSION_ITERATIONS, NUM_PREDICTIONS, TARGET_MEMORY_FRAMES)

    # Release video capture and close all frames
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
