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
    """Convert the frame to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


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
    if contours:
        # Find the contour with the maximum area
        contour = max(contours, key=cv2.contourArea)
        # Return the largest contour if its area is above the threshold
        if cv2.contourArea(contour) >= min_contour_area:
            return contour
    return None


def update_kalman_filter(kalman, contour):
    """Update the Kalman filter with the position of the detected contour."""
    # Get the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(contour)

    # Calculate the center position of the bounding box
    position = (x + w // 2, y + h // 2)

    # Create a measurement matrix with the position
    measurement = np.array([[np.float32(position[0])],
                            [np.float32(position[1])]])

    # Update the Kalman filter with the current measurement
    kalman.correct(measurement)

    return position, (x, y, w, h)


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
        prev_gray = resize_frame(prev_gray, screen_width, screen_height)

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

        prev_gray = curr_gray.copy()



def handle_object_tracking(contours, tracked_object, last_position, target_lost_frames, target_memory_frames, kalman,
                           curr_frame, min_contour_area, trajectory_points, frame_center, last_bbox_area, last_speed):
    """Handle the logic of tracking the object, considering bounding box area and speed check."""

    if tracked_object is None and contours:
        # If no object is being tracked, find the largest contour to start tracking
        tracked_object = find_largest_contour(contours, min_contour_area)
        if tracked_object is not None:
            # Update Kalman filter with the object's position
            last_position, bbox = update_kalman_filter(kalman, tracked_object)
            current_bbox_area = bbox[2] * bbox[3]  # Calculate area of the new bounding box
            current_speed = 0  # Set initial speed to 0 since we just started tracking

            if last_bbox_area is None or current_bbox_area <= 1 * last_bbox_area:
                # Draw the bounding box if the new area is not more than twice the last saved area
                draw_bounding_box(curr_frame, bbox)
                last_bbox_area = current_bbox_area  # Update last_bbox_area with the current one
            trajectory_points.append(last_position)
            target_lost_frames = 0
    elif tracked_object is not None:
        # Update tracking if the object is already being tracked
        tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed = update_tracking_with_contours_and_speed(
            contours, tracked_object, last_position, target_lost_frames, target_memory_frames, kalman, curr_frame,
            trajectory_points, frame_center, last_bbox_area, last_speed
        )

    # Calculate angle of movement if we have at least two points
    if len(trajectory_points) >= 2:
        dx = trajectory_points[-1][0] - trajectory_points[-2][0]
        dy = trajectory_points[-1][1] - trajectory_points[-2][1]
        angle = math.degrees(math.atan2(dy, dx))

        # Display the angle on the frame
        cv2.putText(curr_frame, f"Angle: {angle:.2f} degrees", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2)

    return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed


def update_tracking_with_contours_and_speed(contours, tracked_object, last_position, target_lost_frames,
                                            target_memory_frames, kalman, curr_frame, trajectory_points, frame_center,
                                            last_bbox_area, last_speed):
    """Update the Kalman filter, handle contour matching, and check object speed."""

    # Set the dynamic speed threshold based on the last tracked object's speed
    dynamic_speed_threshold = last_speed * 5 if last_speed > 0 else 100  # If last_speed is 0, use the default max speed

    if contours:
        # Find the closest contour to the last known position
        closest_contour, _ = find_closest_contour(contours, last_position)
        if closest_contour is not None:
            # Save the previous position to calculate speed
            previous_position = last_position if last_position is not None else (0, 0)

            # Update Kalman filter with new measurement
            last_position, bbox = update_kalman_filter(kalman, closest_contour)
            current_bbox_area = bbox[2] * bbox[3]  # Calculate the new bounding box area

            # Calculate the speed of the new object
            dx = last_position[0] - previous_position[0]
            dy = last_position[1] - previous_position[1]
            current_speed = math.sqrt(dx ** 2 + dy ** 2)  # Speed is the Euclidean distance between positions

            # Check if the current speed exceeds the dynamic speed threshold
            if current_speed > dynamic_speed_threshold:
                # If the speed is too high, ignore the object completely and do nothing
                print(f"Ignored object with speed {current_speed:.2f}, too fast compared to dynamic threshold {dynamic_speed_threshold:.2f}")
                return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed

            # Check bounding box size and draw if within the allowed range
            if last_bbox_area is None or current_bbox_area <= 5 * last_bbox_area:
                draw_bounding_box(curr_frame, bbox)  # Draw the bounding box
                last_bbox_area = current_bbox_area  # Update last_bbox_area with the current one
                last_speed = current_speed  # Update the speed of the tracked object

                # Draw the red line from the center of the frame to the tracked object
                bbox_center_x = int(last_position[0])
                bbox_center_y = int(last_position[1])
                cv2.line(curr_frame, frame_center, (bbox_center_x, bbox_center_y), (0, 0, 255), 1)

            else:
                # If the size is too big, ignore it
                print(f"Ignored object with area {current_bbox_area:.2f}, too large compared to last bbox area")
            target_lost_frames = 0
        else:
            target_lost_frames += 1
    else:
        target_lost_frames += 1

    # Reset tracking if the object is lost for too many frames
    if target_lost_frames > target_memory_frames:
        tracked_object = None
        target_lost_frames = 0
    else:
        # Only add the trajectory point if the object is being tracked (i.e., not ignored)
        if tracked_object is not None:
            trajectory_points.append(last_position)

    return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed



def main():
    # Parameters for fine-tuning the algorithm
    THRESHOLD_VALUE = 30  # Threshold value for binary thresholding
    MIN_CONTOUR_AREA = 50  # Minimum contour area to consider for tracking
    MORPH_KERNEL_SIZE = (7, 7)  # Kernel size for morphological operations
    DILATION_ITERATIONS = 3  # Number of dilation iterations
    EROSION_ITERATIONS = 1  # Number of erosion iterations
    TARGET_MEMORY_FRAMES = 5  # Number of frames to "remember" the target before resetting

    # Path to the video file
    video_path = r"C:\Users\User\Desktop\fly\GeneralPT4S4_2023_09_20_15_29_57.avi"

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
