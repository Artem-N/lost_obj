import cv2
import numpy as np
import time
from screeninfo import get_monitors
import math
from collections import deque

# ===========================
# Configuration Parameters
# ===========================

# Video Configuration
VIDEO_PATH = r"C:\Users\User\Desktop\fly\GENERIC_RTSP-realmonitor_2023_09_20_15_31_27.avi"

# Thresholding and Morphological Operations
THRESHOLD_VALUE = 30             # Threshold value for binary thresholding
MIN_CONTOUR_AREA = 10            # Minimum contour area to consider for tracking
MORPH_KERNEL_SIZE = (9, 9)       # Kernel size for morphological operations
DILATION_ITERATIONS = 3          # Number of dilation iterations
EROSION_ITERATIONS = 1           # Number of erosion iterations

# Tracking Parameters
TARGET_MEMORY_FRAMES = 5         # Number of frames to "remember" the target before resetting
MAX_DISTANCE = 1500              # Maximum allowed distance in pixels for contour matching
BBOX_SIZE_THRESHOLD = 5          # Bounding box size threshold multiplier
DYNAMIC_SPEED_MULTIPLIER = 500    # Multiplier for dynamic speed threshold

# Angle Configuration
ANGLE_MIN_DEGREES = -110          # Minimum angle to detect vertical movement
ANGLE_MAX_DEGREES = -70           # Maximum angle to detect vertical movement

# Display Configuration
WINDOW_NAME = 'Frame'             # Name of the display window
FPS_DISPLAY_POSITION = (10, 30)   # Position to display FPS on the frame
CROSSHAIR_COLOR = (0, 255, 0)     # Color of the crosshair (Green)
CROSSHAIR_THICKNESS = 1           # Thickness of the crosshair lines
FPS_TEXT_COLOR = (255, 255, 0)    # Color of the FPS text (Cyan)
FPS_FONT = cv2.FONT_HERSHEY_SIMPLEX
FPS_FONT_SCALE = 1
FPS_FONT_THICKNESS = 2

# Thumbnail Configuration
THUMBNAIL_SIZE = (250, 200)        # Size of the thumbnail (width, height)
THUMBNAIL_PADDING = 10             # Padding from the edges

# Zoom Configuration
ZOOM_SCALE_FACTOR = 2.0             # Factor by which to zoom the ROI
ZOOMED_ROI_SIZE = THUMBNAIL_SIZE    # Size of the zoomed ROI (width, height)

# ===========================
# Helper Functions
# ===========================


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
    resized = cv2.resize(frame, (width, height))
    return resized


def convert_to_grayscale(frame):
    """Convert the frame to grayscale if necessary."""
    # If the frame has 3 channels (e.g., BGR or RGB), convert to grayscale
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray

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
    if contours:
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
    x, y, w, h = cv2.boundingRect(contour)
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
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)


def draw_crosshair(frame, color=CROSSHAIR_COLOR, thickness=CROSSHAIR_THICKNESS):
    """Draw a crosshair across the entire video frame."""
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


def display_fps(frame, fps, position=FPS_DISPLAY_POSITION,
               font=FPS_FONT, font_scale=FPS_FONT_SCALE,
               color=FPS_TEXT_COLOR, thickness=FPS_FONT_THICKNESS):
    """Display the FPS on the frame."""
    cv2.putText(frame, f"FPS: {fps:.2f}", position,
                font, font_scale, color, thickness)


def find_closest_contour(contours, last_position):
    """Find the contour closest to the last known position."""
    if last_position is None or not contours:
        return None  # Return None if last_position or contours are invalid

    closest_contour = None
    min_distance = float("inf")

    for contour in contours:
        # Get the center of the contour's bounding box
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)

        # Calculate the Euclidean distance from the last known position
        distance = np.linalg.norm(np.array(center) - np.array(last_position))
        if distance < min_distance:
            min_distance = distance
            closest_contour = contour  # Assign the closest contour

    return closest_contour  # Return the closest valid contour


def enhance_contrast_linear(roi):
    """Enhance contrast using linear contrast stretching for grayscale images."""

    roi_gray = roi.copy()

    # Find the min and max pixel values in the ROI
    min_val = np.min(roi_gray)
    max_val = np.max(roi_gray)

    # Avoid division by zero in case min_val == max_val
    if max_val - min_val == 0:
        return roi.copy()

    # Scale the pixel values to span the full range [0, 255]
    contrast_stretched = cv2.normalize(roi_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return contrast_stretched


def embed_zoomed_object(frame, bbox):
    """Embed a zoomed image of the object into the frame at a specified position with enhanced contrast."""
    x, y, w, h = bbox

    # Ensure the bounding box is within the frame boundaries
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, frame.shape[1])
    y2 = min(y + h, frame.shape[0])

    # Extract the region of interest (ROI) from the current frame (before any overlays)
    roi = frame[y1:y2, x1:x2]

    # Handle cases where ROI might be too small
    if roi.size == 0:
        return None

    # Enhance contrast using linear stretching
    enhanced_roi = enhance_contrast_linear(roi)

    # Zoom the enhanced ROI by scaling it up
    zoomed_roi = cv2.resize(
        enhanced_roi,
        (0, 0),
        fx=ZOOM_SCALE_FACTOR,
        fy=ZOOM_SCALE_FACTOR,
        interpolation=cv2.INTER_LINEAR
    )

    # Resize the zoomed image to fit into the defined window size
    zoomed_roi = cv2.resize(
        zoomed_roi,
        ZOOMED_ROI_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    # Convert grayscale zoomed ROI back to BGR if necessary
    # if len(zoomed_roi.shape) == 2:
    #     zoomed_roi = cv2.cvtColor(zoomed_roi, cv2.COLOR_GRAY2BGR)

    return zoomed_roi

# ===========================
# Tracking Functions
# ===========================

def track_object_in_frame(cap, kalman, prev_gray, screen_width, screen_height, threshold_value, min_contour_area,
                          morph_kernel_size, dilation_iterations, erosion_iterations, target_memory_frames):
    """Track the object across video frames."""
    tracked_object = None
    last_position = None
    trajectory_points = deque(maxlen=20)
    target_lost_frames = 0
    last_bbox_area = None  # Track the area of the last saved bounding box
    last_speed = 0          # Track the speed of the last tracked object
    frame_center = (screen_width // 2, screen_height // 2)

    while cap.isOpened():
        start_time = time.time()

        ret, curr_frame = cap.read()
        if not ret or curr_frame is None:
            break

        # Resize current frame
        curr_frame = resize_frame(curr_frame, screen_width, screen_height)
        curr_gray = convert_to_grayscale(curr_frame)


        # Calculate difference and apply thresholding
        diff = calculate_difference(prev_gray, curr_gray)
        thresh = apply_thresholding(diff, threshold_value, morph_kernel_size, dilation_iterations, erosion_iterations)

        # Find contours in thresholded frame
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Handle object tracking and draw bounding boxes
        tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed, thumbnail = handle_object_tracking(
            contours, tracked_object, last_position, target_lost_frames, target_memory_frames, kalman, curr_frame,
            min_contour_area, trajectory_points, frame_center, last_bbox_area, last_speed
        )

        # Draw crosshair on the frame
        draw_crosshair(curr_frame)

        # Display the thumbnail if available
        if thumbnail is not None:
            # Define the position for the thumbnail (lower left corner)
            thumbnail_x = THUMBNAIL_PADDING  # 10 pixels from the left edge
            thumbnail_y = screen_height - THUMBNAIL_SIZE[1] - THUMBNAIL_PADDING  # 10 pixels from the bottom edge

            # Ensure the thumbnail fits within the frame
            if thumbnail_y < 0:
                thumbnail_y = THUMBNAIL_PADDING  # If frame height is less than thumbnail height + padding, place it at top-left with padding

            # Overlay the thumbnail on the frame
            # Ensure that the region to place the thumbnail does not exceed frame boundaries
            end_y = thumbnail_y + THUMBNAIL_SIZE[1]
            end_x = thumbnail_x + THUMBNAIL_SIZE[0]
            if end_y > screen_height or end_x > screen_width:
                end_y = min(end_y, screen_height)
                end_x = min(end_x, screen_width)
                thumbnail = thumbnail[0:end_y - thumbnail_y, 0:end_x - thumbnail_x]

            # Overlay the thumbnail
            curr_frame[thumbnail_y:end_y, thumbnail_x:end_x] = thumbnail

        # Calculate and display FPS
        fps = calculate_fps(start_time)
        display_fps(curr_frame, fps)

        # Show the result
        cv2.imshow(WINDOW_NAME, curr_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        prev_gray = curr_gray


def handle_object_tracking(contours, tracked_object, last_position, target_lost_frames, target_memory_frames, kalman,
                           curr_frame, min_contour_area, trajectory_points, frame_center, last_bbox_area, last_speed):
    """Handle the logic of tracking the object, considering bounding box area and speed check."""
    thumbnail = None

    # Initialize tracking if not currently tracking
    if tracked_object is None and contours:
        tracked_object, last_position, last_bbox_area, target_lost_frames = initialize_tracking(
            contours, kalman, min_contour_area, curr_frame, last_bbox_area, target_lost_frames, trajectory_points
        )
    elif tracked_object is not None:
        tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed, thumbnail = update_tracking_with_contours_and_speed(
            contours, tracked_object, last_position, target_lost_frames, target_memory_frames, kalman, curr_frame,
            trajectory_points, frame_center, last_bbox_area, last_speed
        )

    # Perform the angle check
    if len(trajectory_points) >= 2:
        if check_angle_and_stop_tracking(trajectory_points):
            tracked_object = None  # Stop tracking if vertical movement is detected

    return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed, thumbnail


def initialize_tracking(contours, kalman, min_contour_area, curr_frame, last_bbox_area, target_lost_frames, trajectory_points):
    """Initialize object tracking by finding the largest contour and updating Kalman filter."""
    tracked_object = find_largest_contour(contours, min_contour_area)
    if tracked_object is not None:
        # Update Kalman filter and get the object's position and bounding box
        last_position, bbox = update_kalman_filter(kalman, tracked_object)
        if last_position is None:
            return None, None, last_bbox_area, target_lost_frames

        current_bbox_area = bbox[2] * bbox[3]  # Calculate area of the new bounding box

        # Only draw the bounding box if the new area is not more than a specified multiplier of the last saved area
        if last_bbox_area is None or current_bbox_area <= BBOX_SIZE_THRESHOLD * last_bbox_area:
            # Extract and embed the zoomed object thumbnail before drawing the bounding box
            thumbnail = embed_zoomed_object(curr_frame, bbox)

            # Draw the bounding box after extracting the thumbnail to avoid overlay in the thumbnail
            draw_bounding_box(curr_frame, bbox)
            last_bbox_area = current_bbox_area

        # Add the new position to trajectory points and reset lost frames counter
        trajectory_points.append(last_position)
        target_lost_frames = 0

    return tracked_object, last_position, last_bbox_area, target_lost_frames


def check_angle_and_stop_tracking(trajectory_points):
    """Check the angle of movement and stop tracking if vertical movement is detected."""
    dx = trajectory_points[-1][0] - trajectory_points[-2][0]
    dy = trajectory_points[-1][1] - trajectory_points[-2][1]
    angle = math.degrees(math.atan2(dy, dx))

    # Check if the object is moving vertically (between ANGLE_MIN_DEGREES and ANGLE_MAX_DEGREES)
    if ANGLE_MIN_DEGREES <= angle <= ANGLE_MAX_DEGREES:
        return True  # Indicate that tracking should be stopped

    return False


def update_tracking_with_contours_and_speed(contours, tracked_object, last_position, target_lost_frames,
                                            target_memory_frames, kalman, curr_frame, trajectory_points, frame_center,
                                            last_bbox_area, last_speed):
    """Update the Kalman filter, handle contour matching, and check object speed and distance."""
    thumbnail = None

    if contours:
        # Find the closest contour to the last known position
        closest_contour = find_closest_contour(contours, last_position)

        if closest_contour is not None:
            # Update position and bounding box from Kalman filter
            current_position, bbox = update_position_and_bbox(kalman, closest_contour)

            if current_position is None:
                target_lost_frames += 1
                return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed, thumbnail

            # Perform checks for distance, speed, and angle
            if not is_within_distance(current_position, last_position, MAX_DISTANCE):
                target_lost_frames += 1
                return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed, thumbnail

            dynamic_speed_threshold = (last_speed * DYNAMIC_SPEED_MULTIPLIER) if last_speed > 0 else DYNAMIC_SPEED_MULTIPLIER
            if not is_speed_valid(current_position, last_position, last_speed, dynamic_speed_threshold):
                target_lost_frames += 1
                return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed, thumbnail

            # Extract and embed the zoomed object thumbnail before drawing the bounding box
            if bbox is not None:
                thumbnail = embed_zoomed_object(curr_frame, bbox)

            # If all checks passed, draw the bounding box and line
            if should_draw_bbox(last_bbox_area, bbox):
                draw_bounding_box_and_line(curr_frame, bbox, current_position, frame_center)
                last_bbox_area = bbox[2] * bbox[3]  # Update bounding box area

            # Calculate current speed
            dx = current_position[0] - last_position[0]
            dy = current_position[1] - last_position[1]
            last_speed = math.sqrt(dx ** 2 + dy ** 2)

            # Add the current position to the trajectory points for further calculations
            trajectory_points.append(current_position)
            target_lost_frames = 0  # Reset lost frames counter

            # Update last position
            last_position = current_position
    else:
        target_lost_frames += 1

    # Reset tracking if the object is lost for too many frames
    if target_lost_frames > target_memory_frames:
        tracked_object = None
        target_lost_frames = 0

    return tracked_object, last_position, target_lost_frames, last_bbox_area, last_speed, thumbnail


def update_position_and_bbox(kalman, closest_contour):
    """Update the position and bounding box based on Kalman filter and closest contour."""
    position, bbox = update_kalman_filter(kalman, closest_contour)
    return position, bbox


def is_within_distance(current_position, previous_position, max_distance):
    """Check if the current position is within the allowed distance from the previous one."""
    distance = calculate_distance(current_position, previous_position)
    return distance <= max_distance


def calculate_distance(position1, position2):
    """Calculate Euclidean distance between two positions."""
    return math.sqrt((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2)


def is_speed_valid(current_position, previous_position, last_speed, dynamic_speed_threshold):
    """Check if the object's speed is within the allowed dynamic threshold."""
    current_speed = calculate_distance(current_position, previous_position)
    return current_speed <= dynamic_speed_threshold


def should_draw_bbox(last_bbox_area, bbox):
    """Check if the bounding box should be drawn based on its size."""
    current_bbox_area = bbox[2] * bbox[3]
    return last_bbox_area is None or current_bbox_area <= BBOX_SIZE_THRESHOLD * last_bbox_area


def draw_bounding_box_and_line(curr_frame, bbox, last_position, frame_center):
    """Draw the bounding box and red line on the frame."""
    draw_bounding_box(curr_frame, bbox)  # Draw bounding box
    bbox_center_x = int(last_position[0])
    bbox_center_y = int(last_position[1])
    # Uncomment the line below to draw the red line from frame center to bounding box center
    # cv2.line(curr_frame, frame_center, (bbox_center_x, bbox_center_y), (0, 0, 255), 2)  # Draw the red line

# ===========================
# Main Execution Function
# ===========================


def main():
    # Get screen size
    screen_width, screen_height = get_screen_size()

    # Initialize video capture and read the first frame
    cap = initialize_video_capture(VIDEO_PATH)
    first_frame = read_first_frame(cap)

    # Resize the first frame before converting to grayscale
    first_frame = resize_frame(first_frame, screen_width, screen_height)
    prev_gray = convert_to_grayscale(first_frame)

    # Initialize the Kalman filter
    kalman = initialize_kalman_filter()

    # Set the video window to full screen
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Track the object across frames
    try:
        track_object_in_frame(
            cap, kalman, prev_gray, screen_width, screen_height, THRESHOLD_VALUE, MIN_CONTOUR_AREA,
            MORPH_KERNEL_SIZE, DILATION_ITERATIONS, EROSION_ITERATIONS, TARGET_MEMORY_FRAMES
        )
    except Exception as e:
        print(f"An error occurred during tracking: {e}")
    finally:
        # Release video capture and close all frames
        cap.release()
        cv2.destroyAllWindows()
        print("Video capture released and all windows closed.")

# ===========================
# Entry Point
# ===========================


if __name__ == "__main__":
    main()
