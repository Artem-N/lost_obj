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
VIDEO_PATH = r"E:\video_for_test\fly\clear_video\GENERIC_RTSP-realmonitor_2023_09_20_15_30_00.avi"

# Thresholding and Morphological Operations
THRESHOLD_VALUE = 30  # Threshold value for binary thresholding
MIN_CONTOUR_AREA = 10  # Minimum contour area to consider for tracking
MORPH_KERNEL_SIZE = (9, 9)  # Kernel size for morphological operations
DILATION_ITERATIONS = 3  # Number of dilation iterations
EROSION_ITERATIONS = 1  # Number of erosion iterations

# Tracking Parameters
TARGET_MEMORY_FRAMES = 5  # Number of frames to "remember" the target before resetting
MAX_DISTANCE = 1000  # Maximum allowed distance in pixels for contour matching
BBOX_SIZE_THRESHOLD = 5  # Bounding box size threshold multiplier
DYNAMIC_SPEED_MULTIPLIER = 500  # Multiplier for dynamic speed threshold

# Angle Configuration
ANGLE_MIN_DEGREES = -110  # Minimum angle to detect vertical movement
ANGLE_MAX_DEGREES = -70  # Maximum angle to detect vertical movement

# Display Configuration
WINDOW_NAME = 'Frame'  # Name of the display window
FPS_DISPLAY_POSITION = (10, 30)  # Position to display FPS on the frame
CROSSHAIR_COLOR = (0, 255, 0)  # Color of the crosshair (Green)
CROSSHAIR_THICKNESS = 1  # Thickness of the crosshair lines
# FPS_TEXT_COLOR = (255, 255, 0)  # Color of the FPS text (Cyan)
# FPS_FONT = cv2.FONT_HERSHEY_SIMPLEX
# FPS_FONT_SCALE = 1
# FPS_FONT_THICKNESS = 2

# Thumbnail Configuration
THUMBNAIL_SIZE = (430, 370)  # Size of the thumbnail (width, height)
THUMBNAIL_PADDING = 10  # Padding from the edges

# Zoom Configuration
ZOOM_SCALE_FACTOR = 2.0  # Factor by which to zoom the ROI
ZOOMED_ROI_SIZE = THUMBNAIL_SIZE  # Size of the zoomed ROI (width, height)


# ===========================
# Kalman Tracker Class
# ===========================

class KalmanTracker:
    """Encapsulates Kalman filter operations for object tracking."""

    def __init__(self):
        """Initialize the Kalman filter."""
        self.kalman = cv2.KalmanFilter(4, 2)

        # Measurement matrix describes how the measurement relates to the state
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)

        # Transition matrix defines the transition between states (x, y, dx, dy)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)

        # Process noise and measurement noise covariance matrices
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1

    def update(self, contour):
        """
        Update the Kalman filter with the position of the detected contour.

        Args:
            contour (numpy.ndarray): Detected contour.

        Returns:
            tuple: (position, bbox) where position is (x, y) and bbox is (x, y, w, h).
        """
        if contour is None or len(contour) == 0:
            return None, None  # Handle invalid contours

        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        position = (x + w // 2, y + h // 2)  # Center of the bounding box

        # Create a measurement matrix with the position
        measurement = np.array([[np.float32(position[0])],
                                [np.float32(position[1])]])

        # Update the Kalman filter with the current measurement
        self.kalman.correct(measurement)

        return position, (x, y, w, h)  # Return position and bounding box


# ===========================
# Object Tracker Class
# ===========================

class ObjectTracker:
    """Manages video capture, frame processing, object tracking, and thumbnail handling."""

    def __init__(self, video_path):
        """Initialize the ObjectTracker."""
        self.video_path = video_path
        self.cap = self.initialize_video_capture()
        self.kalman_tracker = KalmanTracker()
        self.screen_width, self.screen_height = 1080, 720
        self.prev_gray = self.read_and_prepare_first_frame()
        self.tracked_object = None
        self.last_position = None
        self.trajectory_points = deque(maxlen=20)
        self.target_lost_frames = 0
        self.last_bbox_area = None
        self.last_speed = 0
        self.frame_center = (self.screen_width // 2, self.screen_height // 2)
        self.current_frame = None  # To store the current frame for thumbnail processing

        # Set up the display window
        # cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def initialize_video_capture(self):
        """Initialize video capture from the provided video file path."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise Exception("Error: Could not open video.")
        return cap

    def read_and_prepare_first_frame(self):
        """Read, resize, and convert the first frame to grayscale."""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise Exception("Error: Could not read the first frame.")

        frame = self.resize_frame(frame, self.screen_width, self.screen_height)
        gray = self.convert_to_grayscale(frame)
        return gray

    @staticmethod
    def get_screen_size():
        """Get the screen width and height."""
        monitor = get_monitors()[0]  # Primary monitor
        return monitor.width, monitor.height

    @staticmethod
    def resize_frame(frame, width, height):
        """Resize the frame to the specified width and height."""
        return cv2.resize(frame, (width, height))

    @staticmethod
    def convert_to_grayscale(frame):
        """Convert the frame to grayscale if it has 3 channels."""
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    @staticmethod
    def calculate_difference(prev_frame, curr_frame):
        """Calculate the absolute difference between consecutive frames."""
        diff = cv2.absdiff(prev_frame, curr_frame)
        diff = cv2.GaussianBlur(diff, (7, 7), 0)
        return diff

    @staticmethod
    def apply_thresholding(diff, threshold_value, kernel_size, dilation_iterations, erosion_iterations):
        """Apply thresholding and morphological operations to detect moving objects."""
        _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
        kernel = np.ones(kernel_size, np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=dilation_iterations)
        thresh = cv2.erode(thresh, kernel, iterations=erosion_iterations)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return thresh

    @staticmethod
    def calculate_fps(start_time):
        """Calculate the frames per second (FPS)."""
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        return fps

    @staticmethod
    def display_fps(frame, fps, position, font, font_scale, color, thickness):
        """Display the FPS on the frame."""
        cv2.putText(frame, f"FPS: {fps:.2f}", position,
                    font, font_scale, color, thickness)

    def draw_bounding_box(frame, bbox):
        """Draw a bounding box around the detected object."""
        x, y, w, h = bbox
        # Draw the rectangle with a blue color
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def draw_crosshair(self, frame):
        """Draw a sniper-style crosshair on the video frame with dynamic markings."""
        color = CROSSHAIR_COLOR  # Use the configured crosshair color
        thickness = CROSSHAIR_THICKNESS  # Use the configured thickness
        interval = 50  # Distance between dynamic markings
        num_markings = 5  # Number of markings on each side
        min_marking_length = 5  # Minimum length of markings
        max_marking_length = 50  # Maximum length of markings

        # Calculate the total length of the crosshair lines based on the number of markings and interval
        line_length = num_markings * interval

        # Get the center of the frame
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2

        # Draw the central horizontal crosshair line from center to the last marking on both sides
        cv2.line(frame,
                 (center_x - line_length, center_y),
                 (center_x + line_length, center_y),
                 color, thickness)

        # Draw the central vertical crosshair line from center to the last marking on both sides
        cv2.line(frame,
                 (center_x, center_y - line_length),
                 (center_x, center_y + line_length),
                 color, thickness)

        # Draw dynamic markings at specified intervals
        for i in range(1, num_markings + 1):
            # Calculate the length of the current marking
            length = min_marking_length + int((max_marking_length - min_marking_length) * (i / num_markings))

            # Calculate the offset for horizontal and vertical markings
            offset_x = i * interval
            offset_y = i * interval

            # Horizontal markings (left and right of the central horizontal line)
            cv2.line(frame,
                     (center_x - offset_x, center_y - length // 2),
                     (center_x - offset_x, center_y + length // 2),
                     color, thickness)
            cv2.line(frame,
                     (center_x + offset_x, center_y - length // 2),
                     (center_x + offset_x, center_y + length // 2),
                     color, thickness)

            # Vertical markings (above and below the central vertical line)
            cv2.line(frame,
                     (center_x - length // 2, center_y - offset_y),
                     (center_x + length // 2, center_y - offset_y),
                     color, thickness)
            cv2.line(frame,
                     (center_x - length // 2, center_y + offset_y),
                     (center_x + length // 2, center_y + offset_y),
                     color, thickness)

        return frame

    # def draw_crosshair(self, frame):
    #     """
    #     Draw a sniper-style crosshair on the video frame with concentric circles
    #     and radial lines. The main crosshair lines start from the edge of the first
    #     circle, leaving the central circle empty.
    #     """
    #     color = CROSSHAIR_COLOR  # Use the configured crosshair color
    #     thickness = CROSSHAIR_THICKNESS  # Use the configured thickness
    #     interval = 90  # Distance between dynamic markings (radius of each circle)
    #     num_markings = 4  # Number of concentric circles
    #     min_marking_length = 5  # Minimum length of markings (not used)
    #     max_marking_length = 50  # Maximum length of markings (not used)
    #
    #     # Calculate the total length of the crosshair lines based on the number of markings and interval
    #     line_length = num_markings * interval  # 4 * 90 = 360 pixels
    #
    #     # Get the center of the frame
    #     center_x = frame.shape[1] // 2
    #     center_y = frame.shape[0] // 2
    #
    #     # Draw concentric circles centered at (center_x, center_y)
    #     for i in range(1, num_markings + 1):
    #         radius = i * interval  # Radii: 90, 180, 270, 360 pixels
    #         cv2.circle(frame, (center_x, center_y), radius, color, thickness)
    #
    #     # Draw the main crosshair lines starting from the edge of the first circle
    #     # Horizontal Line (Left and Right)
    #     # Left side: from (center_x - line_length, center_y) to (center_x - interval, center_y)
    #     cv2.line(frame,
    #              (center_x - line_length, center_y),
    #              (center_x - interval, center_y),
    #              color, thickness)
    #
    #     # Right side: from (center_x + interval, center_y) to (center_x + line_length, center_y)
    #     cv2.line(frame,
    #              (center_x + interval, center_y),
    #              (center_x + line_length, center_y),
    #              color, thickness)
    #
    #     # Vertical Line (Top and Bottom)
    #     # Top side: from (center_x, center_y - line_length) to (center_x, center_y - interval)
    #     cv2.line(frame,
    #              (center_x, center_y - line_length),
    #              (center_x, center_y - interval),
    #              color, thickness)
    #
    #     # Bottom side: from (center_x, center_y + interval) to (center_x, center_y + line_length)
    #     cv2.line(frame,
    #              (center_x, center_y + interval),
    #              (center_x, center_y + line_length),
    #              color, thickness)
    #
    #     # Draw eight radial lines (two per quadrant) from the edge of the first circle to the outermost circle
    #     # Define angles for radial lines (in degrees)
    #     radial_angles_deg = [30, 60, 120, 150, 210, 240, 300, 330]
    #
    #     # Convert degrees to radians for calculation
    #     radial_angles_rad = [math.radians(angle) for angle in radial_angles_deg]
    #
    #     # Define start and end radii for radial lines
    #     start_radius = interval  # 90 pixels (edge of the first circle)
    #     end_radius = line_length  # 360 pixels (edge of the outermost circle)
    #
    #     for angle in radial_angles_rad:
    #         # Calculate start point (on the first circle)
    #         start_x = int(center_x + start_radius * math.cos(angle))
    #         start_y = int(center_y + start_radius * math.sin(angle))
    #
    #         # Calculate end point (on the outermost circle)
    #         end_x = int(center_x + end_radius * math.cos(angle))
    #         end_y = int(center_y + end_radius * math.sin(angle))
    #
    #         # Draw the radial line
    #         cv2.line(frame,
    #                  (start_x, start_y),
    #                  (end_x, end_y),
    #                  color, thickness)
    #
    #     return frame

    def enhance_contrast_linear(self, roi):
        """Enhance contrast using linear contrast stretching for grayscale images."""
        roi_gray = roi.copy()
        min_val = np.min(roi_gray)
        max_val = np.max(roi_gray)
        if max_val - min_val == 0:
            return roi.copy()
        contrast_stretched = cv2.normalize(roi_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return contrast_stretched

    def embed_zoomed_object(self, frame, bbox):
        """Embed a zoomed image of the object into the frame at a specified position with enhanced contrast."""
        x, y, w, h = bbox
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + w, self.screen_width)
        y2 = min(y + h, self.screen_height)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        enhanced_roi = self.enhance_contrast_linear(roi)
        zoomed_roi = cv2.resize(
            enhanced_roi,
            (0, 0),
            fx=ZOOM_SCALE_FACTOR,
            fy=ZOOM_SCALE_FACTOR,
            interpolation=cv2.INTER_LINEAR
        )
        zoomed_roi = cv2.resize(
            zoomed_roi,
            ZOOMED_ROI_SIZE,
            interpolation=cv2.INTER_LINEAR
        )

        # if len(zoomed_roi.shape) == 2:
        #     zoomed_roi = cv2.cvtColor(zoomed_roi, cv2.COLOR_GRAY2BGR)

        return zoomed_roi

    def find_closest_contour(self, contours, last_position):
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
            distance = self.calculate_distance(center, last_position)
            if distance < min_distance:
                min_distance = distance
                closest_contour = contour  # Assign the closest contour

        return closest_contour  # Return the closest valid contour

    def handle_object_tracking(self, contours):
        """Handle the logic of tracking the object, considering bounding box area and speed check."""
        thumbnail = None

        if self.tracked_object is None and contours:
            # Initialize tracking
            largest_contour = self.find_largest_contour(contours, MIN_CONTOUR_AREA)
            if largest_contour is not None:
                self.last_position, bbox = self.kalman_tracker.update(largest_contour)
                if bbox:
                    current_bbox_area = bbox[2] * bbox[3]
                    if self.last_bbox_area is None or current_bbox_area <= BBOX_SIZE_THRESHOLD * self.last_bbox_area:
                        thumbnail = self.embed_zoomed_object(self.current_frame, bbox)
                        self.draw_bounding_box(bbox)
                        self.last_bbox_area = current_bbox_area
                    self.trajectory_points.append(self.last_position)
                    self.target_lost_frames = 0
                    self.tracked_object = largest_contour

        elif self.tracked_object is not None:
            # Update tracking
            closest_contour = self.find_closest_contour(contours, self.last_position)
            if closest_contour is not None:
                current_position, bbox = self.kalman_tracker.update(closest_contour)
                if current_position and bbox:
                    if not self.is_within_distance(current_position, self.last_position, MAX_DISTANCE):
                        self.target_lost_frames += 1
                        return thumbnail

                    dynamic_speed_threshold = (
                                self.last_speed * DYNAMIC_SPEED_MULTIPLIER) if self.last_speed > 0 else DYNAMIC_SPEED_MULTIPLIER
                    if not self.is_speed_valid(current_position, self.last_position, self.last_speed,
                                               dynamic_speed_threshold):
                        self.target_lost_frames += 1
                        return thumbnail

                    thumbnail = self.embed_zoomed_object(self.current_frame, bbox)

                    if self.should_draw_bbox(bbox):
                        self.draw_bounding_box_and_line(bbox)
                        self.last_bbox_area = bbox[2] * bbox[3]

                    dx = current_position[0] - self.last_position[0]
                    dy = current_position[1] - self.last_position[1]
                    self.last_speed = math.sqrt(dx ** 2 + dy ** 2)

                    self.trajectory_points.append(current_position)
                    self.target_lost_frames = 0
                    self.last_position = current_position
            else:
                self.target_lost_frames += 1

        # Angle check
        if len(self.trajectory_points) >= 2:
            if self.check_angle_and_stop_tracking():
                self.tracked_object = None

        # Reset tracking if lost
        if self.target_lost_frames > TARGET_MEMORY_FRAMES:
            self.tracked_object = None
            self.target_lost_frames = 0

        return thumbnail

    def find_largest_contour(self, contours, min_contour_area):
        """Find and return the largest contour above a certain area threshold."""
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) >= min_contour_area:
                return contour
        return None

    def draw_bounding_box(self, bbox):
        """Draw a bounding box around the detected object."""
        x, y, w, h = bbox
        cv2.rectangle(self.current_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue color

    def draw_bounding_box_and_line(self, bbox):
        """Draw the bounding box on the frame."""
        self.draw_bounding_box(bbox)
        # Uncomment the following lines to draw a red line from frame center to bounding box center
        # x, y, w, h = bbox
        # bbox_center = (x + w // 2, y + h // 2)
        # cv2.line(self.current_frame, self.frame_center, bbox_center, (0, 0, 255), 2)  # Red line

    def check_angle_and_stop_tracking(self):
        """Check the angle of movement and stop tracking if vertical movement is detected."""
        last_two = list(self.trajectory_points)[-2:]
        dx = last_two[1][0] - last_two[0][0]
        dy = last_two[1][1] - last_two[0][1]
        angle = math.degrees(math.atan2(dy, dx))

        # Check if the object is moving vertically (between ANGLE_MIN_DEGREES and ANGLE_MAX_DEGREES)
        return ANGLE_MIN_DEGREES <= angle <= ANGLE_MAX_DEGREES

    def is_within_distance(self, current_position, previous_position, max_distance):
        """Check if the current position is within the allowed distance from the previous one."""
        distance = self.calculate_distance(current_position, previous_position)
        return distance <= max_distance

    def calculate_distance(self, position1, position2):
        """Calculate Euclidean distance between two positions."""
        return math.sqrt((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2)

    def is_speed_valid(self, current_position, previous_position, last_speed, dynamic_speed_threshold):
        """Check if the object's speed is within the allowed dynamic threshold."""
        current_speed = self.calculate_distance(current_position, previous_position)
        return current_speed <= dynamic_speed_threshold

    def should_draw_bbox(self, bbox):
        """Check if the bounding box should be drawn based on its size."""
        current_bbox_area = bbox[2] * bbox[3]
        return self.last_bbox_area is None or current_bbox_area <= BBOX_SIZE_THRESHOLD * self.last_bbox_area

    def process_thumbnails(self, thumbnail):
        """Process and embed the thumbnail into the current frame."""
        if thumbnail is not None:
            # Define the position for the thumbnail (lower right corner)
            thumbnail_x = self.screen_width - THUMBNAIL_SIZE[0] - THUMBNAIL_PADDING  # 10 pixels from the right edge
            thumbnail_y = self.screen_height - THUMBNAIL_SIZE[1] - THUMBNAIL_PADDING  # 10 pixels from the bottom edge

            # Ensure the thumbnail fits within the frame
            if thumbnail_y < 0:
                thumbnail_y = THUMBNAIL_PADDING  # If frame height is less than thumbnail height + padding, place it at top-right with padding

            # Adjust thumbnail size if it exceeds frame boundaries
            end_y = thumbnail_y + THUMBNAIL_SIZE[1]
            end_x = thumbnail_x + THUMBNAIL_SIZE[0]
            if end_y > self.screen_height or end_x > self.screen_width:
                end_y = min(end_y, self.screen_height)
                end_x = min(end_x, self.screen_width)
                thumbnail = thumbnail[0:end_y - thumbnail_y, 0:end_x - thumbnail_x]

            # Overlay the thumbnail on the bottom right corner
            self.current_frame[thumbnail_y:end_y, thumbnail_x:end_x] = thumbnail

    def run(self):
        """Start the object tracking process."""
        while self.cap.isOpened():
            start_time = time.time()

            ret, frame = self.cap.read()
            if not ret or frame is None:
                break

            # Resize and convert current frame to grayscale
            self.current_frame = self.resize_frame(frame, self.screen_width, self.screen_height)
            curr_gray = self.convert_to_grayscale(self.current_frame)

            # Calculate difference and apply thresholding
            diff = self.calculate_difference(self.prev_gray, curr_gray)
            thresh = self.apply_thresholding(diff, THRESHOLD_VALUE, MORPH_KERNEL_SIZE, DILATION_ITERATIONS,
                                             EROSION_ITERATIONS)

            # Find contours in thresholded frame
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Handle object tracking
            thumbnail = self.handle_object_tracking(contours)

            # Draw crosshair on the frame
            self.draw_crosshair(self.current_frame)

            # Process and embed the thumbnail
            self.process_thumbnails(thumbnail)

            # Calculate and display FPS
            # fps = self.calculate_fps(start_time)
            # self.display_fps(self.current_frame, fps, FPS_DISPLAY_POSITION, FPS_FONT, FPS_FONT_SCALE, FPS_TEXT_COLOR,
            #                  FPS_FONT_THICKNESS)

            # Show the result
            cv2.imshow(WINDOW_NAME, self.current_frame)

            # Exit on 'q' key press
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            # Update previous frame
            self.prev_gray = curr_gray

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()


# ===========================
# Entry Point
# ===========================


def main():
    tracker = ObjectTracker(VIDEO_PATH)
    try:
        tracker.run()
    except Exception as e:
        print(f"An error occurred during tracking: {e}")
    finally:
        if tracker.cap.isOpened():
            tracker.cap.release()
        cv2.destroyAllWindows()
        print("Video capture released and all windows closed.")


if __name__ == "__main__":
    main()
