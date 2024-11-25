import cv2
from ultralytics import YOLO
import numpy as np


class Config:
    """Configuration parameters for the object tracking application."""
    MODEL_PATH = r"D:\pycharm_projects\yolov8\runs\detect\drone_v7_200ep_32bath\weights\best.pt"
    VIDEO_PATH = r"C:\Users\User\Desktop\fly\GENERIC_RTSP-realmonitor_2023_09_20_15_57_55.avi"
    # OUTPUT_VIDEO_PATH = r"C:\Users\User\Desktop\порівняння\Zir\output_tracking_3.mp4"  # New parameter
    CONFIDENCE_THRESHOLD = 0.15
    IOU_THRESHOLD = 0.6
    MOVEMENT_THRESHOLD = 1  # pixels
    STATIONARY_FRAME_LIMIT = 2
    DISPLAY_WINDOW_NAME = "Tracking"
    # VIDEO_CODEC = 'mp4v'  # Codec for MP4 files; can be changed as needed

    # **New Configuration Parameters for Resizing**
    RESIZE_DISPLAY = False  # Flag to control resizing
    RESIZE_WIDTH = 1720  # Desired width after resizing
    RESIZE_HEIGHT = 1080  # Desired height after resizing


class TrackerManager:
    """Manages the creation and updating of the OpenCV tracker."""

    def __init__(self):
        self.tracker = None

    def create_tracker(self):
        """Creates and returns an OpenCV tracker instance."""
        if hasattr(cv2, 'legacy'):
            self.tracker = cv2.legacy.TrackerCSRT_create()
        elif hasattr(cv2, 'TrackerCSRT_create'):
            self.tracker = cv2.TrackerCSRT_create()
        else:
            print("CSRT Tracker is not available in your OpenCV installation.")
            # Fallback to KCF tracker
            if hasattr(cv2, 'legacy'):
                self.tracker = cv2.legacy.TrackerKCF_create()
            elif hasattr(cv2, 'TrackerKCF_create'):
                self.tracker = cv2.TrackerKCF_create()
            else:
                print("KCF Tracker is not available. Exiting.")
                exit()

    def init_tracker(self, frame, bbox):
        """Initializes the tracker with the given bounding box."""
        self.create_tracker()
        return self.tracker.init(frame, bbox)

    def update_tracker(self, frame):
        """Updates the tracker and returns the new bounding box."""
        ok, bbox = self.tracker.update(frame)
        return ok, bbox


class ObjectTracker:
    """Encapsulates the object detection and tracking logic."""

    def __init__(self, config):
        self.config = config
        self.model = YOLO(self.config.MODEL_PATH)
        self.tracker_manager = TrackerManager()
        self.tracking = False
        self.prev_bbox = None
        self.stationary_frame_count = 0

    def perform_detection(self, frame):
        """Performs object detection on the frame and returns bounding boxes."""
        results = self.model.predict(frame, conf=self.config.CONFIDENCE_THRESHOLD, iou=self.config.IOU_THRESHOLD)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                bbox = box.xyxy[0].cpu().numpy()
                detections.append(bbox)
        return detections

    def initialize_tracking(self, frame, bbox):
        """Initializes tracking with the provided bounding box."""
        x1, y1, x2, y2 = bbox.tolist()
        x = max(0, x1)
        y = max(0, y1)
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)

        if w <= 0 or h <= 0:
            print(f"Invalid bounding box dimensions: w={w}, h={h}")
            return False

        # Convert coordinates to integers
        x, y, w, h = map(int, map(round, [x, y, w, h]))
        bbox_tuple = (x, y, w, h)
        ok = self.tracker_manager.init_tracker(frame, bbox_tuple)
        if ok:
            self.tracking = True
            self.prev_bbox = bbox_tuple
            self.stationary_frame_count = 0
            print("Tracker initialized successfully.")
            return True
        else:
            print("Tracker initialization failed.")
            return False

    def update_tracking(self, frame):
        """Updates tracking and handles movement checks."""
        ok, bbox = self.tracker_manager.update_tracker(frame)
        if ok:
            x, y, w, h = map(int, map(round, bbox))
            curr_bbox = (x, y, w, h)
            self.draw_bbox(frame, curr_bbox)

            if self.prev_bbox and self.is_stationary(self.prev_bbox, curr_bbox):
                self.stationary_frame_count += 1
            else:
                self.stationary_frame_count = 0

            self.prev_bbox = curr_bbox

            if self.stationary_frame_count >= self.config.STATIONARY_FRAME_LIMIT:
                print("Object stationary for too many frames. Reinitializing detection.")
                self.reset_tracking()
        else:
            print("Tracking failure detected. Reinitializing detection.")
            self.reset_tracking()

    def is_stationary(self, prev_bbox, curr_bbox):
        """Checks if the object has moved beyond the movement threshold."""
        prev_x, prev_y, prev_w, prev_h = prev_bbox
        curr_x, curr_y, curr_w, curr_h = curr_bbox

        # Compute centers
        prev_center = (prev_x + prev_w / 2, prev_y + prev_h / 2)
        curr_center = (curr_x + curr_w / 2, curr_y + curr_h / 2)

        # Compute the Euclidean distance
        distance = np.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])

        return distance < self.config.MOVEMENT_THRESHOLD

    def reset_tracking(self):
        """Resets tracking variables."""
        self.tracking = False
        self.prev_bbox = None
        self.stationary_frame_count = 0
        self.tracker_manager.tracker = None

    def draw_bbox(self, frame, bbox):
        """Draws the bounding box and tracking status on the frame."""
        x, y, w, h = bbox
        x2, y2 = x + w, y + h
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "target", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


class MainApp:
    """Main application class for running the object tracking."""

    def __init__(self, config):
        self.config = config
        self.object_tracker = ObjectTracker(config)
        self.cap = cv2.VideoCapture(self.config.VIDEO_PATH)
        self.out = None  # Initialize VideoWriter as None

        # Initialize VideoWriter if OUTPUT_VIDEO_PATH is provided
        if hasattr(self.config, 'OUTPUT_VIDEO_PATH') and self.config.OUTPUT_VIDEO_PATH:
            self.initialize_video_writer()

    def initialize_video_writer(self):
        """Initializes the VideoWriter object based on input video properties."""
        if not self.cap.isOpened():
            print(f"Error: Cannot open video file {self.config.VIDEO_PATH}")
            exit()

        # Retrieve properties of the input video
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*self.config.VIDEO_CODEC)

        # Define the output video path
        output_video_path = self.config.OUTPUT_VIDEO_PATH

        # Initialize the VideoWriter
        self.out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        if not self.out.isOpened():
            print(f"Error: Cannot open video writer for file {output_video_path}")
            self.cap.release()
            exit()

        print(f"Recording output video to {output_video_path}")

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("End of video file reached.")
                break

            if not self.object_tracker.tracking:
                # Perform detection
                detections = self.object_tracker.perform_detection(frame)
                if detections:
                    # Initialize tracker with the first detection
                    initialized = self.object_tracker.initialize_tracking(frame, detections[0])
                    if not initialized:
                        # Failed to initialize tracker; display and write frame
                        self.display_and_write_frame(frame)
                        continue
                else:
                    # No detections; display and write frame
                    self.display_and_write_frame(frame)
                    continue
            else:
                # Update tracker
                self.object_tracker.update_tracking(frame)

            # Display and write the frame
            self.display_and_write_frame(frame)

        # After processing all frames, release resources
        self.cleanup()

    def display_and_write_frame(self, frame):
        """Displays the frame and writes it to the output video."""
        # **Conditional Resizing Based on the Flag**
        if self.config.RESIZE_DISPLAY:
            frame_display = cv2.resize(frame, (self.config.RESIZE_WIDTH, self.config.RESIZE_HEIGHT))
        else:
            frame_display = frame

        cv2.imshow(self.config.DISPLAY_WINDOW_NAME, frame_display)

        # Write the original frame to the output video without resizing
        # If you want to write the resized frame instead, uncomment the following lines:
        if self.out:
            if self.config.RESIZE_DISPLAY:
                # Resize frame to match VideoWriter's expected size
                frame_to_write = cv2.resize(frame, (self.config.RESIZE_WIDTH, self.config.RESIZE_HEIGHT))
            else:
                frame_to_write = frame
            self.out.write(frame_to_write)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cleanup()

    def cleanup(self):
        """Releases resources and closes windows."""
        if self.cap.isOpened():
            self.cap.release()
        if self.out and self.out.isOpened():
            self.out.release()
        cv2.destroyAllWindows()
        print(f"Output video saved to {self.config.OUTPUT_VIDEO_PATH}" if self.out else "No output video was saved.")
        exit()


if __name__ == "__main__":
    config = Config()
    app = MainApp(config)
    app.run()
