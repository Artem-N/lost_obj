# detector_manager.py
import threading
import cv2
from video_anomaly_detector import VideoAnomalyDetector

class DetectorManager:
    def __init__(self, video_source=0, persistence_threshold=100, match_distance=50, min_size=10, video_width=640, video_height=480):
        self.lock = threading.Lock()
        self.detector = VideoAnomalyDetector(
            video_source=video_source,
            persistence_threshold=persistence_threshold,
            match_distance=match_distance,
            min_size=min_size
        )
        self.detector.video_width = video_width
        self.detector.video_height = video_height
        # Initialize other necessary attributes

    def set_baseline(self, frame):
        with self.lock:
            resized_frame = cv2.resize(frame, (self.detector.video_width, self.detector.video_height))
            self.detector.baseline_gray = self.detector.tracker.update_baseline(resized_frame)

    def reset_baseline(self):
        with self.lock:
            self.detector.baseline_gray = None
            self.detector.tracker.persistent_objects.clear()
            self.detector.tracker.next_object_id = 1

    def process_frame(self, frame):
        with self.lock:
            # Process frame and return results
            pass

    def release(self):
        with self.lock:
            if self.detector.cap.isOpened():
                self.detector.cap.release()
            self.detector = None