import cv2
import numpy as np

class ObjectTracker:
    def __init__(self, persistence_threshold=100, match_distance=50, min_size=10):
        self.persistence_threshold = persistence_threshold
        self.match_distance = match_distance
        self.min_size = min_size

        # object_id: {"pos":(x,y,w,h), "frames": count_of_persistent_frames}
        self.persistent_objects = {}
        self.next_object_id = 1

    def update_baseline(self, frame):
        """Set a new baseline from the given frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        baseline_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # Reset persistent objects since we have a new baseline
        self.persistent_objects.clear()
        self.next_object_id = 1
        return baseline_gray

    def process_frame(self, frame, baseline_gray):
        """Process a single frame, detect changes, track objects, and identify anomalies."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        diff = cv2.absdiff(baseline_gray, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)  #30 change to more for better background substruct

        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_frame_objects = self._filter_objects(contours)

        self._update_persistent_objects(current_frame_objects)
        self._draw_anomalies(frame)

        return diff, thresh

    def _filter_objects(self, contours):
        """Filter detected contours based on the minimum size threshold."""
        objects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= self.min_size and h >= self.min_size:
                objects.append((x, y, w, h))
        return objects

    def _match_existing_objects(self, x, y, w, h):
        """Match a detected object with an existing tracked object if close enough."""
        cx = x + w / 2.0
        cy = y + h / 2.0
        for obj_id, data in self.persistent_objects.items():
            ox, oy, ow, oh = data["pos"]
            ocx = ox + ow / 2.0
            ocy = oy + oh / 2.0
            dist = np.sqrt((cx - ocx) ** 2 + (cy - ocy) ** 2)
            if dist < self.match_distance:
                return obj_id
        return None

    def _update_persistent_objects(self, current_frame_objects):
        """Update the dictionary of persistent objects based on current detections."""
        found_ids = []
        # Update or add new objects
        for (x, y, w, h) in current_frame_objects:
            matched_id = self._match_existing_objects(x, y, w, h)
            if matched_id is not None:
                self.persistent_objects[matched_id]["pos"] = (x, y, w, h)
                self.persistent_objects[matched_id]["frames"] += 1
                found_ids.append(matched_id)
            else:
                self.persistent_objects[self.next_object_id] = {"pos": (x, y, w, h), "frames": 1}
                found_ids.append(self.next_object_id)
                self.next_object_id += 1

        # Remove objects not detected this frame
        to_remove = [obj_id for obj_id in self.persistent_objects if obj_id not in found_ids]
        for obj_id in to_remove:
            del self.persistent_objects[obj_id]

    def _draw_anomalies(self, frame):
        """Draw bounding boxes and labels for anomalous objects."""
        for obj_id, data in self.persistent_objects.items():
            x, y, w, h = data["pos"]
            frames_count = data["frames"]
            if frames_count > self.persistence_threshold:
                color = (0, 0, 255)
                text = "Anomalous"
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{text} ({frames_count})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
