
import cv2
import numpy as np

# --- Parameters ---
VIDEO_SOURCE = 0   # Use your video source, e.g. 'video.mp4' or 0 for webcam
PERSISTENCE_THRESHOLD = 100
MATCH_DISTANCE = 50        # how close bounding boxes need to be to match existing objects
MIN_SIZE = 10              # Minimum width/height to consider an object, objects smaller than this are ignored

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit(1)

baseline_gray = None

# Structure to hold persistent objects
# object_id: {"pos":(x,y,w,h), "frames": count_of_persistent_frames}
persistent_objects = {}
next_object_id = 1


def match_existing_objects(x, y, w, h, objects_dict):
    """Match the bounding box with an existing object based on center distance."""
    cx = x + w / 2.0
    cy = y + h / 2.0
    for obj_id, data in objects_dict.items():
        ox, oy, ow, oh = data["pos"]
        ocx = ox + ow / 2.0
        ocy = oy + oh / 2.0
        dist = np.sqrt((cx - ocx) ** 2 + (cy - ocy) ** 2)
        if dist < MATCH_DISTANCE:
            return obj_id
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(30)
    if key == ord('q'):
        # Set the current frame as baseline
        baseline_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        baseline_gray = cv2.GaussianBlur(baseline_gray, (5,5), 0)
        # Reset persistent objects since we have a new baseline
        persistent_objects.clear()
        next_object_id = 1

    if baseline_gray is not None:
        # Process only if we have a baseline
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Compute absolute difference between current frame and baseline
        diff = cv2.absdiff(baseline_gray, gray)

        # Threshold the difference
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

        # Find contours of changed areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_frame_objects = []
        # Filter out objects smaller than MIN_SIZE
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < MIN_SIZE or h < MIN_SIZE:
                # Skip if object is too small
                continue
            current_frame_objects.append((x, y, w, h))

        # Update persistent objects
        found_ids = []
        for (x, y, w, h) in current_frame_objects:
            matched_id = match_existing_objects(x, y, w, h, persistent_objects)
            if matched_id is not None:
                # Update existing object
                persistent_objects[matched_id]["pos"] = (x, y, w, h)
                persistent_objects[matched_id]["frames"] += 1
                found_ids.append(matched_id)
            else:
                # Create new object
                persistent_objects[next_object_id] = {"pos": (x, y, w, h), "frames": 1}
                found_ids.append(next_object_id)
                next_object_id += 1

        # Remove objects not seen in this frame
        to_remove = [obj_id for obj_id in persistent_objects if obj_id not in found_ids]
        for obj_id in to_remove:
            del persistent_objects[obj_id]

        # Check for anomalies
        for obj_id, data in persistent_objects.items():
            x, y, w, h = data["pos"]
            frames_count = data["frames"]

            if frames_count > PERSISTENCE_THRESHOLD:
                # Draw and label only if anomalous
                color = (0, 0, 255)  # Red for anomalous
                text = "Anomalous"
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{text} ({frames_count})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Diff", diff)
        cv2.imshow("Thresh", thresh)
    else:
        # No baseline yet, just show the original frame and instructions
        instruction = "Press 'q' to set this frame as baseline."
        cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Frame", frame)

    if key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
