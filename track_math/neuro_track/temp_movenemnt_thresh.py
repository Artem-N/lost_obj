import torch
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model (replace with your custom model path)
model = YOLO(r"D:\pycharm_projects\yolov8\runs\detect\drone_v7_200ep_32bath\weights\best.pt")
model.conf = 0.15  # Adjust confidence threshold as needed

# Open the video file
cap = cv2.VideoCapture(r"C:\Users\User\Desktop\fly\2024.mp4")

# Frame processing parameters
detector_interval = 50  # Run detector every 5 frames
frame_count = 0  # Initialize frame counter
no_movement_count = 0  # Count frames where no movement is detected
movement_threshold = 10  # Number of frames without movement to trigger detection
tracker_initialized = False  # Tracker status

# Initialize OpenCV tracker (CSRT in this case)
tracker = cv2.TrackerCSRT_create()  # New syntax

prev_bbox = None  # Previous bounding box to track movement

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if not tracker_initialized or no_movement_count >= movement_threshold:
        # ===== Run YOLOv8 Detector =====
        results = model(frame)
        result = results[0]  # Get the first (and only) result

        detections = []
        # Check if there are any detections
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes  # Boxes object
            for box in boxes:
                # Extract coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Extract confidence score
                conf = box.conf[0].item()

                # Append detection in format: (x1, y1, x2 - x1, y2 - y1)
                detections.append((x1, y1, x2 - x1, y2 - y1))

            print(f"Frame {frame_count}: Number of detections: {len(detections)}")

            if len(detections) > 0:
                # Initialize the tracker with the first detected object
                tracker = cv2.TrackerCSRT_create()  # Reinitialize the tracker
                bbox = detections[0]  # Assuming the first detected object is the target
                tracker.init(frame, bbox)  # Initialize tracker with the first bounding box
                tracker_initialized = True
                prev_bbox = bbox  # Save initial bounding box
                no_movement_count = 0  # Reset movement counter
                print(f"Tracker initialized with bbox: {bbox}")
        else:
            tracker_initialized = False  # Detection failed
            print(f"Frame {frame_count}: No detections")

    elif tracker_initialized:
        # ===== Use Tracker to Track the Object =====
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Tracking', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Check for movement by comparing current bbox with the previous one
            if prev_bbox is not None and abs(prev_bbox[0] - x) < 5 and abs(prev_bbox[1] - y) < 5:
                no_movement_count += 1  # Increment if the object is stationary
                print(f"Frame {frame_count}: No movement detected, count: {no_movement_count}")
            else:
                no_movement_count = 0  # Reset if movement is detected
                print(f"Frame {frame_count}: Object is moving")

            prev_bbox = bbox  # Update the previous bounding box

        else:
            tracker_initialized = False  # Tracking failed
            print(f"Frame {frame_count}: Tracker lost the object")

    # Display the result
    cv2.imshow('Frame', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
