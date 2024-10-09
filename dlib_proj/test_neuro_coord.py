import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolo11m.pt')  # Replace 'yolov8n.pt' with your model

# Camera credentials and RTSP URL
rtsp_url = "rtsp://username:password@172.16.14.10:554/cam/realmonitor?channel=1&subtype=0"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Configure the tracker
# Set up a DeepSORT tracker by specifying 'tracker' argument in the model's predict method
tracker = 'bytetrack.yaml'  # You can use 'bytetrack.yaml', 'deepsort.yaml', etc. depending on your environment

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Perform object detection and tracking
    results = model.track(source=frame, persist=True, conf=0.5, tracker=tracker)

    # Draw detections and track IDs on the frame
    for result in results:
        for box in result.boxes:
            # Extract box coordinates and tracking information
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = box.cls[0]
            track_id = box.id  # Get the track ID from the tracker

            # Draw the bounding box and tracking ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{model.names[int(cls)]}: {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Print out the detected object, track ID, and coordinates
            print(f"Detected: {model.names[int(cls)]}, Coordinates: ({x1}, {y1}), ({x2}, {y2})")

    # Display the resulting frame with tracking info
    cv2.imshow('YOLOv8 Tracking', frame)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
