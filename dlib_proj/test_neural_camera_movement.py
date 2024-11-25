import cv2
import time
import threading
from ultralytics import YOLO
from requests.auth import HTTPDigestAuth
import requests
import torch

# Enable cuDNN benchmark for potential performance improvements
torch.backends.cudnn.benchmark = True

# PTZ command to move the camera (runs in a separate thread)
def send_ptz_command(code, speed=1):
    def _send_command():
        try:
            camera_ip = "172.16.14.12"
            username = "admin"
            password = "ZirRobotics"

            # PTZ command URL
            url = f"http://{camera_ip}/cgi-bin/ptz.cgi?action=start&channel=0&code={code}&arg1={speed}&arg2={speed}&arg3=0"
            response = requests.get(url, auth=HTTPDigestAuth(username, password), timeout=0.5)

            if response.status_code == 200:
                print(f"Command '{code}' executed successfully.")
            else:
                print(f"Failed to execute '{code}'. Response code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending PTZ command: {e}")

    threading.Thread(target=_send_command).start()

# PTZ command to stop the camera movement (runs in a separate thread)
def stop_ptz_command():
    def _stop_command():
        try:
            camera_ip = "172.16.14.12"
            username = "admin"
            password = "ZirRobotics"

            directions = ["Left", "Right", "Up", "Down"]
            for direction in directions:
                url = f"http://{camera_ip}/cgi-bin/ptz.cgi?action=stop&channel=0&code={direction}&arg1=0&arg2=0&arg3=0"
                requests.get(url, auth=HTTPDigestAuth(username, password), timeout=0.5)
        except requests.exceptions.RequestException as e:
            print(f"Error sending PTZ stop command: {e}")

    threading.Thread(target=_stop_command).start()

# Function to move camera for a specific duration
def move_camera(direction, duration, speed=1):
    def _move():
        send_ptz_command(direction, speed)
        print(f"Started moving '{direction}' at speed {speed} for {duration} seconds.")
        time.sleep(duration)
        stop_ptz_command()
        print(f"Stopped moving '{direction}' after {duration} seconds.")

    threading.Thread(target=_move).start()

# Load the YOLOv8 nano model for faster inference
model = YOLO('yolov8n.pt')

# Use GPU if available
if torch.cuda.is_available():
    model.to('cuda')
    device = 'cuda'
else:
    device = 'cpu'

# Set the model to detect only the 'person' class
model.classes = [0]

# Start video capture
rtsp_url = "rtsp://username:password@172.16.14.12:554/cam/realmonitor?channel=2&subtype=0"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Control parameters
center_tolerance_start = 80    # Pixel tolerance from the center
command_delay = 0.5            # Delay between PTZ commands
stop_cooldown = 0.3            # Cooldown after stopping movement
last_command_time = 0
stop_time = None               # To track when the camera stopped

# Movement duration parameters
min_duration = 1.0             # Minimum movement duration
max_duration = 2.0             # Maximum movement duration
max_speed = 5                  # Maximum camera speed

# Frame dimensions (will be set after the first frame is read)
frame_width = None
frame_height = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Set frame dimensions only once
        if frame_width is None or frame_height is None:
            frame_height, frame_width = frame.shape[:2]
            frame_center_x = frame_width // 2
            frame_center_y = frame_height // 2

        # Optionally resize the frame for faster processing
        # frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))

        # Perform object detection
        results = model.track(frame, persist=True, conf=0.2)

        # Extract person detections
        detected = False
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # Class ID for 'person'
                        detected = True
                        # Extract box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        person_center_x = (x1 + x2) // 2
                        person_center_y = (y1 + y2) // 2

                        # Draw the bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = 'Person'
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Calculate the offsets
                        offset_x = person_center_x - frame_center_x
                        offset_y = person_center_y - frame_center_y

                        current_time = time.time()

                        # Check if we can send a new PTZ command
                        if (current_time - last_command_time) > command_delay and \
                           (stop_time is None or (current_time - stop_time) > stop_cooldown):
                            if abs(offset_x) > center_tolerance_start:
                                # Calculate duration and speed based on offset
                                duration = min_duration + (abs(offset_x) / frame_center_x) * \
                                           (max_duration - min_duration)
                                duration = min(duration, max_duration)
                                speed = min(1 + int((abs(offset_x) / frame_center_x) * \
                                        (max_speed - 1)), max_speed)
                                direction = "Right" if offset_x > 0 else "Left"
                                move_camera(direction, duration, speed)
                                print(f"Moving camera {direction}: offset_x={offset_x}, speed={speed}, duration={duration}")
                            elif abs(offset_y) > center_tolerance_start:
                                duration = min_duration + (abs(offset_y) / frame_center_y) * \
                                           (max_duration - min_duration)
                                duration = min(duration, max_duration)
                                speed = min(1 + int((abs(offset_y) / frame_center_y) * \
                                        (max_speed - 1)), max_speed)
                                direction = "Down" if offset_y > 0 else "Up"
                                move_camera(direction, duration, speed)
                                print(f"Moving camera {direction}: offset_y={offset_y}, speed={speed}, duration={duration}")

                            last_command_time = current_time
                            stop_time = None  # Reset stop_time since we're moving again

                        break  # Only track the first detected person

                if detected:
                    break

        # If no person is detected, stop movement
        if not detected:
            stop_ptz_command()
            last_command_time = time.time()
            stop_time = last_command_time
            print("No person detected, stopping camera.")

        # Display the resulting frame with tracking info
        cv2.imshow('YOLOv8 Tracking', frame)

        # Press 'q' to exit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
