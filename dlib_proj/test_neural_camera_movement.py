import cv2
import time
import threading
import queue
from ultralytics import YOLO
from requests.auth import HTTPDigestAuth
import requests

# Custom class for handling threaded video capture
class VideoCapture:
    def __init__(self, uri):
        self.cap = cv2.VideoCapture(uri)
        self.q = queue.Queue(maxsize=1)
        self.running = True
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # Discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get() if not self.q.empty() else None

    def release(self):
        self.running = False
        self.t.join()
        self.cap.release()

# PTZ command to move the camera
def send_ptz_command(code, speed=1):
    try:
        camera_ip = "172.16.14.10"
        username = "admin"
        password = "admin"

        # PTZ command URL
        url = f"http://{camera_ip}/cgi-bin/ptz.cgi?action=start&channel=0&code={code}&arg1={speed}&arg2={speed}&arg3=0"
        print(f"Sending PTZ command: {url}")
        response = requests.get(url, auth=HTTPDigestAuth(username, password))
        print(f"Response: {response.status_code}, {response.content}")

        if response.status_code == 200:
            print(f"Command '{code}' executed successfully.")
        else:
            print(f"Failed to execute '{code}'. Response code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending PTZ command: {e}")

# PTZ command to stop the camera movement
def stop_ptz_command():
    try:
        camera_ip = "172.16.14.10"  # Replace with your actual IP address
        username = "admin"          # Replace with your username
        password = "admin"          # Replace with your password

        directions = ["Left", "Right", "Up", "Down"]
        for direction in directions:
            url = f"http://{camera_ip}/cgi-bin/ptz.cgi?action=stop&channel=0&code={direction}&arg1=0&arg2=0&arg3=0"
            print(f"Sending stop command: {url}")
            response = requests.get(url, auth=HTTPDigestAuth(username, password))
            print(f"Response for stopping '{direction}': {response.status_code}, {response.content}")
            if response.status_code == 200:
                print(f"Stopped movement '{direction}'.")
            else:
                print(f"Failed to stop '{direction}'. Response code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending PTZ stop command: {e}")

# Function to move camera for a specific duration
def move_camera(direction, duration, speed=1):
    def _move():
        try:
            send_ptz_command(direction, speed)
            print(f"Started moving '{direction}' at speed {speed}.")
            time.sleep(duration)
            stop_ptz_command()
            print(f"Stopped moving '{direction}' after {duration} seconds.")
        except Exception as e:
            print(f"Error moving camera: {e}")
    threading.Thread(target=_move).start()

# Load the YOLOv8 model
model = YOLO('yolo11m.pt')

# Start video capture
cap = VideoCapture('rtsp://172.16.14.10:554/cam/realmonitor?channel=2&subtype=0&unicast=true&proto=Onvif')

# Initialize variables for frame skipping and control
frame_skip = 10  # Adjust as needed
frame_width, frame_height = None, None

# Control parameters
center_tolerance_start = 80    # Adjust as needed
command_delay = 0.5            # Adjust as needed
stop_cooldown = 0.3            # Adjust as needed
last_command_time = 0
stop_time = None               # To track when the camera stopped

# Movement duration parameters
min_duration = 1               # Increased from 0.1
max_duration = 2               # Increased from 0.5
max_speed = 5                  # Maximum speed supported by the camera

# Loop through video frames
try:
    frame_count = 0
    while True:
        img = cap.read()
        if img is None:
            continue

        # Set frame dimensions only on the first frame
        if frame_width is None or frame_height is None:
            frame_height, frame_width = img.shape[:2]
            frame_center_x = frame_width // 2
            frame_center_y = frame_height // 2

        # Skip frames to improve performance
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame_count += 1

        # Perform object detection
        results = model.track(source=img, persist=True, conf=0.5, tracker='bytetrack.yaml')

        # Extract person detections
        detected = False
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Assuming '0' is the class ID for 'person'
                    detected = True
                    # Extract box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_center_x = (x1 + x2) // 2
                    person_center_y = (y1 + y2) // 2

                    # Draw the bounding box and label
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'Person'
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Calculate the offsets
                    offset_x = person_center_x - frame_center_x
                    offset_y = person_center_y - frame_center_y

                    current_time = time.time()

                    # If enough time has passed since the last command and cooldown period is over
                    if (current_time - last_command_time) > command_delay and (stop_time is None or (current_time - stop_time) > stop_cooldown):
                        if abs(offset_x) > center_tolerance_start:
                            # Calculate duration based on offset
                            duration = min_duration + (abs(offset_x) / frame_center_x) * (max_duration - min_duration)
                            duration = min(duration, max_duration)
                            # Calculate speed based on offset
                            speed = min(1 + int((abs(offset_x) / frame_center_x) * (max_speed - 1)), max_speed)
                            if offset_x > 0:
                                move_camera("Right", duration, speed)
                            else:
                                move_camera("Left", duration, speed)
                            print(f"Moving camera to center object: offset_x={offset_x}, offset_y={offset_y}, speed={speed}, duration={duration}")
                        elif abs(offset_y) > center_tolerance_start:
                            duration = min_duration + (abs(offset_y) / frame_center_y) * (max_duration - min_duration)
                            duration = min(duration, max_duration)
                            speed = min(1 + int((abs(offset_y) / frame_center_y) * (max_speed - 1)), max_speed)
                            if offset_y > 0:
                                move_camera("Down", duration, speed)
                            else:
                                move_camera("Up", duration, speed)
                            print(f"Moving camera to center object: offset_x={offset_x}, offset_y={offset_y}, speed={speed}, duration={duration}")
                        else:
                            # If the offset is not significant, no need to move
                            pass

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
        cv2.imshow('YOLOv8 Tracking', img)

        # Press 'q' to exit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()