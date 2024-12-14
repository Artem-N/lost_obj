import cv2
import torch
import time
import threading
import queue

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
        return self.q.get()

    def release(self):
        self.running = False
        self.t.join()
        self.cap.release()

# Start webcam
cap = VideoCapture('rtsp://176.98.28.140:64787')

# Path to the model
path = 'D:\\pycharm_projects\\yolov7\\yolov7\\runs\\train\\shuliavka_trulicy_v4_40b_80ep\\weights\\best.pt'

# Load the YOLO model
model = torch.hub.load("WongKinYiu/yolov7", "custom", path, trust_repo=True)

# Object classes
classNames = ['car', 'person', 'truck']

# Define confidence threshold
confidence_threshold = 0.3

# Initialize variables for FPS calculation
prev_time = 0

# Window name
window_name = 'rtsp'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a resizable window

try:
    while True:
        img = cap.read()
        if img is None:
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Perform inference
        results = model(img)

        # Get the detections
        detections = results.xyxy[0]  # assuming the first image in the batch

        for detection in detections:
            # Bounding box and other details
            x1, y1, x2, y2, conf, cls = detection

            # Filter out low-confidence detections
            if conf < confidence_threshold:
                continue

            # Convert to int values
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Class name
            cls = int(cls.item())
            class_name = classNames[cls]

            # Object details
            org = (x1, y1 - 10)  # Adjusted position to prevent overlap with the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.6
            color = (255, 0, 0)
            thickness = 1

            cv2.putText(img, f'{class_name} {conf:.2f}', org, font, fontScale, color, thickness)

        # Display FPS on the frame
        cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize the window
        cv2.resizeWindow(window_name, 1280, 720)  # Resize window to 1280x720

        # Display the image
        cv2.imshow(window_name, img)
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
