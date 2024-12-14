import cv2
import torch
import time
import threading
import queue
import random

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
cap = VideoCapture('rtsp://admin:ZirRobotics@10.3.1.7:554/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif')

# Get default frame width and height from the capture object
frame_width = int(cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Frame dimensions: {frame_width}x{frame_height}")

# Path to the model
path = r"D:\pycharm_projects\yolov7\yolov7\yolov7.pt"

# Load the YOLO model
model = torch.hub.load("WongKinYiu/yolov7", "custom", path, trust_repo=True, force_reload=False)

# COCO Dataset Class Names
classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Assign a unique color to each class
random.seed(42)  # For reproducibility
classColors = {class_name: tuple(random.randint(0, 255) for _ in range(3)) for class_name in classNames}

# Define confidence threshold
confidence_threshold = 0.5

# Initialize variables for FPS calculation
prev_time = 0

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video5.mp4', fourcc, 20.0, (frame_width, frame_height))

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
        fps = 1 / (current_time - prev_time) if prev_time else 0.0
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

            # Get class name and color
            cls = int(cls.item())
            if cls >= len(classNames):
                continue  # Skip if class index is out of range
            class_name = classNames[cls]
            color = classColors[class_name]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Object details
            org = (x1, y1 - 10)  # Adjusted position to prevent overlap with the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.6
            thickness = 1

            cv2.putText(img, f'{class_name} {conf:.2f}', org, font, fontScale, color, thickness)

        # Display FPS on the frame
        cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write the frame to the video file
        out.write(img)

        # Display the image
        cv2.imshow(window_name, img)
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
