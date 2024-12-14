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
cap = VideoCapture('rtsp://admin:ZirRobotics@10.3.1.7:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif')

# Get default frame width and height from the capture object
frame_width = int(cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Frame dimensions: {frame_width}x{frame_height}")

# Path to the model
path = r"D:\pycharm_projects\yolov7\yolov7\runs\train\fresh_shuliavka_200ep_16batch_v2_model7reg\weights\best.pt"
# Load the YOLO model
model = torch.hub.load("WongKinYiu/yolov7", "custom", path, trust_repo=True, force_reload=False)

# Object classes
# classNames = ['car', 'person', 'truck', 'moto']
classNames = ['person', 'car', 'truck', 'moto']

# Define colors for each class
classColors = {
    'car': (255, 0, 0),
    'person': (0, 255, 0),
    'truck': (0, 0, 255),
    'moto': (0, 200, 200)
}

# Define confidence threshold
confidence_threshold = 0.3

# Initialize variables for FPS calculation
prev_time = 0

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output_video5.mp4', fourcc, 20.0, (frame_width, frame_height))

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

            # Get class name and color
            cls = int(cls.item())
            class_name = classNames[cls]
            color = classColors[class_name]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Object details
            org = (x1, y1 - 10)  # Adjusted position to prevent overlap with the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.6
            thickness = 2

            cv2.putText(img, f'{class_name} {conf:.2f}', org, font, fontScale, color, thickness)

        # Display FPS on the frame
        cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        # Write the frame to the video file
        # out.write(img)

        # Display the image
        cv2.imshow(window_name, img)
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    cap.release()
    # out.release()
    cv2.destroyAllWindows()
