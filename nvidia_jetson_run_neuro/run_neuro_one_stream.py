import cv2
from ultralytics import YOLO
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

# Load the YOLO model
model = YOLO("yolo11m.engine", task='detect')  # Pretrained YOLO11n model

# Start video capture
cap = VideoCapture("rtsp://admin:ZirRobotics@172.16.14.12:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif")

# Get default frame width and height from the capture object
frame_width = int(cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Frame dimensions: {frame_width}x{frame_height}")

# FPS variables
fps = 0
frame_count = 0
start_time = time.time()

while True:
    # Read a frame from the VideoCapture
    frame = cap.read()

    if frame is None:
        break

    # Resize the frame for faster inference
    resized_frame = cv2.resize(frame, (640, 640))

    # Run YOLO inference
    results = model(resized_frame, stream=True)

    # Process results
    for result in results:
        boxes = result.boxes  # Bounding boxes

        for box in boxes:
            # Extract coordinates and convert to integers
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Adjust the bounding box coordinates to the original frame size
            x1 = int(x1 * (frame.shape[1] / 640))
            x2 = int(x2 * (frame.shape[1] / 640))
            y1 = int(y1 * (frame.shape[0] / 640))
            y2 = int(y2 * (frame.shape[0] / 640))

            # Get class ID and confidence
            cls_id = int(box.cls[0])
            conf = box.conf[0]

            # Get class name
            class_name = model.names[cls_id]

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put label with confidence score
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Update frame count and calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLO Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
