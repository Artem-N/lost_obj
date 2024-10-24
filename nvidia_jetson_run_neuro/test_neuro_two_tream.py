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

# Start video captures for two streams
cap1 = VideoCapture("rtsp://admin:ZirRobotics@172.16.14.12:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif")
cap2 = VideoCapture("rtsp://admin:ZirRobotics@172.16.14.12:554/cam/realmonitor?channel=2&subtype=0&unicast=true&proto=Onvif")

# Get default frame width and height from the first capture object
frame_width = int(cap1.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap1.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Frame dimensions: {frame_width}x{frame_height}")

# FPS variables
fps1, fps2 = 0, 0
frame_count1, frame_count2 = 0, 0
start_time1, start_time2 = time.time(), time.time()

while True:
    # Read frames from both VideoCaptures
    frame1 = cap1.read()
    frame2 = cap2.read()

    if frame1 is None and frame2 is None:
        break

    # Resize frames for faster inference
    resized_frame1 = cv2.resize(frame1, (640, 640)) if frame1 is not None else None
    resized_frame2 = cv2.resize(frame2, (640, 640)) if frame2 is not None else None

    # Run YOLO inference
    results1 = model(resized_frame1, stream=True) if resized_frame1 is not None else []
    results2 = model(resized_frame2, stream=True) if resized_frame2 is not None else []

    # Process results for the first stream
    for result in results1:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1 = int(x1 * (frame_width / 640))
            x2 = int(x2 * (frame_width / 640))
            y1 = int(y1 * (frame_height / 640))
            y2 = int(y2 * (frame_height / 640))
            cls_id = int(box.cls[0])
            conf = box.conf[0]
            class_name = model.names[cls_id]
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame1, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Process results for the second stream
    for result in results2:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1 = int(x1 * (frame_width / 640))
            x2 = int(x2 * (frame_width / 640))
            y1 = int(y1 * (frame_height / 640))
            y2 = int(y2 * (frame_height / 640))
            cls_id = int(box.cls[0])
            conf = box.conf[0]
            class_name = model.names[cls_id]
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Update frame count and calculate FPS for the first stream
    frame_count1 += 1
    elapsed_time1 = time.time() - start_time1
    if elapsed_time1 > 0:
        fps1 = frame_count1 / elapsed_time1
    if frame1 is not None:
        cv2.putText(frame1, f"FPS: {fps1:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Update frame count and calculate FPS for the second stream
    frame_count2 += 1
    elapsed_time2 = time.time() - start_time2
    if elapsed_time2 > 0:
        fps2 = frame_count2 / elapsed_time2
    if frame2 is not None:
        cv2.putText(frame2, f"FPS: {fps2:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frames in separate windows
    if frame1 is not None:
        cv2.imshow("YOLO Detection - Stream 1", frame1)
    if frame2 is not None:
        cv2.imshow("YOLO Detection - Stream 2", frame2)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap1.release()
cap2.release()
cv2.destroyAllWindows()

