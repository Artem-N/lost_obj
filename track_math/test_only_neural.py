import cv2
import time
import numpy as np
from screeninfo import get_monitors
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r"D:\pycharm_projects\yolov8\runs\detect\drone_v1_200ep_32bath\weights\best.pt")  # Use the path to your YOLOv8 model

# Replace this with the actual class ID for "plane" from your model's dataset
PLANE_CLASS_ID = 0  # Assuming 4 is the class ID for "plane" (e.g., in the COCO dataset)

def get_screen_size():
    """Get the screen width and height."""
    monitor = get_monitors()[0]  # Get the primary monitor's information
    return monitor.width, monitor.height

def initialize_video_capture(video_path):
    """Initialize video capture from the provided video file path."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")  # Raise an error if the video cannot be opened
    return cap

def detect_objects_yolov8(frame):
    """Detect objects in the frame using YOLOv8 and filter by the 'plane' class."""
    results = model(frame)  # Perform inference with the YOLO model
    detections = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes as NumPy array
    class_ids = results[0].boxes.cls.cpu().numpy()  # Get class IDs for each detection

    # Filter detections to include only those with the class ID for "plane"
    plane_detections = []
    for i, class_id in enumerate(class_ids):
        if int(class_id) == PLANE_CLASS_ID:
            plane_detections.append(detections[i])

    return np.array(plane_detections)

def detect_planes_in_video(cap, screen_width, screen_height):
    """Detect planes in each frame of the video using YOLOv8."""
    frame_center = (screen_width // 2, screen_height // 2)

    while cap.isOpened():
        start_time = time.time()

        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Could not read the frame.")
            break

        frame = cv2.resize(frame, (screen_width, screen_height))

        # Detect planes using YOLOv8
        detections = detect_objects_yolov8(frame)

        if len(detections) > 0:
            for detection in detections:
                x1, y1, x2, y2 = detection[:4]
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                last_position = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                draw_bounding_box(frame, bbox)
                draw_trajectory_line(frame, frame_center, last_position)

        fps = calculate_fps(start_time)
        display_fps(frame, fps)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

def draw_bounding_box(frame, bbox):
    """Draw a bounding box around the detected object."""
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw the rectangle with a blue color

def draw_trajectory_line(frame, frame_center, last_position):
    """Draw a red line from the center of the screen to the detected plane."""
    color = (0, 0, 255)  # Red line
    thickness = 2  # Thickness of the line
    cv2.line(frame, frame_center, last_position, color, thickness)

def calculate_fps(start_time):
    """Calculate the frames per second (FPS)."""
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    return fps

def display_fps(frame, fps):
    """Display the FPS on the frame."""
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

def main():
    # Path to the video file
    video_path = r"C:\Users\User\Desktop\fly\GeneralPT4S4_2023_09_20_15_34_00.avi"

    # Get screen size
    screen_width, screen_height = get_screen_size()

    # Initialize video capture
    cap = initialize_video_capture(video_path)

    cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Detect planes in each frame using YOLOv8
    detect_planes_in_video(cap, screen_width, screen_height)

    # Release video capture and close all frames
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
