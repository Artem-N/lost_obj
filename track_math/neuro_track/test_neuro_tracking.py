import cv2
import numpy as np
import time
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

def initialize_tracker():
    """Initialize an OpenCV tracker (e.g., KCF Tracker)."""
    return cv2.TrackerCSRT_create()

def track_objects_yolov8(cap, screen_width, screen_height, crosshair_length):
    """Track the object across video frames using YOLOv8 and OpenCV trackers."""
    trajectory_points = []
    frame_center = (screen_width // 2, screen_height // 2)
    tracker = None
    bbox = None

    while cap.isOpened():
        start_time = time.time()

        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Could not read the frame.")
            break

        frame = cv2.resize(frame, (screen_width, screen_height))

        if tracker is None or bbox is None:
            # If no tracker or target is lost, use YOLO to detect the plane
            detections = detect_objects_yolov8(frame)

            if len(detections) > 0:
                # Use the first detected plane for tracking
                x1, y1, x2, y2 = detections[0][:4]
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

                # Initialize the tracker with the detected bounding box
                tracker = initialize_tracker()
                tracker.init(frame, bbox)
        else:
            # If the tracker is initialized, update it
            success, bbox = tracker.update(frame)
            if success:
                # Tracking was successful
                x, y, w, h = map(int, bbox)
                last_position = (int(x + w / 2), int(y + h / 2))
                draw_bounding_box(frame, bbox)
                draw_trajectory_line(frame, frame_center, last_position)
                trajectory_points.append(last_position)
            else:
                # Tracking failed, reset the tracker and bbox
                print("Tracker lost the target, invoking YOLO for re-detection.")
                tracker = None
                bbox = None

        draw_crosshair(frame)
        fps = calculate_fps(start_time)
        display_fps(frame, fps)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

def draw_bounding_box(frame, bbox):
    """Draw a bounding box around the detected object."""
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw the rectangle with a blue color

def draw_crosshair(frame):
    """Draw a crosshair across the entire video frame."""
    color = (0, 255, 0)  # Green crosshair
    thickness = 1  # Thickness of the crosshair lines

    # Get the center of the frame
    center_x = frame.shape[1] // 2  # Width of the frame
    center_y = frame.shape[0] // 2  # Height of the frame

    # Draw the horizontal line across the entire width of the frame
    cv2.line(frame, (0, center_y), (frame.shape[1], center_y), color, thickness)

    # Draw the vertical line across the entire height of the frame
    cv2.line(frame, (center_x, 0), (center_x, frame.shape[0]), color, thickness)

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
    # Parameters for fine-tuning the algorithm
    CROSSHAIR_LENGTH = 25  # Length of the crosshair lines in pixels

    # Path to the video file
    video_path = r"C:\Users\User\Desktop\fly\WhatsApp Video 2024-09-14 at 12.40.19.mp4"

    # Get screen size
    screen_width, screen_height = get_screen_size()

    # Initialize video capture
    cap = initialize_video_capture(video_path)

    cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Track the object across frames using YOLOv8 and OpenCV tracker
    track_objects_yolov8(cap, screen_width, screen_height, CROSSHAIR_LENGTH)

    # Release video capture and close all frames
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
