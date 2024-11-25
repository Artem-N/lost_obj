import cv2
import numpy as np
import time
from screeninfo import get_monitors
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r"D:\pycharm_projects\yolov8\runs\detect\drone_v7_200ep_32bath\weights\best.pt")  # Use the path to your YOLOv8 model

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

def track_objects_lucas_kanade(cap, screen_width, screen_height, crosshair_length):
    """Track the object across video frames using YOLOv8 and Lucas-Kanade optical flow."""
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    trajectory_points = []
    frame_center = (screen_width // 2, screen_height // 2)
    old_gray = None
    p0 = None

    while cap.isOpened():
        start_time = time.time()

        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Could not read the frame.")
            break

        frame = cv2.resize(frame, (screen_width, screen_height))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if p0 is None:
            # If no points to track, use YOLO to detect the plane
            detections = detect_objects_yolov8(frame)

            if len(detections) > 0:
                # Use the first detected plane for tracking
                x1, y1, x2, y2 = detections[0][:4]
                bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                p0 = np.array([[bbox_center]], dtype=np.float32)
                old_gray = frame_gray.copy()
        else:
            # If points are available, use Lucas-Kanade to track them
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            if st is not None and len(p1) > 0:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv2.line(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)
                    cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
                    trajectory_points.append((a, b))

                p0 = good_new.reshape(-1, 1, 2)
                old_gray = frame_gray.copy()
            else:
                # Tracking failed, reset points
                print("Tracking lost, invoking YOLO for re-detection.")
                p0 = None

        draw_crosshair(frame)
        fps = calculate_fps(start_time)
        display_fps(frame, fps)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

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
    video_path = r"C:\Users\User\Desktop\bpla\WhatsApp Video 2024-10-24 at 15.16.26.mp4"

    # Get screen size
    screen_width, screen_height = get_screen_size()

    # Initialize video capture
    cap = initialize_video_capture(video_path)

    cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Track the object across frames using YOLOv8 and Lucas-Kanade
    track_objects_lucas_kanade(cap, screen_width, screen_height, CROSSHAIR_LENGTH)

    # Release video capture and close all frames
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()