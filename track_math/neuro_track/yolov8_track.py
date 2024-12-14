import cv2
from ultralytics import YOLO
import numpy as np


# ===========================
# Configuration Parameters
# ===========================

class Config:
    """Configuration parameters for the object tracking application."""
    MODEL_PATH = r"D:\pycharm_projects\yolov8\runs\detect\drone_v7_200ep_32bath\weights\best.pt"  # Ensure this is a valid model file (.pt)
    VIDEO_PATH = r"C:\Users\User\Desktop\bpla\WhatsApp Video 2024-10-24 at 15.16.26 (2).mp4" # Ensure this is the video file
    CONFIDENCE_THRESHOLD = 0.15
    IOU_THRESHOLD = 0.6
    MOVEMENT_THRESHOLD = 1  # pixels
    STATIONARY_FRAME_LIMIT = 2
    DISPLAY_WINDOW_NAME = "Tracking"

    # New Configuration Parameters for Resizing
    RESIZE_DISPLAY = False  # Flag to control resizing
    RESIZE_WIDTH = 1080  # Desired width after resizing
    RESIZE_HEIGHT = 720  # Desired height after resizing

    # Configuration for Output Video
    OUTPUT_VIDEO_PATH = r'C:\Users\User\Desktop\show_thursday\bpla_day_3.mp4'  # Output video file path
    OUTPUT_VIDEO_CODEC = 'mp4v'  # Codec for the output video (e.g., 'mp4v', 'XVID')


# ===========================
# Kalman Filter Initialization
# ===========================

# Load the model
model = YOLO(Config.MODEL_PATH)  # Ensure this is a valid model file (.pt)


# Kalman Filter initialization
def initialize_kalman_filter():
    kalman = cv2.KalmanFilter(4, 2)  # 4 dynamic params (x, y, dx, dy), 2 measurement params (x, y)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kalman


# Initialize Kalman Filter
kalman = initialize_kalman_filter()

# Placeholder for the last known bounding box and a frame counter for the grace period
last_bbox = None
grace_period = 2  # Number of frames to keep the bbox after losing detection
grace_counter = 0

# Set up full-screen display
# cv2.namedWindow("Tracking", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Perform tracking with the model
cap = cv2.VideoCapture(Config.VIDEO_PATH)  # Ensure this is the video file

# Initialize configuration
config = Config()

# Retrieve frame width, height, and fps
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# If RESIZE_DISPLAY is True, adjust frame_width and frame_height to RESIZE_WIDTH and RESIZE_HEIGHT
if config.RESIZE_DISPLAY:
    frame_width = config.RESIZE_WIDTH
    frame_height = config.RESIZE_HEIGHT

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_VIDEO_CODEC)
out = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

def resize_frame_if_needed(frame, config):
    """
    Resizes the frame to the specified dimensions if RESIZE_DISPLAY is True.

    Args:
        frame (numpy.ndarray): The original video frame.
        config (Config): Configuration object containing resizing parameters.

    Returns:
        numpy.ndarray: The resized frame if resizing is enabled; otherwise, the original frame.
    """
    if config.RESIZE_DISPLAY:
        return cv2.resize(frame, (config.RESIZE_WIDTH, config.RESIZE_HEIGHT))
    return frame


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection (pass frame directly to track)
    results = model.track(frame)  # Ensure frame is passed for tracking
    detected = False  # Flag to check if we have detections in the current frame

    if results:  # Check if results contain any detections
        for result in results:
            detections = result.boxes if hasattr(result, 'boxes') else []

            # If detection is found, update the last bounding box
            if len(detections) > 0:
                bbox = detections[0].xyxy[0].cpu().numpy()  # Get the first detected object

                # Update Kalman filter with the detected bounding box center
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                kalman.correct(np.array([[np.float32(center_x)], [np.float32(center_y)]]))

                last_bbox = bbox  # Save the detected bbox
                detected = True  # Mark that a detection occurred
                grace_counter = grace_period  # Reset the grace period counter
                break  # Only interested in the first detection

    # If no detections, decrease the grace counter and predict the next position using Kalman filter
    if not detected:
        if grace_counter > 0:
            grace_counter -= 1
        else:
            if last_bbox is not None:
                # Predict the new position using Kalman filter when no detection
                predicted = kalman.predict()
                predicted_center_x, predicted_center_y = predicted[0], predicted[1]
                width = last_bbox[2] - last_bbox[0]
                height = last_bbox[3] - last_bbox[1]

                # Generate predicted bounding box from predicted center
                predicted_bbox = [
                    predicted_center_x - width / 2,
                    predicted_center_y - height / 2,
                    predicted_center_x + width / 2,
                    predicted_center_y + height / 2
                ]
                last_bbox = predicted_bbox

    # Draw the bounding box (only if we have a valid last known detection or predicted box)
    if last_bbox is not None and grace_counter > 0:
        cv2.rectangle(frame, (int(last_bbox[0]), int(last_bbox[1])),
                      (int(last_bbox[2]), int(last_bbox[3])), (0, 255, 0), 2)

    # Conditional Resizing Before Display
    frame_to_display = resize_frame_if_needed(frame, config)

    # Show the video with detections or the last known bbox
    cv2.imshow(config.DISPLAY_WINDOW_NAME, frame_to_display)

    # Write the frame to the output video
    out.write(frame_to_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
