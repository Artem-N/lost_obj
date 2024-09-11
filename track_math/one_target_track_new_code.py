import cv2
import numpy as np
import time
from screeninfo import get_monitors
import math

video_path = r"C:\Users\User\Desktop\fly\GENERIC_RTSP-realmonitor_2023_09_20_15_33_25.avi"


def get_screen_size():
    """Get the screen width and height of the primary monitor."""
    monitor = get_monitors()[0]  # Get the primary monitor's information
    return monitor.width, monitor.height

# Fetch screen width and height
screen_width, screen_height = 1920, 1080
frame_center = (screen_width // 2, screen_height // 2)
cap = cv2.VideoCapture(video_path)


class Config:
    threshold_value = 30
    min_contour_area = 10
    min_contour_distance = 50
    morph_kernel_size = (7, 7)
    dilation_iterations = 3
    erosion_iterations = 1
    target_memory_frames = 10
    movement_threshold = 700000  # Threshold for detecting camera movement


class Target:
    def __init__(self, contour):
        self.position = None
        self.frames_to_activate = 0
        self.frames_to_kill = 5
        self.active = False
        self.set_contour(contour)

    def set_contour(self, contour):
        self.contour = contour
        if self.contour is not None:
            (x, y, w, h) = cv2.boundingRect(contour)
            self.bbox = (x, y, w, h)
            self.prev_position = self.position
            self.position = (x + w // 2, y + h // 2)
            self.frames_to_activate += 1
            if self.frames_to_activate > 3:
                self.active = True
            self.frames_to_kill = 3


targets = []


def read_frame():
    is_frame_read, frame = cap.read()
    if not is_frame_read or frame is None:
        raise Exception("Error: Could not read the first frame.")

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(frame, (screen_width, screen_height)), cv2.resize(gray_frame, (screen_width, screen_height))


def threshold_frame(prev_frame, curr_frame):
    diff = cv2.absdiff(prev_frame, curr_frame)
    diff = cv2.GaussianBlur(diff, (7, 7), 0)
    _, thresh = cv2.threshold(diff, Config.threshold_value, 255, cv2.THRESH_BINARY)
    kernel = np.ones(Config.morph_kernel_size, np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=Config.dilation_iterations)
    thresh = cv2.erode(thresh, kernel, iterations=Config.erosion_iterations)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh


def initialize_kalman_filter():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]], np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
    return kalman


def update_kalman_filter(kalman, position):
    measurement = np.array([[np.float32(position[0])], [np.float32(position[1])]])
    kalman.correct(measurement)


def predict_kalman(kalman):
    prediction = kalman.predict()
    predicted_position = (int(prediction[0]), int(prediction[1]))
    return predicted_position


def draw_bounding_box(frame, bbox):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


def find_closest_target(contour, targets):
    (x, y, w, h) = cv2.boundingRect(contour)
    center = (x + w // 2, y + h // 2)
    closest_target = None
    min_distance = float("inf")
    for target in targets:
        distance = np.linalg.norm(np.array(center) - np.array(target.position))
        if distance < min_distance:
            min_distance = distance
            closest_target = target
    if min_distance <= Config.min_contour_distance:
        closest_target.set_contour(contour)
        return closest_target
    return None


def calculate_fps(start_time):
    return 1 / (time.time() - start_time)


def display_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def draw_crosshair(frame):
    """Draw a crosshair across the entire video frame."""
    color = (0, 255, 0)  # Green crosshair
    thickness = 1  # Thickness of the crosshair lines

    # Get the center of the frame
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2

    # Draw the horizontal line across the entire width of the frame
    cv2.line(frame, (0, center_y), (frame.shape[1], center_y), color, thickness)
    # Draw the vertical line across the entire height of the frame
    cv2.line(frame, (center_x, 0), (center_x, frame.shape[0]), color, thickness)


def draw_line_to_bbox(frame, bbox):
    """Draw a red line from the center of the resized screen to the center of the bounding box."""
    # Get the center of the frame
    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

    # Get the center of the bounding box
    x, y, w, h = bbox
    bbox_center = (x + w // 2, y + h // 2)

    # Draw a red line from the frame center to the bounding box center
    color = (0, 0, 255)  # Red color for the line
    thickness = 1  # Thickness of the line
    cv2.line(frame, frame_center, bbox_center, color, thickness)


def draw_info(frame, start_time):
    draw_crosshair(frame)
    cv2.imshow('Frame', frame)


def calculate_global_frame_difference(prev_frame, curr_frame):
    """Calculate the global frame difference to detect camera movement."""
    diff = cv2.absdiff(prev_frame, curr_frame)
    return np.sum(diff)


def detect_camera_movement(prev_frame, curr_frame, movement_threshold):
    """Detects camera movement based on frame differences."""
    global_diff = calculate_global_frame_difference(prev_frame, curr_frame)
    print(f"Global Frame Difference: {global_diff}")  # Print the difference value for debugging
    return global_diff > movement_threshold


def main():
    kalman = initialize_kalman_filter()
    _, prev_frame = read_frame()

    cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    camera_moving = False

    while cap.isOpened():
        start_time = time.time()
        frame, gray_frame = read_frame()

        # Detect if the camera is moving
        camera_moving = detect_camera_movement(prev_frame, gray_frame, Config.movement_threshold)

        if camera_moving:
            print("Camera is moving, skipping object tracking...")
        else:
            # print("Camera is stable, tracking objects...")
            thresh_frame = threshold_frame(prev_frame, gray_frame)
            contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            targets_to_kill = []
            for target in targets:
                target.frames_to_kill -= 1
                if target.frames_to_kill <= 0:
                    targets_to_kill.append(target)
            for target in targets_to_kill:
                targets.remove(target)

            if len(contours) + len(targets) <= 3:
                contours_with_no_targets = []
                for contour in contours:
                    target = find_closest_target(contour, targets)
                    if target is None:
                        contours_with_no_targets.append(contour)

                for contour in contours_with_no_targets:
                    targets.append(Target(contour))

                for target in targets:
                    if target.active:
                        draw_bounding_box(frame, target.bbox)
                        update_kalman_filter(kalman, target.position)
                        draw_line_to_bbox(frame, target.bbox)

            # If no contours found, predict using Kalman
            # if len(contours) == 0 and len(targets) > 0:
            #     predicted_position = predict_kalman(kalman)
            #     predicted_bbox = (predicted_position[0] - targets[0].bbox[2] // 2,
            #                       predicted_position[1] - targets[0].bbox[3] // 2,
            #                       targets[0].bbox[2],
            #                       targets[0].bbox[3])
            #     draw_bounding_box(frame, predicted_bbox)

        draw_info(frame, start_time)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        prev_frame = gray_frame

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
