import cv2
import numpy as np

# Path to your video file
video_path = r"C:\Users\User\Desktop\fly\GENERIC_RTSP-realmonitor_2023_09_20_15_33_25.avi"

# Path to save the output video
output_video_path = r"C:\Users\User\Desktop\fly\air3.avi"

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get original video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the number of rows to ignore from the top
IGNORE_ROWS = 200

# Adjust the height after cropping
cropped_height = frame_height - IGNORE_ROWS

# Define the codec and create a VideoWriter object with cropped dimensions
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, cropped_height))

# Read the first frame
ret, prev_frame = cap.read()

if not ret or prev_frame is None:
    print("Error: Could not read the first frame.")
    exit()

# Crop the first frame to ignore the top rows
prev_frame = prev_frame[IGNORE_ROWS:, :]

# Convert to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Initialize Kalman filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.1
kalman.measurementNoiseCov = np.array([[1, 0],
                                       [0, 1]], np.float32) * 0.1
kalman.errorCovPost = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], np.float32) * 0.1

# Parameters for fine-tuning
THRESHOLD_VALUE = 40  # Initial threshold value
MIN_CONTOUR_AREA = 100  # Minimum area of contours to consider as objects
MORPH_KERNEL_SIZE = (5, 5)  # Kernel size for morphological operations
DILATION_ITERATIONS = 3  # Number of dilation iterations
EROSION_ITERATIONS = 1  # Number of erosion iterations

# Number of future frames to predict
NUM_PREDICTIONS = 10
MAX_TRAJECTORY_LENGTH = 100  # Maximum length of trajectory in pixels

# Variables to keep track of the object
tracked_object = None
last_position = None
trajectory_points = []  # List to store the trajectory points

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Read until video is completed
while cap.isOpened():
    # Capture the next frame
    ret, curr_frame = cap.read()

    if not ret or curr_frame is None:
        print("Error: Could not read the frame.")
        break

    # Crop the frame to ignore the top rows
    curr_frame = curr_frame[IGNORE_ROWS:, :]

    # Convert to grayscale
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between consecutive frames
    diff = cv2.absdiff(prev_gray, curr_gray)

    # Apply a Gaussian blur to reduce noise
    diff = cv2.GaussianBlur(diff, (5, 5), 0)

    # Apply thresholding to detect significant changes
    _, thresh = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to clean up the noise
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=DILATION_ITERATIONS)
    thresh = cv2.erode(thresh, kernel, iterations=EROSION_ITERATIONS)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours of the moving object
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if tracked_object is None and contours:
        # Initialize the tracked object with the first detected contour
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) >= MIN_CONTOUR_AREA:
            tracked_object = contour
            (x, y, w, h) = cv2.boundingRect(contour)
            last_position = (x + w / 2, y + h / 2)
            measurement = np.array([[np.float32(last_position[0])],
                                    [np.float32(last_position[1])]])
            kalman.correct(measurement)
            trajectory_points.append(last_position)  # Add to trajectory
    elif tracked_object is not None:
        # Continue tracking the object closest to the last known position
        closest_contour = None
        min_distance = float("inf")
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            center = (x + w / 2, y + h / 2)
            distance = np.linalg.norm(np.array(center) - np.array(last_position))
            if distance < min_distance:
                min_distance = distance
                closest_contour = contour

        if closest_contour is not None:
            (x, y, w, h) = cv2.boundingRect(closest_contour)
            tracked_object = closest_contour
            last_position = (x + w / 2, y + h / 2)
            measurement = np.array([[np.float32(last_position[0])],
                                    [np.float32(last_position[1])]])
            kalman.correct(measurement)
            prediction = kalman.predict()

            # Plot the future trajectory
            future_positions = []
            for _ in range(NUM_PREDICTIONS):
                prediction = kalman.predict()
                future_positions.append((int(prediction[0]), int(prediction[1])))

            # Draw the bounding box and future trajectory
            cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Draw the predicted future position
            # cv2.circle(curr_frame, future_positions[-1], 4, (0, 255, 0), -1)

            # Add the current position to the trajectory
            trajectory_points.append(last_position)

            # Ensure the trajectory length does not exceed MAX_TRAJECTORY_LENGTH
            total_length = 0
            for i in range(len(trajectory_points) - 1, 0, -1):
                pt1 = trajectory_points[i]
                pt2 = trajectory_points[i - 1]
                total_length += calculate_distance(pt1, pt2)
                if total_length > MAX_TRAJECTORY_LENGTH:
                    trajectory_points = trajectory_points[i:]
                    break

            # Draw the trajectory line
            for i in range(1, len(trajectory_points)):
                pt1 = tuple(map(int, trajectory_points[i-1]))
                pt2 = tuple(map(int, trajectory_points[i]))
                cv2.line(curr_frame, pt1, pt2, (0, 255, 255), 2)

    # Write the frame to the output video
    out.write(curr_frame)

    # Display the resulting frame with the detected object and trajectory
    cv2.imshow('Frame', curr_frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Update previous frame
    prev_gray = curr_gray.copy()

# Release the video capture and writer objects
cap.release()
out.release()

# Close all the frames
cv2.destroyAllWindows()
