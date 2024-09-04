import cv2
import numpy as np

# Path to your video file
video_path = r"C:\Users\User\Desktop\fly\GENERIC_RTSP-realmonitor_2023_09_20_15_57_29.avi"

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the number of rows to ignore from the top
IGNORE_ROWS = 150

# Read the first frame
ret, prev_frame = cap.read()

if not ret or prev_frame is None:
    print("Error: Could not read the first frame.")
    exit()

# Crop the first frame to ignore the top 200 rows
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
                                   [0, 0, 0, 1]], np.float32) * 0.03

# Parameters for fine-tuning
THRESHOLD_VALUE = 35  # Initial threshold value
MIN_CONTOUR_AREA = 5  # Minimum area of contours to consider as objects
MORPH_KERNEL_SIZE = (5, 5)  # Kernel size for morphological operations
DILATION_ITERATIONS = 2  # Number of dilation iterations
EROSION_ITERATIONS = 1  # Number of erosion iterations

# Read until video is completed
while cap.isOpened():
    # Capture the next frame
    ret, curr_frame = cap.read()

    if not ret or curr_frame is None:
        print("Error: Could not read the frame.")
        break

    # Crop the frame to ignore the top 200 rows
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

    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:  # Ignore small contours
            continue

        # Get bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Update Kalman filter with the center of the bounding box
        obj_center_x = int(x + w / 2)
        obj_center_y = int(y + h / 2)
        measurement = np.array([[np.float32(obj_center_x)],
                                [np.float32(obj_center_y)]])
        kalman.correct(measurement)
        prediction = kalman.predict()

        # Draw the predicted trajectory
        cv2.circle(curr_frame, (int(prediction[0]), int(prediction[1])), 4, (0, 255, 0), -1)

    # Display the resulting frame with the detected object
    cv2.imshow('Frame', curr_frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Update previous frame
    prev_gray = curr_gray.copy()

# Release the video capture object
cap.release()

# Close all the frames
cv2.destroyAllWindows()
