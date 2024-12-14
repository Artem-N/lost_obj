import cv2
from ultralytics import YOLO
import numpy as np

# Load the model
model = YOLO(
    r"D:\pycharm_projects\yolov8\runs\detect\drone_v2_220ep_40bath\weights\best.pt")  # Ensure this is a valid model file (.pt)


# Kalman Filter initialization
def initialize_kalman_filter():
    kalman = cv2.KalmanFilter(4, 2)  # 4 dynamic params (x, y, dx, dy), 2 measurement params (x, y)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kalman


# Initialize Kalman Filter
kalman = initialize_kalman_filter()

# Placeholder for the last known bounding box and a frame counter for the grace period
last_bbox = None
grace_period = 10  # Number of frames to keep the bbox after losing detection
grace_counter = 0

# Set up full-screen display
# cv2.namedWindow("Tracking", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

try:
    # Perform tracking with the model
    cap = cv2.VideoCapture(r"C:\Users\User\Desktop\fly\GENERIC_RTSP-realmonitor_2023_09_20_15_57_55.avi")  # Ensure this is the video file

    if not cap.isOpened():
        raise Exception("Error opening video file")

    # Get the video writer initialized to save the output video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object to save the output video
    output = cv2.VideoWriter('output_video1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame")
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

        # Write the processed frame to the output video file
        output.write(frame)

        # Show the video with detections or the last known bbox in full screen
        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting on user request")
            break

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    # Release the video capture and writer
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    if 'output' in locals():
        output.release()
    cv2.destroyAllWindows()
    print("Video processing completed and resources released.")
