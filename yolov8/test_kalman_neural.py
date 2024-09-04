import cv2
from filterpy.kalman import KalmanFilter
import numpy as np
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO(r"runs/detect/drone_v1_200ep_32bath/weights/best.pt")

# Initialize the Kalman filter
kf = KalmanFilter(dim_x=8, dim_z=4)
kf.F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                 [0, 1, 0, 0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0, 0, 1, 0],
                 [0, 0, 0, 1, 0, 0, 0, 1],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0]])

kf.P *= 10.  # initial covariance matrix
kf.R *= 0.01  # measurement noise
kf.Q = np.eye(8) * 0.1  # process noise

# Function to convert bounding box to the state vector
def bbox_to_state(bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    x, y = x1 + w / 2, y1 + h / 2
    return np.array([x, y, w, h]).reshape((4, 1))

# Function to convert state vector to bounding box
def state_to_bbox(state):
    x, y, w, h = state[:4].flatten()
    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = int(x + w / 2), int(y + h / 2)
    return x1, y1, x2, y2

# Run inference and Kalman filtering
cap = cv2.VideoCapture(r"C:\Users\User\Desktop\fly\GeneralPT4S4_2023_09_20_15_31_29.avi")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model.predict(frame, conf=0.15)
    results = results[0]  # Access the first (and only) Results object

    # Update Kalman filter
    if len(results.boxes) > 0:  # if YOLO detects any object
        bbox = results.boxes.xyxy[0].cpu().numpy()  # get the first detected bounding box
        state = bbox_to_state(bbox)
        kf.update(state)

    kf.predict()

    # Get the predicted state and convert to bounding box
    predicted_bbox = state_to_bbox(kf.x)
    x1, y1, x2, y2 = predicted_bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
