from ultralytics import YOLO

# # Load a pretrained YOLOv8n model
# # model = YOLO("yolo11m.pt")
# model = YOLO(r"D:\pycharm_projects\yolov8\runs\detect\drone_v9_300ep_32bath\weights\best.pt")
# # model = YOLO("yolo11x.pt")
#
# source = r"E:\video_for_test\fly\clear_video\GENERIC_RTSP-realmonitor_2023_09_20_15_52_11.avi"
# # source = 0
# result = model.predict(source, show=True, save=False, conf=0.15)


import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"D:\pycharm_projects\yolov8\runs\detect\rivne_train_model_L\weights\best.pt")  # Pretrained YOLO11n model
# model = YOLO(r"C:\Users\User\Desktop\WALDO30_yolov8m_640x640.pt")
# Open the webcam
cap = cv2.VideoCapture(r"C:\Users\User\Desktop\20444835-hd_1920_1080_30fps.mp4")

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get class names
class_names = model.names

window_name = "YOLO Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, conf=0.15)

    # Process results
    result = results[0]
    boxes = result.boxes  # Bounding boxes

    for box in boxes:
        # Extract coordinates and convert to integers
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Get class ID and confidence
        cls_id = int(box.cls[0])
        conf = box.conf[0]

        # Get class name
        class_name = class_names[cls_id]

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put label with confidence score
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLO Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
