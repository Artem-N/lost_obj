import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLO model
model = YOLO(r"D:\pycharm_projects\yolov8\runs\detect\drone_v7_200ep_32bath\weights\best.pt")

# Initialize variables
tracker = None  # OpenCV tracker
tracking = False  # Flag to indicate if we are currently tracking

# Perform tracking with the model
cap = cv2.VideoCapture(r"C:\Users\User\Desktop\fly\GeneralPT4S4_2023_09_20_15_28_21.avi")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
output_video_path = r'C:\Users\User\Desktop\show_thursday\bpla_day_4.mp4'  # Change this to your desired output file path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change 'mp4v' to 'XVID' for .avi files
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if not tracking:
        # Perform detection
        results = model.predict(frame, conf=0.15, iou=0.45)

        # Assuming we're tracking a single object, take the first detection
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                bbox = box.xyxy[0].cpu().numpy()
                detections.append(bbox)

        if len(detections) > 0:
            # Initialize tracker with the first detected bounding box
            bbox = detections[0]
            print("bbox:", bbox)
            print("bbox type:", type(bbox))

            # Convert bbox to standard Python floats
            bbox = bbox.tolist() if hasattr(bbox, 'tolist') else bbox
            x1, y1, x2, y2 = [float(coord) for coord in bbox]

            # Compute width and height
            w = x2 - x1
            h = y2 - y1
            x = x1
            y = y1

            # Ensure values are non-negative and dimensions are positive
            x = max(0, x)
            y = max(0, y)
            w = max(0, w)
            h = max(0, h)

            if w <= 0 or h <= 0:
                print("Invalid bounding box dimensions: w =", w, "h =", h)
                tracking = False
                # Write the frame without bounding box to the output video
                out.write(frame)
                continue  # Skip this frame

            # Convert coordinates to integers
            x = int(round(x))
            y = int(round(y))
            w = int(round(w))
            h = int(round(h))

            # Debugging statements
            print("x:", x, "type:", type(x))
            print("y:", y, "type:", type(y))
            print("w:", w, "type:", type(w))
            print("h:", h, "type:", type(h))

            # Create OpenCV tracker
            if hasattr(cv2, 'legacy'):
                tracker = cv2.legacy.TrackerCSRT_create()
            elif hasattr(cv2, 'TrackerCSRT_create'):
                tracker = cv2.TrackerCSRT_create()
            else:
                print("CSRT Tracker is not available in your OpenCV installation.")
                # Fallback to KCF tracker
                if hasattr(cv2, 'legacy'):
                    tracker = cv2.legacy.TrackerKCF_create()
                elif hasattr(cv2, 'TrackerKCF_create'):
                    tracker = cv2.TrackerKCF_create()
                else:
                    print("KCF Tracker is not available. Exiting.")
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                    exit()

            # Initialize tracker with integer bounding box
            bbox_tuple = (x, y, w, h)
            ok = tracker.init(frame, bbox_tuple)
            if ok:
                tracking = True
                frame_count = 0  # Reset frame count after successful initialization
            else:
                print("Tracker initialization failed.")
                tracking = False
                # Write the frame without bounding box to the output video
                out.write(frame)
        else:
            # No detections; display frame as is
            cv2.imshow("Tracking", frame)
            # Write the frame to the output video
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
    else:
        # Update tracker
        ok, bbox = tracker.update(frame)
        if ok:
            # Tracking success
            x, y, w, h = bbox

            # Convert coordinates to integers for drawing
            x1 = int(round(x))
            y1 = int(round(y))
            x2 = int(round(x + w))
            y2 = int(round(y + h))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            # Tracking failure
            print("Tracking failure detected. Reinitializing detection.")
            tracking = False
            tracker = None

        # Display result
        cv2.imshow("Tracking", frame)
        # Write the frame to the output video
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
