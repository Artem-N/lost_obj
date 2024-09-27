import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLO model
model = YOLO(r"D:\pycharm_projects\yolov8\runs\detect\drone_v3_250ep_32bath2\weights\best.pt")

# Initialize variables
tracker = None  # OpenCV tracker
tracking = False  # Flag to indicate if we are currently tracking

# Open the input video
input_video_path = r"C:\Users\User\Desktop\fly\WhatsApp Video 2024-09-09 at 12.08.51.mp4"
cap = cv2.VideoCapture(input_video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Cannot open video file {input_video_path}")
    exit()

# Retrieve properties of the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose other codecs like 'XVID'

# Define the output video path
output_video_path = r"C:\Users\User\Desktop\порівняння\output_tracking_1.mp4"

# Initialize the VideoWriter
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print(f"Error: Cannot open video writer for file {output_video_path}")
    cap.release()
    exit()

print(f"Recording output video to {output_video_path}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video file reached.")
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
                    exit()

            # Initialize tracker with integer bounding box
            bbox_tuple = (x, y, w, h)
            ok = tracker.init(frame, bbox_tuple)
            if ok:
                tracking = True
                frame_count = 0  # Reset frame count after successful initialization
                print("Tracker initialized successfully.")
            else:
                print("Tracker initialization failed.")
                tracking = False
        else:
            # No detections; display and write frame as is
            cv2.imshow("Tracking", frame)
            out.write(frame)  # Write the original frame
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

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        else:
            # Tracking failure
            print("Tracking failure detected. Reinitializing detection.")
            tracking = False
            tracker = None
            # Optionally, you can add text to indicate tracking failure
            cv2.putText(frame, "Tracking Failure", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Display the frame with tracking (or failure)
        cv2.imshow("Tracking", frame)
        out.write(frame)  # Write the processed frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit key pressed by user.")
            break

# Release resources
cap.release()
out.release()  # Make sure to release the VideoWriter
cv2.destroyAllWindows()
print(f"Output video saved to {output_video_path}")
