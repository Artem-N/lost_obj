import supervision as sv
from ultralytics import YOLO
import numpy as np
import cv2

# Load a model
model = YOLO('runs/detect/drone_v1_200ep_32bath/weights/best.pt')  # load an official model

byte_tracker = sv.ByteTrack()
annotator = sv.BoxAnnotator()


def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections)

    labels = []
    for detection in detections:
        xyxy, _, confidence, class_id, tracker_id, data = detection
        label = f"#{tracker_id} {data['class_name']} {confidence:0.2f}"
        labels.append(label)

    annotated_frame = annotator.annotate(scene=frame.copy(), detections=detections)

    # Adding labels manually
    for i, detection in enumerate(detections):
        xyxy = detection[0]
        label = labels[i]
        x1, y1, x2, y2 = map(int, xyxy)
        annotated_frame = cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                      1)

    return annotated_frame


sv.process_video(source_path=r"C:\Users\User\Desktop\fly\GENERIC_RTSP-realmonitor_2023_09_20_15_38_57.avi", target_path="result3.mp4",
                 callback=callback)
