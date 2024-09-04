import supervision as sv
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8m.pt')
tracker = sv.ByteTrack()

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

    annotated_frame = bounding_box_annotator.annotate(
        scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame

sv.process_video(
    source_path=r"D:\video_test_sbu\4825785-hd_1920_1080_30fps.mp4",
    target_path=r'D:\pycharm_projects\yolov8\test1.mp4',
    callback=callback
)
