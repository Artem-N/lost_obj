from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO("yolo11m.pt")

# Export the model to TensorRT format
model.export(format="engine", workspace=2, half=True)  # creates 'yolov8n.engine'

# Load the exported TensorRT model
trt_model = YOLO("yolo11m.engine")

# Run inference
results = trt_model("https://ultralytics.com/images/bus.jpg")
