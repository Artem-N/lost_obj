# from ultralytics import YOLO
#
# # Load the YOLOv8 model
# model = YOLO("yolov8n.pt")
#
# # Export the model to TensorRT format
# model.export(format="engine")  # creates 'yolov8n.engine'
#
# # Load the exported TensorRT model
# tensorrt_model = YOLO("yolov8n.engine")
#
# # Run inference
# results = tensorrt_model("https://ultralytics.com/images/bus.jpg")

from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/drone_v7_200ep_32bath/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")
