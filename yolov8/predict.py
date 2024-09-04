from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO(r"runs/detect/drone_v1_200ep_32bath/weights/best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict(r"C:\Users\User\Desktop\fly\GeneralPT4S4_2023_09_20_15_31_29.avi", save=False, show=True, conf=0.15)
