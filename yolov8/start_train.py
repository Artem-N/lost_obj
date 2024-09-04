from ultralytics import YOLO

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main():
    # Load a model
    model = YOLO('yolov8m.pt')

    # Train the model
    results = model.train(data="drone.yaml", epochs=200, batch=32, imgsz=640, workers=10, device=0, name="drone_v1_200ep_32bath")
    #results = model(r"C:\Users\User\Desktop\tenis\test2_resize - Trim.mp4", show=True)


if __name__ == '__main__':
    main()
