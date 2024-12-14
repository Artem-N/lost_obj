from ultralytics import YOLO

 # Load a model
model = YOLO("yolov8m-pose.pt")  # load an official model

 # Predict with the model
results = model(r"C:\Users\User\Desktop\tenis\image.png", show=True)  # predict on an image


for result in results:
    if hasattr(result, 'keypoints'):
        for person in result.keypoints:
            coord = person.xy
            print(coord)
