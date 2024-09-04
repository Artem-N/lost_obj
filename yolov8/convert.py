import json
import os


def convert_annotations(json_file, output_dir, img_width=1280, img_height=720):
    with open(json_file, 'r') as f:
        data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for item in data:
        img_id = item['id']
        keypoints = item['kps']

        # Normalize keypoints
        normalized_keypoints = []
        for kp in keypoints:
            norm_x = kp[0] / img_width
            norm_y = kp[1] / img_height
            normalized_keypoints.extend([norm_x, norm_y])

        # Create the content for the .txt file
        yolo_format = [1]  # Object class index
        yolo_format.extend([0.5, 0.5, 1.0, 1.0])  # Dummy bbox values (center_x, center_y, width, height)
        yolo_format.extend(normalized_keypoints)

        # Write to .txt file
        txt_file_path = os.path.join(output_dir, f"{img_id}.txt")
        with open(txt_file_path, 'w') as txt_file:
            txt_file.write(' '.join(map(str, yolo_format)) + '\n')


# Convert both training and validation data
convert_annotations(r"C:\Users\User\Desktop\tenis\tennis_court_det_dataset__\data\data_train.json", r"C:\Users\User\Desktop\tenis\tennis_court_det_dataset__\data\train")
convert_annotations(r"C:\Users\User\Desktop\tenis\tennis_court_det_dataset__\data\data_val.json", r"C:\Users\User\Desktop\tenis\tennis_court_det_dataset__\data\val")
