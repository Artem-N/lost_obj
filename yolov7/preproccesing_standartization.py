import cv2
import os
import numpy as np


def preprocess_images(input_folder, output_folder, img_size=(640, 640)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load image {img_name}")
            continue

        # Resize image
        img_resized = cv2.resize(img, img_size)

        # Normalize image
        img_normalized = img_resized / 255.0

        # Standardize image
        img_standardized = (img_normalized - np.mean(img_normalized)) / np.std(img_normalized)

        # Save preprocessed image
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, ((img_standardized - img_standardized.min()) / (
                    img_standardized.max() - img_standardized.min()) * 255).astype(np.uint8))

    print("Preprocessing completed.")


input_folder = r"C:\Users\User\Desktop\dataset_rivne\new_data"
output_folder = r"C:\Users\User\Desktop\dataset_rivne\new_data_standart"
preprocess_images(input_folder, output_folder)
