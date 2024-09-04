import os
import random
from PIL import Image, ImageEnhance
import numpy as np

# Define the path to the images folder
images_folder = r"D:\pycharm_projects\yolov7\yolov7\train_rivne_lotpic\images_fly"  # Replace with the path to your images folder
output_folder = r"D:\pycharm_projects\yolov7\yolov7\train_rivne_lotpic\images_fly"  # Replace with the path to your output folder

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


def apply_saturation(image):
    """Increase saturation by more than 30%"""
    enhancer = ImageEnhance.Color(image)
    factor = 1.35 + random.uniform(0.1, 0.7)  # Increase by 30% to 100%
    return enhancer.enhance(factor)


def apply_noise(image):
    """Add 5% Gaussian noise to the image"""
    np_image = np.array(image)
    noise = np.random.normal(0, 0.05 * 255, np_image.shape).astype(np.uint8)
    noisy_image = np.clip(np_image + noise, 0, 255)
    return Image.fromarray(noisy_image)


# Iterate over all images in the folder
for image_filename in os.listdir(images_folder):
    image_file_path = os.path.join(images_folder, image_filename)

    with Image.open(image_file_path) as image:
        # Randomly choose augmentation type
        if random.random() < 0.5:
            augmented_image = apply_saturation(image)
        else:
            augmented_image = apply_noise(image)

        # Save the augmented image to the output folder
        output_image_path = os.path.join(output_folder, image_filename)
        augmented_image.save(output_image_path)

print("Data augmentation completed successfully.")
