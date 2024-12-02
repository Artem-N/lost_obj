import os
import random
from PIL import Image, ImageEnhance
import numpy as np
import shutil  # Import shutil for file copying
from tqdm import tqdm  # Import tqdm for the progress bar

# Define the paths to the images and annotations folders
images_folder = r"E:\datasets\drone_data\dronevbird\test\images"  # Path to your images folder
annotations_folder = r"E:\datasets\drone_data\dronevbird\test\labels"  # Path to your annotations folder

# Define the paths to the output folders
output_images_folder = r"E:\datasets\drone_data\dronevbird\test\images_noise"  # Path to your output images folder
output_annotations_folder = r"E:\datasets\drone_data\dronevbird\test\labels_noise"  # Path to your output annotations folder

# Create the output directories if they don't exist
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_annotations_folder, exist_ok=True)

def apply_saturation(image):
    """Increase saturation by more than 30%."""
    enhancer = ImageEnhance.Color(image)
    factor = 1.35 + random.uniform(0.1, 0.7)  # Increase by 30% to 100%
    return enhancer.enhance(factor)

def apply_noise(image):
    """Add Gaussian noise to the image."""
    np_image = np.array(image)
    noise = np.random.normal(0, 0.01 * 255, np_image.shape).astype(np.uint8)
    noisy_image = np.clip(np_image + noise, 0, 255)
    return Image.fromarray(noisy_image)

# Get a list of image files in the images folder
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Iterate over all images with a progress bar
for image_filename in tqdm(image_files, desc="Processing images", unit="image"):
    image_file_path = os.path.join(images_folder, image_filename)
    with Image.open(image_file_path) as image:
        # Randomly choose augmentation type
        if random.random() < 0.5:
            augmented_image = apply_saturation(image)
        else:
            augmented_image = apply_noise(image)

        # Save the augmented image to the output images folder
        output_image_path = os.path.join(output_images_folder, image_filename)
        augmented_image.save(output_image_path)

    # Copy the corresponding annotation file to the output annotations folder
    base_name, _ = os.path.splitext(image_filename)
    annotation_filename = base_name + '.txt'
    annotation_file_path = os.path.join(annotations_folder, annotation_filename)

    if os.path.exists(annotation_file_path):
        output_annotation_path = os.path.join(output_annotations_folder, annotation_filename)
        shutil.copy(annotation_file_path, output_annotation_path)
    else:
        print(f"Annotation file for {image_filename} not found.")

print("Data augmentation completed successfully.")
