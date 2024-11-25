import os
import shutil
import cv2
import numpy as np

# Define the paths
source_folder = r"E:\datasets\traiin_rivne\train\images"  # Replace with your source folder
rgb_folder = r"E:\datasets\traiin_rivne\train\images_rgb"
thermal_folder = r"E:\datasets\traiin_rivne\train\images_thermal"

# Create destination folders if they don't exist
os.makedirs(rgb_folder, exist_ok=True)
os.makedirs(thermal_folder, exist_ok=True)


# Function to check if an image is RGB or Grayscale
def check_image_channel(image_path):
    try:
        img = cv2.imread(image_path)

        if len(img.shape) == 2:  # Image has 2 dimensions -> Grayscale
            return "Grayscale"
        elif len(img.shape) == 3 and img.shape[2] == 3:  # Image has 3 channels
            # Check if all channels are identical -> Fake RGB Grayscale
            if np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 1] == img[:, :, 2]):
                return "Grayscale"
            else:
                return "RGB"
        else:
            return "Other"
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "Error"


# Sort the images
for file_name in os.listdir(source_folder):
    file_path = os.path.join(source_folder, file_name)
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp')):
        channel_type = check_image_channel(file_path)
        if channel_type == "RGB":
            shutil.copy(file_path, os.path.join(rgb_folder, file_name))
        elif channel_type == "Grayscale":
            shutil.copy(file_path, os.path.join(thermal_folder, file_name))

print("Images have been sorted successfully.")
