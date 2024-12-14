# import cv2
import os
import requests
from tqdm.auto import tqdm

import zipfile
#
#
# def convert_images_to_rgb(directory):
#     # Ensure the directory exists
#     if not os.path.isdir(directory):
#         print(f"The directory {directory} does not exist.")
#         return
#
#     # Process each file in the directory
#     for filename in os.listdir(directory):
#         if filename.endswith('.jpg'):
#             img_path = os.path.join(directory, filename)
#
#             # Read the image
#             image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#             # Save the image back
#             cv2.imwrite(img_path, image_rgb)
#             print(f"Processed and saved {img_path}")
#         else:
#             print(f"Skipping non-PNG file {filename}")
#
#
# # Define the directory containing the images
# image_directory = "D:\\pycharm_projects\\test_some\\yolov7\\test\\images"
#
# # Convert all images in the directory to RGB
# convert_images_to_rgb(image_directory)

import os
import shutil

# Define the directories
images_folder = "D:\\hituav-a-highaltitude-infrared-thermal-dataset\\hit-uav\\images\\train"
txt_folder = "D:\\hituav-a-highaltitude-infrared-thermal-dataset\\hit-uav\\labels\\train"
output_images_folder = "D:\\hituav-a-highaltitude-infrared-thermal-dataset\\hit-uav\\images\\train_new"
output_txt_folder = "D:\\hituav-a-highaltitude-infrared-thermal-dataset\\hit-uav\\labels\\train_new"

# Create output directories if they don't exist
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_txt_folder, exist_ok=True)

# Define class mappings
class_mapping = {
    '0': '1',
    '1': '0',
    '3': '2'
}

# Function to process a single .txt file
def process_txt_file(txt_file_path, output_txt_file_path):
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    with open(output_txt_file_path, 'w') as file:
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0 and parts[0] in class_mapping:
                parts[0] = class_mapping[parts[0]]
                file.write(' '.join(parts) + '\n')

# Process all files
for txt_filename in os.listdir(txt_folder):
    if txt_filename.endswith('.txt'):
        base_name = os.path.splitext(txt_filename)[0]
        image_filename = base_name + '.jpg'  # Assuming the images are .jpg files

        # File paths
        txt_file_path = os.path.join(txt_folder, txt_filename)
        image_file_path = os.path.join(images_folder, image_filename)
        output_txt_file_path = os.path.join(output_txt_folder, txt_filename)
        output_image_file_path = os.path.join(output_images_folder, image_filename)

        # Process the .txt file
        process_txt_file(txt_file_path, output_txt_file_path)

        # Copy the corresponding image
        if os.path.exists(image_file_path):
            shutil.copy(image_file_path, output_image_file_path)

print("Processing completed.")

