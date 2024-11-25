import os
import shutil
import random

# Set the paths for the source folders and the destination folders
source_images_folder = r"E:\datasets\hituav-a-highaltitude-infrared-thermal-dataset\hit-uav\images\test_new"  # Folder containing the images
source_labels_folder = r"E:\datasets\hituav-a-highaltitude-infrared-thermal-dataset\hit-uav\labels\test_new"  # Folder containing the annotation files
train_images_folder = r"E:\datasets\Odesa\train\images"
val_images_folder = r"E:\datasets\Odesa\test\images"
train_labels_folder = r"E:\datasets\Odesa\train\labels"
val_labels_folder = r"E:\datasets\Odesa\test\labels"

# Create the destination directories if they don't exist
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

# Get a list of all image files in the source folder
image_files = [f for f in os.listdir(source_images_folder) if f.endswith('.jpg') or f.endswith('.png')]

# Shuffle the list of image files
random.shuffle(image_files)

# Define the split ratio
split_ratio = 0.80
split_index = int(len(image_files) * split_ratio)

# Split the image files into training and validation sets
train_files = image_files[:split_index]
val_files = image_files[split_index:]


# Function to copy files
def copy_files(files, src_image_folder, src_label_folder, dest_image_folder, dest_label_folder):
    for file in files:
        image_path = os.path.join(src_image_folder, file)
        label_path = os.path.join(src_label_folder, file.replace('.jpg', '.txt').replace('.png', '.txt'))

        if os.path.exists(image_path) and os.path.exists(label_path):
            shutil.copy(image_path, os.path.join(dest_image_folder, file))
            shutil.copy(label_path, os.path.join(dest_label_folder, os.path.basename(label_path)))

# Copy training files
copy_files(train_files, source_images_folder, source_labels_folder, train_images_folder, train_labels_folder)

# Copy validation files
copy_files(val_files, source_images_folder, source_labels_folder, val_images_folder, val_labels_folder)

print("Files have been copied successfully.")
