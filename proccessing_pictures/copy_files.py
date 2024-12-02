import os
import shutil
import random
from tqdm import tqdm  # Import tqdm for progress tracking

# Set the paths for the source folders and the destination folders
source_images_folder = r"E:\video_for_test\fly\labels\images"  # Folder containing the images
source_labels_folder = r"E:\video_for_test\fly\labels\labels"  # Folder containing the annotation files
train_images_folder = r"E:\datasets\dataset\train\images"
test_images_folder = r"E:\datasets\dataset\valid\images"
train_labels_folder = r"E:\datasets\dataset\train\labels"
test_labels_folder = r"E:\datasets\dataset\valid\labels"

# Create the destination directories if they don't exist
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(test_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(test_labels_folder, exist_ok=True)

# Get a list of all image files in the source folder
image_files = [f for f in os.listdir(source_images_folder) if f.endswith('.jpg') or f.endswith('.png')]

# Shuffle the list of image files
random.shuffle(image_files)

# Define the split ratio
split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)

# Split the image files into training and validation sets
train_files = image_files[:split_index]
val_files = image_files[split_index:]


# Function to copy files with progress tracking
def copy_files_with_progress(files, src_image_folder, src_label_folder, dest_image_folder, dest_label_folder):
    with tqdm(total=len(files), desc="Copying files", unit="file") as pbar:
        for file in files:
            image_path = os.path.join(src_image_folder, file)
            label_path = os.path.join(src_label_folder, file.replace('.jpg', '.txt').replace('.png', '.txt'))

            if os.path.exists(image_path) and os.path.exists(label_path):
                shutil.copy(image_path, os.path.join(dest_image_folder, file))
                shutil.copy(label_path, os.path.join(dest_label_folder, os.path.basename(label_path)))

            # Update progress bar
            pbar.update(1)


# Copy training files with progress
print("Copying training files...")
copy_files_with_progress(train_files, source_images_folder, source_labels_folder, train_images_folder, train_labels_folder)

# Copy validation files with progress
print("Copying validation files...")
copy_files_with_progress(val_files, source_images_folder, source_labels_folder, test_images_folder, test_labels_folder)

print("Files have been copied successfully.")
