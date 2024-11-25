import os
import random
import string

# Define the paths to the folders
images_folder = r"E:\datasets\Odesa\train\images"
labels_folder = r"E:\datasets\Odesa\train\labels"


# Function to generate a random string of 20 characters
def generate_random_name(length=16):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for i in range(length))


# Get the list of image files and label files
image_files = os.listdir(images_folder)
label_files = os.listdir(labels_folder)

# Loop through the image files
for image_file in image_files:
    # Get the file name without extension
    file_name, image_extension = os.path.splitext(image_file)

    # Define the corresponding label file name
    label_file = f"{file_name}.txt"

    # Check if the corresponding label file exists
    if label_file in label_files:
        # Generate a new random name
        new_name = generate_random_name()

        # Define the new file names
        new_image_name = f"{new_name}{image_extension}"
        new_label_name = f"{new_name}.txt"

        # Get the full paths to the current and new files
        old_image_path = os.path.join(images_folder, image_file)
        new_image_path = os.path.join(images_folder, new_image_name)
        old_label_path = os.path.join(labels_folder, label_file)
        new_label_path = os.path.join(labels_folder, new_label_name)

        # Rename the image and label files
        os.rename(old_image_path, new_image_path)
        os.rename(old_label_path, new_label_path)

print("Renaming completed successfully.")
