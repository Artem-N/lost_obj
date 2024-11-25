import os
import shutil

# Define the paths
images_folder = r"C:\Users\User\Desktop\drone_manytrash\valid\images"  # Replace with the path to your images folder
annotations_folder = r"C:\Users\User\Desktop\drone_manytrash\valid\labels"  # Replace with the path to your annotations folder
output_images_folder = r"C:\Users\User\Desktop\drone_manytrash\valid\images_new"  # Replace with the path to your output images folder
output_annotations_folder = r"C:\Users\User\Desktop\drone_manytrash\valid\labels_new"  # Replace with the path to your output annotations folder

# Create the output directories if they don't exist
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_annotations_folder, exist_ok=True)

# Class IDs to check for
target_class_ids = {2, 3}


# Function to check if a file contains any of the target class IDs
def contains_target_class(annotation_file):
    with open(annotation_file, 'r') as file:
        for line in file:
            class_id = int(line.split()[0])
            if class_id in target_class_ids:
                return True
    return False


# Iterate over all annotation files
for annotation_filename in os.listdir(annotations_folder):
    annotation_file_path = os.path.join(annotations_folder, annotation_filename)

    if contains_target_class(annotation_file_path):
        # Get the corresponding image filename
        image_filename = annotation_filename.replace('.txt', '.jpg')  # Assuming images are .jpg files
        image_file_path = os.path.join(images_folder, image_filename)

        # Copy the image and annotation file to the respective output folders
        if os.path.exists(image_file_path):
            shutil.copy(image_file_path, output_images_folder)
        shutil.copy(annotation_file_path, output_annotations_folder)

print("Files copied successfully.")
