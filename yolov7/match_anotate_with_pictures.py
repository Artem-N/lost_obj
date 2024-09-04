import os
import shutil

# Define the paths to the folders
images_folder = r"C:\Users\User\Desktop\temp\images"
annotations_folder = r"C:\Users\User\Desktop\temp\labels_new"
output_folder = r"C:\Users\User\Desktop\temp\labels_new_"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get the list of image and annotation files
image_files = os.listdir(images_folder)
annotation_files = os.listdir(annotations_folder)

# Create a set of image base names without extensions
image_basenames = {os.path.splitext(image_file)[0] for image_file in image_files}

# Iterate over the annotation files
for annotation_file in annotation_files:
    # Extract the base name without extension
    annotation_base = os.path.splitext(annotation_file)[0]

    # Check if there's a corresponding image file
    if annotation_base in image_basenames:
        # Copy the annotation file to the output folder
        shutil.copy(os.path.join(annotations_folder, annotation_file), os.path.join(output_folder, annotation_file))

print(f'Annotation files for matching images have been copied to {output_folder}')
