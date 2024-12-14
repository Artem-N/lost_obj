import os
import shutil

# Define the paths to the folders
images_folder = r"D:\odessa\train\images"
annotations_folder = r"D:\odessa\train\labels"
output_folder = r"D:\odessa\train\images_n"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the set of image file extensions you want to consider
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# Get the list of image and annotation files
image_files = os.listdir(images_folder)
annotation_files = os.listdir(annotations_folder)

# Create a set of annotation base names without extensions
annotation_basenames = {os.path.splitext(annotation_file)[0] for annotation_file in annotation_files}

# Iterate over the image files
matched_images = 0
for image_file in image_files:
    # Get the file extension and convert it to lowercase for consistency
    _, ext = os.path.splitext(image_file)
    ext = ext.lower()

    # Skip files that are not images based on the extension
    if ext not in image_extensions:
        continue

    # Extract the base name without extension
    image_base = os.path.splitext(image_file)[0]

    # Check if there's a corresponding annotation file
    if image_base in annotation_basenames:
        # Copy the image file to the output folder
        shutil.copy(os.path.join(images_folder, image_file), os.path.join(output_folder, image_file))
        matched_images += 1

print(f'{matched_images} image files with matching annotations have been copied to {output_folder}')
