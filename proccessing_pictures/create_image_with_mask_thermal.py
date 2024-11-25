import cv2
import numpy as np
import random
import os
import string

# Paths
base_images_folder = r"C:\Users\User\Desktop\temp\pic\day"  # Folder with base images
mask_image_path = r"C:\Users\User\Desktop\temp\mask\ready_mask_day"  # Path to the mask image
output_folder = r"C:\Users\User\Desktop\temp\pic\day_mask"  # Folder to save images with applied mask
annotations_folder = r"C:\Users\User\Desktop\temp\pic\day_mask_anotate"  # Folder to save annotations

# Ensure output directories exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(annotations_folder, exist_ok=True)

# Load the thermal mask image
mask_image = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)  # Load as is (could be grayscale or RGB)

# If the thermal mask is grayscale, add an alpha channel to allow transparency manipulation
if len(mask_image.shape) == 2:
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGRA)
elif mask_image.shape[2] == 3:
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2BGRA)

# Apply a blur to the mask to make it blend better
# mask_image = cv2.GaussianBlur(mask_image, (15, 15), 0)  # Adjust kernel size for more/less blur

# Get the size of the mask
mask_height, mask_width = mask_image.shape[:2]

# Set to track used positions
used_positions = set()

# Function to generate a random name with 15 characters
def generate_random_name(length=15):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

# Process each image in the base thermal images folder
for image_name in os.listdir(base_images_folder):
    base_image_path = os.path.join(base_images_folder, image_name)

    # Load the base thermal image
    base_image = cv2.imread(base_image_path, cv2.IMREAD_UNCHANGED)  # Load as is (could be grayscale or RGB)

    # If the base thermal image is grayscale, convert it to BGRA to match the mask format
    if len(base_image.shape) == 2:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGRA)
    elif base_image.shape[2] == 3:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2BGRA)

    # Get the size of the base image
    base_height, base_width = base_image.shape[:2]

    # Ensure the mask can fit in the lower half of the base image
    if mask_height > base_height // 2 or mask_width > base_width:
        print(f"Skipping {image_name}: Mask is too large to fit in the lower half.")
        continue

    # Calculate the allowable region in the lower half
    min_x = 0
    max_x = base_width - mask_width
    min_y = base_height // 2
    max_y = base_height - mask_height

    # Generate a unique random position within this region
    while True:
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)
        position = (x, y)

        if position not in used_positions:
            used_positions.add(position)
            break

    # Apply the mask to the base image at the calculated position
    mask_color = mask_image[:, :, :3]
    mask_alpha = mask_image[:, :, 3] / 255.0

    # Reduce the opacity of the mask for better blending
    mask_alpha *= 1  # Adjust this value to control transparency (1.0 is full opacity)

    # Get the region of interest from the base image
    roi = base_image[y:y+mask_height, x:x+mask_width]

    # Blend the mask with the region of interest
    for c in range(0, 3):
        roi[:, :, c] = roi[:, :, c] * (1 - mask_alpha) + mask_color[:, :, c] * mask_alpha

    # Place the blended region back into the base image
    base_image[y:y+mask_height, x:x+mask_width] = roi

    # Generate a random name for the output files
    random_name = generate_random_name()

    # Save the resulting image with the random name in the output folder
    output_image_path = os.path.join(output_folder, f"{random_name}.jpg")
    cv2.imwrite(output_image_path, base_image)

    # Calculate and save the coordinates in YOLO format
    center_x = (x + mask_width / 2) / base_width
    center_y = (y + mask_height / 2) / base_height
    width = mask_width / base_width
    height = mask_height / base_height

    # Prepare the annotation text in YOLO format
    yolo_annotation = f"1 {center_x} {center_y} {width} {height}"

    # Save the annotation with the same random name as the image but with .txt extension
    annotation_path = os.path.join(annotations_folder, f"{random_name}.txt")

    with open(annotation_path, 'w') as f:
        f.write(yolo_annotation)

    print(f"Processed {image_name}: Image and annotation saved with name {random_name}.")

print("Processing complete.")
