import cv2
import numpy as np
import random
import os
import string

# Paths
base_images_folder = r"C:\Users\User\Desktop\temp\pic\new_fone_night"  # Folder with base images
mask_images_folder = r"C:\Users\User\Desktop\temp\mask\resized_masks_night_shahed" # Folder with mask images
output_folder = r"C:\Users\User\Desktop\temp\pic\night_mask"  # Folder to save images with applied mask
annotations_folder = r"C:\Users\User\Desktop\temp\pic\night_mask_anotate"  # Folder to save annotations

# Ensure output directories exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(annotations_folder, exist_ok=True)

# Function to generate a random name with 15 characters
def generate_random_name(length=18):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

# Process each mask in the mask images folder
for mask_name in os.listdir(mask_images_folder):
    mask_image_path = os.path.join(mask_images_folder, mask_name)

    # Check if the mask file exists and is accessible
    if not os.path.exists(mask_image_path):
        print(f"Error: Mask image file not found at {mask_image_path}")
        continue

    # Try to load the mask image
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
    if mask_image is None:
        print(f"Error: Unable to load mask image from {mask_image_path}")
        continue
    else:
        print(f"Mask image '{mask_name}' loaded successfully")

    # Get the size of the mask
    mask_height, mask_width = mask_image.shape[:2]

    # Process each image in the base images folder
    for image_name in os.listdir(base_images_folder):
        base_image_path = os.path.join(base_images_folder, image_name)

        # Load the base image
        base_image = cv2.imread(base_image_path)

        if base_image is None:
            print(f"Skipping {image_name}: Unable to load image.")
            continue

        # Get the size of the base image
        base_height, base_width = base_image.shape[:2]

        # Ensure the mask can fit inside the base image
        if mask_height > base_height or mask_width > base_width:
            print(f"Skipping {image_name}: Mask is too large to fit in the base image.")
            continue

        # Calculate the allowable region for placing the mask in the full image
        min_x = 0
        max_x = base_width - mask_width
        min_y = 0
        max_y = base_height - mask_height

        # Generate a random position within this region
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)

        # Apply the mask to the base image at the calculated position
        if mask_image.shape[2] == 4:  # Check if the mask has an alpha channel
            # Separate the color and alpha channels
            mask_color = mask_image[:, :, :3]
            mask_alpha = mask_image[:, :, 3] / 255.0

            # Reduce the opacity of the mask for better blending
            mask_alpha *= 0.5  # Adjust this value to control transparency (1.0 is full opacity)

            # Get the region of interest from the base image
            roi = base_image[y:y+mask_height, x:x+mask_width]

            # Blend the mask with the region of interest
            for c in range(0, 3):
                roi[:, :, c] = roi[:, :, c] * (1 - mask_alpha) + mask_color[:, :, c] * mask_alpha

            # Place the blended region back into the base image
            base_image[y:y+mask_height, x:x+mask_width] = roi
        else:
            # If no alpha channel, simply overlay the mask
            base_image[y:y+mask_height, x:x+mask_width] = mask_image

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
        yolo_annotation = f"0 {center_x} {center_y} {width} {height}"

        # Save the annotation with the same random name as the image but with .txt extension
        annotation_path = os.path.join(annotations_folder, f"{random_name}.txt")

        with open(annotation_path, 'w') as f:
            f.write(yolo_annotation)

        # print(f"Processed {image_name} with {mask_name}: Image and annotation saved with name {random_name}.")

print("Processing complete.")
