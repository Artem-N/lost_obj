import os
from PIL import Image


def resize_image(input_path, output_path, new_width, new_height):
    # Open the image
    with Image.open(input_path) as img:
        # Resize the image
        resized_img = img.resize((new_width, new_height))

        # Save the resized image
        resized_img.save(output_path)

    print(f"Image resized to {new_width}x{new_height} and saved as {output_path}")


def process_folder(input_folder, output_folder, new_width, new_height):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):  # Add more formats if necessary
                input_path = os.path.join(root, file)

                # Create the same directory structure in the output folder
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                output_path = os.path.join(output_dir, file)

                # Resize the image
                resize_image(input_path, output_path, new_width, new_height)


# Example usage
input_folder = r"C:\Users\User\Desktop\temp\mask\resized_masks_night_shahed"
output_folder = r"C:\Users\User\Desktop\temp\mask\resized_masks_night_shahed"
width, height = 45, 38

process_folder(input_folder, output_folder, width, height)
