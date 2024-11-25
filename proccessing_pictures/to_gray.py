import os
import cv2
import shutil

# Define input and output directories
input_image_dir = r"D:\train_shuliavka\pictures_correct"
input_text_dir = r"D:\train_shuliavka\labels"
output_image_dir = r"D:\train_shuliavka\pictures_correct_gray"
output_text_dir = r"D:\train_shuliavka\labels_gray"

# Create output directories if they don't exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_text_dir, exist_ok=True)

# List and sort all image files in the input directory
image_files = [f for f in os.listdir(input_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()


# Take the first 3000 image files
image_files_to_process = image_files[200:800]

# Process each image file
for image_file in image_files_to_process:
    # Construct the full file paths for the input image and corresponding text file
    input_image_path = os.path.join(input_image_dir, image_file)

    # Check if the corresponding text file exists
    corresponding_text_file = image_file.rsplit('.', 1)[0] + '.txt'
    input_text_path = os.path.join(input_text_dir, corresponding_text_file)

    if os.path.exists(input_text_path):
        # Read the image
        image = cv2.imread(input_image_path)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create new filenames for the output image and text file
        base_filename = os.path.splitext(image_file)[0] + '_'
        output_image_path = os.path.join(output_image_dir, base_filename + '.jpg')
        output_text_path = os.path.join(output_text_dir, base_filename + '.txt')

        # Save the grayscale image
        cv2.imwrite(output_image_path, gray_image)

        # Copy the corresponding text file to the output directory with the new name
        shutil.copy(input_text_path, output_text_path)

print("Conversion and file copying completed.")
