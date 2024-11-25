import os
import shutil


def copy_images_with_annotations(image_dir, annotation_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List all annotation files in the annotation directory
    annotation_files = [f for f in os.listdir(annotation_dir) if f.lower().endswith('.txt')]

    # Process each annotation file
    for annotation_file in annotation_files:
        # Construct the base filename without the extension
        base_filename = os.path.splitext(annotation_file)[0]

        # Look for corresponding image files with common image extensions
        found = False
        for ext in ['.png', '.jpg', '.jpeg']:
            image_file = base_filename + ext
            image_path = os.path.join(image_dir, image_file)

            if os.path.exists(image_path):
                output_image_path = os.path.join(output_dir, image_file)
                shutil.copy(image_path, output_image_path)
                print(f"Copied image: {image_path} to {output_image_path}")
                found = True
                break

        if not found:
            print(f"No corresponding image found for annotation: {annotation_file}")


# Parameters
image_directory = r"C:\Users\User\Desktop\work_test_new_data"
annotation_directory = r"C:\Users\User\Desktop\labels"
output_directory = r"C:\Users\User\Desktop\labels_new"

# Call the function to copy images
copy_images_with_annotations(image_directory, annotation_directory, output_directory)

print("Image copying completed.")
