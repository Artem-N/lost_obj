import os


def delete_files(image_dir, text_dir, start_index, end_index):
    # List all files in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    # List all files in the text directory
    text_files = [f for f in os.listdir(text_dir) if f.lower().endswith('.txt')]
    text_files.sort()

    # Define the range of files to delete
    files_to_delete = image_files[start_index:end_index]

    # Process each file in the range
    for image_file in files_to_delete:
        # Construct the full file paths for the image and corresponding text file
        image_path = os.path.join(image_dir, image_file)
        text_file = image_file.rsplit('.', 1)[0] + '.txt'
        text_path = os.path.join(text_dir, text_file)

        # Delete the image file
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted image: {image_path}")
        else:
            print(f"Image file not found: {image_path}")

        # Delete the corresponding text file
        if os.path.exists(text_path):
            os.remove(text_path)
            print(f"Deleted annotation: {text_path}")
        else:
            print(f"Annotation file not found: {text_path}")


# Parameters
image_directory = "D:\\pycharm_projects\\yolov7\\yolov7\\train\\images"
text_directory = "D:\\pycharm_projects\\yolov7\\yolov7\\train\\labels"
start_index = 4000  # starting index (inclusive)
end_index = 6000  # ending index (exclusive)

# Call the function to delete files
delete_files(image_directory, text_directory, start_index, end_index)

print("Deletion completed.")


import os

def remove_class_id_from_annotations(annotation_folder, target_class_id=4):
    # Iterate through each annotation file in the folder
    for annotation_filename in os.listdir(annotation_folder):
        if annotation_filename.endswith('.txt'):
            annotation_path = os.path.join(annotation_folder, annotation_filename)

            # Read the annotation file
            with open(annotation_path, 'r') as file:
                lines = file.readlines()

            # Filter out lines with the target class ID
            lines_to_keep = [line for line in lines if int(line.strip().split()[0]) != target_class_id]

            # If there are changes, overwrite the file with filtered lines
            if len(lines_to_keep) != len(lines):
                with open(annotation_path, 'w') as file:
                    file.writelines(lines_to_keep)
                print(f"Removed class ID {target_class_id} from {annotation_filename}")

# Example usage
annotation_folder = r"C:\Users\User\Desktop\dataset\train\labels"

remove_class_id_from_annotations(annotation_folder)
