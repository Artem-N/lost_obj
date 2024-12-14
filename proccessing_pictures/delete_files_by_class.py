import os

def delete_files_with_class_id(folder_path, target_class_id):
    """
    Delete .txt files in a folder that contain a specific YOLO class ID.

    Args:
        folder_path (str): Path to the folder containing .txt files.
        target_class_id (int): The YOLO class ID to search for and delete files if found.
    """
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    deleted_files = 0

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Process only .txt files
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            delete_file = False  # Flag to decide whether to delete the file

            # Open the file and process its content
            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        # Check if the line starts with the target class ID
                        if line.startswith(f"{target_class_id} "):
                            delete_file = True
                            break  # No need to check further lines
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")
                continue

            # Delete the file if the target class ID is found
            if delete_file:
                try:
                    os.remove(file_path)
                    deleted_files += 1
                    print(f"Deleted file: {file_name}")
                except Exception as e:
                    print(f"Error deleting file {file_name}: {e}")

    print(f"Completed. Deleted {deleted_files} files containing class ID {target_class_id}.")

# Parameters
folder_path = r"D:\odessa\train\labels"  # Replace with the path to your folder
target_class_id = 3  # Class ID to search for

# Run the function
delete_files_with_class_id(folder_path, target_class_id)
