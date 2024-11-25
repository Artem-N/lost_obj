import os

def get_file_names(folder):
    """Get a dictionary of file names without extensions mapped to their full paths."""
    files_dict = {}
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_name, _ = os.path.splitext(file)  # Extract file name without extension
            files_dict[file_name] = os.path.join(root, file)  # Store the full path for reference
    return files_dict

def compare_folders(folder1, folder2):
    # Get file names without extensions from both folders
    folder1_files = get_file_names(folder1)
    folder2_files = get_file_names(folder2)

    # Files in folder1 but not in folder2
    for file_name in folder1_files:
        if file_name not in folder2_files:
            print(f"File '{file_name}' is in {folder1} but not in {folder2}: {folder1_files[file_name]}")

    # Files in folder2 but not in folder1
    for file_name in folder2_files:
        if file_name not in folder1_files:
            print(f"File '{file_name}' is in {folder2} but not in {folder1}: {folder2_files[file_name]}")

# Example usage
folder1 = r"C:\Users\User\Desktop\temp\pic\day_mask"
folder2 = r"C:\Users\User\Desktop\temp\pic\day_mask_anotate"

compare_folders(folder1, folder2)
