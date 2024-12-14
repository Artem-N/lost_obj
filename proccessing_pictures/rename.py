import os


def rename_files(folder_path, new_name_pattern):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter out image and text files
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    text_extensions = ['.txt']

    image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    text_files = [f for f in files if os.path.splitext(f)[1].lower() in text_extensions]

    # Make sure there are equal numbers of image and text files
    if len(image_files) != len(text_files):
        print("The number of image files and text files do not match!")
        return

    # Rename each file
    for index, (image_file, text_file) in enumerate(zip(image_files, text_files)):
        # Construct the new file name
        new_image_name = f"{new_name_pattern}{index + 1}{os.path.splitext(image_file)[1]}"
        new_text_name = f"{new_name_pattern}{index + 1}{os.path.splitext(text_file)[1]}"

        # Get full file paths
        src_image = os.path.join(folder_path, image_file)
        dst_image = os.path.join(folder_path, new_image_name)
        src_text = os.path.join(folder_path, text_file)
        dst_text = os.path.join(folder_path, new_text_name)

        # Rename the files
        os.rename(src_image, dst_image)
        os.rename(src_text, dst_text)
        print(f"Renamed: {src_image} -> {dst_image}")
        print(f"Renamed: {src_text} -> {dst_text}")


# Example usage
folder_path = "D:\\train shuliavka\\labels_pic"
new_name_pattern = "test"
rename_files(folder_path, new_name_pattern)
