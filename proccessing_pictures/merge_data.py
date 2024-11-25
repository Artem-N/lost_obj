import os

def merge_files(folder1, folder2, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of files from both folders
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # Find common files
    common_files = files1.intersection(files2)

    # Merge data of common files
    for file_name in common_files:
        file_path1 = os.path.join(folder1, file_name)
        file_path2 = os.path.join(folder2, file_name)
        output_file_path = os.path.join(output_folder, file_name)

        with open(file_path1, 'r') as file1, open(file_path2, 'r') as file2, open(output_file_path, 'w') as outfile:
            data1 = file1.read().strip()
            data2 = file2.read().strip()
            outfile.write(data1 + '\n' + data2)

folder1 = "D:\\dataset\\Images _people\\Annotations_people\\Annotations\\Yolo\\Train_visdrone"
folder2 = "D:\\dataset\\Images _people\\Images\\labels_train"
output_folder = "D:\\dataset\\Images _people\\Images\\labels_train_correct_combine"

merge_files(folder1, folder2, output_folder)
