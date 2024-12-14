import os
from collections import Counter


def read_classes_from_files(directory):
    class_counts = Counter()

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as f:
                for line in f:
                    class_id = line.split()[0]  # Read the first element (class)
                    class_counts[class_id] += 1

    return class_counts


# Example usage
directory = r"D:\pycharm_projects\yolov7\yolov7\test\labels"
class_counts = read_classes_from_files(directory)
print(class_counts)


def compute_class_weights(class_counts):
    total_count = sum(class_counts.values())
    class_weights = {class_id: total_count / count for class_id, count in class_counts.items()}
    return class_weights

# Example usage
class_weights = compute_class_weights(class_counts)
print(class_weights)

# def create_data_file(data_file_path, class_weights, classes):
#     with open(data_file_path, 'w') as f:
#         f.write(f"train = data/train.txt\n")
#         f.write(f"val = data/val.txt\n")
#         f.write(f"names = data/obj.names\n")
#         f.write(f"nc = {len(classes)}\n")
#         weights = [class_weights.get(str(i), 1.0) for i in range(len(classes))]
#         f.write(f"cls_weights = {weights}\n")
#
# # Example usage
# classes = ["person", "car", "truck"]  # Add all your class names here
# data_file_path = r"D:\pycharm_projects\yolov7\yolov7\test_200kpic_origin\labels\obj.data"
# create_data_file(data_file_path, class_weights, classes)
