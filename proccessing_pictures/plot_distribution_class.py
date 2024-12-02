import os
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm  # Import tqdm for progress bar


def read_classes_from_files(directory):
    class_counts = Counter()
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    # Use tqdm to display progress
    with tqdm(total=len(txt_files), desc="Processing files") as pbar:
        for filename in txt_files:
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as f:
                for line in f:
                    class_id = line.split()[0]  # Read the first element (class)
                    class_counts[class_id] += 1

            # Update the progress bar
            pbar.update(1)

    return class_counts


def plot_class_distribution(class_counts):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Class ID')
    plt.ylabel('Count')
    plt.title('Class Distribution in YOLO Annotations')
    plt.show()


# Example usage
directory = r"E:\video_for_test\fly\labels\labels_noise"
class_counts = read_classes_from_files(directory)
plot_class_distribution(class_counts)
