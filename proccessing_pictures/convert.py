import os
import pandas as pd
from PIL import Image

def convert_to_yolo_format(annotation_line, image_width, image_height):
    """
    Converts a single annotation line to YOLO format.

    Parameters:
    - annotation_line: str, e.g., "0 1498 703 1625 840"
    - image_width: int
    - image_height: int

    Returns:
    - str: YOLO formatted annotation
    """
    parts = annotation_line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid annotation format: {annotation_line}")

    class_id, xmin, ymin, xmax, ymax = map(float, parts)

    # Calculate center coordinates, width, and height
    x_center = (xmin + xmax) / 2.0 / image_width
    y_center = (ymin + ymax) / 2.0 / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height

    return f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def convert_annotations(input_file, output_file, image_file=None):
    """
    Converts annotation from <class_id> <xmin> <ymin> <xmax> <ymax> to YOLO format.

    Parameters:
    - input_file: str, path to the input annotation file
    - output_file: str, path to save the YOLO formatted annotations
    - image_file: str, path to the image file to get dimensions. If None, user must provide image dimensions.
    """
    if image_file:
        with Image.open(image_file) as img:
            image_width, image_height = img.size
    else:
        raise ValueError("Image file path must be provided to obtain image dimensions.")

    yolo_annotations = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() == "":
                continue  # Skip empty lines
            yolo_line = convert_to_yolo_format(line, image_width, image_height)
            yolo_annotations.append(yolo_line)

    with open(output_file, 'w') as f:
        f.write('\n'.join(yolo_annotations))

    print(f"Converted {len(yolo_annotations)} annotations to YOLO format and saved to {output_file}")

def batch_convert_annotations(annotations_folder, images_folder, output_folder):
    """
    Batch converts multiple annotation files to YOLO format.

    Parameters:
    - annotations_folder: str, path to folder containing annotation .txt files
    - images_folder: str, path to folder containing image files
    - output_folder: str, path to save YOLO formatted annotations
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    annotation_files = [f for f in os.listdir(annotations_folder) if f.endswith('.txt')]

    for ann_file in annotation_files:
        # Assuming annotation file has the same name as image file but different extension
        image_filename = os.path.splitext(ann_file)[0] + ".jpg"  # Change extension if needed
        image_path = os.path.join(images_folder, image_filename)

        if not os.path.exists(image_path):
            print(f"Image file {image_filename} not found. Skipping {ann_file}.")
            continue

        input_ann_path = os.path.join(annotations_folder, ann_file)
        output_ann_path = os.path.join(output_folder, ann_file)  # Keeping same filename

        convert_annotations(input_ann_path, output_ann_path, image_path)

# Example Usage:

# Single File Conversion
# convert_annotations(
#     input_file='path_to_input_annotations/annotations.txt',
#     output_file='path_to_output_annotations/annotations_yolo.txt',
#     image_file='path_to_image/image.jpg'
# )

# Batch Conversion
batch_convert_annotations(
    annotations_folder=r"C:\Users\User\Desktop\EVD4UAV\EVD4UAV\bb",
    images_folder=r"C:\Users\User\Desktop\EVD4UAV\EVD4UAV\images",
    output_folder=r"C:\Users\User\Desktop\EVD4UAV\EVD4UAV\bb_yolo"
)
