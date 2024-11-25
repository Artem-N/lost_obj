import os
import pandas as pd
from PIL import Image


def convert_to_yolo(row, img_width, img_height, class_id=0):
    """
    Converts a single CSV annotation row to YOLO format.

    Parameters:
    - row: pandas Series containing ['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
    - img_width: Width of the image
    - img_height: Height of the image
    - class_id: Integer class ID to assign

    Returns:
    - str: YOLO formatted annotation line
    """
    xmin = float(row['xmin'])
    ymin = float(row['ymin'])
    xmax = float(row['xmax'])
    ymax = float(row['ymax'])

    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height
    bbox_width = (xmax - xmin) / img_width
    bbox_height = (ymax - ymin) / img_height

    # Debugging: Print original and converted coordinates
    print(f"Image: {row['image']}")
    print(f"Original BBox: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
    print(
        f"Converted YOLO: x_center={x_center:.6f}, y_center={y_center:.6f}, width={bbox_width:.6f}, height={bbox_height:.6f}")

    # Ensure the values are within [0,1]
    x_center = min(max(x_center, 0), 1)
    y_center = min(max(y_center, 0), 1)
    bbox_width = min(max(bbox_width, 0), 1)
    bbox_height = min(max(bbox_height, 0), 1)

    return f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"


def process_annotations(csv_path, images_dir, output_dir, class_id=0):
    """
    Processes the CSV annotations and converts them to YOLO format.

    Parameters:
    - csv_path: Path to the CSV annotation file
    - images_dir: Directory containing the images
    - output_dir: Directory to save YOLO formatted annotation files
    - class_id: Integer class ID to assign
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Validate required columns
    required_columns = {'image', 'xmin', 'ymin', 'xmax', 'ymax', 'label'}
    if not required_columns.issubset(df.columns):
        print(f"CSV file is missing required columns. Required columns are: {required_columns}")
        return

    # Iterate over each image
    grouped = df.groupby('image')
    for image_name, group in grouped:
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            print(f"Warning: Image file '{image_name}' not found in '{images_dir}'. Skipping.")
            continue

        # Open the image to get its size
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening image '{image_name}': {e}")
            continue

        print(f"\nProcessing Image: {image_name} ({img_width}x{img_height})")

        yolo_annotations = []
        for _, row in group.iterrows():
            yolo_line = convert_to_yolo(row, img_width, img_height, class_id)
            yolo_annotations.append(yolo_line)

        # Prepare YOLO annotation file path
        annotation_filename = os.path.splitext(image_name)[0] + '.txt'
        annotation_path = os.path.join(output_dir, annotation_filename)

        # Write YOLO annotations to the file
        try:
            with open(annotation_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            print(f"Saved YOLO annotations to '{annotation_filename}'")
        except Exception as e:
            print(f"Error writing to file '{annotation_filename}': {e}")

    print(f"\nConversion completed. YOLO annotations are saved in '{output_dir}'.")


# Example Usage
if __name__ == "__main__":
    # Paths (Update these paths according to your directory structure)
    csv_annotation_path = r"D:\dataset\archive\ntut_drone_test\ntut_drone_test\Drone_004\vott-csv-export\Drone_004-export.csv"  # Path to your CSV file
    images_directory = r"D:\dataset\archive\ntut_drone_test\ntut_drone_test\Drone_004\vott-csv-export"  # Path to your images folder
    yolo_output_directory = r"D:\dataset\archive\ntut_drone_test\ntut_drone_test\Drone_004\vott-csv-export_anotate"  # Path to save YOLO .txt files

    # Convert annotations
    process_annotations(csv_annotation_path, images_directory, yolo_output_directory, class_id=0)
