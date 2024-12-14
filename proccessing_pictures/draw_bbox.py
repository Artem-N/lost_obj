import cv2
import os
from tqdm import tqdm  # Import tqdm for the progress bar


def draw_bounding_boxes(image_path, annotation_path, output_path):
    """Draw bounding boxes on the image based on annotation data and save the result."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image {image_path}")
        return

    height, width, _ = image.shape

    # Read the annotation file
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    # Iterate over each line in the annotation file
    for line in lines:
        class_id, center_x, center_y, w, h = map(float, line.strip().split())

        # Convert normalized coordinates to pixel coordinates
        center_x *= width
        center_y *= height
        w *= width
        h *= height

        # Calculate the top-left and bottom-right coordinates
        x1 = int(center_x - w / 2)
        y1 = int(center_y - h / 2)
        x2 = int(center_x + w / 2)
        y2 = int(center_y + h / 2)

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(int(class_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Save the image with bounding boxes
    output_image_path = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)
    # print(f"Output saved to {output_image_path}")


# Example usage
image_folder = r"E:\video_for_test\fly\labels\images"
annotation_folder = r"E:\video_for_test\fly\labels\labels"
output_folder = r"C:\Users\User\Desktop\check_bbox_correct"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get a list of image files to process
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

# Iterate through each image file with a progress bar
for image_filename in tqdm(image_files, desc="Processing images", unit="image"):
    image_path = os.path.join(image_folder, image_filename)
    annotation_path = os.path.join(annotation_folder, os.path.splitext(image_filename)[0] + '.txt')

    if os.path.exists(annotation_path):
        draw_bounding_boxes(image_path, annotation_path, output_folder)
    else:
        print(f"Annotation file for {image_filename} not found.")
