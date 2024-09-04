import json
import os

# Load the JSON file
with open(r"D:\dataset\PeopleOnGrass\instances_train.json") as f:
    data = json.load(f)

# Create a dictionary for image info
image_info = {img['id']: img for img in data['images']}

# Create output directory if it doesn't exist
output_dir = r'D:\dataset\PeopleOnGrass_yolo_annotations_train'
os.makedirs(output_dir, exist_ok=True)


# Function to convert bbox to YOLO format
def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2] / 2.0) * dw
    y = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return (x, y, w, h)


# Process annotations and write to YOLO format
for ann in data['annotations']:
    image_id = ann['image_id']
    bbox = ann['bbox']
    category_id = ann['category_id']

    # Get image size
    img_info = image_info[image_id]
    image_width = img_info['width']
    image_height = img_info['height']

    # Convert bbox
    yolo_bbox = convert_bbox((image_width, image_height), bbox)

    # Prepare the annotation line
    yolo_line = f"{category_id} " + " ".join(map(str, yolo_bbox)) + "\n"

    # Write to corresponding txt file
    output_file = os.path.join(output_dir, f"{image_id}.txt")
    with open(output_file, 'a') as f:
        f.write(yolo_line)
