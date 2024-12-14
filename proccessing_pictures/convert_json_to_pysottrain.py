import json

# Load the provided COCO annotations file
file_path = r"C:\Users\User\Desktop\fly\annotations\instances_default.json"
with open(file_path, 'r') as file:
    coco_data = json.load(file)

# Prepare groundtruth format for each annotated image
groundtruth_data = {}

# Create a dictionary for each image based on ID
image_dict = {image['id']: image['file_name'] for image in coco_data['images']}

# Extract annotations in required PySOT format (x1,y1,width,height) for each image frame
for annotation in coco_data['annotations']:
    image_id = annotation['image_id']
    x, y, width, height = annotation['bbox']  # COCO uses [x, y, width, height]

    # Append data in the format PySOT expects
    if image_id in image_dict:
        filename = image_dict[image_id]
        if filename not in groundtruth_data:
            groundtruth_data[filename] = []
        groundtruth_data[filename].append(f"{int(x)},{int(y)},{int(width)},{int(height)}")

# Convert groundtruth data into PySOT-compatible format
groundtruth_output = []
for frame, annotations in sorted(groundtruth_data.items()):
    groundtruth_output.append("\n".join(annotations))

# Save groundtruth.txt in the appropriate PySOT format
output_path = r"C:\Users\User\Desktop\fly\annotations\groundtruth.txt"
with open(output_path, 'w') as file:
    file.write("\n".join(groundtruth_output))

output_path
