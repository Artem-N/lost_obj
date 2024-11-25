import os


def convert_bbox_to_center_and_keypoint(class_id, x_center, y_center, width, height, img_width=640, img_height=640):
    # Normalize coordinates to the image size
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    # Use the center point as the keypoint
    keypoint_x = x_center
    keypoint_y = y_center

    # Keypoint visibility is set to 2 (visible)
    keypoint_visible = 2

    return f"{class_id} {x_center} {y_center} {width} {height} {keypoint_x} {keypoint_y} {keypoint_visible}"


def process_annotation_file(annotation_file, output_file, img_width=640, img_height=640):
    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f:
        for line in lines:
            parts = line.strip().split()
            class_id = parts[0]
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height

            new_annotation = convert_bbox_to_center_and_keypoint(class_id, x_center, y_center, width, height, img_width,
                                                                 img_height)
            f.write(new_annotation + '\n')


def process_annotations_folder(input_folder, output_folder, img_width=1280, img_height=720):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for annotation_file in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, annotation_file)
        output_file_path = os.path.join(output_folder, annotation_file)

        process_annotation_file(input_file_path, output_file_path, img_width, img_height)


if __name__ == "__main__":
    input_folder = r"C:\Users\User\Desktop\tenis\Tennis Match.v3i.yolov8\valid\labels"
    output_folder = r"C:\Users\User\Desktop\tenis\Tennis Match.v3i.yolov8\valid\labels_new"
    img_width = 640  # Image width
    img_height = 640  # Image height

    process_annotations_folder(input_folder, output_folder, img_width, img_height)
