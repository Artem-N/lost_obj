import os
import cv2


def draw_annotations(image, annotations, img_width, img_height):
    for annotation in annotations:
        parts = annotation.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * img_width
        y_center = float(parts[2]) * img_height
        width = float(parts[3]) * img_width
        height = float(parts[4]) * img_height
        keypoint_x = float(parts[5]) * img_width
        keypoint_y = float(parts[6]) * img_height
        visibility = int(parts[7])

        # Calculate the bounding box coordinates
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Draw the keypoint if visible
        if visibility == 2:
            cv2.circle(image, (int(keypoint_x), int(keypoint_y)), 5, (0, 255, 0), -1)

    return image


def process_images_and_annotations(images_folder, annotations_folder, output_folder, img_width, img_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_file)
        annotation_file = os.path.splitext(image_file)[0] + ".txt"
        annotation_path = os.path.join(annotations_folder, annotation_file)

        if os.path.exists(annotation_path):
            image = cv2.imread(image_path)
            with open(annotation_path, 'r') as f:
                annotations = f.readlines()

            annotated_image = draw_annotations(image, annotations, img_width, img_height)
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, annotated_image)


if __name__ == "__main__":
    images_folder = r"C:\Users\User\Desktop\tenis\Tennis Match.v3i.yolov8\test\images_standart"
    annotations_folder = r"C:\Users\User\Desktop\tenis\Tennis Match.v3i.yolov8\test\labels_new"
    output_folder = r"C:\Users\User\Desktop\tenis\Tennis Match.v3i.yolov8\test\labels_new_pic"
    img_width = 640  # Replace with your image width
    img_height = 640  # Replace with your image height

    process_images_and_annotations(images_folder, annotations_folder, output_folder, img_width, img_height)
