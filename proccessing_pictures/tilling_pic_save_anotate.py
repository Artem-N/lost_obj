import cv2
import os
import numpy as np


def load_annotations(annotation_file):
    with open(annotation_file, 'r') as file:
        annotations = file.readlines()
    return [list(map(float, line.strip().split())) for line in annotations]


def save_annotations(annotation_file, annotations):
    with open(annotation_file, 'w') as file:
        for annotation in annotations:
            file.write(' '.join(map(str, annotation)) + '\n')


def adjust_annotations(annotations, x_offset, y_offset, tile_width, tile_height, original_width, original_height):
    adjusted_annotations = []
    for annotation in annotations:
        cls, x, y, w, h = annotation

        # Convert YOLO coordinates to pixel coordinates
        x_center = x * original_width
        y_center = y * original_height
        box_width = w * original_width
        box_height = h * original_height

        # Calculate new coordinates relative to the tile
        new_x_center = x_center - x_offset
        new_y_center = y_center - y_offset

        # Check if the bounding box center is within the tile
        if 0 <= new_x_center <= tile_width and 0 <= new_y_center <= tile_height:
            # Adjust coordinates to be relative to the new tile size
            adjusted_x = new_x_center / tile_width
            adjusted_y = new_y_center / tile_height
            adjusted_w = box_width / tile_width
            adjusted_h = box_height / tile_height

            adjusted_annotations.append([cls, adjusted_x, adjusted_y, adjusted_w, adjusted_h])

    return adjusted_annotations


def tile_image(image, annotations, output_prefix, output_dir):
    height, width = image.shape[:2]
    tile_width, tile_height = width // 2, height // 2

    tiles = [
        (0, 0), (tile_width, 0), (0, tile_height), (tile_width, tile_height)
    ]

    for idx, (x_offset, y_offset) in enumerate(tiles):
        tile = image[y_offset:y_offset + tile_height, x_offset:x_offset + tile_width]
        adjusted_annotations = adjust_annotations(annotations, x_offset, y_offset, tile_width, tile_height, width, height)
        if adjusted_annotations:  # Only save tiles that have at least one annotation
            cv2.imwrite(os.path.join(output_dir, f'{output_prefix}_{idx}.jpg'), tile)
            save_annotations(os.path.join(output_dir, f'{output_prefix}_{idx}.txt'), adjusted_annotations)


def process_dataset(image_dir, annotation_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for image_filename in os.listdir(image_dir):
        if image_filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_dir, image_filename)
            annotation_path = os.path.join(annotation_dir, image_filename.replace('.jpg', '.txt'))
            if os.path.exists(annotation_path):
                image = cv2.imread(image_path)
                annotations = load_annotations(annotation_path)
                tile_image(image, annotations, os.path.splitext(image_filename)[0], output_dir)


# Usage example:
image_dir = "D:\\pycharm_projects\\yolov7\\yolov7\\train\\images"
annotation_dir = "D:\\pycharm_projects\\yolov7\\yolov7\\train\\labels"
output_dir = "D:\\pycharm_projects\\yolov7\\yolov7\\train\\images_tilling"

process_dataset(image_dir, annotation_dir, output_dir)

print('Finished')
