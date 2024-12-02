import os
import random
import cv2
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar


def rotate_box(corners, angle, cx, cy, h, w):
    """Rotate the bounding box."""
    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to account for translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    # Perform the rotation on the corners
    calculated = np.dot(M, corners.T).T
    calculated = calculated.reshape(-1, 8)

    return calculated


# Define paths
images_folder = r"E:\video_for_test\fly\labels\images_noise"
annotations_folder = r"E:\video_for_test\fly\labels\labels_noise"

rotated_images_folder = r"E:\video_for_test\fly\labels\images_noise_rotate"
rotated_annotations_folder = r"E:\video_for_test\fly\labels\labels_noise_rotate"

# Create output directories if they don't exist
os.makedirs(rotated_images_folder, exist_ok=True)
os.makedirs(rotated_annotations_folder, exist_ok=True)

# Get list of image files
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Initialize tqdm progress bar
with tqdm(total=len(image_files), desc="Processing images") as pbar:
    for img_file in image_files:
        # Read the image
        img_path = os.path.join(images_folder, img_file)
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2  # Center of the image

        # Randomly select rotation angle
        rotation = random.choice([90, 180])
        angle = -rotation  # Negative for clockwise rotation in OpenCV

        # Rotate the image
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to account for translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy

        rotated_image = cv2.warpAffine(image, M, (nW, nH))

        # Save the rotated image
        rotated_img_path = os.path.join(rotated_images_folder, img_file)
        cv2.imwrite(rotated_img_path, rotated_image)

        # Process the corresponding annotation file
        base_name = os.path.splitext(img_file)[0]
        ann_file = base_name + '.txt'
        ann_path = os.path.join(annotations_folder, ann_file)

        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                class_id = parts[0]
                x_norm = float(parts[1])
                y_norm = float(parts[2])
                w_norm = float(parts[3])
                h_norm = float(parts[4])

                # Convert normalized coordinates to pixel coordinates
                x = x_norm * w
                y = y_norm * h
                bw = w_norm * w
                bh = h_norm * h

                # Get the coordinates of the bounding box corners
                x_min = x - bw / 2
                y_min = y - bh / 2
                x_max = x + bw / 2
                y_max = y + bh / 2

                # Corners of the bounding box
                corners = np.array([[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]], dtype=np.float32)

                # Rotate the bounding box
                rotated_corners = rotate_box(corners, angle, cx, cy, h, w)

                # Get the new bounding box coordinates
                x_coords = rotated_corners[0][0::2]
                y_coords = rotated_corners[0][1::2]
                x_min_new = min(x_coords)
                y_min_new = min(y_coords)
                x_max_new = max(x_coords)
                y_max_new = max(y_coords)

                # Calculate new center, width, and height
                x_new = (x_min_new + x_max_new) / 2
                y_new = (y_min_new + y_max_new) / 2
                bw_new = x_max_new - x_min_new
                bh_new = y_max_new - y_min_new

                # Normalize the coordinates
                x_new_norm = x_new / nW
                y_new_norm = y_new / nH
                bw_new_norm = bw_new / nW
                bh_new_norm = bh_new / nH

                # Ensure new coordinates are within [0, 1]
                x_new_norm = min(max(x_new_norm, 0.0), 1.0)
                y_new_norm = min(max(y_new_norm, 0.0), 1.0)
                bw_new_norm = min(max(bw_new_norm, 0.0), 1.0)
                bh_new_norm = min(max(bh_new_norm, 0.0), 1.0)

                new_line = f"{class_id} {x_new_norm:.6f} {y_new_norm:.6f} {bw_new_norm:.6f} {bh_new_norm:.6f}\n"
                new_lines.append(new_line)

            # Save the updated annotation file
            rotated_ann_path = os.path.join(rotated_annotations_folder, ann_file)
            with open(rotated_ann_path, 'w') as f:
                f.writelines(new_lines)
        else:
            print(f"Annotation file not found for image {img_file}.")

        # Update tqdm progress bar
        pbar.update(1)

print("Rotation and annotation update completed successfully.")
