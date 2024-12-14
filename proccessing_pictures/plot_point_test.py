import os
import matplotlib.pyplot as plt


def plot_keypoints_on_image(img_path, txt_path, img_width=1280, img_height=720):
    # Read the image
    img = plt.imread(img_path)

    # Read the keypoints from the .txt file
    with open(txt_path, 'r') as f:
        line = f.readline().strip().split()

    # Extract the keypoints (skip the first 5 elements which are class index, center_x, center_y, width, height)
    keypoints = list(map(float, line[5:]))

    # Separate the keypoints into x and y coordinates
    x_coords = keypoints[0::2]
    y_coords = keypoints[1::2]

    # Convert normalized coordinates back to image coordinates
    x_coords = [x * img_width for x in x_coords]
    y_coords = [y * img_height for y in y_coords]

    # Plot the image
    plt.imshow(img)
    plt.scatter(x_coords, y_coords, color='red', s=20)
    plt.title('Keypoints on Image')
    plt.show()


# Example usage
img_path = r"C:\Users\User\Desktop\tenis\tennis_court_det_dataset__\data\images\_22eJvtP_lc_50.png"  # Update this path to the image you want to check
txt_path = r"C:\Users\User\Desktop\tenis\tennis_court_det_dataset__\data\val\_22eJvtP_lc_50.txt"  # Update this path to the corresponding .txt file

plot_keypoints_on_image(img_path, txt_path)
