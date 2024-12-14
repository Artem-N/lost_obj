import cv2
import torch

# Example tensor of keypoints
keypoints = torch.tensor([[[0.0000, 0.0000],
                           [0.0000, 0.0000],
                           [0.0000, 0.0000],
                           [0.0000, 0.0000],
                           [853.8406, 931.4277],
                           [821.7547, 944.7402],
                           [847.3018, 960.7969],
                           [832.5792, 963.1834],
                           [877.9713, 995.2988],
                           [875.1199, 973.6946],
                           [913.4692, 989.1684],
                           [798.1732, 1023.5155],
                           [813.7925, 1030.7797],
                           [811.5422, 1080.4456],
                           [822.8441, 1093.1005],
                           [772.0675, 1117.6449],
                           [780.1394, 1135.8016]]])

# Load your custom image
image_path = r"C:\Users\User\Desktop\tenis\image.png"
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to load image.")
else:
    # Plot points on the image
    for kp in keypoints[0]:
        x, y = kp
        if x > 0 and y > 0:  # Only plot points with valid coordinates
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw a green circle

    # Display the image with keypoints
    cv2.imshow('Image with Keypoints', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the image with keypoints if needed
    cv2.imwrite('output_image_with_keypoints.jpg', image)
