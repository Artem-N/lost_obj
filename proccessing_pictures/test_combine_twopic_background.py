import cv2
import numpy as np

def overlay_images(background_image, overlay_image, position, alpha_foreground=0.5, alpha_background=0.8):
    # Get the dimensions of the overlay image
    overlay_height, overlay_width, _ = overlay_image.shape

    # Extract the region of interest (ROI) from the background image
    y1, y2 = position[1], position[1] + overlay_height
    x1, x2 = position[0], position[0] + overlay_width
    roi = background_image[y1:y2, x1:x2]

    # Resize overlay image to match ROI size
    overlay_image_resized = cv2.resize(overlay_image, (roi.shape[1], roi.shape[0]))

    # Perform alpha blending for foreground (overlay)
    blended_roi_foreground = cv2.addWeighted(roi, 1 - alpha_foreground, overlay_image_resized, alpha_foreground, 0)

    # Perform alpha blending for background
    blended_roi_background = cv2.addWeighted(blended_roi_foreground, 1 - alpha_background, background_image[y1:y2, x1:x2], alpha_background, 0)

    # Replace the region of interest (ROI) in the background image with the blended ROI
    background_image[y1:y2, x1:x2] = blended_roi_background

    return background_image

# Load the background and overlay images
background_image = cv2.imread("C:\\Users\\User\\Pictures\\Screenshots\\q.png")
overlay_image = cv2.imread("D:\\pycharm_projects\\yolov7\\yolov7\\test_16kpic\\images\\58_peoplepegia_2650.jpg")
print("Background Image Shape:", background_image.shape)
print("Overlay Image Shape:", overlay_image.shape)

# Specify the position where the overlay image should be placed on the background image
position = (0, 0)

# Overlay the images with increased transparency for foreground and background
result_image = overlay_images(background_image, overlay_image, position, alpha_foreground=1, alpha_background=0.8)

# Display the result
cv2.imshow("Result", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
