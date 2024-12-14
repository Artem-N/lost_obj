import cv2
import os

# Step 1: Define the paths for input and output folders
input_folder = r"C:\Users\User\Desktop\temp\mask\shahed_night"
output_folder = r"C:\Users\User\Desktop\temp\mask\ready_mask_night_shahed"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Step 2: Loop through all the images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Filter image files
        image_path = os.path.join(input_folder, filename)

        # Load the image
        image = cv2.imread(image_path)

        # Manually select the object using OpenCV's selectROI
        roi = cv2.selectROI('Select Object', image, fromCenter=False, showCrosshair=True)

        # Extract ROI coordinates and size
        x, y, w, h = roi

        # Step 3: Crop the image to the selected ROI (the selected area)
        cropped_image = image[y:y + h, x:x + w]

        # Step 4: Save the cropped image in the output folder
        output_path = os.path.join(output_folder, f"cropped_{filename}")
        cv2.imwrite(output_path, cropped_image)

        # Display the result (optional)
        cv2.imshow('Cropped Image', cropped_image)
        cv2.waitKey(0)

# Cleanup
cv2.destroyAllWindows()
