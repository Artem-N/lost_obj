import cv2
import os

def extract_frames(video_path, output_folder, frame_interval):
    # Create the output directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Save the frame as an image file
            frame_filename = os.path.join(output_folder, f"frame___{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {saved_count} frames and saved to {output_folder}.")

# Example usage
video_path = "C:\\Users\\User\\Documents\\00000003633000000.mp4"
output_folder = "C:\\Users\\User\\Desktop\\video_korobochka_"
frame_interval = 500

extract_frames(video_path, output_folder, frame_interval)
