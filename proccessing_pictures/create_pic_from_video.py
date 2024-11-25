import cv2
import os
import random
import string

# Function to save frames from video
def save_frames_from_video(video_path, output_folder, frame_interval=10):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frame_count = 0
    existing_files = set(os.listdir(output_folder))  # To avoid overwriting existing files

    while True:
        # Read a frame from the video
        success, frame = video_capture.read()

        # Break the loop when no more frames are available
        if not success:
            break

        # Save the frame every 'frame_interval' frames
        if frame_count % frame_interval == 0:
            # Generate a unique random filename of 20 characters
            while True:
                random_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=20)) + '.jpg'
                if random_filename not in existing_files:
                    existing_files.add(random_filename)
                    break  # Exit the loop when a unique filename is generated

            output_filename = os.path.join(output_folder, random_filename)
            cv2.imwrite(output_filename, frame)
            print(f"Saved {output_filename}")
            saved_frame_count += 1

        frame_count += 1

    # Release the video capture object
    video_capture.release()
    print("All frames saved.")


# Example usage:
video_path = r"C:\Users\User\Desktop\fly\not_ready_video_air\_Орлан-30_ долітався в небі над Харківщиною завдяки воїнам 127-ї бригади ТрО.mp4"
output_folder = r"C:\Users\User\Desktop\fly\wing_train_picture"
save_frames_from_video(video_path, output_folder, frame_interval=5)
