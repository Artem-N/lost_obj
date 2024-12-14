import cv2
import os

def extract_frames(video_path, output_folder, prefix, frame_interval):
    """
    Extract every `frame_interval` frame from a video and save as images.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where extracted frames will be saved.
        prefix (str): Prefix for the output image file names.
        frame_interval (int): Extract every nth frame.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_count = 0
    saved_count = 0

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Save every nth frame
        if frame_count % frame_interval == 0:
            file_name = f"{prefix}_{saved_count + 1}.jpg"
            file_path = os.path.join(output_folder, file_name)
            cv2.imwrite(file_path, frame)
            saved_count += 1
            print(f"Saved frame {frame_count} as {file_name}")

        frame_count += 1

    video_capture.release()
    print(f"Done! Extracted {saved_count} frames to {output_folder}")

# Parameters
video_path = r"E:\video_for_test\fly\not_ready_video_air\_Холодноярці_ збили ворожий _Орлан_ на Донеччині.mp4"
output_folder = r"E:\video_for_test\fly\temp_wing_pic_to_train"  # Replace with the desired output folder
prefix = "bpla_16"
frame_interval = 4

# Run the function
extract_frames(video_path, output_folder, prefix, frame_interval)
