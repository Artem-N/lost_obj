import cv2
import os

video_path = r"C:\Users\User\Desktop\fly\GENERIC_RTSP-realmonitor_2023_09_20_15_57_55 - Trim.avi"
output_dir = r"C:\Users\User\Desktop\fly\test_val"
os.makedirs(output_dir, exist_ok=True)

vidcap = cv2.VideoCapture(video_path)
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite(os.path.join(output_dir, f"frame_{count:05d}.jpg"), image)
    success, image = vidcap.read()
    count += 1
