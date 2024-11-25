from ultralytics import YOLO
from multiprocessing import Process
import cv2
import time


def worker(video_path, window_name):
    # Load the model once per process
    model = YOLO("runs/detect/drone_v7_200ep_32bath/weights/best.pt")
    cap = cv2.VideoCapture(video_path)
    prev_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Run inference on the frame
        results = model.predict(source=frame, save=False, conf=0.15, verbose=False)
        # Get the annotated frame
        result_frame = results[0].plot()

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Overlay FPS on the frame
        cv2.putText(result_frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame in the specified window
        cv2.imshow(window_name, result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyWindow(window_name)


def main():
    # List of video paths
    video_paths = [
        r"C:\Users\User\Desktop\fly\drive-download-20241007T130521Z-001\Video_3.mp4",
        r"C:\Users\User\Desktop\fly\drive-download-20241007T130521Z-001\Video_2.mp4",
        r"C:\Users\User\Desktop\fly\drive-download-20241007T130521Z-001\Video_1.mp4",
        r"C:\Users\User\Desktop\fly\2024.mp4",
        r"C:\Users\User\Desktop\fly\Velora 3D Printed Plane.mp4",
        r"C:\Users\User\Desktop\fly\High Speed FPV Drone .mp4",
        r"D:\video_test_sbu\6472912-hd_1920_1080_24fps.mp4",
        r"D:\video_test_sbu\2829264-hd_1080_1920_24fps.mp4",
        r"D:\video_test_sbu\3723537-uhd_4096_2160_24fps.mp4",
        r"D:\video_test_sbu\12026119_1080_1920_30fps.mp4"
    ]

    processes = []
    window_names = [f"YOLO Output {i + 1}" for i in range(len(video_paths))]

    # Start worker processes
    for i, video_path in enumerate(video_paths):
        p = Process(target=worker, args=(video_path, window_names[i]))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
