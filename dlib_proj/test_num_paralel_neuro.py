from ultralytics import YOLO
from multiprocessing import Process, Queue
import cv2
import time


def worker(queue, window_name):
    # Load the model once per process
    model = YOLO("yolo11m.pt")
    prev_time = time.time()
    while True:
        frame = queue.get()
        if frame is None:
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
    cv2.destroyWindow(window_name)


def main():
    num_workers = 8
    queues = [Queue(maxsize=20) for _ in range(num_workers)]
    processes = []
    window_names = [f"YOLO Output {i + 1}" for i in range(num_workers)]

    # Start worker processes
    for i in range(num_workers):
        p = Process(target=worker, args=(queues[i], window_names[i]))
        p.start()
        processes.append(p)

    vid = r"C:\Users\User\Desktop\fly\drive-download-20241007T130521Z-001\Video_2.mp4"
    cap = cv2.VideoCapture(vid)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Distribute the frame to all workers
            for q in queues:
                # Ensure each worker gets a separate copy of the frame
                q.put(frame.copy())
            # Optional: Break the loop if 'q' is pressed in the main window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        # Send termination signal to workers
        for q in queues:
            q.put(None)
        for p in processes:
            p.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
