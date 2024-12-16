import cv2
import time
from threading import Lock

# Ensure this lock is created if used outside the main app
detector_lock = Lock()
detector = None  # Placeholder for the global detector instance

def gen_frames():
    """Generator function that yields video frames for the video_feed route."""
    global detector
    frame_count = 0  # Counter to limit debug prints
    while True:
        with detector_lock:
            if detector is None:
                # If no detector is initialized, wait and continue
                time.sleep(0.1)
                continue

            ret, frame = detector.cap.read()
            if not ret:
                print("No frame retrieved.")
                # Release the detector and set it to None
                detector.cap.release()
                detector = None
                # Yield an empty frame to inform the client
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
                continue

        # Retrieve necessary attributes outside the lock to minimize lock holding time
        video_width = detector.video_width
        video_height = detector.video_height
        baseline_gray = detector.baseline_gray

        try:
            # Resize the frame based on user settings
            frame = cv2.resize(frame, (video_width, video_height))
        except Exception as e:
            print(f"Error resizing frame: {e}")
            continue

        if baseline_gray is not None:
            try:
                # Process the frame for anomaly detection
                diff, thresh = detector.tracker.process_frame(frame, baseline_gray)
                # You can choose to overlay 'diff' and 'thresh' on the frame if desired
                # For simplicity, we'll just display the original frame with annotations
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame.")
            continue

        frame_bytes = buffer.tobytes()
        # Yield the frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Limit debug prints to every 50 frames to reduce I/O load
        frame_count += 1
        if frame_count % 50 == 0:
            print("50 frames processed.")

def gen_check_camera_frames():
    """Generator function that yields camera frames for the check_camera_feed route."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Adjust the index if needed

    if not cap.isOpened():
        # If the camera cannot be opened, yield a plain text message
        while True:
            try:
                yield (b'--frame\r\n'
                       b'Content-Type: text/plain\r\n\r\n'
                       b'Camera not accessible\r\n\r\n')
                time.sleep(1)  # Prevent tight loop
            except GeneratorExit:
                # Handle generator close (client disconnect)
                break
    else:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()
                try:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except GeneratorExit:
                    # Handle generator close (client disconnect)
                    break
        finally:
            cap.release()
