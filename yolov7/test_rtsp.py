import cv2
import time

# RTSP stream URL (replace with your stream URL)
rtsp_url = 0

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the stream is opened successfully
if not cap.isOpened():
    print("Error: Cannot open the RTSP stream")
    exit()

# Initialize variables for FPS calculation
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) != 0 else 25  # Default to 25 if FPS is not available
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        current_fps = frame_count / elapsed_time
    else:
        current_fps = fps

    # Get frame size
    height, width, _ = frame.shape
    frame_info = f"FPS: {current_fps:.2f} | Size: {width}x{height}"

    # Display FPS and size on the video
    cv2.putText(frame, frame_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the video
    cv2.imshow("RTSP Stream", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()
