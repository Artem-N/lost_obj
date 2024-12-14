import cv2

# Dictionary of OpenCV object tracker constructors
OPENCV_OBJECT_TRACKERS = {
    'csrt': cv2.legacy.TrackerCSRT_create,
    'kcf': cv2.legacy.TrackerKCF_create,
    'mil': cv2.legacy.TrackerMIL_create,
    'tld': cv2.legacy.TrackerTLD_create,
    'medianflow': cv2.legacy.TrackerMedianFlow_create,
    'mosse': cv2.legacy.TrackerMOSSE_create
}

def main():
    # Initialize video capture
    video_path = r"D:\video_test_sbu\5192703-uhd_3840_2160_24fps.mp4"  # Set to 0 for webcam
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Could not open video")
        return

    # Read the first frame
    ok, frame = video.read()
    if not ok:
        print("Cannot read video file")
        return

    # Allow user to select the bounding box
    bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Frame")

    # Initialize tracker
    tracker_type = 'csrt'  # Choose tracker type: 'csrt', 'kcf', etc.
    tracker = OPENCV_OBJECT_TRACKERS[tracker_type]()
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Draw bounding box
        if ok:
            # Tracking success
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type.upper() + " Tracker", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    # Release resources
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
