import cv2
from object_tracker import ObjectTracker

class VideoAnomalyDetector:
    def __init__(self, video_source=0, persistence_threshold=100, match_distance=50, min_size=10):
        self.video_source = video_source
        self.tracker = ObjectTracker(persistence_threshold, match_distance, min_size)
        # Try using DirectShow backend on Windows
        self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video source.")
        self.baseline_gray = None

    def run(self):
        """Run the anomaly detection until ESC key is pressed."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("No frame received from the camera.")
                break

            key = cv2.waitKey(30)

            if key == ord('q'):
                # Set the current frame as baseline
                self.baseline_gray = self.tracker.update_baseline(frame)

            if self.baseline_gray is not None:
                diff, thresh = self.tracker.process_frame(frame, self.baseline_gray)
                cv2.imshow("Diff", diff)
                cv2.imshow("Thresh", thresh)
            else:
                instruction = "Press 'q' to set this frame as baseline."
                cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv2.imshow("Frame", frame)

            if key == 27:  # ESC key to quit
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = VideoAnomalyDetector(
        video_source=0,
        persistence_threshold=100,
        match_distance=50,
        min_size=10
    )
    detector.run()
