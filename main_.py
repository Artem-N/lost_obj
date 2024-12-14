from video_anomaly_detector import VideoAnomalyDetector

if __name__ == "__main__":
    detector = VideoAnomalyDetector(
        video_source=0,
        persistence_threshold=100,
        match_distance=50,
        min_size=10
    )
    detector.run()
