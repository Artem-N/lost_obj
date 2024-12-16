from flask import Flask, render_template, Response, redirect, url_for, request, session, flash
from flask_wtf import FlaskForm
from flask_wtf import CSRFProtect
from wtforms import IntegerField, BooleanField, SubmitField, HiddenField
from wtforms.validators import DataRequired, NumberRange
import cv2
from video_anomaly_detector import VideoAnomalyDetector  # Ensure this does not define conflicting routes
import threading
import time
import os
import atexit

app = Flask(__name__, static_folder='static')
app.secret_key = os.urandom(24)  # Replace with a secure key in production

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Global detector instance and a lock for thread safety
detector = None
detector_lock = threading.Lock()


# Define the SettingsForm
# Define the SettingsForm
class SettingsForm(FlaskForm):
    persistence_threshold = IntegerField(
        'Persistence Threshold',
        validators=[DataRequired(message="Persistence Threshold is required."), NumberRange(min=1, message="Must be at least 1.")]
    )
    min_size = IntegerField(
        'Minimum Object Size',
        validators=[DataRequired(message="Minimum Object Size is required."), NumberRange(min=1, message="Must be at least 1.")]
    )
    use_default_camera_params = BooleanField('Use Default Camera Parameters')
    video_width = IntegerField(
        'Video Width',
        validators=[NumberRange(min=1, message="Must be at least 1.")],
        default=640
    )
    video_height = IntegerField(
        'Video Height',
        validators=[NumberRange(min=1, message="Must be at least 1.")],
        default=480
    )
    redirect_to = HiddenField('Redirect To')  # Renamed Hidden Field
    submit = SubmitField('Save Settings')



class CheckCameraForm(FlaskForm):
    submit = SubmitField('Check the Camera')


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


@app.route('/', methods=['GET', 'POST'])
def welcome():
    form = CheckCameraForm()
    if form.validate_on_submit():
        # Attempt to access the camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Adjust the index if needed
        if not cap.isOpened():
            flash('Cannot access the camera. Please ensure it is connected and not in use.', 'error')
        else:
            cap.release()
            flash('Camera is accessible.', 'success')
            return redirect(url_for('check_camera_page'))
    return render_template('welcome.html', form=form)


@app.route('/check_camera_page')
def check_camera_page():
    """Render the check camera page which displays the live video feed."""
    return render_template('check_camera.html')


@app.route('/check_camera', methods=['POST'])
def check_camera():
    """Attempt to access the camera and redirect based on availability."""
    global detector
    with detector_lock:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Adjust index if needed
        if not cap.isOpened():
            flash('Cannot access the camera. Please ensure it is connected and not in use.', 'error')
            return redirect(url_for('welcome'))
        else:
            cap.release()
            flash('Camera is accessible.', 'success')
            return redirect(url_for('check_camera_page'))


@app.route('/check_camera_feed')
def check_camera_feed():
    """Provide the video feed as a multipart response for the check camera page."""
    return Response(gen_check_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    global detector
    form = SettingsForm()

    if form.validate_on_submit():
        redirect_to_page = form.redirect_to.data or 'welcome'  # Default to 'welcome' if 'redirect_to' is not provided
        flash(f"Redirecting to: {redirect_to_page}", 'info')  # Debugging Flash Message
        print(f"Redirecting to: {redirect_to_page}")  # Debugging Print Statement

        with detector_lock:
            # Release existing detector if any
            if detector is not None:
                detector.cap.release()
                detector = None
                print("Old detector released.")
                flash('Old detector released.', 'success')

            # Retrieve and validate form data
            persistence_threshold = form.persistence_threshold.data
            min_size = form.min_size.data
            use_default_camera = form.use_default_camera_params.data

            if use_default_camera:
                video_width = 640
                video_height = 480
            else:
                video_width = form.video_width.data
                video_height = form.video_height.data

            # Store parameters in the session
            session['persistence_threshold'] = persistence_threshold
            session['min_size'] = min_size
            session['video_width'] = video_width
            session['video_height'] = video_height

            # Initialize the detector with the chosen settings
            match_distance = 50  # Can be made configurable if desired
            try:
                detector = VideoAnomalyDetector(
                    video_source=0,
                    persistence_threshold=persistence_threshold,
                    match_distance=match_distance,
                    min_size=min_size
                )
                detector.video_width = video_width
                detector.video_height = video_height
                print("Detector initialized with new settings.")
                flash('Settings have been successfully saved.', 'success')
            except RuntimeError as e:
                print(f"Error initializing VideoAnomalyDetector: {e}")
                flash('Error initializing video detector. Please try again.', 'error')
                return redirect(url_for('settings'))

        # Redirect based on the 'redirect_to' parameter
        if redirect_to_page == 'video':
            return redirect(url_for('video_page'))
        else:
            return redirect(url_for('welcome'))
    else:
        if request.method == 'GET':
            # Retrieve 'redirect_to' from query parameters; default to 'welcome'
            redirect_to_page = request.args.get('redirect_to', 'welcome')
            form.redirect_to.data = redirect_to_page  # Set the hidden 'redirect_to' field

            # Pre-populate form with existing session data
            form.persistence_threshold.data = session.get('persistence_threshold', 100)
            form.min_size.data = session.get('min_size', 10)
            video_width = session.get('video_width', 640)
            video_height = session.get('video_height', 480)
            form.video_width.data = video_width
            form.video_height.data = video_height
            form.use_default_camera_params.data = (video_width == 640 and video_height == 480)

    # Render the settings form with the form object
    return render_template('settings.html', form=form)



@app.route('/video')
def video_page():
    """Render the video page where the live feed and controls are displayed."""
    global detector
    print("Entering /video route")  # Debug statement
    print(f"Detector is None? {detector is None}")  # Debug statement

    with detector_lock:
        if detector is None:
            # If detector is not initialized, attempt to initialize it with session parameters
            persistence_threshold = session.get('persistence_threshold', 100)
            min_size = session.get('min_size', 10)
            match_distance = 50  # This can also be made configurable if desired
            video_width = session.get('video_width', 640)
            video_height = session.get('video_height', 480)

            try:
                detector = VideoAnomalyDetector(
                    video_source=0,
                    persistence_threshold=persistence_threshold,
                    match_distance=match_distance,
                    min_size=min_size
                )
                detector.video_width = video_width
                detector.video_height = video_height
                print("Detector re-initialized.")
                flash('Video detector has been re-initialized.', 'success')
            except RuntimeError as e:
                print(f"Error re-initializing VideoAnomalyDetector: {e}")
                flash('Error initializing video detector. Please check your settings.', 'error')
                return redirect(url_for('settings'))

    # Render the video page with the current baseline status
    return render_template('video.html', baseline_set=(detector.baseline_gray is not None))


@app.route('/video_feed')
def video_feed():
    """Provide the video feed as a multipart response."""
    with detector_lock:
        if detector is None:
            print("Video feed requested but detector is None.")
            flash('Video detector is not initialized. Please configure settings first.', 'error')
            return "No video detector initialized.", 500
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_baseline')
def set_baseline():
    """Capture the current frame and set it as the baseline."""
    global detector
    with detector_lock:
        if detector is None:
            print("Set Baseline called but detector is None.")
            flash('Video detector is not initialized. Please configure settings first.', 'error')
            return redirect(url_for('video_page'))

        # Capture the current frame
        ret, frame = detector.cap.read()
        if ret:
            video_width = session.get('video_width', 640)
            video_height = session.get('video_height', 480)
            try:
                frame = cv2.resize(frame, (video_width, video_height))
            except Exception as e:
                print(f"Error resizing frame for baseline: {e}")
                flash('Error resizing frame for baseline.', 'error')
                return redirect(url_for('video_page'))
            detector.baseline_gray = detector.tracker.update_baseline(frame)
            print("Baseline set successfully.")

            # Additional debug: Verify baseline
            if detector.baseline_gray is not None:
                print(f"Baseline shape: {detector.baseline_gray.shape}")
                print(f"Baseline pixel value at (0,0): {detector.baseline_gray[0,0]}")
                flash('Baseline has been successfully set.', 'success')
            else:
                print("Failed to set the baseline.")
                flash('Failed to set the baseline.', 'error')
        else:
            print("Failed to capture frame for baseline.")
            flash('Failed to capture frame for baseline.', 'error')

    return redirect(url_for('video_page'))


@app.route('/reset_baseline')
def reset_baseline():
    """Reset the baseline and clear tracked objects."""
    global detector
    with detector_lock:
        if detector is None:
            print("Reset Baseline called but detector is None.")
            flash('Video detector is not initialized.', 'error')
            return redirect(url_for('video_page'))

        # Reset baseline and tracked anomalies
        detector.baseline_gray = None
        detector.tracker.persistent_objects.clear()
        detector.tracker.next_object_id = 1
        print("Baseline and persistent objects reset.")
        flash('Baseline and anomalies have been successfully reset.', 'success')

    return redirect(url_for('video_page'))


@app.route('/reset_anomalies')
def reset_anomalies():
    """Reset all detected anomalies and set the current frame as the new baseline."""
    global detector
    with detector_lock:
        if detector is None:
            print("Reset Anomalies called but detector is None.")
            flash('Video detector is not initialized.', 'error')
            return redirect(url_for('video_page'))

        # Clear all detected anomalies
        detector.tracker.persistent_objects.clear()
        detector.tracker.next_object_id = 1
        print("Anomalies reset while keeping the baseline.")
        flash('Anomalies have been successfully reset.', 'success')

        # Capture the current frame to set as the new baseline
        ret, frame = detector.cap.read()
        if ret:
            video_width = session.get('video_width', 640)
            video_height = session.get('video_height', 480)
            try:
                frame = cv2.resize(frame, (video_width, video_height))
            except Exception as e:
                print(f"Error resizing frame for baseline during anomaly reset: {e}")
                flash('Error resizing frame during anomaly reset.', 'error')
                return redirect(url_for('video_page'))
            detector.baseline_gray = detector.tracker.update_baseline(frame)
            print("New baseline set successfully during anomaly reset.")

            # Additional debug: Verify baseline
            if detector.baseline_gray is not None:
                print(f"Baseline shape: {detector.baseline_gray.shape}")
                print(f"Baseline pixel value at (0,0): {detector.baseline_gray[0, 0]}")
                flash('New baseline has been set after resetting anomalies.', 'success')
            else:
                print("Failed to set the new baseline during anomaly reset.")
                flash('Failed to set the new baseline during anomaly reset.', 'error')
        else:
            print("Failed to capture frame for baseline during anomaly reset.")
            flash('Failed to capture frame for baseline during anomaly reset.', 'error')

    return redirect(url_for('video_page'))


@app.route('/help')
def help_page():
    """Render the help and instructions page."""
    return render_template('help.html')


@atexit.register
def cleanup():
    """Release the camera resource when the app is shutting down."""
    global detector
    with detector_lock:
        if detector is not None:
            detector.cap.release()
            detector = None
            print("Detector released on shutdown.")
            # Note: Flash messages here won't be displayed as the app is shutting down


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
