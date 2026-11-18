

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from collections import deque
import subprocess, webbrowser, os, tempfile

CAREERS_URL = "https://i.pinimg.com/736x/2a/cf/0d/2acf0da8b459e64c95b3c5033836278c.jpg"
BROWSER = "chrome"  # Change this to your preferred browser
PROFILE_DIR = os.path.join(tempfile.gettempdir(), "lockin-profile")


recent = deque(maxlen=10)  # Store the last 10 pitch ratios
calib_values = []  # Store calibration values for pitch ratio
baseline = None  # Initialize baseline for pitch ratio
CALIB_SECONDS = 3
calib_start = time.time()  # Start time for calibration
first_time_face_detected = None  # before the first face is detected, this will be None
THRESHOLD = 0.04  # Threshold for pitch ratio change
HOLD_SECONDS = 4 # Time to hold the pitch ratio change before triggering an alert
COOL_DOWN_SECONDS = 12 # Cool down period after an alert is triggered
SHOW_COLOR = False  # Set to True to show the color of the detected points

state = "FOCUSED"
down_since = None  # Time when the pitch ratio change was first detected
cooldown_start = None  # Time when the cooldown period started

def fire_alert():
    print("LOCK IN ALERT: Please focus on the screen!")
    try:
        subprocess.Popen([
            "cmd", "/c", "start", BROWSER, 
            "--kiosk", "--new-window", "--no-first-run", CAREERS_URL,
            f"--user-data-dir={PROFILE_DIR}"
        ])
    except OSError:
        webbrowser.open(CAREERS_URL, new=2)  # Open in a new tab, if possible

def get_pitch_ratio(landmarks):
    nose_y = landmarks[1].y
    eyes_y = (landmarks[33].y + landmarks[263].y) / 2  # Average y of left and right eyes
    chin_y = landmarks[152].y
    forehead_y = landmarks[10].y

    # Calculate the vertical distances
    ratio = (nose_y - eyes_y) / (chin_y - forehead_y) if (chin_y - forehead_y) != 0 else 0
    return ratio


# Initialize the FaceLandmarker with the model and options
options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
landmarker = vision.FaceLandmarker.create_from_options(options)
webcam = cv2.VideoCapture(0)  # Open the default webcam

# Define the points of interest and their corresponding colors
points = {
    1: (0, 255, 0),  # Green. for nose
    152: (0, 0, 255),  # Red, for chin
    10: (255, 0, 0),  # Blue, for forehead
    33: (0, 255, 255),  # yellow, for left eye
}

try:
    while webcam.isOpened():
        ret, frame = webcam.read()
        if not ret:
            break

        # Convert the frame to RGB format for MediaPipe processing    
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms=timestamp_ms)

        # Draw the detected landmarks on the frame
        if result.face_landmarks:
            now = time.monotonic()
            if first_time_face_detected is None:
                first_time_face_detected = now  # Set the time when the first face is detected

            elapsed_time = time.monotonic() - first_time_face_detected
            remaining_time = max(0, CALIB_SECONDS - elapsed_time)

            landmarks = result.face_landmarks[0]
            h, w = frame.shape[:2]
            if SHOW_COLOR:
                for idx, color in points.items():
                    lm = landmarks[idx]
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (px, py), 5, color, -1)  # Draw a circle for each landmark
  
            pitch_ratio = get_pitch_ratio(landmarks)
            recent.append(pitch_ratio)  # Add the current pitch ratio to the deque
            smooth = sum(recent) / len(recent)  # Calculate the average of the last 10 pitch ratios

            # Display the smoothed pitch ratio on the frame
            cv2.putText(frame, f"Pitch Ratio: {pitch_ratio:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)        

            if baseline is None:
                calib_values.append(smooth)
                cv2.putText(frame, f"Calibrating... {remaining_time:.1f}s, sit nice and still!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                if elapsed_time > CALIB_SECONDS:
                    baseline = sum(calib_values) / len(calib_values)  # Set the baseline after calibration
            else:
                delta = smooth - baseline
                looking_down = delta > THRESHOLD

                if state == "FOCUSED":
                    if looking_down:
                        down_since = now
                        state = "POSSIBLE_ALERT"

                elif state == "POSSIBLE_ALERT":
                    if not looking_down:
                        down_since = None
                        state = "FOCUSED"
                    elif (now - down_since) > HOLD_SECONDS:
                        fire_alert()
                        state = "COOLDOWN"
                        cooldown_start = now

                elif state == "COOLDOWN":
                    if (now - cooldown_start) > COOL_DOWN_SECONDS:
                        state = "FOCUSED"

                cv2.putText(frame, f"Delta: {delta:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"State: {state}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    
        else:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if state == "POSSIBLE_ALERT":
                down_since = None
                state = "FOCUSED"
        
        cv2.imshow("Webcam Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            baseline = None
            calib_values.clear()
            first_time_face_detected = None
            recent.clear()
            down_since = None
            state = "FOCUSED"
            print("Calibration reset. Please sit still for 3 seconds to recalibrate.")

finally:
    webcam.release()
    cv2.destroyAllWindows()
    landmarker.close()  # Close the landmarker to release resources