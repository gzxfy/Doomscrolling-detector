import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from collections import deque

recent = deque(maxlen=10)  # Store the last 10 pitch ratios
calib_values = []  # Store calibration values for pitch ratio
baseline = None  # Initialize baseline for pitch ratio
CALIB_SECONDS = 3
calib_start = time.time()  # Start time for calibration
first_time_face_detected = None  # before the first face is detected, this will be None

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
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms=timestamp_ms)

        # Draw the detected landmarks on the frame
        if result.face_landmarks:
            now = time.monotonic()
            if first_time_face_detected is None:
                first_time_face_detected = now  # Set the time when the first face is detected

            elapsed_time = time.time() - first_time_face_detected
            remaining_time = max(0, CALIB_SECONDS - elapsed_time)

            landmarks = result.face_landmarks[0]
            h, w = frame.shape[:2]
            for idx, color in points.items():
                lm = landmarks[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 5, color, -1)  # Draw a circle for each landmark
  
            pitch_ratio = get_pitch_ratio(landmarks)
            recent.append(pitch_ratio)  # Add the current pitch ratio to the deque
            smooth = sum(recent) / len(recent)  # Calculate the average of the last 10 pitch ratios

       

            if baseline is None:
                calib_values.append(smooth)
                cv2.putText(frame, f"Calibrating, sit nice and still!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if time.time() - calib_start > CALIB_SECONDS:
                    baseline = sum(calib_values) / len(calib_values)  # Set the baseline after calibration
            else:
                delta = smooth - baseline
                cv2.putText(frame, f"Delta: {delta:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Pitch Ratio: {pitch_ratio:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)        
        
        cv2.imshow("Webcam Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    webcam.release()
    cv2.destroyAllWindows()
    landmarker.close()  # Close the landmarker to release resources