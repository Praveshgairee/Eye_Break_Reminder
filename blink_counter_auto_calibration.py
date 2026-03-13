import os
import sys
import cv2
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Face Landmarker setup
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Auto-calibration blink counter started.")
print("Press q to quit.")

# -----------------------------
# Auto calibration settings
# -----------------------------
BLINK_COOLDOWN_SECONDS = 0.25

# How many recent EAR values to keep
BASELINE_BUFFER_SIZE = 120

# Minimum number of good samples before blink detection starts
MIN_BASELINE_SAMPLES = 30

# Baseline update speed after initialization
BASELINE_SMOOTHING = 0.05

# Threshold ratios from personal open-eye baseline
CLOSE_RATIO = 0.85
REOPEN_RATIO = 0.90

# -----------------------------
# Blink state
# -----------------------------
blink_count = 0
eyes_currently_closed = False
last_blink_time = 0.0

# -----------------------------
# Auto-calibration state
# -----------------------------
ear_buffer = []
open_eye_baseline = None
close_threshold = None
reopen_threshold = None
is_calibrated = False


def euclidean_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def eye_aspect_ratio(landmarks, eye_indices):
    """
    EAR = (vertical1 + vertical2) / (2 * horizontal)

    eye_indices order:
    [left_corner, top1, top2, right_corner, bottom1, bottom2]
    """
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    horizontal = euclidean_distance(p1, p4)
    vertical1 = euclidean_distance(p2, p5)
    vertical2 = euclidean_distance(p3, p6)

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def update_thresholds():
    global close_threshold, reopen_threshold
    close_threshold = open_eye_baseline * CLOSE_RATIO
    reopen_threshold = open_eye_baseline * REOPEN_RATIO


# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        status = "No face detected"
        instruction = "Looking for face..."
        avg_ear = None

        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]
            h, w, _ = frame.shape

            # Draw eye landmarks
            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(face_landmarks[idx].x * w)
                y = int(face_landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            status = "Face detected"

            # -----------------------------
            # Auto-calibration logic
            # -----------------------------
            # Keep only recent EAR values
            ear_buffer.append(avg_ear)
            if len(ear_buffer) > BASELINE_BUFFER_SIZE:
                ear_buffer.pop(0)

            # Initial calibration:
            # use the top 30% values from recent history as "likely open-eye" values
            if len(ear_buffer) >= MIN_BASELINE_SAMPLES:
                sorted_ears = sorted(ear_buffer, reverse=True)
                top_count = max(10, int(len(sorted_ears) * 0.3))
                likely_open_values = sorted_ears[:top_count]
                estimated_open = sum(likely_open_values) / len(likely_open_values)

                if open_eye_baseline is None:
                    open_eye_baseline = estimated_open
                    update_thresholds()
                    is_calibrated = True
                    print(f"\nInitial baseline set: {open_eye_baseline:.3f}")
                    print(f"Close threshold: {close_threshold:.3f}")
                    print(f"Reopen threshold: {reopen_threshold:.3f}")
                else:
                    # Slowly adapt baseline upward/downward to match normal open-eye state
                    open_eye_baseline = (
                        (1 - BASELINE_SMOOTHING) * open_eye_baseline
                        + BASELINE_SMOOTHING * estimated_open
                    )
                    update_thresholds()

            # -----------------------------
            # Blink logic
            # -----------------------------
            if is_calibrated and close_threshold is not None and reopen_threshold is not None:
                current_time = time.time()

                if avg_ear < close_threshold:
                    eyes_currently_closed = True
                elif avg_ear > reopen_threshold:
                    if eyes_currently_closed:
                        if current_time - last_blink_time > BLINK_COOLDOWN_SECONDS:
                            blink_count += 1
                            last_blink_time = current_time
                    eyes_currently_closed = False

        # -----------------------------
        # Draw information
        # -----------------------------
        cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        ear_text = f"{avg_ear:.3f}" if avg_ear is not None else "--"
        cv2.putText(
            frame,
            f"Eye openness: {ear_text}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 0),
            2
        )

        baseline_text = f"{open_eye_baseline:.3f}" if open_eye_baseline is not None else "--"
        cv2.putText(
            frame,
            f"Baseline: {baseline_text}",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        close_text = f"{close_threshold:.3f}" if close_threshold is not None else "--"
        reopen_text = f"{reopen_threshold:.3f}" if reopen_threshold is not None else "--"
        cv2.putText(
            frame,
            f"Close: {close_text}  Reopen: {reopen_text}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 255, 200),
            2
        )

        if is_calibrated:
            instruction = "Auto calibration active"
        else:
            instruction = "Auto calibrating... keep looking normally"

        cv2.putText(
            frame,
            instruction,
            (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            status,
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 128, 255),
            2
        )

        cv2.imshow("Blink Counter Auto Calibration", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
