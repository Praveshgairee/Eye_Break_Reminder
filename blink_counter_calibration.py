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

print("Blink counter test started.")
print("Press c to calibrate with eyes open.")
print("Press q to quit.")

# -----------------------------
# Blink detection settings
# -----------------------------
CALIBRATION_SECONDS = 3
BLINK_COOLDOWN_SECONDS = 0.25

# Personalized thresholds will be created after calibration
open_eye_baseline = None
close_threshold = None
reopen_threshold = None

blink_count = 0
eyes_currently_closed = False
last_blink_time = 0.0

# Calibration state
is_calibrating = False
calibration_start_time = None
calibration_samples = []


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
            # Calibration logic
            # -----------------------------
            if is_calibrating:
                calibration_samples.append(avg_ear)
                elapsed = time.time() - calibration_start_time

                if elapsed >= CALIBRATION_SECONDS:
                    open_eye_baseline = sum(calibration_samples) / len(calibration_samples)

                    # Personalized thresholds
                    # Close threshold: eye is considered closed below this
                    close_threshold = open_eye_baseline * 0.82

                    # Reopen threshold: eye must come back above this
                    reopen_threshold = open_eye_baseline * 0.88

                    is_calibrating = False
                    calibration_samples = []

                    print(f"\nCalibration done. Baseline: {open_eye_baseline:.3f}")
                    print(f"Close threshold: {close_threshold:.3f}")
                    print(f"Reopen threshold: {reopen_threshold:.3f}")

            # -----------------------------
            # Blink logic
            # -----------------------------
            if open_eye_baseline is not None:
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
            0.7,
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

        if is_calibrating:
            remaining = max(0, CALIBRATION_SECONDS - int(time.time() - calibration_start_time))
            cv2.putText(
                frame,
                f"Calibrating... keep eyes open naturally ({remaining}s)",
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )
        else:
            cv2.putText(
                frame,
                "Press c to calibrate",
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (180, 180, 180),
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

        cv2.imshow("Blink Counter Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            if avg_ear is not None:
                is_calibrating = True
                calibration_start_time = time.time()
                calibration_samples = []
                blink_count = 0
                eyes_currently_closed = False
                print("\nStarted calibration. Keep your eyes open naturally.")
        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
