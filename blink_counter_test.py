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
print("Press q to quit.")

# -----------------------------
# Blink detection settings
# -----------------------------
# EAR threshold:
# Lower than this means eye is likely closed.
EAR_THRESHOLD = 0.42

# How many consecutive processed frames must stay closed
# before we count it as a blink event.
CONSECUTIVE_CLOSED_FRAMES = 1
BLINK_COOLDOWN_SECONDS = 0.25

blink_count = 0
closed_frames = 0
eyes_currently_closed = False
last_blink_time = 0.0


def euclidean_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def eye_aspect_ratio(landmarks, eye_indices):
    """
    EAR = (vertical1 + vertical2) / (2 * horizontal)

    eye_indices order used here:
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


# Approximate eye landmark groups from MediaPipe Face Mesh / Landmarker topology.
# These work well as a practical starting point for blink testing.
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
        ear_text = "--"

        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]
            h, w, _ = frame.shape

            # Draw a few eye landmarks for visual debugging
            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(face_landmarks[idx].x * w)
                y = int(face_landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            ear_text = f"{avg_ear:.3f}"
            status = "Face landmarks detected"

            # Blink logic
            current_time = time.time()

            if avg_ear < EAR_THRESHOLD:
                closed_frames += 1
                if closed_frames >= CONSECUTIVE_CLOSED_FRAMES:
                    eyes_currently_closed = True
            else:
                if eyes_currently_closed:
                    if current_time - last_blink_time > BLINK_COOLDOWN_SECONDS:
                        blink_count += 1
                        last_blink_time = current_time
            
                closed_frames = 0
                eyes_currently_closed = False

        cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"EAR: {ear_text}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            status,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 128, 255),
            2
        )

        cv2.imshow("Blink Counter Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
