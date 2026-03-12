import cv2
import time
import tkinter as tk
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Settings
# -----------------------------
WORK_TIME_SECONDS = 20   # use 20 * 60 for real use
BREAK_TIME_SECONDS = 5   # use 20 for real use
FACE_TIMEOUT_SECONDS = 2  # allow short face-detection misses
PROCESS_EVERY_N_FRAMES = 3  # reduce CPU usage a bit

# -----------------------------
# MediaPipe face detector setup
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
FaceDetector = vision.FaceDetector
FaceDetectorOptions = vision.FaceDetectorOptions
VisionRunningMode = vision.RunningMode

MODEL_PATH = "models/blaze_face_short_range.tflite"

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    min_detection_confidence=0.5,
)

# -----------------------------
# Open camera
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Lower resolution to reduce CPU / battery usage
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# -----------------------------
# Tkinter root for popup
# -----------------------------
root = tk.Tk()
root.withdraw()  # hide main tkinter window


def show_break_overlay():
    overlay = tk.Toplevel()
    overlay.title("Eye Break Reminder")
    overlay.configure(bg="red")
    overlay.attributes("-topmost", True)

    screen_width = overlay.winfo_screenwidth()
    screen_height = overlay.winfo_screenheight()

    width = int(screen_width * 0.7)
    height = int(screen_height * 0.4)

    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    overlay.geometry(f"{width}x{height}+{x}+{y}")
    overlay.resizable(False, False)

    title_label = tk.Label(
        overlay,
        text="TIME TO REST YOUR EYES",
        font=("Arial", 28, "bold"),
        bg="red",
        fg="white"
    )
    title_label.pack(pady=30)

    message_label = tk.Label(
        overlay,
        text="Look away from the screen",
        font=("Arial", 20),
        bg="red",
        fg="white"
    )
    message_label.pack(pady=10)

    countdown_label = tk.Label(
        overlay,
        text="",
        font=("Arial", 40, "bold"),
        bg="red",
        fg="yellow"
    )
    countdown_label.pack(pady=20)

    overlay.update()

    for remaining in range(BREAK_TIME_SECONDS, 0, -1):
        countdown_label.config(text=f"{remaining} seconds")
        overlay.update()
        time.sleep(1)

    overlay.destroy()


# -----------------------------
# Timer state
# -----------------------------
face_present_start = None
last_face_seen_time = None
alert_shown = False
frame_count = 0
last_display_value = None

print("Eye break reminder started in background mode.")
print("Camera preview is hidden.")
print("Press Ctrl + C in Terminal to stop.")

with FaceDetector.create_from_options(options) as detector:
    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("\nError: Could not read frame.")
                break

            current_time = time.time()
            frame_count += 1
            detections = []

            # Run detection every few frames to save CPU
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = int(current_time * 1000)
                result = detector.detect_for_video(mp_image, timestamp_ms)
                detections = result.detections if result.detections else []

                if len(detections) > 0:
                    last_face_seen_time = current_time

                    if face_present_start is None:
                        face_present_start = current_time

            # Timing logic
            if last_face_seen_time is not None and (current_time - last_face_seen_time) < FACE_TIMEOUT_SECONDS:
                elapsed_time = int(current_time - face_present_start) if face_present_start is not None else 0
                remaining_time = max(0, WORK_TIME_SECONDS - elapsed_time)

                # Show only milestone values in same terminal line
                if remaining_time > 15:
                    display_value = 20
                elif remaining_time > 10:
                    display_value = 15
                elif remaining_time > 5:
                    display_value = 10
                elif remaining_time > 0:
                    display_value = 5
                else:
                    display_value = 0

                if display_value != last_display_value:
                    print(f"\rBreak in: {display_value}s   ", end="", flush=True)
                    last_display_value = display_value

                if elapsed_time >= WORK_TIME_SECONDS and not alert_shown:
                    alert_shown = True
                    print("\rBreak in: 0s   ", end="", flush=True)

                    show_break_overlay()

                    # Reset after break
                    face_present_start = None
                    last_face_seen_time = None
                    alert_shown = False
                    last_display_value = None

            else:
                face_present_start = None
                last_face_seen_time = None
                alert_shown = False
                last_display_value = None

            root.update()

    except KeyboardInterrupt:
        print("\nStopped by user.")

cap.release()
root.destroy()