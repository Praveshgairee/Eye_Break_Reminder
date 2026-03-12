import cv2
import time
import tkinter as tk
from tkinter import messagebox

# -----------------------------
# Settings
# -----------------------------
WORK_TIME_SECONDS = 60   # 20 minutes, 60 sec for test
BREAK_TIME_SECONDS = 10       # 20 seconds 10 sec for test

# -----------------------------
# Load face detector
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Open camera
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# -----------------------------
# Tkinter root for popup
# -----------------------------
root = tk.Tk()
root.withdraw()  # hide main tkinter window

# -----------------------------
# Timer state
# -----------------------------
face_present_start = None
alert_shown = False

print("Eye break reminder started.")
print("Press q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    current_time = time.time()

    if len(faces) > 0:
        # Start timing only when face is first detected
        if face_present_start is None:
            face_present_start = current_time

        elapsed_time = int(current_time - face_present_start)
        remaining_time = max(0, WORK_TIME_SECONDS - elapsed_time)

        status = f"Face detected | Break in: {remaining_time}s"

        # Draw face rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show popup after 20 minutes
        if elapsed_time >= WORK_TIME_SECONDS and not alert_shown:
            alert_shown = True
            messagebox.showwarning(
                "Eye Break Reminder",
                "20 minutes completed.\nLook away from the screen for 20 seconds."
            )

            # Break countdown
            break_start = time.time()
            while time.time() - break_start < BREAK_TIME_SECONDS:
                break_remaining = int(BREAK_TIME_SECONDS - (time.time() - break_start))
                temp_frame = frame.copy()
                cv2.putText(
                    temp_frame,
                    f"BREAK: {break_remaining}s",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                cv2.imshow("Eye Break Reminder", temp_frame)
                cv2.waitKey(1)

            # Reset timer after break
            face_present_start = None
            alert_shown = False

    else:
        status = "No face detected"
        face_present_start = None
        alert_shown = False

    # Show status text
    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow("Eye Break Reminder", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
root.destroy()
