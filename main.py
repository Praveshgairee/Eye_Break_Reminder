import cv2
import time
import tkinter as tk


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

        # Show popup after work time is completed
            if elapsed_time >= WORK_TIME_SECONDS and not alert_shown:
             alert_shown = True

             show_break_overlay()

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
