import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit

print("MediaPipe test started. Press q to quit.")

with FaceDetector.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.detections:
            for detection in result.detections:
                bbox = detection.bounding_box
                x = bbox.origin_x
                y = bbox.origin_y
                box_w = bbox.width
                box_h = bbox.height

                cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)

            status = "Face detected"
        else:
            status = "No face detected"

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imshow("MediaPipe Face Detector Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
