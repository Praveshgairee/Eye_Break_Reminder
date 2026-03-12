from setuptools import setup

APP = ["main.py"]
DATA_FILES = [("models", ["models/blaze_face_short_range.tflite"])]
OPTIONS = {
    "argv_emulation": False,
    "plist": {
        "CFBundleName": "Eye Break Reminder",
        "CFBundleDisplayName": "Eye Break Reminder",
        "CFBundleIdentifier": "com.praveshgaire.eyebreakreminder",
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "LSUIElement": True,
        "NSCameraUsageDescription": "This app uses the camera to detect when you are in front of the screen and remind you to rest your eyes."
    },
    "packages": ["mediapipe", "cv2", "tkinter"],
    "includes": [
        "mediapipe",
        "mediapipe.tasks",
        "mediapipe.tasks.python",
        "mediapipe.tasks.python.vision",
        "cv2",
        "tkinter"
    ],
}

setup(
    app=APP,
    name="Eye Break Reminder",
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
