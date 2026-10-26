# Eye Break Reminder

A macOS background app that uses your webcam to measure how long you've **actually** been looking at the screen, then reminds you to follow the **20-20-20 rule**: every 20 minutes, look away for 20 seconds.

Fixed-interval timers fire even when you've stepped away. This app only counts time while your face is in front of the camera, and the timer resets when you leave.

## How it works

```
Webcam (320x240) ──► MediaPipe BlazeFace ──► face present? ──► work timer
                                                  │                 │
                                     absent > 3 s │                 │ 20 min reached
                                                  ▼                 ▼
                                             reset timer     full-screen overlay
                                                             (20 s countdown)
```

1. **Face detection.** Frames go through MediaPipe's BlazeFace short-range model in `VIDEO` running mode.
2. **Presence tracking.** The work timer runs while a face has been seen in the last 3 seconds. That grace period stops one missed detection (a head turn, bad lighting) from resetting your progress.
3. **Break overlay.** After 20 minutes of continuous presence, an always-on-top Tkinter window shows a 20-second countdown. It can't be closed until the countdown finishes.
4. **Live countdown.** The terminal shows the time left until the next break (`Break in: 14:32`).

### Battery and CPU optimizations

- The camera captures at **320x240**, which is plenty for detecting a face at desk distance.
- **Adaptive detection frequency.** Inference runs on every 5th frame while you're present and every 12th frame while you're away, so the model rarely runs when nobody is there.
- **No camera preview** window is rendered in background mode.

### Privacy

All processing happens locally and in memory. No frames are saved to disk or sent over the network.

## Project evolution

Each stage of the project lives on its own branch, so you can see how the design changed:

| Branch | What changed |
|---|---|
| `haar-cascade-v1` | First prototype: OpenCV Haar cascade face detection test |
| `haar-cascade-v2` / `v3` | Face-based work timer, topmost break overlay, grace period for missed detections |
| `mediapipe-face-detector` | Replaced Haar cascades with MediaPipe BlazeFace for more robust detection |
| `mediapipe-background-mode` | Headless detection (no preview window), fixed an overlay crash on close |
| `app-packaging` | Packaged as a standalone macOS `.app`, lower resolution plus adaptive detection to save battery |
| `v3-face-landmarker-blink` | **Experimental:** blink detection with MediaPipe Face Landmarker (see below) |

### Experimental: blink detection

Staring at a screen lowers your blink rate, and that causes eye strain. The `blink_counter_*.py` scripts explore tracking blinks as a future signal for when to suggest a break:

- **Eye Aspect Ratio (EAR).** Computed from Face Landmarker eye landmarks; the value drops sharply when the eye closes.
- **Auto-calibration** (`blink_counter_auto_calibration.py`). Instead of a fixed EAR threshold, the script learns your personal open-eye baseline from a rolling buffer of samples. It then sets separate close (85% of baseline) and reopen (90%) thresholds, which keeps one blink from being counted twice, and adds a cooldown.

These scripts open a preview window and aren't wired into `main.py` yet.

## Getting started

**Requirements:** macOS, Python 3.13, a webcam.

```bash
git clone https://github.com/Praveshgairee/Eye_Break_Reminder.git
cd Eye_Break_Reminder

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python main.py
```

On first run, macOS asks for camera permission for your terminal. Press `Ctrl+C` to stop.

### Configuration

The timing constants are at the top of `main.py`:

```python
WORK_TIME_SECONDS = 20 * 60    # screen time before a break
BREAK_TIME_SECONDS = 20        # break countdown length
FACE_TIMEOUT_SECONDS = 3       # grace period for missed detections
```

### Build the macOS app

```bash
pip install pyinstaller
pyinstaller "Eye Break Reminder.spec"
```

The app is written to `dist/Eye Break Reminder.app`.

## Project structure

```
main.py                            # the app: detection loop, timer, break overlay
Eye Break Reminder.spec            # PyInstaller build config
models/
  blaze_face_short_range.tflite    # MediaPipe face detector (used by main.py)
  face_landmarker.task             # MediaPipe face landmarker (blink detection)
blink_counter_test.py              # blink detection, fixed EAR threshold
blink_counter_calibration.py       # blink detection, manual calibration
blink_counter_auto_calibration.py  # blink detection, automatic per-user calibration
face_landmarker_test.py            # Face Landmarker sanity check
mediapipe_test.py                  # BlazeFace detector sanity check
camera_test.py                     # OpenCV camera sanity check
```

## Tech stack

Python · OpenCV · MediaPipe Tasks (BlazeFace, Face Landmarker) · Tkinter · PyInstaller
