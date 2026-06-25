# people-counter

Real-time person detection and tracking for video files and live webcam feeds.

Built with YOLOv8 and a custom centroid tracker. Draws a persistent bounding box around every person in frame for as long as they remain visible, and keeps a live count in the corner.

---

## What it does

- Detects people in a video file or webcam stream using YOLOv8
- Assigns each person a persistent ID that follows them across frames
- Draws a clean bounding box around each person while they're in frame
- Shows a live "people in frame" counter in the top-right corner
- Saves an annotated copy of the video when processing a file
- Optional perspective calibration for more accurate spatial measurements

---

## Requirements

- Python 3.10+
- A GPU is not required — runs on CPU

---

## Installation

```bash
pip install ultralytics opencv-python numpy
```

---

## Usage

```bash
# Process a video file (saves annotated output automatically)
python people_counter.py --source video.mp4

# Live webcam
python people_counter.py

# Save annotated output to a specific path
python people_counter.py --source video.mp4 --output result.mp4

# Adjust detection confidence (default 0.40)
python people_counter.py --source video.mp4 --conf 0.5

# Run without opening a display window (file output only)
python people_counter.py --source video.mp4 --no-display

# Run the perspective calibration wizard first
python people_counter.py --calibrate
```

---

## Controls (live window)

| Key | Action |
|---|---|
| `Q` / `Esc` | Quit |
| `R` | Reset counts and tracking |
| `+` / `-` | Raise / lower confidence threshold |
| `C` | Re-run calibration wizard (webcam only) |

---

## Perspective calibration

By default the script uses a bounding-box height heuristic to estimate depth — people further from the camera appear smaller, so the scale factor adjusts accordingly. This is good enough for most fixed-angle cameras.

For higher accuracy, run `--calibrate` once. You'll click 4 floor points in the frame whose real-world positions you know (e.g. corners of a mat you've measured). The script computes a homography matrix, saves it to `calibration.npz`, and loads it automatically on every subsequent run.

```bash
python people_counter.py --calibrate
```

Tips:
- Use floor markings or tile corners — something you can physically measure
- Place P0 at (0, 0), measure X rightward and Y away from the camera
- The four points must form a proper quadrilateral (not collinear)

---

## Output

When processing a video file, an annotated copy is saved automatically next to the source file with `_counted` appended to the filename. A summary is printed to the terminal when the run finishes:

```
  Frames processed  : 3240
  Peak in frame     : 7
  Velocity mode     : fallback
  Output video      : corridor_counted.mp4
```

---

## Model

Uses `yolov8n.pt` by default (the smallest/fastest YOLOv8 variant). Pass `--model` to use a different set of weights:

```bash
python people_counter.py --source video.mp4 --model yolov8s.pt
```

Larger models (`yolov8s`, `yolov8m`) are more accurate but slower on CPU.

---

## License

For educational and research use.