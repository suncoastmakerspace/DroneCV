DroneCV — People Counter

Real-time person detection and tracking for drone footage, video files, and live streams.

Built with YOLOv8 and ByteTrack. Draws a persistent bounding box around every person in frame, logs detections to CSV, generates a heatmap of where people spent time, and supports RTSP for live drone feeds.


What it does


Detects people using YOLOv8, filters by minimum box size (useful for high-altitude footage where noise gets misclassified)
Tracks each person with ByteTrack — IDs survive occlusions and path crossings without swapping
Confidence smoothing — a person must appear in 3 consecutive frames before their box is shown, eliminating single-frame false positives
Live "people in frame" counter with 15-frame smoothing to prevent flicker
Frame timestamp burned into every frame so you know exactly when each detection happened
CSV log — one row per detection per frame with timestamp, track ID, position, and confidence
Heatmap — accumulates where people appeared across the whole video, toggle with [H]
Saves annotated output video automatically
Auto-reconnect for RTSP streams that drop mid-flight
Optional perspective calibration for accurate spatial measurements



Requirements


Python 3.10+
GPU not required — runs on CPU



Installation

bashpip install ultralytics opencv-python numpy


Usage

bash# Process a video file (annotated output + CSV saved automatically)
python people_counter_v2.py --source video.mp4

# Live webcam
python people_counter_v2.py

# Live RTSP stream from Raspberry Pi on drone
python people_counter_v2.py --source rtsp://192.168.1.x:8554/stream

# Save to a specific output path
python people_counter_v2.py --source video.mp4 --output result.mp4

# Adjust confidence (default 0.40)
python people_counter_v2.py --source video.mp4 --conf 0.5

# Raise minimum box height for high-altitude footage
python people_counter_v2.py --source video.mp4 --min-box 50

# Headless — no display window, file output only
python people_counter_v2.py --source video.mp4 --no-display

# Force centroid tracker instead of ByteTrack
python people_counter_v2.py --source video.mp4 --no-bytetrack

# Run perspective calibration wizard first
python people_counter_v2.py --calibrate


Controls (live window)

KeyActionQ / EscQuitRReset counts, tracking, and heatmapHToggle heatmap overlay+ / -Raise / lower confidence thresholdCRe-run calibration wizard (live / stream only)


Output files

When processing a video file, three files are saved automatically next to the source:

FileContentsvideo_counted.mp4Annotated video with bounding boxes and HUDvideo_counted.csvPer-frame detection log (timestamp, ID, position, confidence)video_counted_heatmap.pngHeatmap image showing where people appeared


Drone / Raspberry Pi setup

The Pi streams video over WiFi; the laptop runs all detection locally.

On the Raspberry Pi:

bashlibcamera-vid -t 0 --width 1280 --height 720 --framerate 30 \
  --codec h264 -o - | ffmpeg -i - -c copy -f rtsp rtsp://0.0.0.0:8554/stream

On the laptop:

bashpython people_counter_v2.py --source rtsp://192.168.1.x:8554/stream

Replace 192.168.1.x with the Pi's IP address. If the stream drops, the program automatically attempts to reconnect.

Tips for drone footage:


Use --min-box 50 or higher — people look small from altitude
Use --conf 0.45 or higher to cut false positives
Run --calibrate with known ground measurements for accurate spatial data
yolov8s.pt or yolov8m.pt are more accurate than the default yolov8n.pt if speed allows



Perspective calibration

Run --calibrate once. Click 4 floor points whose real-world positions you know. A homography matrix is saved to calibration.npz and loaded automatically on every subsequent run.


Model options

ModelSpeedAccuracyyolov8n.ptFastestLoweryolov8s.ptFastBetteryolov8m.ptMediumGood

Pass --model yolov8s.pt to switch. All models download automatically on first run.


License

For educational and research use.
