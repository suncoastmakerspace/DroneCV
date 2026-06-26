"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              DRONE CV — People Counter v1                                    ║
║   YOLOv8 · ByteTrack · Confidence Smoothing · CSV Logging · Heatmap         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  INSTALL (once):                                                             ║
║    pip install ultralytics opencv-python numpy                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USAGE:                                                                      ║
║    python people_counter_v1.py --source video.mp4                           ║
║    python people_counter_v1.py --source video.mp4 --output result.mp4       ║
║    python people_counter_v1.py --source 0                  # webcam          ║
║    python people_counter_v1.py --source rtsp://x.x.x.x:8554/stream         ║
║    python people_counter_v1.py --source video.mp4 --no-display              ║
║    python people_counter_v1.py --calibrate                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  LIVE CONTROLS:                                                              ║
║    [Q / Esc]   Quit                                                          ║
║    [R]         Reset counts and tracking                                     ║
║    [H]         Toggle heatmap overlay                                        ║
║    [+/-]       Raise / lower confidence threshold                            ║
║    [C]         Re-run calibration wizard (live / stream only)               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import csv
import sys
import time
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("\n[ERROR] ultralytics is not installed.")
    print("        Run:  pip install ultralytics opencv-python numpy\n")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (BGR)
# ─────────────────────────────────────────────────────────────────────────────

C_WHITE      = (255, 255, 255)
C_BLACK      = (0,   0,   0)
C_PANEL_DARK = (18,  18,  18)
C_RULE       = (60,  60,  60)
C_DIM        = (140, 140, 140)
C_ACCENT     = (200, 200, 200)
C_PRESENT    = (255, 255, 255)
C_BOX        = (160, 210, 160)
C_CALIB      = (80,  200, 255)

# ─────────────────────────────────────────────────────────────────────────────
#  TUNABLE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PIXELS_PER_METER_BASELINE: float = 200.0
REFERENCE_BOX_HEIGHT_PX:   float = 300.0
CALIBRATION_FILE                 = "calibration.npz"

# Confidence smoothing — a track must appear in this many consecutive frames
# before its box is shown. Eliminates single-frame false positives.
CONFIRM_FRAMES: int = 3

# Minimum bounding-box height in pixels. Detections smaller than this are
# discarded — useful for high-altitude drone footage where distant noise
# gets misclassified as people.
MIN_BOX_HEIGHT_PX: int = 30


# ─────────────────────────────────────────────────────────────────────────────
#  PERSPECTIVE MAPPER
# ─────────────────────────────────────────────────────────────────────────────

class PerspectiveMapper:
    """
    Maps pixel foot-positions to real-world ground-plane coordinates (metres).
    Uses a homography if calibration.npz exists, otherwise falls back to a
    depth-corrected single-ratio estimate based on bounding-box height.
    """

    def __init__(self, calibration_file: str = CALIBRATION_FILE):
        self.H: np.ndarray | None = None
        self.mode = "fallback"
        self._load(calibration_file)

    def _load(self, path: str):
        if Path(path).exists():
            try:
                data = np.load(path)
                self.H = data["H"]
                self.mode = "homography"
                print(f"[CALIB] Loaded homography from {path}")
                return
            except Exception as exc:
                print(f"[CALIB] Could not load {path}: {exc} — using fallback")
        print("[CALIB] No calibration file — using depth-corrected fallback.")
        print("        Run with --calibrate for best accuracy.")

    def save(self, H: np.ndarray, path: str = CALIBRATION_FILE):
        np.savez(path, H=H)
        self.H = H
        self.mode = "homography"
        print(f"[CALIB] Homography saved to {path}")

    def pixel_to_world(self, px: float, py: float,
                       box_height_px: float = 0.0) -> tuple[float, float]:
        if self.mode == "homography" and self.H is not None:
            pt = np.array([px, py, 1.0], dtype=np.float64)
            world = self.H @ pt
            if abs(world[2]) < 1e-9:
                return (0.0, 0.0)
            return (world[0] / world[2], world[1] / world[2])
        bh = box_height_px if box_height_px > 10 else REFERENCE_BOX_HEIGHT_PX
        effective_ppm = PIXELS_PER_METER_BASELINE * (bh / REFERENCE_BOX_HEIGHT_PX)
        return (px / effective_ppm, py / effective_ppm)


# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION WIZARD
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration_wizard(cap: cv2.VideoCapture,
                           mapper: PerspectiveMapper,
                           calib_file: str = CALIBRATION_FILE) -> bool:
    print("\n" + "═" * 60)
    print("  CALIBRATION WIZARD")
    print("  Click 4 floor corners of a rectangle you have measured.")
    print("  Press [Esc] to cancel.\n")

    for _ in range(10):
        ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read frame for calibration.")
        return False

    frame_display = frame.copy()
    h, w = frame.shape[:2]
    points_px: list[tuple[int, int]] = []
    NEEDED = 4

    def _draw_state(img):
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 56), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.putText(img, f"CALIBRATION: click point {len(points_px)+1} of {NEEDED}",
                    (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_CALIB, 1, cv2.LINE_AA)
        cv2.putText(img, "Click 4 floor corners. Esc = cancel.",
                    (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
        for i, (px, py) in enumerate(points_px):
            cv2.circle(img, (px, py), 7, C_CALIB, -1)
            cv2.circle(img, (px, py), 9, C_WHITE, 1)
            cv2.putText(img, f"P{i}", (px + 11, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_CALIB, 1, cv2.LINE_AA)
            if i > 0:
                cv2.line(img, points_px[i - 1], (px, py), C_CALIB, 1)
        if len(points_px) == NEEDED:
            cv2.line(img, points_px[-1], points_px[0], C_CALIB, 1)

    window = "Calibration — Drone CV"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(w, 1280), min(h, 720))

    def _on_click(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points_px) < NEEDED:
            points_px.append((x, y))

    cv2.setMouseCallback(window, _on_click)

    while len(points_px) < NEEDED:
        disp = frame_display.copy()
        _draw_state(disp)
        cv2.imshow(window, disp)
        if cv2.waitKey(30) & 0xFF == 27:
            cv2.destroyWindow(window)
            print("[CALIB] Cancelled.")
            return False

    disp = frame_display.copy()
    _draw_state(disp)
    cv2.imshow(window, disp)
    cv2.waitKey(300)
    cv2.destroyWindow(window)

    print("\n  Enter real-world (X, Y) in METRES for each point.")
    print("  Tip: set P0 at (0,0); measure X rightward, Y away from camera.\n")
    world_pts = []
    for i, (px, py) in enumerate(points_px):
        while True:
            try:
                raw = input(f"  P{i} pixel=({px},{py})  →  X Y (metres): ")
                if raw.strip().lower() in ("q", "esc", "cancel"):
                    print("[CALIB] Cancelled.")
                    return False
                parts = raw.strip().split()
                world_pts.append((float(parts[0]), float(parts[1])))
                break
            except (ValueError, IndexError):
                print("    [!] Enter two numbers e.g.  0 0")

    src = np.array(points_px, dtype=np.float32)
    dst = np.array(world_pts,  dtype=np.float32)
    H, status = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        print("[CALIB] Failed — ensure 4 points form a proper quadrilateral.")
        return False

    n_inliers = int(status.sum()) if status is not None else "?"
    print(f"\n[CALIB] Homography computed ({n_inliers}/4 inliers).")
    mapper.save(H, calib_file)

    print("\n  Reprojection check:")
    for i, (ppx, ppy) in enumerate(points_px):
        wx, wy = mapper.pixel_to_world(ppx, ppy)
        ex, ey = world_pts[i]
        print(f"    P{i}: ({ppx},{ppy}) → ({wx:.3f}, {wy:.3f}) m  "
              f"[expected ({ex:.3f}, {ey:.3f})]")
    print()
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIDENCE SMOOTHER
#  Requires a detection to appear CONFIRM_FRAMES times in a row before the
#  bounding box is shown. Kills single-frame false positives cold.
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceSmoother:
    def __init__(self, confirm_frames: int = CONFIRM_FRAMES):
        self._confirm = confirm_frames
        self._streak: dict[int, int] = {}   # tid → consecutive-hit count

    def update(self, tracked: list) -> list:
        """Return only tracks that have been seen for CONFIRM_FRAMES frames."""
        current_ids = {t[0] for t in tracked}

        # Increment streak for present tracks, reset absent ones
        for tid in list(self._streak.keys()):
            if tid not in current_ids:
                del self._streak[tid]
        for tid in current_ids:
            self._streak[tid] = self._streak.get(tid, 0) + 1

        return [t for t in tracked if self._streak.get(t[0], 0) >= self._confirm]

    def reset(self):
        self._streak = {}


# ─────────────────────────────────────────────────────────────────────────────
#  HEATMAP ACCUMULATOR
#  Accumulates foot-point positions across every frame and renders a colour
#  heatmap overlay showing where people spent the most time.
# ─────────────────────────────────────────────────────────────────────────────

class HeatmapAccumulator:
    def __init__(self, w: int, h: int, decay: float = 0.998):
        self._map   = np.zeros((h, w), dtype=np.float32)
        self._decay = decay   # slight decay so old paths fade slowly

    def update(self, tracked: list):
        self._map *= self._decay
        for _tid, cx, cy, _conf, _x1, _y1, _x2, y2 in tracked:
            # Use foot point (bottom-centre) — same ground reference as tracker
            fx, fy = int(cx), int(y2)
            if 0 <= fy < self._map.shape[0] and 0 <= fx < self._map.shape[1]:
                cv2.circle(self._map, (fx, fy), 15, 1.0, -1)

    def render(self, frame: np.ndarray, alpha: float = 0.45):
        """Blend the heatmap onto frame in-place."""
        if self._map.max() < 1e-6:
            return
        norm = cv2.normalize(self._map, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)
        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        # Mask out near-zero areas so the background shows through cleanly
        mask = norm > 8
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        blended = cv2.addWeighted(coloured, alpha, frame, 1 - alpha, 0)
        frame[mask_3ch] = blended[mask_3ch]

    def save(self, path: str, frame_shape: tuple):
        """Save a standalone heatmap PNG next to the output video."""
        if self._map.max() < 1e-6:
            return
        norm = cv2.normalize(self._map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        cv2.imwrite(path, coloured)
        print(f"[HEATMAP] Saved to {path}")

    def reset(self):
        self._map[:] = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  CSV LOGGER
#  Writes one row per frame: timestamp, frame number, people in frame,
#  and each detected person's ID + bounding-box centre.
# ─────────────────────────────────────────────────────────────────────────────

class CSVLogger:
    def __init__(self, path: str):
        self._path = path
        self._fh   = open(path, "w", newline="")
        self._w    = csv.writer(self._fh)
        self._w.writerow([
            "timestamp", "frame", "people_in_frame",
            "track_id", "cx", "cy", "conf", "x1", "y1", "x2", "y2"
        ])
        print(f"[CSV] Logging to {path}")

    def log(self, frame_num: int, tracked: list):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        n  = len(tracked)
        if not tracked:
            self._w.writerow([ts, frame_num, n, "", "", "", "", "", "", "", ""])
        for tid, cx, cy, conf, x1, y1, x2, y2 in tracked:
            self._w.writerow([ts, frame_num, n, tid, cx, cy, f"{conf:.3f}",
                              x1, y1, x2, y2])

    def close(self):
        self._fh.close()


# ─────────────────────────────────────────────────────────────────────────────
#  CENTROID TRACKER  (fallback when ByteTrack unavailable)
# ─────────────────────────────────────────────────────────────────────────────

class CentroidTracker:
    def __init__(self, max_distance: int = 80, max_absent: int = 15):
        self.tracks: dict  = {}
        self.next_id: int  = 1
        self.max_distance  = max_distance
        self.max_absent    = max_absent

    def update(self, detections):
        det_centroids = []
        for x1, y1, x2, y2, conf in detections:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            det_centroids.append((cx, cy, conf, x1, y1, x2, y2))

        assigned_dets  = set()
        updated_tracks = {}

        for tid, t in self.tracks.items():
            best_idx, best_dist = None, self.max_distance
            for i, (cx, cy, conf, x1, y1, x2, y2) in enumerate(det_centroids):
                if i in assigned_dets:
                    continue
                d = np.hypot(cx - t["cx"], cy - t["cy"])
                if d < best_dist:
                    best_dist, best_idx = d, i
            if best_idx is not None:
                cx, cy, conf, x1, y1, x2, y2 = det_centroids[best_idx]
                assigned_dets.add(best_idx)
                updated_tracks[tid] = dict(cx=cx, cy=cy, absent=0,
                                           conf=conf, x1=x1, y1=y1, x2=x2, y2=y2)
            else:
                if t["absent"] < self.max_absent:
                    updated_tracks[tid] = {**t, "absent": t["absent"] + 1}

        for i, (cx, cy, conf, x1, y1, x2, y2) in enumerate(det_centroids):
            if i not in assigned_dets:
                updated_tracks[self.next_id] = dict(cx=cx, cy=cy, absent=0,
                                                    conf=conf, x1=x1, y1=y1,
                                                    x2=x2, y2=y2)
                self.next_id += 1

        self.tracks = updated_tracks
        result = []
        for tid, t in self.tracks.items():
            if t["absent"] == 0:
                result.append((tid, t["cx"], t["cy"], t["conf"],
                               t["x1"], t["y1"], t["x2"], t["y2"]))
        return result

    def reset(self):
        self.tracks  = {}
        self.next_id = 1


# ─────────────────────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_bounding_box(frame, tid, cx, cy, conf, x1, y1, x2, y2):
    """Corner-bracket bounding box with ID + confidence label."""
    color = C_BOX
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    L, T = 12, 2
    for (px, py, dx, dy) in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx * L, py),          color, T)
        cv2.line(frame, (px, py), (px,          py + dy * L), color, T)

    fs     = 0.36
    label  = f"ID {tid}  {conf:.2f}"
    (iw, ih), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    pill_w   = iw + 8
    pill_h   = ih + 6
    lx       = x1
    pill_bot = max(y1 - 5, pill_h + 4)
    pill_top = pill_bot - pill_h

    cv2.rectangle(frame, (lx - 2, pill_top), (lx + pill_w + 2, pill_bot),
                  C_PANEL_DARK, -1)
    cv2.putText(frame, label, (lx + 2, pill_top + ih + 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, color, -1)


def blend_overlay(frame, x1, y1, x2, y2, color, alpha=0.80):
    y1, y2 = max(0, y1), min(frame.shape[0], y2)
    x1, x2 = max(0, x1), min(frame.shape[1], x2)
    if y2 <= y1 or x2 <= x1:
        return
    roi   = frame[y1:y2, x1:x2]
    solid = np.full_like(roi, color, dtype=np.uint8)
    frame[y1:y2, x1:x2] = cv2.addWeighted(solid, 1 - alpha, roi, alpha, 0)


def _text(frame, txt, pos, scale, color, thickness=1):
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_timestamp(frame, frame_num: int, source_fps: float):
    """Burn a frame timestamp into the bottom-left corner of the frame."""
    elapsed_s = frame_num / max(source_fps, 1.0)
    mins, secs = divmod(int(elapsed_s), 60)
    hrs,  mins = divmod(mins, 60)
    ts_str = f"{hrs:02d}:{mins:02d}:{secs:02d}  f{frame_num}"
    cv2.putText(frame, ts_str, (8, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_DIM, 1, cv2.LINE_AA)


def draw_hud(frame, n_present, fps, conf_thresh, w, h,
             calib_mode: str = "fallback", show_heatmap: bool = False):
    BAR_H = 40
    blend_overlay(frame, 0, 0, w, BAR_H, C_PANEL_DARK, alpha=0.15)
    cv2.line(frame, (0, BAR_H), (w, BAR_H), C_RULE, 1)
    _text(frame, "DRONE CV", (16, 26), 0.55, C_ACCENT, 1)

    calib_tag = "HOMOGRAPHY" if calib_mode == "homography" else "DEPTH-EST"
    heat_tag  = "HEAT:ON" if show_heatmap else "HEAT:OFF"
    meta = (f"{datetime.now().strftime('%H:%M:%S')}   "
            f"FPS {fps:4.1f}   "
            f"CONF {conf_thresh:.2f}   "
            f"CAL {calib_tag}   "
            f"{heat_tag}")
    (mw, _), _ = cv2.getTextSize(meta, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    _text(frame, meta, (w - mw - 14, 26), 0.38, C_DIM)

    # People in frame counter
    PW, PH, PM = 210, 100, 10
    px = w - PW - PM
    py = BAR_H + 14
    blend_overlay(frame, px - 1, py, w - PM + 1, py + PH, C_PANEL_DARK, alpha=0.18)
    cv2.rectangle(frame, (px - 1, py), (w - PM + 1, py + PH), C_RULE, 1)
    _text(frame, "PEOPLE IN FRAME", (px + 8, py + 18), 0.36, C_DIM)
    cv2.line(frame, (px, py + 26), (w - PM, py + 26), C_RULE, 1)
    val_str = str(n_present)
    (vw2, vh2), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_DUPLEX, 2.0, 3)
    val_x = w - PM - vw2 - 10
    _text(frame, val_str, (val_x, py + PH - 10), 2.0, C_PRESENT, 3)

    # Footer
    blend_overlay(frame, 0, h - 24, w, h, C_PANEL_DARK, alpha=0.15)
    cv2.line(frame, (0, h - 24), (w, h - 24), C_RULE, 1)
    _text(frame,
          "Q/Esc Quit   R Reset   H Heatmap   +/- Conf   C Calibrate",
          (12, h - 7), 0.32, C_DIM)


# ─────────────────────────────────────────────────────────────────────────────
#  CAMERA / STREAM OPEN  (with auto-reconnect for live streams)
# ─────────────────────────────────────────────────────────────────────────────

def open_source(src, is_file: bool) -> cv2.VideoCapture:
    """Open a video file, webcam index, or RTSP/network stream."""
    if is_file or isinstance(src, str):
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {src}")
            sys.exit(1)
        return cap

    # Webcam — try platform backends in order
    backends = ([cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                if sys.platform == "win32"
                else [cv2.CAP_V4L2, cv2.CAP_ANY])
    indexes  = [src] + [i for i in range(5) if i != src]
    print(f"[INFO] Scanning for webcam (requested index {src}) …")
    for idx in indexes:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"[INFO] Camera found → index {idx}")
                    return cap
                cap.release()
    print("[ERROR] No webcam detected.")
    sys.exit(1)


def try_reconnect(src, is_file: bool,
                  retries: int = 10, delay: float = 2.0) -> cv2.VideoCapture | None:
    """
    Auto-reconnect for live streams that drop. Tries up to `retries` times
    with `delay` seconds between attempts. Returns None if all attempts fail.
    """
    for attempt in range(1, retries + 1):
        print(f"[RECONNECT] Attempt {attempt}/{retries} in {delay:.0f}s …")
        time.sleep(delay)
        try:
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print("[RECONNECT] Stream restored.")
                    return cap
            cap.release()
        except Exception:
            pass
    print("[RECONNECT] Failed — giving up.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def process_video(cap, model, conf_thresh, mapper: PerspectiveMapper,
                  output_path=None, is_file=False, display=True,
                  calib_file=CALIBRATION_FILE,
                  csv_path: str | None = None,
                  use_bytetrack: bool = True,
                  src=None):

    w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # ── Tracker ──────────────────────────────────────────────────────────────
    # ByteTrack is built into ultralytics — pass tracker="bytetrack.yaml" to
    # model.track(). If that fails we fall back to the centroid tracker.
    _use_bt = use_bytetrack
    tracker  = CentroidTracker(max_distance=80, max_absent=15)

    # ── Confidence smoother ───────────────────────────────────────────────────
    smoother = ConfidenceSmoother(confirm_frames=CONFIRM_FRAMES)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    heatmap      = HeatmapAccumulator(w, h)
    show_heatmap = False

    # ── CSV logger ────────────────────────────────────────────────────────────
    csv_logger = CSVLogger(csv_path) if csv_path else None

    # ── Video writer ──────────────────────────────────────────────────────────
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, source_fps, (w, h))

    if display:
        cv2.namedWindow("Drone CV", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Drone CV", min(w, 1280), min(h, 720))

    frame_num     = 0
    fps_avg       = 0.0
    t_last        = time.time()
    conf          = conf_thresh
    SMOOTH_WINDOW = 15
    count_history = []
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
    peak          = 0

    print("\n" + "═" * 70)
    mode_str = f"FILE: {output_path or 'display only'}" if is_file else f"LIVE: {src}"
    print(f"  MODE         : {mode_str}")
    print(f"  RES          : {w}×{h}  |  FPS source: {source_fps:.1f}")
    print(f"  MODEL        : YOLO  |  CONF: {conf:.2f}")
    print(f"  TRACKER      : {'ByteTrack' if _use_bt else 'Centroid'}")
    print(f"  CONFIRM FRAMES: {CONFIRM_FRAMES}  |  MIN BOX H: {MIN_BOX_HEIGHT_PX}px")
    print(f"  CSV LOG      : {csv_path or 'disabled'}")
    print(f"  CALIBRATION  : {mapper.mode.upper()}")
    print("═" * 70 + "\n")

    while True:
        ret, frame = cap.read()

        # ── Handle dropped frames / stream disconnects ────────────────────────
        if not ret:
            if is_file:
                break
            # Live stream — attempt reconnect
            print("[WARN] Frame read failed — attempting reconnect …")
            cap.release()
            cap = try_reconnect(src, is_file)
            if cap is None:
                break
            continue

        frame_num += 1

        # ── YOLO inference ────────────────────────────────────────────────────
        detections = []
        try:
            if _use_bt:
                # ByteTrack path — model.track() returns tracked results directly
                results = model.track(frame, classes=[0], conf=conf,
                                      tracker="bytetrack.yaml", persist=True,
                                      verbose=False)
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        c    = float(box.conf[0])
                        # ByteTrack assigns IDs directly — extract if available
                        tid  = int(box.id[0]) if box.id is not None else (i + 1)
                        bh   = y2 - y1
                        if bh < MIN_BOX_HEIGHT_PX:
                            continue
                        detections.append((tid, (x1 + x2) // 2, (y1 + y2) // 2,
                                           c, x1, y1, x2, y2))
                # ByteTrack gives us IDs directly — skip centroid tracker
                tracked_raw = detections
            else:
                raise RuntimeError("centroid fallback")

        except Exception:
            # Centroid tracker fallback
            _use_bt = False
            results = model(frame, classes=[0], conf=conf, verbose=False)
            raw_dets = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    c  = float(box.conf[0])
                    bh = y2 - y1
                    if bh < MIN_BOX_HEIGHT_PX:
                        continue
                    raw_dets.append((x1, y1, x2, y2, c))
            tracked_raw = tracker.update(raw_dets)

        # ── Confidence smoothing (require CONFIRM_FRAMES consecutive hits) ────
        tracked = smoother.update(tracked_raw)

        # ── Heatmap accumulation ──────────────────────────────────────────────
        heatmap.update(tracked)

        # ── Smoothed count ────────────────────────────────────────────────────
        count_history.append(len(tracked))
        if len(count_history) > SMOOTH_WINDOW:
            count_history.pop(0)
        smoothed_count = round(sum(count_history) / len(count_history))
        peak = max(peak, smoothed_count)

        # ── Render heatmap (behind boxes) ─────────────────────────────────────
        if show_heatmap:
            heatmap.render(frame)

        # ── Draw bounding boxes ───────────────────────────────────────────────
        for tid, cx, cy, c, x1, y1, x2, y2 in tracked:
            draw_bounding_box(frame, tid, cx, cy, c, x1, y1, x2, y2)

        # ── Burn frame timestamp into video ───────────────────────────────────
        draw_timestamp(frame, frame_num, source_fps)

        # ── FPS ───────────────────────────────────────────────────────────────
        t_now   = time.time()
        elapsed = max(t_now - t_last, 1e-9)
        fps_avg = 0.9 * fps_avg + 0.1 * (1.0 / elapsed)
        t_last  = t_now

        # ── HUD ───────────────────────────────────────────────────────────────
        draw_hud(frame, smoothed_count, fps_avg, conf, w, h,
                 calib_mode=mapper.mode, show_heatmap=show_heatmap)

        # ── Progress bar (file mode) ───────────────────────────────────────────
        if is_file and total_frames > 0:
            progress = int(w * frame_num / total_frames)
            cv2.rectangle(frame, (0, h - 4), (progress, h), C_ACCENT, -1)

        # ── CSV logging ───────────────────────────────────────────────────────
        if csv_logger:
            csv_logger.log(frame_num, tracked)

        # ── Write output video ────────────────────────────────────────────────
        if writer:
            writer.write(frame)

        # ── Console log every 60 frames ───────────────────────────────────────
        if frame_num % 60 == 0:
            ts  = datetime.now().strftime("%H:%M:%S")
            pct = f"{100 * frame_num / total_frames:.1f}%" if total_frames else "live"
            print(f"  [{ts}]  frame {frame_num:6d} ({pct})  "
                  f"in frame: {smoothed_count:2d}  fps: {fps_avg:5.1f}")

        # ── Display + key handling ────────────────────────────────────────────
        if display:
            cv2.imshow("Drone CV", frame)
            key = cv2.waitKey(1 if not is_file else 15) & 0xFF

            if cv2.getWindowProperty("Drone CV", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), 27, ord("x")):
                break
            elif key == ord("r"):
                tracker.reset()
                smoother.reset()
                heatmap.reset()
                count_history.clear()
                print("[RESET] Counts and heatmap cleared.")
            elif key == ord("h"):
                show_heatmap = not show_heatmap
                print(f"[HEATMAP] {'ON' if show_heatmap else 'OFF'}")
            elif key in (ord("+"), ord("=")):
                conf = min(0.95, conf + 0.05)
                print(f"[CONF] → {conf:.2f}")
            elif key == ord("-"):
                conf = max(0.05, conf - 0.05)
                print(f"[CONF] → {conf:.2f}")
            elif key == ord("c") and not is_file:
                print("[CALIB] Re-running calibration wizard …")
                run_calibration_wizard(cap, mapper, calib_file)
        else:
            # Headless — still break on file end (handled above)
            pass

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()
    if csv_logger:
        csv_logger.close()

    # Save standalone heatmap image
    if output_path:
        hmap_path = str(Path(output_path).with_suffix("")) + "_heatmap.png"
        heatmap.save(hmap_path, (h, w))

    return {
        "frames_processed": frame_num,
        "peak_in_frame":    peak,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Drone CV — People Counter v1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--source", "-s", default="0",
                        help="Webcam index, video file path, or RTSP stream URL")
    parser.add_argument("--output", "-o", default=None,
                        help="Save annotated video to this path")
    parser.add_argument("--conf", "-c", type=float, default=0.40,
                        help="YOLOv8 confidence threshold (default 0.40)")
    parser.add_argument("--model", "-m", default="yolov8n.pt",
                        help="YOLOv8 model weights (default: yolov8n.pt)")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode — requires --output")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run the calibration wizard before processing")
    parser.add_argument("--calib-file", default=CALIBRATION_FILE,
                        help=f"Path to calibration file (default: {CALIBRATION_FILE})")
    parser.add_argument("--csv", default=None,
                        help="Path for CSV detection log (e.g. log.csv). "
                             "Auto-generated next to output video if not set.")
    parser.add_argument("--no-bytetrack", action="store_true",
                        help="Force centroid tracker instead of ByteTrack")
    parser.add_argument("--min-box", type=int, default=MIN_BOX_HEIGHT_PX,
                        help=f"Minimum bounding-box height in pixels "
                             f"(default: {MIN_BOX_HEIGHT_PX}). Raise for drone footage.")
    args = parser.parse_args()

    # Apply CLI overrides to module-level constants
    global MIN_BOX_HEIGHT_PX
    MIN_BOX_HEIGHT_PX = args.min_box

    print(f"\n[INFO] Loading {args.model} …")
    try:
        model = YOLO(args.model)
    except Exception as exc:
        print(f"[ERROR] Cannot load model: {exc}")
        sys.exit(1)
    print("[INFO] Model ready.\n")

    # Determine source type
    raw_src = args.source
    try:
        src     = int(raw_src)
        is_file = False
    except ValueError:
        src     = raw_src
        is_file = not (raw_src.startswith("rtsp://") or
                       raw_src.startswith("http://") or
                       raw_src.startswith("https://"))

    is_stream = not is_file and isinstance(src, str)

    cap    = open_source(src, is_file)
    mapper = PerspectiveMapper(calibration_file=args.calib_file)

    if args.calibrate:
        ok = run_calibration_wizard(cap, mapper, args.calib_file)
        if not ok:
            print("[WARN] Calibration skipped — using fallback.")

    # Auto-generate output path for file mode
    output_path = args.output
    if is_file and output_path is None:
        p           = Path(src)
        output_path = str(p.parent / (p.stem + "_counted" + p.suffix))
        print(f"[INFO] Output will be saved to: {output_path}")

    # Auto-generate CSV path
    csv_path = args.csv
    if csv_path is None and output_path:
        csv_path = str(Path(output_path).with_suffix(".csv"))

    display = not args.no_display

    try:
        stats = process_video(
            cap, model,
            conf_thresh=args.conf,
            mapper=mapper,
            output_path=output_path if (is_file or args.output) else None,
            is_file=is_file,
            display=display,
            calib_file=args.calib_file,
            csv_path=csv_path,
            use_bytetrack=not args.no_bytetrack,
            src=src,
        )
    finally:
        cap.release()

    print("\n" + "═" * 50)
    print("  SESSION SUMMARY")
    print("═" * 50)
    print(f"  Frames processed  : {stats['frames_processed']}")
    print(f"  Peak in frame     : {stats['peak_in_frame']}")
    print(f"  Calibration       : {mapper.mode}")
    if output_path:
        print(f"  Output video      : {output_path}")
    if csv_path:
        print(f"  CSV log           : {csv_path}")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    main()
