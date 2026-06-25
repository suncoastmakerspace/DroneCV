"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           HALLWAY PEOPLE COUNTER — School Edition                           ║
║   YOLOv8 Detection · Centroid Tracking · Bounding Box Overlay               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  INSTALL (once):                                                             ║
║    pip install ultralytics opencv-python numpy                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USAGE:                                                                      ║
║    python people_counter.py                        # webcam (live)           ║
║    python people_counter.py --source video.mp4     # process video file      ║
║    python people_counter.py --source video.mp4 --output result.mp4          ║
║    python people_counter.py --source 0 --conf 0.4  # custom confidence       ║
║    python people_counter.py --calibrate            # run calibration wizard  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  LIVE CONTROLS:                                                              ║
║    [Q / Esc]   Quit                                                          ║
║    [R]         Reset counts                                                  ║
║    [+/-]       Raise / lower confidence threshold                            ║
║    [C]         Re-run calibration wizard (live mode only)                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import argparse
import sys
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("\n[ERROR] ultralytics is not installed.")
    print("        Run:  pip install ultralytics opencv-python\n")
    sys.exit(1)

try:
    import pybodytrack
    from pybodytrack import BodyTracker
    BODYTRACK_AVAILABLE = True
except ImportError:
    BODYTRACK_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (BGR)
# ─────────────────────────────────────────────────────────────────────────────

C_WHITE       = (255, 255, 255)
C_BLACK       = (0,   0,   0)
C_PANEL_DARK  = (18,  18,  18)
C_RULE        = (60,  60,  60)
C_DIM         = (140, 140, 140)
C_ACCENT      = (200, 200, 200)
C_PRESENT     = (255, 255, 255)
C_BOX         = (160, 210, 160)
C_CALIB       = (80,  200, 255)   # cyan — calibration overlay
C_DIST_OK     = (180, 180, 180)   # grey  — safe distance line
C_DIST_ALERT  = (60,  60,  220)   # red   — close-pair alert line


# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION CONSTANTS  (fallback when no .npz file is present)
# ─────────────────────────────────────────────────────────────────────────────

PIXELS_PER_METER_BASELINE: float = 200.0
REFERENCE_BOX_HEIGHT_PX:   float = 300.0
CALIBRATION_FILE = "calibration.npz"

DEFAULT_ALERT_DISTANCE_M: float = 1.5   # metres — pairs closer than this are flagged


# ─────────────────────────────────────────────────────────────────────────────
#  PERSPECTIVE MAPPER
# ─────────────────────────────────────────────────────────────────────────────

class PerspectiveMapper:
    """
    Maps pixel foot-positions to real-world ground-plane coordinates (metres).

    If a calibration .npz file exists it uses the stored 3×3 homography H.
    Otherwise it falls back to a depth-corrected single-ratio estimate:

        effective_ppm = BASELINE_PPM × (box_height / REFERENCE_BOX_HEIGHT)
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
                print(f"[CALIB] Loaded homography from {path}  (perspective-correct)")
                return
            except Exception as exc:
                print(f"[CALIB] Could not load {path}: {exc} — using fallback")
        print(f"[CALIB] No calibration file found — using depth-corrected fallback")
        print(f"        Run with --calibrate to generate one for best accuracy.")

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
        else:
            bh = box_height_px if box_height_px > 10 else REFERENCE_BOX_HEIGHT_PX
            effective_ppm = PIXELS_PER_METER_BASELINE * (bh / REFERENCE_BOX_HEIGHT_PX)
            wx = px / effective_ppm
            wy = py / effective_ppm
            return (wx, wy)

    def pixel_displacement_to_metres(self,
                                     px0: float, py0: float,
                                     px1: float, py1: float,
                                     box_height_px: float = 0.0) -> float:
        wx0, wy0 = self.pixel_to_world(px0, py0, box_height_px)
        wx1, wy1 = self.pixel_to_world(px1, py1, box_height_px)
        return float(np.hypot(wx1 - wx0, wy1 - wy0))


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE CALIBRATION WIZARD
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration_wizard(cap: cv2.VideoCapture,
                           mapper: PerspectiveMapper,
                           calib_file: str = CALIBRATION_FILE) -> bool:
    """
    Grabs a frame and lets the user click 4 ground-plane control points.
    Returns True if calibration succeeded and was saved.
    """
    print("\n" + "═"*60)
    print("  CALIBRATION WIZARD")
    print("  You will click 4 floor points (corners of a known rectangle).")
    print("  Press [Esc] at any time to cancel.\n")

    for _ in range(10):
        ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read frame for calibration.")
        return False

    frame_display = frame.copy()
    h, w = frame.shape[:2]
    points_px: list[tuple[int,int]] = []
    NEEDED = 4

    def _draw_state(img):
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 56), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.putText(img, f"CALIBRATION: click point {len(points_px)+1} of {NEEDED}",
                    (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "Click 4 floor corners of a rectangle you know in real life. Esc = cancel.",
                    (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
        for i, (px, py) in enumerate(points_px):
            cv2.circle(img, (px, py), 7, C_CALIB, -1)
            cv2.circle(img, (px, py), 9, (255,255,255), 1)
            cv2.putText(img, f"P{i}", (px+11, py-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_CALIB, 1, cv2.LINE_AA)
            if i > 0:
                cv2.line(img, points_px[i-1], (px, py), C_CALIB, 1)
        if len(points_px) == NEEDED:
            cv2.line(img, points_px[-1], points_px[0], C_CALIB, 1)

    window = "Calibration — People Counter"
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
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyWindow(window)
            print("[CALIB] Cancelled.")
            return False

    disp = frame_display.copy()
    _draw_state(disp)
    cv2.imshow(window, disp)
    cv2.waitKey(300)
    cv2.destroyWindow(window)

    print("\n  Enter the real-world (X, Y) in METRES for each point.")
    print("  Tip: place P0 at (0, 0); measure X rightward, Y away from camera.\n")
    world_pts = []
    for i, (px, py) in enumerate(points_px):
        while True:
            try:
                raw = input(f"  P{i} pixel=({px},{py})  →  X Y (metres, space-separated): ")
                if raw.strip().lower() in ("q", "esc", "cancel"):
                    print("[CALIB] Cancelled.")
                    return False
                parts = raw.strip().split()
                wx, wy = float(parts[0]), float(parts[1])
                world_pts.append((wx, wy))
                break
            except (ValueError, IndexError):
                print("    [!] Enter two numbers, e.g.  0 0")

    src = np.array(points_px,  dtype=np.float32)
    dst = np.array(world_pts,  dtype=np.float32)
    H, status = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        print("[CALIB] Homography computation failed — check that your 4 points "
              "form a proper non-degenerate quadrilateral.")
        return False

    n_inliers = int(status.sum()) if status is not None else "?"
    print(f"\n[CALIB] Homography computed ({n_inliers}/4 inliers).")
    mapper.save(H, calib_file)

    print("\n  Reprojection check (should match your entered values):")
    for i, (ppx, ppy) in enumerate(points_px):
        wx, wy = mapper.pixel_to_world(ppx, ppy)
        expected = world_pts[i]
        print(f"    P{i}: pixel=({ppx},{ppy}) → world=({wx:.3f}, {wy:.3f}) m  "
              f"[expected ({expected[0]:.3f}, {expected[1]:.3f})]")
    print()
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  CENTROID TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class CentroidTracker:
    def __init__(self, max_distance: int = 80, max_absent: int = 10):
        self.tracks: dict = {}
        self.next_id: int = 1
        self.max_distance = max_distance
        self.max_absent = max_absent

    def update(self, detections):
        det_centroids = []
        for x1, y1, x2, y2, conf in detections:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            det_centroids.append((cx, cy, conf, x1, y1, x2, y2))

        assigned_dets = set()
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
        self.tracks = {}
        self.next_id = 1


# ─────────────────────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_bounding_box(frame, tid, cx, cy, conf, x1, y1, x2, y2):
    """Corner-bracket bounding box with ID label."""
    box_color = C_BOX

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)

    L, T = 12, 2
    for (px, py, dx, dy) in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx * L, py),          box_color, T)
        cv2.line(frame, (px, py), (px,          py + dy * L), box_color, T)

    fs     = 0.36
    id_str = f"ID {tid}"
    (iw, ih), _ = cv2.getTextSize(id_str, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    pill_w   = iw + 8
    pill_h   = ih + 6
    lx       = x1
    pill_bot = max(y1 - 5, pill_h + 4)
    pill_top = pill_bot - pill_h

    cv2.rectangle(frame,
                  (lx - 2,          pill_top),
                  (lx + pill_w + 2, pill_bot),
                  C_PANEL_DARK, -1)
    cv2.putText(frame, id_str,
                (lx + 2, pill_top + ih + 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, box_color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, box_color, -1)


def blend_overlay(frame, x1, y1, x2, y2, color, alpha=0.80):
    y1, y2 = max(0, y1), min(frame.shape[0], y2)
    x1, x2 = max(0, x1), min(frame.shape[1], x2)
    if y2 <= y1 or x2 <= x1:
        return
    roi = frame[y1:y2, x1:x2]
    solid = np.full_like(roi, color, dtype=np.uint8)
    frame[y1:y2, x1:x2] = cv2.addWeighted(solid, 1 - alpha, roi, alpha, 0)


def _text(frame, txt, pos, scale, color, thickness=1):
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_hud(frame, n_present, fps, conf_thresh, w, h,
             calib_mode: str = "fallback"):
    """HUD showing in-frame people count."""
    BAR_H = 40
    blend_overlay(frame, 0, 0, w, BAR_H, C_PANEL_DARK, alpha=0.15)
    cv2.line(frame, (0, BAR_H), (w, BAR_H), C_RULE, 1)
    _text(frame, "OCCUPANCY MONITOR", (16, 26), 0.55, C_ACCENT, 1)

    calib_tag = "HOMOGRPHY" if calib_mode == "homography" else "DEPTH-EST"
    meta = (f"{datetime.now().strftime('%H:%M:%S')}   "
            f"FPS {fps:4.1f}   "
            f"CONF {conf_thresh:.2f}   "
            f"CAL {calib_tag}")
    (mw, _), _ = cv2.getTextSize(meta, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    _text(frame, meta, (w - mw - 14, 26), 0.38, C_DIM)

    # ── Large "PEOPLE IN FRAME" counter panel ────────────────────────
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

    # ── Footer ───────────────────────────────────────────────────────
    blend_overlay(frame, 0, h - 24, w, h, C_PANEL_DARK, alpha=0.15)
    cv2.line(frame, (0, h - 24), (w, h - 24), C_RULE, 1)
    _text(frame,
          "Q/Esc Quit   R Reset   +/- Conf   C Calibrate",
          (12, h - 7), 0.32, C_DIM)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def process_video(cap, model, conf_thresh, mapper: PerspectiveMapper,
                  output_path=None, is_file=False, display=True,
                  calib_file=CALIBRATION_FILE):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracker      = CentroidTracker(max_distance=80, max_absent=15)

    body_tracker = None
    if BODYTRACK_AVAILABLE:
        try:
            body_tracker = BodyTracker()
        except Exception:
            pass

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, source_fps, (w, h))

    if display:
        cv2.namedWindow("People Counter", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("People Counter", min(w, 1280), min(h, 720))

    frame_num      = 0
    fps_avg        = 0.0
    t_last         = time.time()
    conf           = conf_thresh

    SMOOTH_WINDOW  = 15
    count_history  = []
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0

    print("\n" + "═" * 70)
    mode_str = f"FILE: {output_path or 'display only'}" if is_file else "LIVE WEBCAM"
    print(f"  MODE         : {mode_str}")
    print(f"  RES          : {w}×{h}  |  FPS source: {source_fps:.1f}")
    print(f"  MODEL        : YOLO  |  CONF: {conf:.2f}")
    print(f"  VELOCITY     : {mapper.mode.upper()} calibration")
    print("═" * 70 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_file:
                break
            else:
                continue

        frame_num += 1

        # ── YOLO inference ────────────────────────────────────────────
        results = model(frame, classes=[0], conf=conf, verbose=False)
        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                c = float(box.conf[0])
                detections.append((x1, y1, x2, y2, c))

        # ── Track ─────────────────────────────────────────────────────
        tracked = tracker.update(detections)

        # ── Smoothed in-frame count ───────────────────────────────────
        count_history.append(len(tracked))
        if len(count_history) > SMOOTH_WINDOW:
            count_history.pop(0)
        smoothed_count = round(sum(count_history) / len(count_history))

        # ── Draw bounding boxes ───────────────────────────────────────
        for tid, cx, cy, c, x1, y1, x2, y2 in tracked:
            draw_bounding_box(frame, tid, cx, cy, c, x1, y1, x2, y2)

        # ── FPS ───────────────────────────────────────────────────────
        t_now   = time.time()
        elapsed = max(t_now - t_last, 1e-9)
        fps_avg = 0.9 * fps_avg + 0.1 * (1.0 / elapsed)
        t_last  = t_now

        # ── HUD ───────────────────────────────────────────────────────
        draw_hud(frame, smoothed_count, fps_avg, conf, w, h,
                 calib_mode=mapper.mode)

        if is_file and total_frames > 0:
            progress = int(w * frame_num / total_frames)
            cv2.rectangle(frame, (0, h - 4), (progress, h), C_ACCENT, -1)

        if writer:
            writer.write(frame)

        if frame_num % 60 == 0:
            ts  = datetime.now().strftime("%H:%M:%S")
            pct = f"{100*frame_num/total_frames:.1f}%" if total_frames else "live"
            print(f"  [{ts}]  frame {frame_num:6d} ({pct})  "
                  f"in frame: {smoothed_count:2d}  "
                  f"fps: {fps_avg:5.1f}")

        if display:
            cv2.imshow("People Counter", frame)
            key = cv2.waitKey(1 if not is_file else 15) & 0xFF

            if cv2.getWindowProperty("People Counter", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), 27, ord("x")):
                break
            elif key == ord("r"):
                tracker.reset()
                count_history.clear()
                print("[RESET] Counts cleared.")
            elif key in (ord("+"), ord("=")):
                conf = max(0.05, conf - 0.05)
                print(f"[CONF] → {conf:.2f}")
            elif key == ord("-"):
                conf = min(0.95, conf + 0.05)
                print(f"[CONF] → {conf:.2f}")
            elif key == ord("c") and not is_file:
                print("[CALIB] Re-running calibration wizard …")
                run_calibration_wizard(cap, mapper, calib_file)

    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    return {
        "frames_processed": frame_num,
        "peak_in_frame":    max(count_history) if count_history else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CAMERA AUTO-DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def open_camera(src, is_file: bool) -> cv2.VideoCapture:
    if is_file:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video file: {src}")
            sys.exit(1)
        return cap

    backends = []
    if sys.platform == "win32":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    indexes = [src] + [i for i in range(5) if i != src]
    print(f"[INFO] Scanning for webcam (requested index {src}) …")

    for idx in indexes:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    backend_name = {
                        cv2.CAP_DSHOW: "DirectShow",
                        cv2.CAP_MSMF:  "MSMF",
                        cv2.CAP_V4L2:  "V4L2",
                        cv2.CAP_ANY:   "Auto",
                    }.get(backend, str(backend))
                    print(f"[INFO] Camera found → index {idx}, backend {backend_name}  "
                          f"({int(cap.get(3))}×{int(cap.get(4))})")
                    return cap
                cap.release()

    print("[ERROR] No webcam detected.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🎓 Hallway People Counter — YOLOv8 + Centroid Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--source", "-s", default="0",
                        help="Webcam index or video file path")
    parser.add_argument("--output", "-o", default=None,
                        help="Save annotated video to this path")
    parser.add_argument("--conf", "-c", type=float, default=0.40,
                        help="YOLOv8 confidence threshold (default 0.40)")
    parser.add_argument("--model", "-m", default="yolov8n.pt",
                        help="YOLOv8 model weights (default: yolov8n.pt)")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode — only valid with --output")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run the calibration wizard before processing")
    parser.add_argument("--calib-file", default=CALIBRATION_FILE,
                        help=f"Path to calibration file (default: {CALIBRATION_FILE})")
    args = parser.parse_args()

    print(f"\n[INFO] Loading {args.model} …")
    try:
        model = YOLO(args.model)
    except Exception as exc:
        print(f"[ERROR] Cannot load model: {exc}")
        sys.exit(1)
    print("[INFO] Model ready.\n")

    try:
        src     = int(args.source)
        is_file = False
    except ValueError:
        src     = args.source
        is_file = True

    cap = open_camera(src, is_file)

    mapper = PerspectiveMapper(calibration_file=args.calib_file)

    if args.calibrate:
        ok = run_calibration_wizard(cap, mapper, args.calib_file)
        if not ok:
            print("[WARN] Calibration skipped — using fallback.")

    output_path = args.output
    if is_file and output_path is None:
        p = Path(src)
        output_path = str(p.parent / (p.stem + "_counted" + p.suffix))
        print(f"[INFO] Output will be saved to: {output_path}")

    display = not args.no_display

    try:
        stats = process_video(
            cap, model,
            conf_thresh=args.conf,
            mapper=mapper,
            output_path=output_path if is_file else None,
            is_file=is_file,
            display=display,
            calib_file=args.calib_file,
        )
    finally:
        cap.release()

    print("\n" + "═" * 50)
    print("  SESSION SUMMARY")
    print("═" * 50)
    print(f"  Frames processed  : {stats['frames_processed']}")
    print(f"  Peak in frame     : {stats['peak_in_frame']}")
    print(f"  Velocity mode     : {mapper.mode}")
    if output_path and is_file:
        print(f"  Output video      : {output_path}")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    main()