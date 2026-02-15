"""
Jetson Nano bike detection + annotation + recording + auto-upload to desktop.

Runs inside the jetson-containers NanoOWL container. Launch with:

    jetson-containers run --workdir /opt/nanoowl \
      -v ~/bike_clips:/bike_clips \
      -v ~/detect_and_record.py:/opt/nanoowl/detect_and_record.py \
      --device /dev/video0 \
      $(autotag nanoowl) \
      python3 detect_and_record.py

Set DESKTOP_URL below to your laptop's IP before running.
"""

import cv2
import time
import os
import json
import queue
import threading
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image
from nanoowl.owl_predictor import OwlPredictor

# ── Configuration ────────────────────────────────────────────────────────────

# Desktop Flask server URL (set to your laptop's local IP)
DESKTOP_URL = "http://10.19.182.128:8081/api/upload"

# NanoOWL detection
ENGINE_PATH = "/opt/nanoowl/data/owl_image_encoder_patch32.engine"
TEXT_PROMPTS = ["a face"]  # ["a bicycle", "a bike", "a person riding a bicycle"]
CONFIDENCE_THRESHOLD = 0.15

# Anti-flicker: frames needed to confirm start/stop
START_FRAMES = 5   # consecutive detections to begin recording
STOP_FRAMES = 30   # consecutive absences to stop recording (~1 sec at 30fps)

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Recording
OUTPUT_DIR = Path("/bike_clips")
CODEC = "MJPG"
FPS = 30

# Disk safety: stop recording if free space drops below this (GB)
MIN_FREE_GB = 1.0

# Upload retry backoff (seconds)
UPLOAD_RETRY_DELAY = 5

# Annotation
TRACK_COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 165, 255),  # Orange
    (255, 255, 0),  # Cyan
    (128, 0, 128),  # Purple
    (0, 128, 255),  # Orange-red
]


# ── IoU Tracker ──────────────────────────────────────────────────────────────

def calculate_iou(box1, box2):
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_disappeared=30):
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.next_id = 1
        self.tracks = {}  # id -> {bbox, first_seen, last_seen, disappeared, frame_history}

    def update(self, detections, frame_idx):
        """Update with detections [(x1,y1,x2,y2,score),...]. Returns list of track_ids."""
        if not detections:
            for tid in list(self.tracks):
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]
            return []

        if not self.tracks:
            ids = []
            for det in detections:
                ids.append(self._new_track(det, frame_idx))
            return ids

        # IoU matching
        det_boxes = [d[:4] for d in detections]
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[t]["bbox"][:4] for t in track_ids]

        iou_matrix = np.zeros((len(det_boxes), len(track_boxes)))
        for i, db in enumerate(det_boxes):
            for j, tb in enumerate(track_boxes):
                iou_matrix[i, j] = calculate_iou(db, tb)

        matches = {}
        used_tracks = set()
        for idx in np.argsort(-iou_matrix.ravel()):
            di = idx // len(track_boxes)
            ti = idx % len(track_boxes)
            if iou_matrix[di, ti] < self.iou_threshold:
                break
            if di not in matches and ti not in used_tracks:
                matches[di] = track_ids[ti]
                used_tracks.add(ti)

        result_ids = []
        matched_tids = set()
        for di, det in enumerate(detections):
            if di in matches:
                tid = matches[di]
                self.tracks[tid]["bbox"] = det
                self.tracks[tid]["last_seen"] = frame_idx
                self.tracks[tid]["disappeared"] = 0
                self.tracks[tid]["frame_history"].append(frame_idx)
                result_ids.append(tid)
                matched_tids.add(tid)
            else:
                result_ids.append(self._new_track(det, frame_idx))

        for tid in list(self.tracks):
            if tid not in matched_tids:
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]

        return result_ids

    def _new_track(self, det, frame_idx):
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = {
            "bbox": det,
            "first_seen": frame_idx,
            "last_seen": frame_idx,
            "disappeared": 0,
            "frame_history": [frame_idx],
        }
        return tid


# ── Detection + Annotation ───────────────────────────────────────────────────

def detect_bikes(predictor, frame, text_encodings):
    """Run NanoOWL on a frame, return list of (x1, y1, x2, y2, score)."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    output = predictor.predict(
        image=image,
        text=TEXT_PROMPTS,
        text_encodings=text_encodings,
        threshold=CONFIDENCE_THRESHOLD,
    )
    boxes = []
    for i in range(len(output.boxes)):
        x1, y1, x2, y2 = output.boxes[i].int().tolist()
        score = float(output.scores[i])
        boxes.append((x1, y1, x2, y2, score))
    return boxes


def annotate_frame(frame, detections, track_ids):
    """Draw bounding boxes + track ID labels on frame."""
    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x1, y1, x2, y2, score), tid in zip(detections, track_ids):
        color = TRACK_COLORS[(tid - 1) % len(TRACK_COLORS)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"Bike #{tid}"
        label_size = cv2.getTextSize(label, font, 0.6, 1)[0]
        label_y = y1 - 10 if y1 > 30 else y1 + label_size[1] + 10
        cv2.rectangle(
            annotated,
            (x1, label_y - label_size[1] - 5),
            (x1 + label_size[0] + 10, label_y + 5),
            color, -1,
        )
        cv2.putText(annotated, label, (x1 + 5, label_y), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated


def build_report(tracker, total_frames, clip_name):
    """Build a JSON report from tracker data."""
    tracks = tracker.tracks
    total_bikes = len(tracks)
    bikes = []
    for tid, data in sorted(tracks.items()):
        first = data["first_seen"]
        last = data["last_seen"]
        appearances = len(data["frame_history"])
        bikes.append({
            "track_id": tid,
            "first_seen_frame": first,
            "last_seen_frame": last,
            "first_seen_timestamp_seconds": round(first / FPS, 2),
            "last_seen_timestamp_seconds": round(last / FPS, 2),
            "duration_seconds": round((last - first + 1) / FPS, 2),
            "total_appearances": appearances,
            "lights": {
                "has_front_light": False,
                "has_rear_light": False,
                "has_both_lights": False,
                "compliance_status": "UNKNOWN",
            },
        })

    return {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "report_version": "1.0",
            "source": "jetson_nanoowl",
        },
        "video_metadata": {
            "time_of_day": "auto",
        },
        "video_info": {
            "filename": clip_name,
            "fps": FPS,
            "resolution": [FRAME_WIDTH, FRAME_HEIGHT],
            "total_frames": total_frames,
            "processed_frames": total_frames,
            "duration_seconds": round(total_frames / FPS, 2),
        },
        "summary": {
            "total_bikes_detected": total_bikes,
            "bikes_with_front_lights": 0,
            "bikes_with_rear_lights": 0,
            "bikes_with_both_lights": 0,
            "bikes_with_no_lights": total_bikes,
            "compliance_rate": 0.0,
        },
        "violations": [],
        "bikes": bikes,
        "processing_stats": {
            "processed_frames": total_frames,
            "total_detections": sum(b["total_appearances"] for b in bikes),
        },
    }


# ── Upload worker ────────────────────────────────────────────────────────────

def upload_worker(upload_queue):
    """Background thread: POST completed clips + reports to the desktop."""
    while True:
        item = upload_queue.get()
        if item is None:
            break

        video_path, json_path = item
        print(f"[upload] Sending {video_path} + {json_path} to {DESKTOP_URL}")
        try:
            files = {}
            with open(video_path, "rb") as vf:
                files["video"] = (os.path.basename(video_path), vf.read())
            with open(json_path, "rb") as jf:
                files["report"] = (os.path.basename(json_path), jf.read())

            resp = requests.post(DESKTOP_URL, files=files, timeout=120)
            if resp.status_code in (200, 202):
                print(f"[upload] Success: {resp.json()}")
                os.remove(video_path)
                os.remove(json_path)
            else:
                print(f"[upload] Server returned {resp.status_code}, re-queuing")
                upload_queue.put(item)
                time.sleep(UPLOAD_RETRY_DELAY)
        except requests.RequestException as e:
            print(f"[upload] Failed: {e}, re-queuing")
            upload_queue.put(item)
            time.sleep(UPLOAD_RETRY_DELAY)


def check_disk_space():
    usage = shutil.disk_usage(str(OUTPUT_DIR))
    return usage.free / (1024 ** 3) >= MIN_FREE_GB


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize NanoOWL
    print(f"[init] Loading NanoOWL engine from {ENGINE_PATH}")
    predictor = OwlPredictor(
        "google/owlvit-base-patch32",
        image_encoder_engine=ENGINE_PATH,
    )
    print("[init] NanoOWL loaded")

    print(f"[init] Encoding text prompts: {TEXT_PROMPTS}")
    text_encodings = predictor.encode_text(TEXT_PROMPTS)
    print("[init] Text encodings ready")

    # Initialize camera
    print(f"[init] Opening camera index {CAMERA_INDEX}")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Failed to open USB camera")
    print(f"[init] Camera opened: {FRAME_WIDTH}x{FRAME_HEIGHT}")

    # Upload thread
    upload_queue = queue.Queue()
    uploader = threading.Thread(target=upload_worker, args=(upload_queue,), daemon=True)
    uploader.start()

    # State machine
    state = "IDLE"
    consecutive_detect = 0
    consecutive_absent = 0
    video_writer = None
    current_clip_path = None
    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    frame_count = 0
    camera_fail_count = 0
    idle_counter = 0
    tracker = None

    print("[main] Starting detection loop (Ctrl+C to stop)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                camera_fail_count += 1
                if camera_fail_count > 100:
                    print("[main] Camera read failed 100 times, attempting reopen")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(CAMERA_INDEX)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                    camera_fail_count = 0
                time.sleep(0.01)
                continue
            camera_fail_count = 0

            # Throttle in IDLE: skip every other frame to reduce CPU/GPU load
            if state == "IDLE":
                idle_counter += 1
                if idle_counter % 3 != 0:
                    continue

            if state == "IDLE":
                detections = detect_bikes(predictor, frame, text_encodings)
                bike_detected = len(detections) > 0

                if bike_detected:
                    consecutive_detect += 1
                    if consecutive_detect >= START_FRAMES:
                        if not check_disk_space():
                            print("[main] Low disk space, skipping recording")
                            consecutive_detect = 0
                            continue

                        now = datetime.now()
                        date_str = now.strftime("%m-%d-%Y")
                        time_str = now.strftime("%I-%M-%S%p")
                        current_clip_path = str(OUTPUT_DIR / f"Biking-{date_str}-{time_str}.avi")
                        video_writer = cv2.VideoWriter(
                            current_clip_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT)
                        )
                        tracker = SimpleTracker()
                        state = "RECORDING"
                        consecutive_detect = 0
                        consecutive_absent = 0
                        frame_count = 0
                        detect_skip = 0
                        print(f"[main] RECORDING started: {current_clip_path}")
                else:
                    consecutive_detect = 0

            elif state == "RECORDING":
                # Write RAW frame every time (cheap, no annotation on Jetson)
                video_writer.write(frame)
                frame_count += 1

                # Only run detection every 5th frame to save GPU
                detect_skip += 1
                if detect_skip % 5 == 0:
                    detections = detect_bikes(predictor, frame, text_encodings)
                    bike_detected = len(detections) > 0
                    track_ids = tracker.update(detections, frame_count)

                    if not bike_detected:
                        consecutive_absent += 5
                    else:
                        consecutive_absent = 0
                else:
                    # Between detections, just keep recording
                    continue

                if consecutive_absent >= STOP_FRAMES:
                    # Finalize clip
                    video_writer.release()
                    video_writer = None
                    duration = frame_count / FPS
                    print(f"[main] RECORDING stopped: {frame_count} frames ({duration:.1f}s)")

                    # Build and save JSON report
                    clip_stem = Path(current_clip_path).stem
                    json_path = str(OUTPUT_DIR / f"{clip_stem}_report.json")
                    report = build_report(tracker, frame_count, Path(current_clip_path).name)
                    with open(json_path, "w") as f:
                        json.dump(report, f, indent=2)

                    upload_queue.put((current_clip_path, json_path))
                    current_clip_path = None
                    tracker = None
                    state = "IDLE"
                    consecutive_absent = 0

    except KeyboardInterrupt:
        print("\n[main] Shutting down...")
    finally:
        if video_writer is not None:
            video_writer.release()
            if current_clip_path and tracker:
                clip_stem = Path(current_clip_path).stem
                json_path = str(OUTPUT_DIR / f"{clip_stem}_report.json")
                report = build_report(tracker, frame_count, Path(current_clip_path).name)
                with open(json_path, "w") as f:
                    json.dump(report, f, indent=2)
                print(f"[main] Finalizing last clip: {current_clip_path}")
                upload_queue.put((current_clip_path, json_path))
        cap.release()
        upload_queue.put(None)
        uploader.join(timeout=10)
        print("[main] Done")


if __name__ == "__main__":
    main()
