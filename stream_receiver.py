"""
Laptop: receives video stream from Jetson, records when bike detected,
processes with YOLO + depth after recording stops.

Usage:
    python stream_receiver.py [JETSON_IP]

Default Jetson IP is read from ~/.ssh/config (Host jetson).
"""

import cv2
import json
import sys
import struct
import socket
import time
import yaml
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from src.light_detector import LightDetector

# ── Configuration ────────────────────────────────────────────────────────────

JETSON_PORT = 9000
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Anti-flicker
START_FRAMES = 5
STOP_FRAMES = 30

# Reconnect
RECONNECT_DELAY = 1.0
MAX_RECONNECT_DELAY = 10.0
RECV_TIMEOUT = 10.0

OUTPUT_DIR = Path("./output")
INPUT_DIR = Path("./input")

# Shared latest frame for live feed (single frame, no accumulation)
_live_frame_lock = threading.Lock()
_live_frame_jpeg = None  # bytes: most recent JPEG-encoded frame


def get_live_frame():
    """Return the latest JPEG frame bytes (or None if no stream)."""
    with _live_frame_lock:
        return _live_frame_jpeg


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_jetson_ip():
    """Try to read Jetson IP from ~/.ssh/config."""
    ssh_config = Path.home() / ".ssh" / "config"
    if ssh_config.exists():
        lines = ssh_config.read_text().splitlines()
        in_jetson = False
        for line in lines:
            stripped = line.strip().lower()
            if stripped.startswith("host ") and "jetson" in stripped:
                in_jetson = True
            elif stripped.startswith("host "):
                in_jetson = False
            elif in_jetson and stripped.startswith("hostname"):
                return stripped.split()[-1]
    return None


def configure_socket(sock):
    """Enable TCP keepalive with aggressive timeouts (cross-platform)."""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    if sys.platform == "linux":
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 5)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 2)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
    elif sys.platform == "darwin":
        TCP_KEEPALIVE = 0x10
        sock.setsockopt(socket.IPPROTO_TCP, TCP_KEEPALIVE, 5)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 2)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)

    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(RECV_TIMEOUT)


def connect_to_jetson(jetson_ip):
    """Create and configure a TCP connection to the Jetson."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect((jetson_ip, JETSON_PORT))
    configure_socket(sock)
    return sock


def recv_exact(sock, n):
    """Receive exactly n bytes from socket."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed")
        buf += chunk
    return buf


def finalize_recording(video_writer):
    """Release video writer."""
    if video_writer is not None:
        video_writer.release()


def run_depth(video_path, stem):
    """Run depth estimation on a recorded video (background thread)."""
    try:
        from src.depth_estimator import DepthEstimator
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        depth_config = config.get("depth_estimation", {"enabled": False})
        if not depth_config.get("enabled", False):
            return
        estimator = DepthEstimator(depth_config)
        if not estimator.enabled:
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or FPS
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        depth_dir = OUTPUT_DIR / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        depth_path = depth_dir / f"{stem}_depth.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(str(depth_path), fourcc, fps, (w, h))

        fidx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            depth_map = estimator.estimate_depth(frame)
            if depth_map is not None:
                colored = estimator.render_depth_colormap(depth_map)
                out.write(colored)
                if fidx == 0:
                    cv2.imwrite(str(depth_dir / f"{stem}_depth.png"), colored)
            fidx += 1
        cap.release()
        out.release()
        print(f"[depth] Saved: {depth_path}")
    except Exception as e:
        print(f"[depth] Error: {e}")


def save_report(clip_path, frame_count, recording_data):
    """Generate and save a report JSON from accumulated recording data."""
    # stem is like "Biking-02-15-2026-06-30-21AM_annotated"
    # strip "_annotated" to match dashboard expectations
    full_stem = Path(clip_path).stem  # e.g. Biking-..._annotated
    stem = full_stem.replace("_annotated", "")
    reports_dir = OUTPUT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Count unique bikes from max simultaneous detections
    total_detections = sum(len(d["boxes"]) for d in recording_data)
    max_simultaneous = max((len(d["boxes"]) for d in recording_data), default=0)

    # Per-bike light tallies (by box index across frames)
    bike_front = [0] * max_simultaneous
    bike_rear = [0] * max_simultaneous
    bike_appearances = [0] * max_simultaneous

    for d in recording_data:
        for i, light in enumerate(d["lights"]):
            if i >= max_simultaneous:
                break
            bike_appearances[i] += 1
            if light.get("has_front_light", False):
                bike_front[i] += 1
            if light.get("has_rear_light", False):
                bike_rear[i] += 1

    # Build bike entries
    bikes = []
    bikes_with_front = 0
    bikes_with_rear = 0
    bikes_with_both = 0
    bikes_with_none = 0
    violations = []
    duration = round(frame_count / FPS, 2)

    for i in range(max_simultaneous):
        apps = max(bike_appearances[i], 1)
        has_front = bike_front[i] / apps > 0.3
        has_rear = bike_rear[i] / apps > 0.3
        has_both = has_front and has_rear

        if has_front:
            bikes_with_front += 1
        if has_rear:
            bikes_with_rear += 1
        if has_both:
            bikes_with_both += 1
        if not has_front and not has_rear:
            bikes_with_none += 1
            violations.append({
                "type": "NO_LIGHTS",
                "track_id": i + 1,
                "severity": "HIGH",
                "description": "Bike detected without front or rear lights",
            })
        elif not has_front:
            violations.append({
                "type": "MISSING_FRONT_LIGHT",
                "track_id": i + 1,
                "severity": "MEDIUM",
                "description": "Bike missing front light",
            })
        elif not has_rear:
            violations.append({
                "type": "MISSING_REAR_LIGHT",
                "track_id": i + 1,
                "severity": "MEDIUM",
                "description": "Bike missing rear light",
            })

        compliance_status = "COMPLIANT" if has_both else "NON-COMPLIANT"
        bikes.append({
            "track_id": i + 1,
            "first_seen_frame": 0,
            "last_seen_frame": frame_count - 1,
            "duration_seconds": duration,
            "total_appearances": bike_appearances[i],
            "color": {"primary_color": "unknown", "confidence": 0.0},
            "lights": {
                "has_front_light": has_front,
                "has_rear_light": has_rear,
                "has_both_lights": has_both,
                "front_light_detection_rate": round(bike_front[i] / apps, 2),
                "rear_light_detection_rate": round(bike_rear[i] / apps, 2),
                "compliance_status": compliance_status,
            },
        })

    compliance_rate = round(bikes_with_both / max(max_simultaneous, 1) * 100, 1)

    report = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "report_version": "1.0",
        },
        "video_metadata": {
            "location": {
                "lat": 37.4275,
                "lng": -122.1697,
                "name": "Palm Drive, Stanford, CA",
            },
            "time_of_day": "night",
        },
        "video_info": {
            "filename": Path(clip_path).name,
            "fps": FPS,
            "resolution": [FRAME_WIDTH, FRAME_HEIGHT],
            "total_frames": frame_count,
            "processed_frames": frame_count,
            "duration_seconds": duration,
        },
        "summary": {
            "total_bikes_detected": max_simultaneous,
            "bikes_with_front_lights": bikes_with_front,
            "bikes_with_rear_lights": bikes_with_rear,
            "bikes_with_both_lights": bikes_with_both,
            "bikes_with_no_lights": bikes_with_none,
            "bikes_missing_front_light": max_simultaneous - bikes_with_front,
            "bikes_missing_rear_light": max_simultaneous - bikes_with_rear,
            "compliance_rate": compliance_rate,
        },
        "violations": violations,
        "bikes": bikes,
        "processing_stats": {
            "total_processing_time_seconds": 0,
            "processed_frames": frame_count,
            "total_detections": total_detections,
            "average_detections_per_frame": round(total_detections / max(frame_count, 1), 2),
        },
    }

    report_path = reports_dir / f"{stem}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] Saved: {report_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def reencode_to_mp4(avi_path, real_fps, stem):
    """Re-encode MJPG .avi to H.264 .mp4, then run depth on the .mp4."""
    try:
        import os
        mp4_path = avi_path.replace(".avi", ".mp4")
        cap = cv2.VideoCapture(avi_path)
        if not cap.isOpened():
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = max(int(round(real_fps)), 1)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
        out.release()
        os.remove(avi_path)
        print(f"[reencode] {mp4_path} ({fps} fps)")

        # Now run depth on the finished .mp4
        run_depth(mp4_path, stem)
    except Exception as e:
        print(f"[reencode] Error: {e}")


def main():
    # Get Jetson IP
    if len(sys.argv) > 1:
        jetson_ip = sys.argv[1]
    else:
        jetson_ip = get_jetson_ip()
        if not jetson_ip:
            print("Usage: python stream_receiver.py <JETSON_IP>")
            print("Or configure 'Host jetson' in ~/.ssh/config")
            sys.exit(1)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load light detector once
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    light_det = LightDetector(config["light_detection"])

    # Recording state persists across reconnects
    state = "IDLE"
    consecutive_detect = 0
    consecutive_absent = 0
    video_writer = None
    current_clip_path = None
    frame_count = 0
    recording_data = []
    recording_start_time = 0.0
    last_recording_end = 0.0  # cooldown: no new recording within 5s
    RECORDING_COOLDOWN = 5.0
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # fast for real-time recording

    delay = RECONNECT_DELAY
    sock = None

    try:
        while True:
            # Connect (with retry)
            try:
                print(f"[stream] Connecting to Jetson at {jetson_ip}:{JETSON_PORT}...")
                sock = connect_to_jetson(jetson_ip)
                print(f"[stream] Connected!")
                delay = RECONNECT_DELAY
            except (ConnectionError, socket.timeout, OSError) as e:
                print(f"[stream] Connection failed: {e}, retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * 2, MAX_RECONNECT_DELAY)
                continue

            print("[main] Receiving stream (Ctrl+C to stop)")

            try:
                while True:
                    # Read header: 4B frame_size + 4B boxes_size + 1B bike flag
                    header = recv_exact(sock, 9)
                    frame_size, boxes_size, bike_flag = struct.unpack(">IIB", header)
                    bike_detected = bool(bike_flag)

                    # Read JPEG data + box coordinates
                    jpeg_data = recv_exact(sock, frame_size)
                    boxes_data = recv_exact(sock, boxes_size)
                    boxes = json.loads(boxes_data)

                    frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    # Run light detection on each detected bike box
                    for (x1, y1, x2, y2) in boxes:
                        crop = frame[y1:y2, x1:x2]
                        lights = light_det.detect_bike_lights(crop)
                        label = ""
                        if lights["has_front_light"]:
                            label += "F "
                        if lights["has_rear_light"]:
                            label += "R"
                        if not label:
                            label = "NO LIGHTS"
                        cv2.putText(frame, label.strip(), (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Update shared live frame for web UI
                    global _live_frame_jpeg
                    with _live_frame_lock:
                        _live_frame_jpeg = jpeg_data

                    if state == "IDLE":
                        if bike_detected:
                            consecutive_detect += 1
                            cooldown_ok = (time.time() - last_recording_end) >= RECORDING_COOLDOWN
                            if consecutive_detect >= START_FRAMES and cooldown_ok:
                                now = datetime.now()
                                date_str = now.strftime("%m-%d-%Y")
                                time_str = now.strftime("%I-%M-%S%p")
                                stem = f"Biking-{date_str}-{time_str}"
                                videos_dir = OUTPUT_DIR / "videos"
                                videos_dir.mkdir(parents=True, exist_ok=True)
                                current_clip_path = str(videos_dir / f"{stem}_annotated.avi")
                                h, w = frame.shape[:2]
                                video_writer = cv2.VideoWriter(
                                    current_clip_path, fourcc, FPS, (w, h)
                                )
                                state = "RECORDING"
                                consecutive_detect = 0
                                consecutive_absent = 0
                                frame_count = 0
                                recording_data = []
                                recording_start_time = time.time()
                                print(f"[main] RECORDING started: {current_clip_path}")
                        else:
                            consecutive_detect = 0

                    elif state == "RECORDING":
                        video_writer.write(frame)
                        frame_count += 1

                        # Accumulate light detection results for report
                        frame_lights = []
                        for (x1, y1, x2, y2) in boxes:
                            crop = frame[y1:y2, x1:x2]
                            frame_lights.append(light_det.detect_bike_lights(crop))
                        recording_data.append({"boxes": boxes, "lights": frame_lights})

                        if not bike_detected:
                            consecutive_absent += 1
                            if consecutive_absent >= STOP_FRAMES:
                                video_writer.release()
                                video_writer = None
                                elapsed = max(time.time() - recording_start_time, 0.1)
                                real_fps = frame_count / elapsed
                                print(f"[main] RECORDING stopped: {frame_count} frames in {elapsed:.1f}s ({real_fps:.1f} fps)")

                                save_report(current_clip_path, frame_count, recording_data)
                                # Re-encode to MP4 then depth in background
                                clip_stem = Path(current_clip_path).stem.replace("_annotated", "")
                                threading.Thread(
                                    target=reencode_to_mp4,
                                    args=(current_clip_path, real_fps, clip_stem),
                                    daemon=True,
                                ).start()
                                current_clip_path = None
                                recording_data = []
                                state = "IDLE"
                                consecutive_absent = 0
                                last_recording_end = time.time()
                        else:
                            consecutive_absent = 0

            except KeyboardInterrupt:
                raise
            except (ConnectionError, socket.timeout, OSError) as e:
                print(f"[stream] Disconnected: {e}, will reconnect...")
                sock.close()
                sock = None
                # Do NOT reset recording state — persist across reconnects
                time.sleep(delay)
                delay = min(delay * 2, MAX_RECONNECT_DELAY)
                continue

    except KeyboardInterrupt:
        print("\n[main] Shutting down...")
    finally:
        if video_writer is not None:
            video_writer.release()
            if current_clip_path and frame_count > 0:
                elapsed = max(time.time() - recording_start_time, 0.1)
                real_fps = frame_count / elapsed
                save_report(current_clip_path, frame_count, recording_data)
                clip_stem = Path(current_clip_path).stem.replace("_annotated", "")
                reencode_to_mp4(current_clip_path, real_fps, clip_stem)
        if sock:
            sock.close()
        print("[main] Done")


if __name__ == "__main__":
    main()
