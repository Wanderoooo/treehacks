"""
Laptop: receives video stream from Jetson, records when bike detected,
processes with YOLO + depth after recording stops.

Usage:
    python stream_receiver.py [JETSON_IP]

Default Jetson IP is read from ~/.ssh/config (Host jetson).
"""

import cv2
import sys
import struct
import socket
import time
import threading
import numpy as np
from datetime import datetime
from pathlib import Path

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


def process_clip(video_path, stem):
    """Run YOLO annotation + depth on a saved clip (background thread)."""
    try:
        import yaml
        from src.detector import BikeDetector
        from src.utils import SimpleTracker
        from src.annotator import VideoAnnotator
        from src.report_generator import ReportGenerator
        from src.depth_estimator import DepthEstimator

        config_path = "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        detector = BikeDetector(
            model_path=config["yolo"]["model"],
            confidence=config["yolo"]["confidence"],
            bike_class_id=config["yolo"]["bike_class_id"],
            device="auto" if config["performance"]["use_gpu"] else "cpu",
        )
        tracker = SimpleTracker(iou_threshold=0.3)
        annotator = VideoAnnotator(config)
        report_gen = ReportGenerator()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[process] Could not open: {video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or FPS
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        videos_dir = OUTPUT_DIR / "videos"
        reports_dir = OUTPUT_DIR / "reports"
        depth_dir = OUTPUT_DIR / "depth"
        videos_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        out_path = videos_dir / f"{stem}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        frame_idx = 0
        all_detections = []
        track_info = {}

        print(f"[process] Running YOLO on {video_path} ({total_frames} frames)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            bikes = detector.detect_bikes(frame)
            track_ids = tracker.update(bikes, frame_idx)

            bikes_with_data = []
            for (x1, y1, x2, y2, conf), tid in zip(bikes, track_ids):
                bikes_with_data.append({
                    "bbox": (x1, y1, x2, y2, conf),
                    "track_id": tid,
                    "lights": {"has_front": False, "has_rear": False},
                })

            all_detections.append({"frame": frame_idx, "bikes": bikes_with_data})
            annotated = annotator.annotate_frame(frame.copy(), bikes_with_data)
            out.write(annotated)
            frame_idx += 1

        cap.release()
        out.release()

        # Build report
        all_tracks = tracker.get_all_tracks()
        track_data = {}
        for tid, tdata in all_tracks.items():
            track_data[tid] = {
                "first_seen": tdata["first_seen"],
                "last_seen": tdata["last_seen"],
                "frame_history": tdata["frame_history"],
                "lights": {
                    "has_front_light": False,
                    "has_rear_light": False,
                    "front_detection_rate": 0.0,
                    "rear_detection_rate": 0.0,
                },
            }

        video_info = {
            "filename": Path(video_path).name,
            "fps": fps,
            "resolution": [w, h],
            "total_frames": total_frames,
            "processed_frames": frame_idx,
            "duration_seconds": round(total_frames / fps, 2),
        }

        report_path = reports_dir / f"{stem}_report.json"
        report = report_gen.generate_report(
            video_info=video_info,
            all_detections=all_detections,
            track_data=track_data,
            output_path=str(report_path),
        )

        print(f"[process] Annotated video: {out_path}")
        print(f"[process] Report: {report_path}")
        print(f"[process] Bikes detected: {len(all_tracks)}")

        # Depth estimation
        depth_config = config.get("depth_estimation", {"enabled": False})
        if depth_config.get("enabled", False):
            print(f"[depth] Running depth estimation...")
            estimator = DepthEstimator(depth_config)
            if estimator.enabled:
                depth_dir.mkdir(parents=True, exist_ok=True)
                cap2 = cv2.VideoCapture(str(out_path))
                depth_video_path = depth_dir / f"{stem}_depth.mp4"
                depth_out = cv2.VideoWriter(str(depth_video_path), fourcc, fps, (w, h))

                fidx = 0
                while True:
                    ret, frame = cap2.read()
                    if not ret:
                        break
                    depth_map = estimator.estimate_depth(frame)
                    if depth_map is not None:
                        colored = estimator.render_depth_colormap(depth_map)
                        depth_out.write(colored)
                        if fidx == 0:
                            cv2.imwrite(str(depth_dir / f"{stem}_depth.png"), colored)
                    fidx += 1

                cap2.release()
                depth_out.release()
                print(f"[depth] Depth video: {depth_video_path}")

    except Exception as e:
        print(f"[process] Error: {e}")
        import traceback
        traceback.print_exc()


def finalize_recording(video_writer, clip_path):
    """Release video writer and kick off background processing."""
    if video_writer is not None:
        video_writer.release()
    if clip_path:
        clip_stem = Path(clip_path).stem
        threading.Thread(
            target=process_clip,
            args=(clip_path, clip_stem),
            daemon=True,
        ).start()


# ── Main ─────────────────────────────────────────────────────────────────────

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

            # Reset state for this connection
            state = "IDLE"
            consecutive_detect = 0
            consecutive_absent = 0
            video_writer = None
            current_clip_path = None
            frame_count = 0
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")

            print("[main] Receiving stream (Ctrl+C to stop)")

            try:
                while True:
                    # Read header: 4 bytes size + 1 byte bike flag
                    header = recv_exact(sock, 5)
                    frame_size, bike_flag = struct.unpack(">IB", header)
                    bike_detected = bool(bike_flag)

                    # Read JPEG data
                    jpeg_data = recv_exact(sock, frame_size)
                    frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    if state == "IDLE":
                        if bike_detected:
                            consecutive_detect += 1
                            if consecutive_detect >= START_FRAMES:
                                now = datetime.now()
                                date_str = now.strftime("%m-%d-%Y")
                                time_str = now.strftime("%I-%M-%S%p")
                                stem = f"Biking-{date_str}-{time_str}"
                                current_clip_path = str(INPUT_DIR / f"{stem}.avi")
                                h, w = frame.shape[:2]
                                video_writer = cv2.VideoWriter(
                                    current_clip_path, fourcc, FPS, (w, h)
                                )
                                state = "RECORDING"
                                consecutive_detect = 0
                                consecutive_absent = 0
                                frame_count = 0
                                print(f"[main] RECORDING started: {current_clip_path}")
                        else:
                            consecutive_detect = 0

                    elif state == "RECORDING":
                        video_writer.write(frame)
                        frame_count += 1

                        if not bike_detected:
                            consecutive_absent += 1
                            if consecutive_absent >= STOP_FRAMES:
                                video_writer.release()
                                video_writer = None
                                duration = frame_count / FPS
                                print(f"[main] RECORDING stopped: {frame_count} frames ({duration:.1f}s)")

                                # Process in background
                                clip_path = current_clip_path
                                clip_stem = Path(clip_path).stem
                                thread = threading.Thread(
                                    target=process_clip,
                                    args=(clip_path, clip_stem),
                                    daemon=True,
                                )
                                thread.start()

                                current_clip_path = None
                                state = "IDLE"
                                consecutive_absent = 0
                        else:
                            consecutive_absent = 0

            except KeyboardInterrupt:
                raise
            except (ConnectionError, socket.timeout, OSError) as e:
                print(f"[stream] Disconnected: {e}, will reconnect...")
                sock.close()
                sock = None
                # Finalize any in-progress recording
                finalize_recording(video_writer, current_clip_path)
                video_writer = None
                current_clip_path = None
                time.sleep(delay)
                delay = min(delay * 2, MAX_RECONNECT_DELAY)
                continue

    except KeyboardInterrupt:
        print("\n[main] Shutting down...")
    finally:
        if video_writer is not None:
            finalize_recording(video_writer, current_clip_path)
        if sock:
            sock.close()
        print("[main] Done")


if __name__ == "__main__":
    main()
