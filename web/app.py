"""Flask server for the Bike Safety Dashboard."""

import sys
import os
import time
import threading
from flask import Flask, Response, jsonify, request, send_from_directory, render_template
from werkzeug.utils import secure_filename
from pathlib import Path
import json

app = Flask(__name__,
            template_folder="templates",
            static_folder="static")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
VIDEOS_DIR = OUTPUT_DIR / "videos"
DEPTH_DIR = OUTPUT_DIR / "depth"
HEATMAPS_DIR = OUTPUT_DIR / "heatmaps"
INPUT_DIR = PROJECT_ROOT / "input"

# Import agent and video processor
sys.path.insert(0, str(PROJECT_ROOT))
from agent import agent_chat, _build_seed_messages
from src.video_processor import VideoProcessor
from stream_receiver import get_live_frame, main as stream_receiver_main

# Shared conversation state (single-user; fine for demo)
_chat_messages = _build_seed_messages()

# Video processor (initialized once)
_processor = None
_processor_lock = threading.Lock()


def _get_processor():
    global _processor
    if _processor is None:
        with _processor_lock:
            if _processor is None:
                _processor = VideoProcessor(config_path=str(PROJECT_ROOT / "config.yaml"))
    return _processor


@app.route("/")
def index():
    return render_template("index.html")


def _find_file(directory, stem, suffix):
    """Find a file matching stem+suffix, checking multiple extensions for videos."""
    exact = directory / f"{stem}{suffix}"
    if exact.exists():
        return f"{stem}{suffix}"
    # For video files, also check .avi
    if suffix.endswith((".mp4", ".avi")):
        base = suffix.rsplit(".", 1)[0]
        for ext in (".mp4", ".avi"):
            candidate = directory / f"{stem}{base}{ext}"
            if candidate.exists():
                return f"{stem}{base}{ext}"
    return None


@app.route("/api/reports")
def list_reports():
    reports = []
    # Sort by modification time, newest first
    report_files = sorted(
        REPORTS_DIR.glob("*_report.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    for report_file in report_files:
        with open(report_file) as f:
            data = json.load(f)
        video_stem = report_file.stem.replace("_report", "")

        ann = _find_file(VIDEOS_DIR, video_stem, "_annotated.mp4")
        dep_vid = _find_file(DEPTH_DIR, video_stem, "_depth.mp4")

        data["_files"] = {
            "annotated_video": f"/media/videos/{ann}" if ann else None,
            "depth_video": f"/media/depth/{dep_vid}" if dep_vid else None,
            "depth_image": f"/media/depth/{video_stem}_depth.png"
                if (DEPTH_DIR / f"{video_stem}_depth.png").exists() else None,
            "heatmap": f"/media/heatmaps/{video_stem}_heatmap.png"
                if (HEATMAPS_DIR / f"{video_stem}_heatmap.png").exists() else None,
            "heatmap_overlay": f"/media/heatmaps/{video_stem}_heatmap_overlay.png"
                if (HEATMAPS_DIR / f"{video_stem}_heatmap_overlay.png").exists() else None,
            "original_video": f"/media/input/{video_stem}.mp4"
                if (INPUT_DIR / f"{video_stem}.mp4").exists() else None,
        }
        data["_video_stem"] = video_stem
        reports.append(data)
    return jsonify(reports)


@app.route("/media/videos/<path:filename>")
def serve_video(filename):
    return send_from_directory(str(VIDEOS_DIR), filename)


@app.route("/media/depth/<path:filename>")
def serve_depth(filename):
    return send_from_directory(str(DEPTH_DIR), filename)


@app.route("/media/heatmaps/<path:filename>")
def serve_heatmap(filename):
    return send_from_directory(str(HEATMAPS_DIR), filename)


@app.route("/media/input/<path:filename>")
def serve_input(filename):
    return send_from_directory(str(INPUT_DIR), filename)


@app.route("/api/live")
def live_feed():
    """MJPEG stream of the latest frame from the Jetson."""
    def generate():
        while True:
            frame = get_live_frame()
            if frame is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.033)  # ~30fps cap
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Receive video (+ optional report) from Jetson or manual upload."""
    if "video" not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    report_file = request.files.get("report")
    filename = secure_filename(video_file.filename)
    stem = Path(filename).stem

    if report_file:
        # Pre-processed from Jetson: save annotated video + report directly
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        INPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Save raw upload temporarily
        orig_ext = Path(filename).suffix or ".avi"
        raw_path = VIDEOS_DIR / f"{stem}_raw{orig_ext}"
        counter = 1
        while raw_path.exists() or (VIDEOS_DIR / f"{stem}_annotated.mp4").exists():
            stem = f"{Path(filename).stem}_{counter}"
            raw_path = VIDEOS_DIR / f"{stem}_raw{orig_ext}"
            counter += 1

        video_file.save(str(raw_path))

        report_path = REPORTS_DIR / f"{stem}_report.json"
        report_file.save(str(report_path))

        print(f"[upload] Jetson pre-processed: {raw_path.name}, {report_path.name}")

        # Re-encode to H.264 MP4 + run depth in background
        thread = threading.Thread(
            target=_reencode_and_depth,
            args=(str(raw_path), stem, str(report_path)),
            daemon=True,
        )
        thread.start()

        return jsonify({
            "status": "accepted",
            "filename": f"{stem}_annotated.mp4",
            "message": "Pre-processed video saved, re-encoding + depth started",
        }), 202
    else:
        # Manual upload: save to input/ and run full pipeline
        ext = Path(filename).suffix or ".avi"
        save_path = INPUT_DIR / f"{stem}{ext}"
        counter = 1
        while save_path.exists():
            save_path = INPUT_DIR / f"{stem}_{counter}{ext}"
            counter += 1

        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        video_file.save(str(save_path))

        thread = threading.Thread(
            target=_process_uploaded_video,
            args=(str(save_path),),
            daemon=True,
        )
        thread.start()

        return jsonify({
            "status": "accepted",
            "filename": save_path.name,
            "message": "Video saved and full processing started",
        }), 202


def _reencode_and_depth(raw_path, stem, report_path):
    """Re-encode MJPG/AVI to H.264 MP4 with YOLO annotation, then depth."""
    try:
        import cv2
        import yaml
        from src.detector import BikeDetector
        from src.utils import SimpleTracker
        from src.annotator import VideoAnnotator

        # Load config for YOLO + annotator
        config_path = str(PROJECT_ROOT / "config.yaml")
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

        cap = cv2.VideoCapture(raw_path)
        if not cap.isOpened():
            print(f"[reencode] Could not open: {raw_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = VIDEOS_DIR / f"{stem}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        frame_idx = 0
        all_detections = []
        track_info = {}
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO detection + tracking + annotation
            bikes = detector.detect_bikes(frame)
            track_ids = tracker.update(bikes, frame_idx)

            bikes_with_data = []
            for (x1, y1, x2, y2, conf), tid in zip(bikes, track_ids):
                bikes_with_data.append({
                    "bbox": (x1, y1, x2, y2, conf),
                    "track_id": tid,
                    "lights": {"has_front": False, "has_rear": False},
                })

            annotated = annotator.annotate_frame(frame.copy(), bikes_with_data)
            out.write(annotated)
            frame_idx += 1

        cap.release()
        out.release()

        # Update report with YOLO detection data
        try:
            import json as json_mod
            all_tracks = tracker.get_all_tracks()
            total_bikes = len(all_tracks)

            report_data = json_mod.load(open(report_path))
            report_data["summary"]["total_bikes_detected"] = total_bikes
            report_data["bikes"] = []
            for tid, tdata in sorted(all_tracks.items()):
                report_data["bikes"].append({
                    "track_id": tid,
                    "first_seen_frame": tdata["first_seen"],
                    "last_seen_frame": tdata["last_seen"],
                    "duration_seconds": round((tdata["last_seen"] - tdata["first_seen"] + 1) / fps, 2),
                    "total_appearances": len(tdata["frame_history"]),
                    "lights": {
                        "has_front_light": False,
                        "has_rear_light": False,
                        "has_both_lights": False,
                        "compliance_status": "UNKNOWN",
                    },
                })
            with open(report_path, "w") as f:
                json_mod.dump(report_data, f, indent=2)
        except Exception as e:
            print(f"[reencode] Warning: could not update report: {e}")

        # Remove raw file
        try:
            os.remove(raw_path)
        except OSError:
            pass

        print(f"[reencode] Annotated H.264 video saved: {out_path} ({frame_idx} frames)")

        # Now run depth on the annotated video
        _run_depth_only(str(out_path), stem, report_path)

    except Exception as e:
        print(f"[reencode] Error: {e}")


def _run_depth_only(video_path, stem, report_path):
    """Run only depth estimation on a video and update the report."""
    try:
        import cv2
        import yaml
        import numpy as np
        from src.depth_estimator import DepthEstimator

        config_path = str(PROJECT_ROOT / "config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        depth_config = config.get("depth_estimation", {"enabled": False})
        if not depth_config.get("enabled", False):
            print(f"[depth] Depth estimation disabled in config, skipping")
            return

        estimator = DepthEstimator(depth_config)
        if not estimator.enabled:
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[depth] Could not open video: {video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        DEPTH_DIR.mkdir(parents=True, exist_ok=True)
        depth_video_path = DEPTH_DIR / f"{stem}_depth.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        depth_out = cv2.VideoWriter(str(depth_video_path), fourcc, fps, (w, h))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            depth_map = estimator.estimate_depth(frame)
            if depth_map is not None:
                colored = estimator.render_depth_colormap(depth_map)
                depth_out.write(colored)
                if frame_idx == 0:
                    depth_img_path = DEPTH_DIR / f"{stem}_depth.png"
                    cv2.imwrite(str(depth_img_path), colored)
            frame_idx += 1

        cap.release()
        depth_out.release()
        print(f"[depth] Depth video saved: {depth_video_path}")

    except Exception as e:
        print(f"[depth] Error: {e}")


def _process_uploaded_video(video_path):
    """Run the full pipeline on an uploaded video (background thread)."""
    try:
        processor = _get_processor()
        results = processor.process_video(video_path, output_dir=str(OUTPUT_DIR))
        print(f"[upload] Processing complete: {video_path}")
        print(f"[upload]   Bikes detected: {results['report']['summary']['total_bikes_detected']}")
    except Exception as e:
        print(f"[upload] Error processing {video_path}: {e}")


@app.route("/api/chat", methods=["POST"])
def chat():
    global _chat_messages
    data = request.get_json()
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"reply": "Please enter a message."}), 400
    try:
        reply = agent_chat(user_msg, _chat_messages)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {e}"}), 500


@app.route("/api/chat/reset", methods=["POST"])
def chat_reset():
    global _chat_messages
    _chat_messages = _build_seed_messages()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Start stream receiver in background thread
    jetson_ip = sys.argv[1] if len(sys.argv) > 1 else None
    if jetson_ip:
        sys.argv = [sys.argv[0], jetson_ip]  # pass IP to stream_receiver
        stream_thread = threading.Thread(target=stream_receiver_main, daemon=True)
        stream_thread.start()
        print(f"[live] Stream receiver started for Jetson at {jetson_ip}")
    else:
        print("[live] No Jetson IP provided, live feed disabled. Usage: python web/app.py <JETSON_IP>")

    print(f"Bike Safety Dashboard: http://localhost:8081")
    print(f"Output directory: {OUTPUT_DIR}")
    app.run(debug=False, host="0.0.0.0", port=8081)
