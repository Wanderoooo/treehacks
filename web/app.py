"""Flask server for the Bike Safety Dashboard."""

from flask import Flask, jsonify, send_from_directory, render_template
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reports")
def list_reports():
    reports = []
    for report_file in sorted(REPORTS_DIR.glob("*_report.json")):
        with open(report_file) as f:
            data = json.load(f)
        video_stem = report_file.stem.replace("_report", "")

        data["_files"] = {
            "annotated_video": f"/media/videos/{video_stem}_annotated.mp4"
                if (VIDEOS_DIR / f"{video_stem}_annotated.mp4").exists() else None,
            "depth_video": f"/media/depth/{video_stem}_depth.mp4"
                if (DEPTH_DIR / f"{video_stem}_depth.mp4").exists() else None,
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


if __name__ == "__main__":
    print(f"Bike Safety Dashboard: http://localhost:8080")
    print(f"Output directory: {OUTPUT_DIR}")
    app.run(debug=True, port=8081)
