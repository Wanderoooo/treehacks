# Bike Detection and Analysis System

A computer vision system that detects bikes in videos, identifies bike lights, and describes bike colors using YOLO11 and LLaVA vision language model.

## Features

- **Bike Detection**: Uses YOLO11 to detect bicycles in video frames
- **Light Detection**: Identifies front and rear bike lights using brightness-based heuristics
- **Color Analysis**: Describes bike colors using LLaVA VLM running locally via Ollama
- **Multi-bike Tracking**: Tracks multiple bikes across frames with unique IDs
- **Dual Output**: Generates both annotated videos and detailed JSON reports

## Requirements

- Python 3.9 or higher
- Ollama installed locally
- Optional: GPU (CUDA or Apple Silicon) for faster processing

## Installation

1. **Clone or navigate to project directory:**
   ```bash
   cd /Users/alissalol/treehacks
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install Ollama and LLaVA model:**
   - Download Ollama from https://ollama.ai/download
   - After installation, run:
     ```bash
     ollama pull llava:latest
     ```

5. **Test installation:**
   ```bash
   python main.py --help
   ```

## Usage

### Basic Usage

```bash
python main.py input/your_video.mp4
```

This will:
- Detect bikes in the video
- Process light detection (once implemented)
- Generate color analysis (once implemented)
- Save annotated video to `output/videos/`
- Save JSON report to `output/reports/`

### Advanced Options

```bash
# Specify output directory
python main.py input/video.mp4 -o /path/to/output

# Override confidence threshold
python main.py input/video.mp4 --confidence 0.5

# Use larger YOLO model (more accurate, slower)
python main.py input/video.mp4 --model yolo11s

# Skip video generation (faster)
python main.py input/video.mp4 --no-video

# Custom configuration file
python main.py input/video.mp4 -c custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:

- **YOLO settings**: Model variant, confidence threshold
- **Light detection**: Brightness thresholds, ROI zones, shape filters
- **Color analysis**: Sampling frequency, consensus parameters
- **Video output**: Annotation style, frame skipping
- **Performance**: GPU usage, batch processing

## Project Structure

```
treehacks/
├── main.py                   # CLI entry point
├── config.yaml               # Configuration parameters
├── requirements.txt          # Python dependencies
├── src/
│   ├── detector.py          # YOLO bike detection
│   ├── light_detector.py    # Light detection algorithm (Phase 2)
│   ├── color_analyzer.py    # LLaVA integration (Phase 3)
│   ├── video_processor.py   # Main pipeline
│   ├── annotator.py         # Video annotation (Phase 4)
│   ├── report_generator.py  # JSON reports (Phase 4)
│   └── utils.py             # Utilities and tracking (Phase 4)
├── input/                    # Place input videos here
└── output/
    ├── videos/              # Annotated videos
    └── reports/             # JSON reports
```

## Current Status

**Phase 1 (Foundation): ✅ Complete**
- Project structure created
- YOLO bike detection implemented
- Basic video processing pipeline functional
- CLI interface ready

**Phase 2 (Light Detection): ✅ Complete**
- Brightness-based light detection algorithm with multi-stage filtering
- ROI extraction (front/rear zones) and shape filtering
- Circularity and brightness validation
- Full integration with pipeline

**Phase 3 (Color Analysis): ✅ Complete**
- LLaVA/Ollama integration
- Sparse sampling (every 30th frame)
- Consensus voting for reliable color detection
- Color normalization

**Phase 4 (Tracking & Output): ✅ Complete**
- Multi-bike tracking with IoU matching
- Enhanced video annotation with track IDs, colors, and lights
- Comprehensive JSON report generation
- Per-track aggregation of light and color data

## Output Format

### Annotated Video
- Bounding boxes around detected bikes
- Light indicators (green for front, red for rear)
- Color labels
- Track IDs

### JSON Report
```json
{
  "video_info": {
    "filename": "input.mp4",
    "fps": 30,
    "total_frames": 900
  },
  "summary": {
    "total_bikes_detected": 3,
    "bikes_with_front_lights": 2,
    "bikes_with_rear_lights": 1
  },
  "bikes": [
    {
      "track_id": 1,
      "color": {"primary_color": "red", "confidence": 0.87},
      "lights": {"has_front_light": true, "has_rear_light": true}
    }
  ]
}
```

## Performance

Expected processing speed:
- **With GPU** (M1/M2 Mac or NVIDIA): 15-20 fps
- **CPU only**: 5-8 fps

For 1080p 30fps video:
- With GPU: ~1.5-2x slower than real-time
- With CPU: ~4-6x slower than real-time

## Troubleshooting

**"Ollama not found" error:**
- Make sure Ollama is installed and running
- Test with: `ollama list`

**"Model not found" error:**
- Download LLaVA: `ollama pull llava:latest`
- Download YOLO model (auto-downloads on first run)

**Slow processing:**
- Use smaller YOLO model (yolo11n)
- Enable frame skipping in config.yaml
- Reduce video resolution before processing

**GPU not detected:**
- Check CUDA installation (NVIDIA)
- Check torch.backends.mps (Apple Silicon)
- System will automatically fallback to CPU

## License

MIT License

## Acknowledgments

- YOLO11 by Ultralytics
- LLaVA vision language model
- Ollama for local LLM/VLM inference
