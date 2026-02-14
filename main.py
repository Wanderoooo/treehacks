#!/usr/bin/env python3
"""
Bike Detection and Analysis System

Main entry point for processing videos to detect bikes, identify lights,
and analyze colors using YOLO11 and LLaVA.
"""

import argparse
import sys
import os
from pathlib import Path

from src.video_processor import VideoProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Bike Detection and Analysis System - Detect bikes, lights, and colors in videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input/bike_video.mp4
  python main.py input/bike_video.mp4 -o /path/to/output
  python main.py input/bike_video.mp4 --confidence 0.5
  python main.py input/bike_video.mp4 --no-video
        """
    )

    parser.add_argument(
        "input_video",
        help="Path to input video file"
    )

    parser.add_argument(
        "-o", "--output-dir",
        default="./output",
        help="Output directory for results (default: ./output)"
    )

    parser.add_argument(
        "-c", "--config",
        default="./config.yaml",
        help="Path to configuration file (default: ./config.yaml)"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        help="Override YOLO confidence threshold (0-1)"
    )

    parser.add_argument(
        "--brightness",
        type=int,
        help="Override brightness threshold for light detection (0-255)"
    )

    parser.add_argument(
        "--model",
        choices=["yolo11n", "yolo11s", "yolo11m", "yolo11l"],
        help="YOLO model variant to use (n=nano, s=small, m=medium, l=large)"
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip annotated video generation (faster, JSON only)"
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input_video):
        print(f"Error: Input video not found: {args.input_video}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Bike Detection and Analysis System")
    print("=" * 60)

    try:
        # Initialize processor
        processor = VideoProcessor(config_path=args.config)

        # Apply CLI overrides if provided
        if args.confidence:
            processor.config['yolo']['confidence'] = args.confidence
            processor.detector.confidence = args.confidence
            print(f"Override: confidence = {args.confidence}")

        if args.brightness:
            processor.config['light_detection']['brightness_threshold'] = args.brightness
            print(f"Override: brightness_threshold = {args.brightness}")

        if args.model:
            print(f"Override: model = {args.model}.pt")
            # Note: Would need to reload detector with new model

        # Process video
        results = processor.process_video(args.input_video, args.output_dir)

        print("\n" + "=" * 60)
        print("Processing Summary")
        print("=" * 60)
        print(f"Input: {results['input_path']}")
        print(f"Output video: {results['output_video']}")
        print(f"Total frames: {results['total_frames']}")
        print(f"Processed frames: {results['processed_frames']}")
        print(f"Total detections: {results['total_detections']}")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
