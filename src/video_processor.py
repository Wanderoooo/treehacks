"""Main video processing pipeline."""

import cv2
import os
import yaml
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from .detector import BikeDetector
from .light_detector import LightDetector
from .color_analyzer import ColorAnalyzer
from .utils import SimpleTracker
from .annotator import VideoAnnotator
from .report_generator import ReportGenerator
from .depth_estimator import DepthEstimator
from .speed_estimator import SpeedEstimator
from .heatmap_generator import HeatmapGenerator


class VideoProcessor:
    """Main pipeline for processing videos."""

    def __init__(self, config_path="config.yaml"):
        """
        Initialize the video processor.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.detector = BikeDetector(
            model_path=self.config['yolo']['model'],
            confidence=self.config['yolo']['confidence'],
            bike_class_id=self.config['yolo']['bike_class_id'],
            device='auto' if self.config['performance']['use_gpu'] else 'cpu'
        )

        self.light_detector = LightDetector(self.config['light_detection'])
        self.color_analyzer = ColorAnalyzer(self.config['color_analysis'])
        self.tracker = SimpleTracker(iou_threshold=0.3)
        self.annotator = VideoAnnotator(self.config)
        self.report_generator = ReportGenerator()

        # Depth estimator (model loads once, no video-specific params)
        depth_config = self.config.get('depth_estimation', {'enabled': False})
        self.depth_estimator = DepthEstimator(depth_config) if depth_config.get('enabled', False) else None

        # Speed/heatmap configs stored for instantiation in process_video() (need fps/resolution)
        self.speed_config = self.config.get('speed_estimation', {'enabled': False})
        self.heatmap_config = self.config.get('heatmap', {'enabled': False})

        print("Video processor initialized with all components")

    def process_video(self, input_path, output_dir="./output"):
        """
        Process a video file with full pipeline.

        Args:
            input_path: Path to input video
            output_dir: Directory for output files

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()

        # Validate input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        # Create output directories
        output_dir = Path(output_dir)
        video_out_dir = output_dir / "videos"
        report_out_dir = output_dir / "reports"
        video_out_dir.mkdir(parents=True, exist_ok=True)
        report_out_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nVideo Info:")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.2f} seconds\n")

        # Prepare output video
        input_filename = Path(input_path).stem
        output_video_path = video_out_dir / f"{input_filename}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out_fps = fps if self.config['video']['output_fps'] == -1 else self.config['video']['output_fps']
        out = cv2.VideoWriter(str(output_video_path), fourcc, out_fps, (frame_width, frame_height))

        # Prepare depth video writer
        depth_video_path = None
        depth_out = None
        depth_config = self.config.get('depth_estimation', {})
        if self.depth_estimator and self.depth_estimator.enabled:
            depth_out_dir = output_dir / "depth"
            depth_out_dir.mkdir(parents=True, exist_ok=True)
            depth_video_path = depth_out_dir / f"{input_filename}_depth.mp4"
            depth_out = cv2.VideoWriter(str(depth_video_path), fourcc, out_fps, (frame_width, frame_height))

        # Initialize runtime components (need video properties)
        speed_estimator = SpeedEstimator(self.speed_config, fps) if self.speed_config.get('enabled', False) else None
        heatmap_generator = HeatmapGenerator(self.heatmap_config, frame_width, frame_height) if self.heatmap_config.get('enabled', False) else None

        # Track data across frames
        track_crops = defaultdict(dict)  # track_id -> {frame_idx: crop}
        track_lights = defaultdict(list)  # track_id -> [light_results]
        track_depths = defaultdict(list)  # track_id -> [depth_info_dicts]
        track_info = {}  # track_id -> metadata
        reference_frame = None

        frame_idx = 0
        processed_frames = 0
        total_detections = 0
        frame_skip = self.config['video']['frame_skip']
        sample_frequency = self.config['color_analysis']['sample_frequency']

        all_detections = []

        print("Processing frames...")
        with tqdm(total=total_frames, desc="Processing video", unit="frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Frame skipping
                if frame_idx % frame_skip == 0:
                    # Detect bikes
                    bikes = self.detector.detect_bikes(frame)
                    total_detections += len(bikes)

                    # Update tracker
                    track_ids = self.tracker.update(bikes, frame_idx)

                    # Run depth estimation once per frame
                    depth_map = None
                    if self.depth_estimator and self.depth_estimator.enabled:
                        depth_map = self.depth_estimator.estimate_depth(frame)
                        if depth_map is not None:
                            depth_colored = self.depth_estimator.render_depth_colormap(depth_map)
                            if depth_out is not None:
                                depth_out.write(depth_colored)
                            # Save first depth frame as standalone image
                            if processed_frames == 0:
                                depth_img_path = output_dir / "depth" / f"{input_filename}_depth.png"
                                cv2.imwrite(str(depth_img_path), depth_colored)
                                print(f"Depth image saved to: {depth_img_path}")

                    # Capture first frame for heatmap background
                    if reference_frame is None:
                        reference_frame = frame.copy()

                    # Process each bike
                    bikes_with_data = []
                    for (x1, y1, x2, y2, conf), track_id in zip(bikes, track_ids):
                        # Crop bike region
                        bike_crop = frame[y1:y2, x1:x2]

                        # Detect lights
                        light_result = self.light_detector.detect_bike_lights(bike_crop)

                        # Adjust light coordinates to frame coordinates
                        front_lights_frame = [(x + x1, y + y1) for x, y in light_result['front_light_coords']]
                        rear_lights_frame = [(x + x1, y + y1) for x, y in light_result['rear_light_coords']]

                        # Store light results for aggregation
                        track_lights[track_id].append(light_result)

                        # Store bike crop for color analysis (sampled)
                        if frame_idx % sample_frequency == 0 and bike_crop.size > 0:
                            track_crops[track_id][frame_idx] = bike_crop.copy()

                        # Depth estimation for this bike
                        depth_info = None
                        if depth_map is not None and self.depth_estimator:
                            depth_info = self.depth_estimator.get_depth_for_bbox(depth_map, (x1, y1, x2, y2))
                            track_depths[track_id].append(depth_info)

                        # Speed estimation
                        speed_kmh = None
                        if speed_estimator:
                            speed_kmh = speed_estimator.update(track_id, (x1, y1, x2, y2), frame_idx)

                        # Heatmap accumulation
                        if heatmap_generator:
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            heatmap_generator.add_centroid(cx, cy)

                        bikes_with_data.append({
                            'bbox': (x1, y1, x2, y2, conf),
                            'track_id': track_id,
                            'lights': {
                                'has_front': light_result['has_front_light'],
                                'has_rear': light_result['has_rear_light'],
                                'front_coords': front_lights_frame,
                                'rear_coords': rear_lights_frame
                            },
                            'depth': depth_info,
                            'speed_kmh': speed_kmh,
                        })

                    # Store detection data
                    all_detections.append({
                        'frame': frame_idx,
                        'bikes': bikes_with_data
                    })

                    # Annotate frame
                    annotated_frame = self.annotator.annotate_frame(frame.copy(), bikes_with_data)

                    # Write frame
                    out.write(annotated_frame)
                    processed_frames += 1
                else:
                    # Skip this frame, but write original
                    out.write(frame)

                frame_idx += 1
                pbar.update(1)

        # Cleanup video
        cap.release()
        out.release()
        if depth_out is not None:
            depth_out.release()
            print(f"Depth video saved to: {depth_video_path}")

        # Save heatmap outputs
        heatmap_out_dir = output_dir / "heatmaps"
        if heatmap_generator and heatmap_generator.enabled:
            heatmap_out_dir.mkdir(parents=True, exist_ok=True)
            if heatmap_generator.save_standalone:
                heatmap_generator.save_heatmap_image(str(heatmap_out_dir / f"{input_filename}_heatmap.png"))
            if reference_frame is not None:
                heatmap_generator.save_heatmap_with_background(
                    reference_frame, str(heatmap_out_dir / f"{input_filename}_heatmap_overlay.png")
                )

        # Post-processing: Color analysis per track
        print("\nAnalyzing bike colors...")
        track_colors = {}
        for track_id, crops in tqdm(track_crops.items(), desc="Color analysis"):
            if len(crops) > 0:
                color_result = self.color_analyzer.analyze_bike_colors_from_frames(
                    crops,
                    sample_frequency=1  # Already sampled, so use all collected crops
                )
                track_colors[track_id] = color_result

        # Aggregate track data
        print("\nAggregating results...")
        for track_id in self.tracker.get_all_tracks().keys():
            track_data = self.tracker.get_track_info(track_id)

            # Aggregate lights
            if track_id in track_lights and track_lights[track_id]:
                lights_list = track_lights[track_id]
                front_count = sum(1 for l in lights_list if l['has_front_light'])
                rear_count = sum(1 for l in lights_list if l['has_rear_light'])
                total_count = len(lights_list)

                track_info[track_id] = {
                    'first_seen': track_data['first_seen'],
                    'last_seen': track_data['last_seen'],
                    'frame_history': track_data['frame_history'],
                    'lights': {
                        'has_front_light': front_count > total_count * 0.5,
                        'has_rear_light': rear_count > total_count * 0.5,
                        'front_detection_rate': front_count / total_count,
                        'rear_detection_rate': rear_count / total_count
                    }
                }
            else:
                track_info[track_id] = {
                    'first_seen': track_data['first_seen'],
                    'last_seen': track_data['last_seen'],
                    'frame_history': track_data['frame_history'],
                    'lights': {
                        'has_front_light': False,
                        'has_rear_light': False,
                        'front_detection_rate': 0.0,
                        'rear_detection_rate': 0.0
                    }
                }

            # Add color data
            if track_id in track_colors:
                track_info[track_id]['color'] = track_colors[track_id]

            # Add depth data
            if track_id in track_depths and track_depths[track_id]:
                depth_samples = track_depths[track_id]
                avg_depth = np.mean([d['centroid_depth'] for d in depth_samples])
                track_info[track_id]['depth'] = {
                    'avg_centroid_depth': round(float(avg_depth), 4),
                    'samples': len(depth_samples)
                }

            # Add speed data
            if speed_estimator:
                track_info[track_id]['speed'] = speed_estimator.get_track_speed_summary(track_id)

        # Generate JSON report
        processing_time = time.time() - start_time
        output_report_path = report_out_dir / f"{input_filename}_report.json"

        video_info = {
            'filename': Path(input_path).name,
            'duration_seconds': total_frames / fps,
            'fps': fps,
            'resolution': [frame_width, frame_height],
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'processing_time': round(processing_time, 2)
        }

        # Build video metadata (location, time of day)
        metadata_config = self.config.get('video_metadata', {})
        time_of_day = metadata_config.get('time_of_day', 'auto')
        if time_of_day == 'auto' and reference_frame is not None:
            avg_brightness = np.mean(cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY))
            if avg_brightness < 80:
                time_of_day = 'night'
            elif avg_brightness < 160:
                time_of_day = 'dusk/dawn'
            else:
                time_of_day = 'day'
        video_metadata = {
            'location': metadata_config.get('location', {}),
            'time_of_day': time_of_day,
            'recording_timestamp': metadata_config.get('recording_timestamp')
        }

        report = self.report_generator.generate_report(
            video_info=video_info,
            all_detections=all_detections,
            track_data=track_info,
            output_path=str(output_report_path),
            video_metadata=video_metadata
        )

        # Print summary
        print(f"\n{'='*60}")
        print("Processing Complete!")
        print(f"{'='*60}")
        print(f"Time elapsed: {processing_time:.2f} seconds")
        print(f"Processing speed: {processed_frames/processing_time:.2f} fps")
        print(f"\nResults:")
        print(f"  Total bikes detected: {report['summary']['total_bikes_detected']}")
        print(f"  Bikes with front lights: {report['summary']['bikes_with_front_lights']}")
        print(f"  Bikes with rear lights: {report['summary']['bikes_with_rear_lights']}")
        print(f"  Bikes with both lights: {report['summary']['bikes_with_both_lights']}")
        print(f"\nOutputs:")
        print(f"  Annotated video: {output_video_path}")
        print(f"  JSON report: {output_report_path}")
        if heatmap_generator and heatmap_generator.enabled:
            print(f"  Heatmap outputs: {heatmap_out_dir}")
        if speed_estimator:
            print(f"  Speed estimation: enabled")
        if depth_video_path:
            print(f"  Depth video: {depth_video_path}")
        print(f"{'='*60}\n")

        return {
            'input_path': input_path,
            'output_video': str(output_video_path),
            'output_report': str(output_report_path),
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'total_detections': total_detections,
            'processing_time': processing_time,
            'report': report
        }
