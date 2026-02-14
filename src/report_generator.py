"""JSON report generation."""

import json
from typing import Dict, List
from pathlib import Path
from collections import defaultdict
from datetime import datetime


class ReportGenerator:
    """Generates JSON reports from bike detection results."""

    def __init__(self):
        """Initialize report generator."""
        pass

    def generate_report(
        self,
        video_info: Dict,
        all_detections: List[Dict],
        track_data: Dict[int, Dict],
        output_path: str
    ):
        """
        Generate and save JSON report.

        Args:
            video_info: Video metadata
            all_detections: Frame-by-frame detection results
            track_data: Per-track aggregated data
            output_path: Path to save JSON report
        """
        # Build report structure
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
                'report_version': '1.0'
            },
            'video_info': video_info,
            'summary': self._generate_summary(track_data),
            'violations': self._generate_violations(track_data, video_info),
            'bikes': self._generate_bike_details(track_data, video_info),
            'processing_stats': self._generate_stats(video_info, all_detections)
        }

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def _generate_summary(self, track_data: Dict[int, Dict]) -> Dict:
        """Generate summary statistics."""
        total_bikes = len(track_data)
        bikes_with_front = sum(1 for t in track_data.values()
                               if t.get('lights', {}).get('has_front_light', False))
        bikes_with_rear = sum(1 for t in track_data.values()
                              if t.get('lights', {}).get('has_rear_light', False))
        bikes_with_both = sum(1 for t in track_data.values()
                              if (t.get('lights', {}).get('has_front_light', False) and
                                  t.get('lights', {}).get('has_rear_light', False)))
        bikes_with_no_lights = sum(1 for t in track_data.values()
                                   if (not t.get('lights', {}).get('has_front_light', False) and
                                       not t.get('lights', {}).get('has_rear_light', False)))
        bikes_missing_front = sum(1 for t in track_data.values()
                                 if not t.get('lights', {}).get('has_front_light', False))
        bikes_missing_rear = sum(1 for t in track_data.values()
                                if not t.get('lights', {}).get('has_rear_light', False))

        return {
            'total_bikes_detected': total_bikes,
            'bikes_with_front_lights': bikes_with_front,
            'bikes_with_rear_lights': bikes_with_rear,
            'bikes_with_both_lights': bikes_with_both,
            'bikes_with_no_lights': bikes_with_no_lights,
            'bikes_missing_front_light': bikes_missing_front,
            'bikes_missing_rear_light': bikes_missing_rear,
            'compliance_rate': round((bikes_with_both / max(total_bikes, 1)) * 100, 2)
        }

    def _generate_violations(self, track_data: Dict[int, Dict], video_info: Dict) -> List[Dict]:
        """Generate list of bikes with lighting violations for ticketing."""
        violations = []
        fps = video_info.get('fps', 25)

        for track_id, data in sorted(track_data.items()):
            lights_data = data.get('lights', {})
            has_front = lights_data.get('has_front_light', False)
            has_rear = lights_data.get('has_rear_light', False)

            # Determine if there's a violation
            if not has_front or not has_rear:
                first_seen_frame = data.get('first_seen', 0)
                last_seen_frame = data.get('last_seen', 0)
                duration_frames = last_seen_frame - first_seen_frame + 1

                # Convert to timestamps
                first_seen_time = round(first_seen_frame / fps, 2)
                last_seen_time = round(last_seen_frame / fps, 2)
                duration_seconds = round(duration_frames / fps, 2)

                # Determine violation type and severity
                missing_lights = []
                if not has_front:
                    missing_lights.append('front')
                if not has_rear:
                    missing_lights.append('rear')

                severity = 'HIGH' if not has_front and not has_rear else 'MEDIUM'
                violation_code = 'NO_LIGHTS' if not has_front and not has_rear else 'PARTIAL_LIGHTS'

                violation = {
                    'track_id': track_id,
                    'violation_type': violation_code,
                    'severity': severity,
                    'missing_lights': missing_lights,
                    'description': f"Bike missing {' and '.join(missing_lights)} light(s)",
                    'detection_confidence': {
                        'front_light_detection_rate': round(lights_data.get('front_detection_rate', 0.0), 2),
                        'rear_light_detection_rate': round(lights_data.get('rear_detection_rate', 0.0), 2)
                    },
                    'time_in_video': {
                        'first_seen_seconds': first_seen_time,
                        'last_seen_seconds': last_seen_time,
                        'duration_seconds': duration_seconds,
                        'first_seen_frame': first_seen_frame,
                        'last_seen_frame': last_seen_frame
                    },
                    'bike_details': {
                        'color': data.get('color', {}).get('primary_color', 'unknown'),
                        'color_confidence': round(data.get('color', {}).get('confidence', 0.0), 2),
                        'total_appearances': len(data.get('frame_history', []))
                    }
                }

                violations.append(violation)

        return violations

    def _generate_bike_details(self, track_data: Dict[int, Dict], video_info: Dict) -> List[Dict]:
        """Generate per-bike detailed information."""
        bikes = []
        fps = video_info.get('fps', 25)

        for track_id, data in sorted(track_data.items()):
            first_seen_frame = data.get('first_seen', 0)
            last_seen_frame = data.get('last_seen', 0)
            duration_frames = last_seen_frame - first_seen_frame + 1

            bike_info = {
                'track_id': track_id,
                'first_seen_frame': first_seen_frame,
                'last_seen_frame': last_seen_frame,
                'first_seen_timestamp_seconds': round(first_seen_frame / fps, 2),
                'last_seen_timestamp_seconds': round(last_seen_frame / fps, 2),
                'duration_seconds': round(duration_frames / fps, 2),
                'total_appearances': len(data.get('frame_history', [])),
            }

            # Color information
            color_data = data.get('color', {})
            if color_data:
                bike_info['color'] = {
                    'primary_color': color_data.get('primary_color', 'unknown'),
                    'confidence': round(color_data.get('confidence', 0.0), 2),
                    'sample_responses': color_data.get('all_responses', []),
                    'frames_analyzed': color_data.get('frames_analyzed', [])
                }

            # Light information
            lights_data = data.get('lights', {})
            if lights_data:
                has_front = lights_data.get('has_front_light', False)
                has_rear = lights_data.get('has_rear_light', False)

                bike_info['lights'] = {
                    'has_front_light': has_front,
                    'has_rear_light': has_rear,
                    'has_both_lights': has_front and has_rear,
                    'front_light_detection_rate': round(lights_data.get('front_detection_rate', 0.0), 2),
                    'rear_light_detection_rate': round(lights_data.get('rear_detection_rate', 0.0), 2),
                    'compliance_status': 'COMPLIANT' if (has_front and has_rear) else 'NON_COMPLIANT'
                }

            bikes.append(bike_info)

        return bikes

    def _generate_stats(self, video_info: Dict, all_detections: List[Dict]) -> Dict:
        """Generate processing statistics."""
        total_detections = sum(len(frame_data.get('bikes', [])) for frame_data in all_detections)
        processed_frames = len(all_detections)

        return {
            'total_processing_time_seconds': video_info.get('processing_time', 0),
            'processed_frames': processed_frames,
            'total_detections': total_detections,
            'average_detections_per_frame': round(total_detections / max(processed_frames, 1), 2)
        }
