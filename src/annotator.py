"""Video annotation utilities."""

import cv2
import numpy as np
from typing import List, Dict, Tuple


class VideoAnnotator:
    """Handles video frame annotation with bike detection results."""

    def __init__(self, config):
        """
        Initialize annotator.

        Args:
            config: Configuration dictionary
        """
        self.thickness = config['video']['annotation_thickness']
        self.font_scale = config['video']['annotation_font_scale']
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def annotate_frame(
        self,
        frame: np.ndarray,
        bikes_data: List[Dict]
    ) -> np.ndarray:
        """
        Annotate frame with bike detection, tracking, lights, and colors.

        Args:
            frame: Input frame
            bikes_data: List of bike data dictionaries with:
                - 'bbox': (x1, y1, x2, y2, conf)
                - 'track_id': int
                - 'lights': light detection results
                - 'color': color analysis results (optional)

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for bike_data in bikes_data:
            x1, y1, x2, y2, conf = bike_data['bbox']
            track_id = bike_data.get('track_id', 0)
            lights = bike_data.get('lights', {})
            color_data = bike_data.get('color', {})

            # Draw bounding box (color varies by track_id)
            box_color = self._get_track_color(track_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, self.thickness)

            # Build label
            label_parts = [f"Bike #{track_id}"]

            # Add color if available
            if color_data.get('primary_color') and color_data['primary_color'] != 'unknown':
                label_parts.append(f"{color_data['primary_color']}")

            # Add light status
            if lights.get('has_front') or lights.get('has_rear'):
                label_parts.append("[Lights on]")
            else:
                label_parts.append("[No lights]")

            label = " ".join(label_parts)

            # Draw label background
            label_size = cv2.getTextSize(label, self.font, self.font_scale, 1)[0]
            label_y = y1 - 10 if y1 > 30 else y1 + label_size[1] + 10

            cv2.rectangle(
                annotated,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0] + 10, label_y + 5),
                box_color,
                -1
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1 + 5, label_y),
                self.font,
                self.font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

            # Draw light indicators
            if lights.get('front_coords'):
                for lx, ly in lights['front_coords']:
                    cv2.circle(annotated, (lx, ly), 6, (0, 255, 0), 2)
                    cv2.circle(annotated, (lx, ly), 2, (0, 255, 0), -1)

            if lights.get('rear_coords'):
                for lx, ly in lights['rear_coords']:
                    cv2.circle(annotated, (lx, ly), 6, (0, 0, 255), 2)
                    cv2.circle(annotated, (lx, ly), 2, (0, 0, 255), -1)

            # Draw speed indicator bar below bbox
            speed_kmh = bike_data.get('speed_kmh')
            if speed_kmh is not None and speed_kmh > 0:
                bar_width = int(min(speed_kmh / 40.0, 1.0) * (x2 - x1))
                if speed_kmh < 15:
                    bar_color = (0, 255, 0)      # Green (slow)
                elif speed_kmh < 30:
                    bar_color = (0, 255, 255)    # Yellow (medium)
                else:
                    bar_color = (0, 0, 255)      # Red (fast)
                cv2.rectangle(annotated, (x1, y2 + 2), (x1 + bar_width, y2 + 6), bar_color, -1)

        return annotated

    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Get a consistent color for a track ID.

        Args:
            track_id: Track identifier

        Returns:
            BGR color tuple
        """
        # Generate distinct colors for different tracks
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 165, 255),  # Orange
            (255, 255, 0),  # Cyan
            (128, 0, 128),  # Purple
            (0, 128, 255),  # Orange-red
        ]

        return colors[(track_id - 1) % len(colors)]
