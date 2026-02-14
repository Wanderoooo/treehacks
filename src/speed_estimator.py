"""Speed estimation from bounding box centroid displacement."""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple


class SpeedEstimator:
    """Estimates bike speed using centroid displacement between frames."""

    def __init__(self, config: dict, fps: int):
        """
        Initialize speed estimator.

        Args:
            config: speed_estimation section from config.yaml
            fps: Video frames per second
        """
        self.enabled = config.get('enabled', True)
        self.pixels_per_meter = config.get('pixels_per_meter', 50.0)
        self.smoothing_window = config.get('smoothing_window', 7)
        self.min_displacement = config.get('min_displacement', 2.0)
        self.max_speed_kmh = config.get('max_speed_kmh', 80.0)
        self.fps = fps

        # Per-track state
        self.track_centroids: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.smoothing_window + 1)
        )
        self.track_speeds: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.smoothing_window)
        )
        self.track_speed_history: Dict[int, list] = defaultdict(list)

        if self.enabled:
            print(f"Speed estimator initialized (pixels_per_meter={self.pixels_per_meter}, "
                  f"smoothing={self.smoothing_window}, fps={self.fps})")

    def update(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        frame_idx: int
    ) -> Optional[float]:
        """
        Update centroid for a track and compute current speed.

        Args:
            track_id: Unique track identifier
            bbox: (x1, y1, x2, y2) bounding box
            frame_idx: Current frame index

        Returns:
            Smoothed speed in km/h, or None if not enough data yet
        """
        if not self.enabled:
            return None

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        centroids = self.track_centroids[track_id]
        centroids.append((frame_idx, cx, cy))

        if len(centroids) < 2:
            return None

        # Compute instantaneous speed from last two entries
        prev_frame, prev_cx, prev_cy = centroids[-2]
        curr_frame, curr_cx, curr_cy = centroids[-1]

        frame_delta = curr_frame - prev_frame
        if frame_delta <= 0:
            return None

        # Pixel displacement
        dx = curr_cx - prev_cx
        dy = curr_cy - prev_cy
        pixel_disp = np.sqrt(dx**2 + dy**2)

        # Noise filter
        if pixel_disp < self.min_displacement:
            pixel_disp = 0.0

        # Convert pixels -> meters -> km/h
        meters = pixel_disp / self.pixels_per_meter
        time_delta = frame_delta / self.fps
        mps = meters / time_delta if time_delta > 0 else 0.0
        kmh = min(mps * 3.6, self.max_speed_kmh)

        # Add to smoothing window
        self.track_speeds[track_id].append(kmh)
        self.track_speed_history[track_id].append(kmh)

        # Return moving average
        speeds = self.track_speeds[track_id]
        return round(sum(speeds) / len(speeds), 1)

    def get_track_speed_summary(self, track_id: int) -> Dict:
        """
        Get speed summary for a track (for report generation).

        Returns:
            Dict with avg_speed_kmh, max_speed_kmh, min_speed_kmh, speed_samples
        """
        history = self.track_speed_history.get(track_id, [])
        if not history:
            return {
                'avg_speed_kmh': 0.0,
                'max_speed_kmh': 0.0,
                'min_speed_kmh': 0.0,
                'speed_samples': 0
            }

        return {
            'avg_speed_kmh': round(float(np.mean(history)), 1),
            'max_speed_kmh': round(float(np.max(history)), 1),
            'min_speed_kmh': round(float(np.min(history)), 1),
            'speed_samples': len(history)
        }
