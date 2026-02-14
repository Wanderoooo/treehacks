"""Utility functions for tracking and helpers."""

import numpy as np
from typing import List, Tuple, Dict


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.

    Args:
        box1: (x1, y1, x2, y2) coordinates
        box2: (x1, y1, x2, y2) coordinates

    Returns:
        IoU score (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


class SimpleTracker:
    """Simple IoU-based tracker for multi-bike tracking."""

    def __init__(self, iou_threshold: float = 0.3):
        """
        Initialize tracker.

        Args:
            iou_threshold: Minimum IoU for matching bikes across frames
        """
        self.iou_threshold = iou_threshold
        self.next_track_id = 1
        self.active_tracks: Dict[int, Dict] = {}  # track_id -> track_info
        self.max_disappeared = 30  # Max frames a track can be missing before deletion

    def update(self, detections: List[Tuple[int, int, int, int, float]], frame_idx: int) -> List[int]:
        """
        Update tracker with new detections.

        Args:
            detections: List of (x1, y1, x2, y2, conf) detections
            frame_idx: Current frame index

        Returns:
            List of track IDs corresponding to each detection
        """
        if not detections:
            # Increment disappeared counter for all tracks
            for track_id in list(self.active_tracks.keys()):
                self.active_tracks[track_id]['disappeared'] += 1
                if self.active_tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.active_tracks[track_id]
            return []

        detection_boxes = [det[:4] for det in detections]

        if not self.active_tracks:
            # First detections - create new tracks
            track_ids = []
            for det in detections:
                track_id = self._create_new_track(det, frame_idx)
                track_ids.append(track_id)
            return track_ids

        # Match detections to existing tracks using IoU
        matches = self._match_detections_to_tracks(detection_boxes)

        track_ids = []
        matched_track_ids = set()

        for det_idx, det in enumerate(detections):
            if det_idx in matches:
                # Match found
                track_id = matches[det_idx]
                self._update_track(track_id, det, frame_idx)
                track_ids.append(track_id)
                matched_track_ids.add(track_id)
            else:
                # No match - create new track
                track_id = self._create_new_track(det, frame_idx)
                track_ids.append(track_id)

        # Mark unmatched tracks as disappeared
        for track_id in list(self.active_tracks.keys()):
            if track_id not in matched_track_ids:
                self.active_tracks[track_id]['disappeared'] += 1
                if self.active_tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.active_tracks[track_id]

        return track_ids

    def _match_detections_to_tracks(self, detection_boxes: List[Tuple]) -> Dict[int, int]:
        """
        Match detections to tracks using IoU.

        Args:
            detection_boxes: List of detection bounding boxes

        Returns:
            Dictionary mapping detection_idx to track_id
        """
        if not self.active_tracks or not detection_boxes:
            return {}

        # Build IoU matrix
        track_ids = list(self.active_tracks.keys())
        track_boxes = [self.active_tracks[tid]['bbox'][:4] for tid in track_ids]

        iou_matrix = np.zeros((len(detection_boxes), len(track_boxes)))

        for i, det_box in enumerate(detection_boxes):
            for j, track_box in enumerate(track_boxes):
                iou_matrix[i, j] = calculate_iou(det_box, track_box)

        # Greedy matching (could use Hungarian algorithm for optimal matching)
        matches = {}
        used_tracks = set()

        # Sort by IoU score (highest first)
        indices = np.argsort(-iou_matrix.ravel())

        for idx in indices:
            det_idx = idx // len(track_boxes)
            track_idx = idx % len(track_boxes)
            iou = iou_matrix[det_idx, track_idx]

            if iou < self.iou_threshold:
                break  # No more valid matches

            if det_idx not in matches and track_idx not in used_tracks:
                matches[det_idx] = track_ids[track_idx]
                used_tracks.add(track_idx)

        return matches

    def _create_new_track(self, detection: Tuple, frame_idx: int) -> int:
        """Create a new track."""
        track_id = self.next_track_id
        self.next_track_id += 1

        self.active_tracks[track_id] = {
            'bbox': detection,
            'first_seen': frame_idx,
            'last_seen': frame_idx,
            'disappeared': 0,
            'frame_history': [frame_idx],
            'bbox_history': [detection[:4]]
        }

        return track_id

    def _update_track(self, track_id: int, detection: Tuple, frame_idx: int):
        """Update an existing track."""
        self.active_tracks[track_id]['bbox'] = detection
        self.active_tracks[track_id]['last_seen'] = frame_idx
        self.active_tracks[track_id]['disappeared'] = 0
        self.active_tracks[track_id]['frame_history'].append(frame_idx)
        self.active_tracks[track_id]['bbox_history'].append(detection[:4])

    def get_track_info(self, track_id: int) -> Dict:
        """Get information about a track."""
        return self.active_tracks.get(track_id, None)

    def get_all_tracks(self) -> Dict[int, Dict]:
        """Get all active tracks."""
        return self.active_tracks.copy()
