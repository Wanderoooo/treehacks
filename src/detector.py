"""YOLO-based bike detector."""

import torch
from ultralytics import YOLO
import numpy as np


class BikeDetector:
    """Wrapper for YOLO bike detection."""

    def __init__(self, model_path="yolo11n.pt", confidence=0.4, bike_class_id=1, device='auto'):
        """
        Initialize the bike detector.

        Args:
            model_path: Path to YOLO model weights
            confidence: Detection confidence threshold (0-1)
            bike_class_id: COCO class ID for bicycle (default: 1)
            device: Device to run inference on ('auto', 'cpu', 'cuda', 'mps')
        """
        self.confidence = confidence
        self.bike_class_id = bike_class_id
        self.device = self._get_device(device)

        print(f"Loading YOLO model: {model_path}")
        print(f"Using device: {self.device}")

        self.model = YOLO(model_path)
        self.model.to(self.device)

    def _get_device(self, device):
        """Auto-detect best available device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device

    def detect_bikes(self, frame):
        """
        Detect bikes in a frame.

        Args:
            frame: Input image (numpy array, BGR format)

        Returns:
            List of detections: [(x1, y1, x2, y2, confidence), ...]
            Empty list if no bikes detected
        """
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=[self.bike_class_id],  # Only detect bikes
            verbose=False
        )

        # Extract bike detections
        bikes = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # Get coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                # Double-check it's a bike (should be filtered by YOLO already)
                if cls == self.bike_class_id:
                    bikes.append((int(x1), int(y1), int(x2), int(y2), conf))

        return bikes

    def detect_bikes_batch(self, frames):
        """
        Detect bikes in multiple frames (batch processing).

        Args:
            frames: List of input images

        Returns:
            List of detection lists, one per frame
        """
        results = self.model.predict(
            frames,
            conf=self.confidence,
            classes=[self.bike_class_id],
            verbose=False
        )

        all_bikes = []
        for result in results:
            bikes = []
            if result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())

                    if cls == self.bike_class_id:
                        bikes.append((int(x1), int(y1), int(x2), int(y2), conf))
            all_bikes.append(bikes)

        return all_bikes
