"""Brightness-based bike light detection."""

import cv2
import numpy as np
from typing import Dict, List, Tuple


class LightDetector:
    """Detects bike lights using brightness and shape analysis."""

    def __init__(self, config):
        """
        Initialize light detector.

        Args:
            config: Configuration dictionary with light_detection parameters
        """
        self.config = config
        self.brightness_threshold = config['brightness_threshold']
        self.min_light_area = config['min_light_area']
        self.max_light_area = config['max_light_area']
        self.circularity_threshold = config['circularity_threshold']
        self.aspect_ratio_range = config['aspect_ratio_range']
        self.roi_zones = config['roi_zones']

    def detect_bike_lights(self, bike_crop: np.ndarray, bike_bbox: Tuple[int, int, int, int] = None) -> Dict:
        """
        Detect lights on a bike using multi-stage filtering.

        Args:
            bike_crop: Cropped image of the bike (BGR format)
            bike_bbox: Optional bounding box (x1, y1, x2, y2) for additional context

        Returns:
            Dictionary with detection results:
            {
                'has_front_light': bool,
                'has_rear_light': bool,
                'front_light_coords': [(x, y), ...],
                'rear_light_coords': [(x, y), ...],
                'front_confidence': float,
                'rear_confidence': float
            }
        """
        if bike_crop is None or bike_crop.size == 0:
            return self._empty_result()

        h, w = bike_crop.shape[:2]

        # Step 1: Preprocessing
        gray = cv2.cvtColor(bike_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, self.brightness_threshold, 255, cv2.THRESH_BINARY)

        # Step 2: Define ROI zones (front and rear)
        front_roi_mask = self._create_roi_mask(w, h, 'front')
        rear_roi_mask = self._create_roi_mask(w, h, 'rear')

        # Step 3: Detect lights in each zone
        front_lights, front_conf = self._detect_lights_in_zone(
            binary, gray, front_roi_mask, bike_crop
        )
        rear_lights, rear_conf = self._detect_lights_in_zone(
            binary, gray, rear_roi_mask, bike_crop
        )

        return {
            'has_front_light': len(front_lights) > 0,
            'has_rear_light': len(rear_lights) > 0,
            'front_light_coords': front_lights,
            'rear_light_coords': rear_lights,
            'front_confidence': front_conf,
            'rear_confidence': rear_conf
        }

    def _create_roi_mask(self, width: int, height: int, zone: str) -> np.ndarray:
        """
        Create a binary mask for ROI zone (front or rear).

        Args:
            width: Image width
            height: Image height
            zone: 'front' or 'rear'

        Returns:
            Binary mask (255 in ROI, 0 elsewhere)
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        x_range = self.roi_zones[zone]['x_range']
        y_range = self.roi_zones[zone]['y_range']

        x1 = int(width * x_range[0])
        x2 = int(width * x_range[1])
        y1 = int(height * y_range[0])
        y2 = int(height * y_range[1])

        mask[y1:y2, x1:x2] = 255

        return mask

    def _detect_lights_in_zone(
        self,
        binary: np.ndarray,
        gray: np.ndarray,
        roi_mask: np.ndarray,
        original: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], float]:
        """
        Detect lights in a specific zone with multi-factor filtering.

        Args:
            binary: Binary thresholded image
            gray: Grayscale original image
            roi_mask: ROI zone mask
            original: Original color image (for debugging)

        Returns:
            Tuple of (light_coordinates, confidence_score)
        """
        # Apply ROI mask
        zone_binary = cv2.bitwise_and(binary, roi_mask)

        # Find contours
        contours, _ = cv2.findContours(zone_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_light_area or area > self.max_light_area:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue

            # Calculate circularity: 4π * area / perimeter²
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)

            if circularity < self.circularity_threshold:
                continue

            # Brightness validation
            roi = gray[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            mean_brightness = np.mean(roi)
            max_brightness = np.max(roi)

            # Must be very bright
            if mean_brightness < self.brightness_threshold * 0.9:
                continue
            if max_brightness < 240:  # Must have very bright pixels
                continue

            # Calculate confidence score
            confidence = self._calculate_confidence(
                area, circularity, aspect_ratio, mean_brightness, max_brightness
            )

            # Get center point
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx = x + w // 2
                cy = y + h // 2

            candidates.append({
                'coords': (cx, cy),
                'confidence': confidence,
                'area': area,
                'brightness': mean_brightness
            })

        # Sort by confidence and keep top 2
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        top_candidates = candidates[:2]

        # Extract coordinates and average confidence
        light_coords = [c['coords'] for c in top_candidates]
        avg_confidence = np.mean([c['confidence'] for c in top_candidates]) if top_candidates else 0.0

        return light_coords, float(avg_confidence)

    def _calculate_confidence(
        self,
        area: float,
        circularity: float,
        aspect_ratio: float,
        mean_brightness: float,
        max_brightness: float
    ) -> float:
        """
        Calculate confidence score for a light candidate.

        Args:
            area: Contour area
            circularity: Shape circularity (0-1)
            aspect_ratio: Width/height ratio
            mean_brightness: Mean pixel brightness
            max_brightness: Maximum pixel brightness

        Returns:
            Confidence score (0-1)
        """
        # Normalize area score (prefer 50-500 pixels)
        ideal_area = 200
        area_score = 1.0 - min(abs(area - ideal_area) / ideal_area, 1.0)

        # Circularity score (already 0-1, higher is better)
        shape_score = circularity

        # Aspect ratio score (prefer close to 1.0, i.e., square/circular)
        aspect_score = 1.0 - min(abs(aspect_ratio - 1.0), 1.0)

        # Brightness scores (normalized)
        mean_brightness_score = (mean_brightness - 180) / (255 - 180)  # 180-255 range
        mean_brightness_score = np.clip(mean_brightness_score, 0, 1)

        max_brightness_score = (max_brightness - 240) / (255 - 240)  # 240-255 range
        max_brightness_score = np.clip(max_brightness_score, 0, 1)

        # Weighted combination
        confidence = (
            0.2 * area_score +
            0.3 * shape_score +
            0.1 * aspect_score +
            0.2 * mean_brightness_score +
            0.2 * max_brightness_score
        )

        return np.clip(confidence, 0, 1)

    def _empty_result(self) -> Dict:
        """Return empty detection result."""
        return {
            'has_front_light': False,
            'has_rear_light': False,
            'front_light_coords': [],
            'rear_light_coords': [],
            'front_confidence': 0.0,
            'rear_confidence': 0.0
        }

    def visualize_detection(self, bike_crop: np.ndarray, detection_result: Dict) -> np.ndarray:
        """
        Visualize light detection results on bike crop (for debugging).

        Args:
            bike_crop: Original bike crop
            detection_result: Result from detect_bike_lights()

        Returns:
            Annotated image
        """
        vis = bike_crop.copy()

        # Draw front lights (green)
        for x, y in detection_result['front_light_coords']:
            cv2.circle(vis, (x, y), 8, (0, 255, 0), 2)
            cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)

        # Draw rear lights (red)
        for x, y in detection_result['rear_light_coords']:
            cv2.circle(vis, (x, y), 8, (0, 0, 255), 2)
            cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

        # Add text labels
        if detection_result['has_front_light']:
            cv2.putText(vis, f"Front: {detection_result['front_confidence']:.2f}",
                       (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if detection_result['has_rear_light']:
            cv2.putText(vis, f"Rear: {detection_result['rear_confidence']:.2f}",
                       (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return vis
