"""Depth estimation using Depth Anything V2."""

import torch
import numpy as np
import cv2
from typing import Dict, Tuple, Optional


class DepthEstimator:
    """Estimates relative depth using Depth Anything V2 Small model."""

    def __init__(self, config: dict):
        """
        Initialize depth estimator.

        Args:
            config: depth_estimation section from config.yaml
        """
        self.enabled = config.get('enabled', True)
        self.model_name = config.get('model', 'depth-anything/Depth-Anything-V2-Small-hf')

        if not self.enabled:
            print("Depth estimation disabled in config.")
            self.pipe = None
            return

        try:
            from transformers import pipeline
            device_str = self._get_device()
            print(f"Loading Depth Anything V2 model: {self.model_name}")
            print(f"Depth estimation device: {device_str}")
            self.pipe = pipeline(
                task="depth-estimation",
                model=self.model_name,
                device=device_str
            )
            print("Depth estimator initialized successfully.")
        except Exception as e:
            print(f"Warning: Could not initialize depth estimator: {e}")
            self.pipe = None
            self.enabled = False

    def _get_device(self) -> str:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run depth estimation on a full frame.

        Args:
            frame: BGR numpy array (full frame)

        Returns:
            2D float32 array (H x W), normalized 0.0-1.0. None if disabled/error.
        """
        if not self.enabled or self.pipe is None:
            return None

        try:
            from PIL import Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            result = self.pipe(pil_image)
            depth_pil = result["depth"]  # PIL Image

            # Convert PIL depth image to numpy - handle different modes
            depth_map = np.array(depth_pil.convert("F"), dtype=np.float32)

            # Resize to match original frame if needed
            h, w = frame.shape[:2]
            if depth_map.shape[:2] != (h, w):
                depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

            # Normalize to 0-1
            d_min, d_max = depth_map.min(), depth_map.max()
            if d_max - d_min > 0:
                depth_map = (depth_map - d_min) / (d_max - d_min)
            else:
                depth_map = np.zeros_like(depth_map)

            return depth_map

        except Exception as e:
            print(f"Warning: Depth estimation failed: {e}")
            return None

    def get_depth_for_bbox(
        self,
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Dict[str, float]:
        """
        Get depth info for a bounding box.

        Args:
            depth_map: Normalized depth map (0.0-1.0)
            bbox: (x1, y1, x2, y2)

        Returns:
            Dict with centroid_depth and mean_depth
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]

        # Centroid depth (averaged over small region)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = 3
        ry1, ry2 = max(0, cy - r), min(h, cy + r + 1)
        rx1, rx2 = max(0, cx - r), min(w, cx + r + 1)
        region = depth_map[ry1:ry2, rx1:rx2]
        centroid_depth = float(np.mean(region)) if region.size > 0 else 0.0

        # Mean depth over full bbox
        bx1, by1 = max(0, x1), max(0, y1)
        bx2, by2 = min(w, x2), min(h, y2)
        bbox_region = depth_map[by1:by2, bx1:bx2]
        mean_depth = float(np.mean(bbox_region)) if bbox_region.size > 0 else 0.0

        return {
            'centroid_depth': round(centroid_depth, 4),
            'mean_depth': round(mean_depth, 4)
        }

    def render_depth_colormap(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Render a depth map as a colored BGR image for visualization.

        Args:
            depth_map: Normalized depth map (0.0-1.0), shape (H, W)

        Returns:
            Colored BGR image (H, W, 3)
        """
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
