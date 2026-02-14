"""Trajectory heatmap generation from bike centroids."""

import cv2
import numpy as np
from typing import List, Tuple


class HeatmapGenerator:
    """Accumulates bike centroids and generates heatmap visualizations."""

    def __init__(self, config: dict, frame_width: int, frame_height: int):
        """
        Initialize heatmap generator.

        Args:
            config: heatmap section from config.yaml
            frame_width: Video frame width
            frame_height: Video frame height
        """
        self.enabled = config.get('enabled', True)
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Config
        colormap_name = config.get('colormap', 'COLORMAP_JET')
        self.colormap = getattr(cv2, colormap_name, cv2.COLORMAP_JET)
        self.blur_kernel = config.get('gaussian_blur_kernel', 25)
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1
        self.overlay_alpha = config.get('overlay_alpha', 0.4)
        self.point_radius = config.get('point_radius', 5)
        self.save_standalone = config.get('save_standalone', True)
        self.generate_overlay_video = config.get('generate_overlay_video', True)

        # Accumulator
        self.accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)

        if self.enabled:
            print(f"Heatmap generator initialized ({frame_width}x{frame_height})")

    def add_centroid(self, cx: int, cy: int):
        """Add a single bike centroid to the accumulator."""
        if not self.enabled:
            return
        cv2.circle(self.accumulator, (int(cx), int(cy)), self.point_radius, 1.0, -1)

    def get_heatmap_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate a heatmap overlay on a given frame.

        Returns:
            Frame with semi-transparent heatmap overlay
        """
        if not self.enabled or self.accumulator.max() == 0:
            return frame

        heatmap_color = self._render_heatmap()
        return cv2.addWeighted(frame, 1.0 - self.overlay_alpha, heatmap_color, self.overlay_alpha, 0)

    def save_heatmap_image(self, output_path: str):
        """Save the final standalone heatmap image."""
        if not self.enabled or self.accumulator.max() == 0:
            print("Warning: No centroids accumulated, skipping heatmap save.")
            return

        heatmap_color = self._render_heatmap()
        cv2.imwrite(output_path, heatmap_color)
        print(f"Heatmap saved to: {output_path}")

    def save_heatmap_with_background(self, background_frame: np.ndarray, output_path: str):
        """Save heatmap overlaid on a background frame."""
        if not self.enabled or self.accumulator.max() == 0:
            return

        overlay = self.get_heatmap_overlay(background_frame)
        cv2.imwrite(output_path, overlay)
        print(f"Heatmap overlay image saved to: {output_path}")

    def _render_heatmap(self) -> np.ndarray:
        """Render the accumulator into a colored heatmap image."""
        blurred = cv2.GaussianBlur(self.accumulator, (self.blur_kernel, self.blur_kernel), 0)

        max_val = blurred.max()
        if max_val > 0:
            normalized = (blurred / max_val * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(blurred, dtype=np.uint8)

        heatmap_color = cv2.applyColorMap(normalized, self.colormap)

        # Make zero-areas black so overlay doesn't tint the entire frame
        mask = normalized == 0
        heatmap_color[mask] = [0, 0, 0]

        return heatmap_color
