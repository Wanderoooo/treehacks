"""LLaVA-based color analysis using Ollama."""

import os
import tempfile
from typing import Dict, List
from collections import Counter
import cv2
import numpy as np

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama package not installed. Color analysis will be disabled.")


class ColorAnalyzer:
    """Analyzes bike colors using LLaVA vision language model via Ollama."""

    def __init__(self, config):
        """
        Initialize color analyzer.

        Args:
            config: Configuration dictionary with color_analysis parameters
        """
        self.config = config
        self.ollama_model = config['ollama_model']
        self.prompt = config['prompt']
        self.temperature = config['temperature']
        self.timeout = config['timeout']

        if not OLLAMA_AVAILABLE:
            print("Warning: Ollama not available. Skipping color analysis.")
            self.enabled = False
            return

        # Check if Ollama is running and model is available
        try:
            models = ollama.list()
            model_names = [m['name'] for m in models.get('models', [])]
            if not any(self.ollama_model in name for name in model_names):
                print(f"Warning: Model '{self.ollama_model}' not found in Ollama.")
                print(f"Available models: {model_names}")
                print("Run 'ollama pull llava:latest' to install.")
                self.enabled = False
            else:
                self.enabled = True
                print(f"Color analyzer initialized with model: {self.ollama_model}")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
            print("Make sure Ollama is running.")
            self.enabled = False

    def analyze_bike_colors_from_frames(
        self,
        frame_crops: Dict[int, np.ndarray],
        sample_frequency: int = 30
    ) -> Dict:
        """
        Analyze bike color from multiple frame crops using consensus.

        Args:
            frame_crops: Dictionary mapping frame_idx to cropped bike images
            sample_frequency: Process every Nth frame

        Returns:
            Dictionary with color analysis results:
            {
                'primary_color': str,
                'confidence': float,
                'all_responses': [str, ...],
                'frames_analyzed': [int, ...]
            }
        """
        if not self.enabled:
            return self._empty_result()

        # Sample frames
        sampled_frames = sorted(frame_crops.keys())[::sample_frequency]

        if not sampled_frames:
            return self._empty_result()

        # Limit to consensus window (e.g., 3-5 samples)
        max_samples = self.config.get('consensus_window', 3)
        sampled_frames = sampled_frames[:max_samples]

        # Query LLaVA for each sample
        responses = []
        analyzed_frames = []

        for frame_idx in sampled_frames:
            crop = frame_crops[frame_idx]

            # Prepare image
            prepared_crop = self._prepare_image(crop)

            # Query Ollama
            color_desc = self._query_ollama_for_color(prepared_crop)

            if color_desc:
                responses.append(color_desc)
                analyzed_frames.append(frame_idx)

        if not responses:
            return self._empty_result()

        # Consensus voting
        primary_color, confidence = self._consensus_vote(responses)

        return {
            'primary_color': primary_color,
            'confidence': confidence,
            'all_responses': responses,
            'frames_analyzed': analyzed_frames
        }

    def _prepare_image(self, crop: np.ndarray, max_size: int = 512) -> np.ndarray:
        """
        Prepare bike crop for VLM analysis.

        Args:
            crop: Bike crop image
            max_size: Maximum dimension size

        Returns:
            Prepared image
        """
        h, w = crop.shape[:2]

        # Add 10% padding
        pad_h = int(h * 0.1)
        pad_w = int(w * 0.1)
        padded = cv2.copyMakeBorder(
            crop,
            pad_h, pad_h, pad_w, pad_w,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        # Resize to max_size maintaining aspect ratio
        h, w = padded.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(padded, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = padded

        return resized

    def _query_ollama_for_color(self, image: np.ndarray) -> str:
        """
        Query LLaVA via Ollama for color description.

        Args:
            image: Prepared bike crop

        Returns:
            Color description string, or empty string on error
        """
        if not self.enabled:
            return ""

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])

        try:
            # Query Ollama
            response = ollama.generate(
                model=self.ollama_model,
                prompt=self.prompt,
                images=[tmp_path],
                stream=False,
                options={
                    'temperature': self.temperature,
                    'num_predict': 10  # Limit response length
                }
            )

            color_desc = response['response'].strip().lower()

            # Clean up response (remove common prefixes/suffixes)
            color_desc = color_desc.replace('the bicycle is', '').strip()
            color_desc = color_desc.replace('the bike is', '').strip()
            color_desc = color_desc.replace('color:', '').strip()
            color_desc = color_desc.rstrip('.')

            return color_desc

        except Exception as e:
            print(f"Warning: Ollama query failed: {e}")
            return ""

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

    def _consensus_vote(self, responses: List[str]) -> tuple:
        """
        Perform consensus voting on color responses.

        Args:
            responses: List of color descriptions from LLaVA

        Returns:
            Tuple of (primary_color, confidence)
        """
        if not responses:
            return "unknown", 0.0

        # Normalize colors
        normalized = [self._normalize_color(r) for r in responses]

        # Count occurrences
        counts = Counter(normalized)
        most_common = counts.most_common(1)[0]

        primary_color = most_common[0]
        vote_count = most_common[1]
        confidence = vote_count / len(responses)

        return primary_color, confidence

    def _normalize_color(self, color_desc: str) -> str:
        """
        Normalize color descriptions to standard colors.

        Args:
            color_desc: Raw color description

        Returns:
            Normalized color name
        """
        color_desc = color_desc.lower().strip()

        # Color mapping
        color_map = {
            'red': ['red', 'crimson', 'scarlet', 'burgundy', 'maroon'],
            'blue': ['blue', 'navy', 'azure', 'cobalt', 'teal', 'cyan'],
            'green': ['green', 'lime', 'olive', 'forest green', 'emerald'],
            'yellow': ['yellow', 'gold', 'golden', 'amber'],
            'orange': ['orange', 'tangerine'],
            'black': ['black', 'dark', 'ebony'],
            'white': ['white', 'pearl', 'ivory', 'cream'],
            'gray': ['gray', 'grey', 'silver', 'metallic'],
            'brown': ['brown', 'tan', 'beige', 'bronze'],
            'pink': ['pink', 'rose', 'magenta'],
            'purple': ['purple', 'violet', 'lavender']
        }

        # Try to match to standard color
        for standard_color, variations in color_map.items():
            for variation in variations:
                if variation in color_desc:
                    return standard_color

        # If no match, return first word as color
        words = color_desc.split()
        return words[0] if words else "unknown"

    def _empty_result(self) -> Dict:
        """Return empty result."""
        return {
            'primary_color': 'unknown',
            'confidence': 0.0,
            'all_responses': [],
            'frames_analyzed': []
        }
