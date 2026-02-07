import yaml
import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AppConfig:
    # Camera
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30

    # Virtual camera backend
    vcam_backend: str = "obs"

    # Hair color target in LAB (default: Dark Brown)
    target_color_lab: Tuple[int, int, int] = (45, 138, 150)
    color_intensity: float = 0.85

    # Segmentation
    model_path: str = "models/hair_segmenter.tflite"
    mask_blur_kernel: int = 15
    mask_threshold: float = 0.5
    temporal_alpha: float = 0.7

    # Performance: process segmentation at this fraction of full resolution
    processing_scale: float = 0.35

    @classmethod
    def load(cls, path: str = "config.yaml") -> "AppConfig":
        if os.path.exists(path):
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            if "target_color_lab" in data and isinstance(data["target_color_lab"], list):
                data["target_color_lab"] = tuple(data["target_color_lab"])
            return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        return cls()

    def save(self, path: str = None):
        if path is None:
            path = getattr(self, '_save_path', 'config.yaml')
        data = {
            "camera_index": self.camera_index,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "vcam_backend": self.vcam_backend,
            "target_color_lab": list(self.target_color_lab),
            "color_intensity": self.color_intensity,
            "model_path": "models/hair_segmenter.tflite",  # Always save relative path
            "mask_blur_kernel": self.mask_blur_kernel,
            "mask_threshold": self.mask_threshold,
            "temporal_alpha": self.temporal_alpha,
            "processing_scale": self.processing_scale,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
