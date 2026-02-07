import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from typing import Optional


class HairSegmenter:
    """
    Wraps MediaPipe Image Segmenter for hair mask extraction.

    Post-processing pipeline for high-quality masks:
    1. Confidence threshold to remove noise
    2. Morphological operations to close holes and remove specks
    3. Edge-aware bilateral filter (softens mask while preserving hair boundary)
    4. Temporal EMA smoothing to reduce flicker
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        blur_kernel: int = 15,
        temporal_alpha: float = 0.7,
    ):
        self._threshold = confidence_threshold
        self._blur_kernel = blur_kernel
        self._temporal_alpha = temporal_alpha
        self._prev_mask: Optional[np.ndarray] = None
        self._frame_count = 0

        # Pre-create morphological kernels (avoid re-creating every frame)
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)

    def get_hair_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Returns a float32 mask [0.0-1.0] of hair region, same size as input.
        """
        h, w = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self._frame_count += 1
        timestamp_ms = int(self._frame_count * (1000 / 30))

        result = self._segmenter.segment_for_video(mp_image, timestamp_ms)

        if len(result.confidence_masks) < 2:
            return np.zeros((h, w), dtype=np.float32)

        # Get raw mask and ensure it's a 2D float32 array matching input size
        raw_mask = result.confidence_masks[1].numpy_view().copy().astype(np.float32)
        if raw_mask.ndim > 2:
            raw_mask = raw_mask[:, :, 0]
        if raw_mask.shape[:2] != (h, w):
            raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # === Post-processing pipeline ===

        # 1. Threshold to create a clean confidence mask
        mask = np.where(raw_mask > self._threshold, raw_mask, 0.0).astype(np.float32)

        # 2. Morphological cleanup on a binary version
        #    - Close: fills small holes inside hair region
        #    - Open: removes small noise specks outside hair
        binary = (mask > 0.3).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._kernel_close)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._kernel_open)

        # Combine: keep confidence gradients where binary says hair, zero elsewhere
        mask = mask * binary.astype(np.float32)

        # 3. Edge-aware smoothing
        k = self._blur_kernel
        if k > 0:
            # Ensure d is odd for bilateral filter
            d = k if k % 2 == 1 else k + 1
            mask_u8 = (mask * 255).astype(np.uint8)
            gray_guide = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            if hasattr(cv2, 'ximgproc'):
                # Joint bilateral: uses image edges to guide mask smoothing
                mask_u8 = cv2.ximgproc.jointBilateralFilter(
                    gray_guide, mask_u8, d=d, sigmaColor=50, sigmaSpace=d
                )
            else:
                # Fallback: regular bilateral (still edge-aware, just less precise)
                mask_u8 = cv2.bilateralFilter(mask_u8, d=d, sigmaColor=50, sigmaSpace=d)

            mask = mask_u8.astype(np.float32) / 255.0

        # 4. Temporal smoothing via exponential moving average
        if self._prev_mask is not None and self._prev_mask.shape == mask.shape:
            cv2.addWeighted(
                mask, self._temporal_alpha,
                self._prev_mask, 1.0 - self._temporal_alpha,
                0.0, dst=mask,
            )
        self._prev_mask = mask.copy()

        return mask

    @property
    def temporal_alpha(self) -> float:
        return self._temporal_alpha

    @temporal_alpha.setter
    def temporal_alpha(self, value: float):
        self._temporal_alpha = np.clip(value, 0.0, 1.0)

    @property
    def blur_kernel(self) -> int:
        return self._blur_kernel

    @blur_kernel.setter
    def blur_kernel(self, value: int):
        self._blur_kernel = max(0, value)

    def close(self):
        self._segmenter.close()
