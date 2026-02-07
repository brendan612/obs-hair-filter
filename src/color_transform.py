import cv2
import numpy as np
from typing import Tuple, Dict


# Color presets: name -> (L, A, B) in OpenCV LAB range [0-255]
HAIR_COLOR_PRESETS: Dict[str, Tuple[int, int, int]] = {
    "Jet Black":        (15,  128, 128),
    "Dark Brown":       (45,  138, 150),
    "Medium Brown":     (60,  138, 150),
    "Auburn":           (50,  160, 165),
    "Fiery Red":        (55,  180, 170),
    "Strawberry Blonde": (70, 145, 160),
    "Platinum Blonde":  (85,  127, 130),
    "Ash Blonde":       (75,  125, 120),
    "Pastel Pink":      (70,  165, 128),
    "Vivid Blue":       (45,  120,  80),
    "Emerald Green":    (50,   90, 140),
    "Purple":           (40,  160,  90),
}


class ColorTransformer:
    """
    Transforms hair color in CIE LAB color space while preserving
    luminance, texture, strand detail, and natural color variation.

    Improvements over basic version:
    - Specular highlight preservation (shiny spots stay white/bright)
    - Local contrast enhancement to maintain strand visibility
    - Saturation-aware blending for natural-looking results
    """

    def __init__(self):
        self._target_l: int = 50
        self._target_a: int = 128
        self._target_b: int = 128
        self._intensity: float = 0.85
        self._luminance_strength: float = 0.4

    def set_target_color_lab(self, l: int, a: int, b: int):
        self._target_l = l
        self._target_a = a
        self._target_b = b

    def set_target_color_rgb(self, r: int, g: int, b: int):
        """Set target from an RGB color value."""
        pixel = np.uint8([[[r, g, b]]])
        lab = cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)[0][0]
        self._target_l = int(lab[0])
        self._target_a = int(lab[1])
        self._target_b = int(lab[2])

    def set_intensity(self, intensity: float):
        """Color shift intensity: 0.0 = no change, 1.0 = full replacement."""
        self._intensity = float(np.clip(intensity, 0.0, 1.0))

    @property
    def intensity(self) -> float:
        return self._intensity

    @property
    def target_lab(self) -> Tuple[int, int, int]:
        return (self._target_l, self._target_a, self._target_b)

    def apply(self, frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply hair color transformation with highlight preservation.
        """
        # Find bounding box of hair region to minimize work
        hair_rows = np.any(mask > 0.01, axis=1)
        if not np.any(hair_rows):
            return frame_bgr

        hair_cols = np.any(mask > 0.01, axis=0)
        r_min, r_max = np.where(hair_rows)[0][[0, -1]]
        c_min, c_max = np.where(hair_cols)[0][[0, -1]]

        pad = 5
        r_min = max(0, r_min - pad)
        r_max = min(frame_bgr.shape[0], r_max + pad + 1)
        c_min = max(0, c_min - pad)
        c_max = min(frame_bgr.shape[1], c_max + pad + 1)

        # Crop to hair region only
        roi_bgr = frame_bgr[r_min:r_max, c_min:c_max]
        roi_mask = mask[r_min:r_max, c_min:c_max]

        # Convert to LAB
        roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        lab_f = roi_lab.astype(np.float32)

        L = lab_f[:, :, 0]
        A = lab_f[:, :, 1]
        B = lab_f[:, :, 2]

        # === Specular highlight detection ===
        # Very bright pixels (high L) with low saturation are specular highlights.
        # We want to preserve these — applying color to a shiny white spot
        # looks fake. Create a "highlight mask" that reduces color application
        # on specular areas.
        #
        # Saturation in LAB = distance from neutral (128, 128)
        sat = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)
        # Highlights: high luminance + low saturation
        # Normalize L to 0-1 range, sat inversely
        highlight_strength = np.clip((L - 180) / 75.0, 0, 1) * np.clip((1.0 - sat / 40.0), 0, 1)
        # This gives 0 for normal hair, approaching 1 for bright shiny spots
        color_reduction = 1.0 - highlight_strength * 0.7  # Reduce intensity by up to 70% on highlights

        # === Chrominance shift with highlight awareness ===
        effective_intensity = self._intensity * color_reduction
        A_shifted = A + effective_intensity * (self._target_a - A)
        B_shifted = B + effective_intensity * (self._target_b - B)

        # === Luminance adjustment with local detail preservation ===
        hair_pixels = roi_mask > 0.3
        hair_pixel_count = hair_pixels.sum()

        if hair_pixel_count > 100:
            L_mean = float(L[hair_pixels].mean())
        else:
            L_mean = 128.0

        # Decompose luminance into mean + detail
        L_detail = L - L_mean

        # Boost local contrast slightly to enhance strand visibility
        # This counteracts the slight flattening that color shifting can cause
        contrast_boost = 1.05
        L_detail = L_detail * contrast_boost

        L_mean_shifted = L_mean + self._luminance_strength * self._intensity * (
            self._target_l - L_mean
        )
        L_shifted = L_mean_shifted + L_detail

        # Clamp
        np.clip(L_shifted, 0, 255, out=L_shifted)
        np.clip(A_shifted, 0, 255, out=A_shifted)
        np.clip(B_shifted, 0, 255, out=B_shifted)

        # === Composite using soft mask ===
        m = roi_mask[:, :, np.newaxis]
        inv_m = 1.0 - m

        lab_f[:, :, 0] = L * inv_m[:, :, 0] + L_shifted * m[:, :, 0]
        lab_f[:, :, 1] = A * inv_m[:, :, 0] + A_shifted * m[:, :, 0]
        lab_f[:, :, 2] = B * inv_m[:, :, 0] + B_shifted * m[:, :, 0]

        # Convert back
        roi_result = cv2.cvtColor(lab_f.astype(np.uint8), cv2.COLOR_LAB2BGR)

        result = frame_bgr.copy()
        result[r_min:r_max, c_min:c_max] = roi_result
        return result
