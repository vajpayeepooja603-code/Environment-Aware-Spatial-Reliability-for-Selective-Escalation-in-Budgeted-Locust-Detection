"""
Synthetic Environmental Perturbation Engine
Implements the 5 perturbation families used in training and evaluation:
1. Illumination (underexposure / overexposure / glare)
2. Haze (atmospheric scattering simulation)
3. Shadow (spatially non-uniform cast shadow)
4. Noise (Gaussian / sensor noise)
5. Occlusion (simulated vegetation / crop leaf occlusions)
Perturbation magnitudes: [0.1, 0.3, 0.5] as per Table II.
"""

import numpy as np
import cv2
import torch
from typing import Tuple, List, Optional


class SyntheticPerturbationEngine:
    """
    Generates synthetic spatial degradations for training the survival head
    and evaluating robustness under leave-one-family-out protocols.
    """
    FAMILIES = ["illumination", "haze", "shadow", "noise", "occlusion"]

    def __init__(self, magnitudes: List[float] = [0.1, 0.3, 0.5], seed: Optional[int] = None):
        self.magnitudes = magnitudes
        self.rng = np.random.RandomState(seed if seed is not None else 42)

    def set_seed(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def apply_illumination(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Spatially uneven lighting change: overexposure or deep shade."""
        h, w = img.shape[:2]
        # Create non-uniform illumination gradient
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        gradient = np.exp(-((xx - 0.2)**2 + (yy + 0.3)**2) / (0.8 + magnitude))
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)

        factor = 1.0 + (magnitude * 2.0) * (gradient[:, :, None] - 0.5)
        out = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return out

    def apply_haze(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Atmospheric haze model: I = J * t(x) + A * (1 - t(x))."""
        h, w = img.shape[:2]
        # Depth/transmission map
        y = np.linspace(0.2, 1.0, h)[:, None]
        transmission = np.exp(-magnitude * 2.5 * y)
        transmission = np.repeat(transmission, w, axis=1)[:, :, None]
        atmospheric_light = 220.0
        out = img.astype(np.float32) * transmission + atmospheric_light * (1.0 - transmission)
        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_shadow(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Spatially localized cast shadow across a portion of the field."""
        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        # Random polygon shadow
        pts = np.array([
            [int(w * 0.1), int(h * 0.2)],
            [int(w * 0.85), int(h * 0.05)],
            [int(w * 0.95), int(h * 0.7)],
            [int(w * 0.35), int(h * 0.9)]
        ], np.int32)
        cv2.fillPoly(mask, [pts], 1.0 - (0.4 + 0.5 * magnitude))
        # Blur the shadow boundary
        mask = cv2.GaussianBlur(mask, (31, 31), 10)
        out = img.astype(np.float32) * mask[:, :, None]
        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_noise(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Sensor Gaussian noise."""
        sigma = magnitude * 40.0
        noise = self.rng.normal(0, sigma, img.shape).astype(np.float32)
        out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out

    def apply_occlusion(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Simulates foreground foliage / leaf occlusions."""
        out = img.copy()
        h, w = img.shape[:2]
        num_patches = int(5 + 15 * magnitude)
        for _ in range(num_patches):
            pw = int(self.rng.uniform(15, 45) * (1.0 + magnitude))
            ph = int(self.rng.uniform(15, 45) * (1.0 + magnitude))
            px = self.rng.randint(0, max(1, w - pw))
            py = self.rng.randint(0, max(1, h - ph))
            color = [int(c) for c in self.rng.randint(30, 110, size=3)]
            cv2.ellipse(out, (px + pw // 2, py + ph // 2), (pw // 2, ph // 2),
                        self.rng.randint(0, 180), 0, 360, color, -1)
        return out

    def apply_random_perturbation(self, img: np.ndarray,
                                  exclude_family: Optional[str] = None) -> Tuple[np.ndarray, str, float]:
        """
        Applies a random perturbation from available families (optionally excluding one for LOPO).
        """
        families = [f for f in self.FAMILIES if f != exclude_family]
        family = self.rng.choice(families)
        magnitude = float(self.rng.choice(self.magnitudes))

        if family == "illumination":
            pert_img = self.apply_illumination(img, magnitude)
        elif family == "haze":
            pert_img = self.apply_haze(img, magnitude)
        elif family == "shadow":
            pert_img = self.apply_shadow(img, magnitude)
        elif family == "noise":
            pert_img = self.apply_noise(img, magnitude)
        elif family == "occlusion":
            pert_img = self.apply_occlusion(img, magnitude)
        else:
            pert_img = img.copy()

        return pert_img, family, magnitude
