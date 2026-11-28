"""
Image Descriptors Extractor
Computes the 8 environmental descriptors specified in Section 4.3 of the manuscript:
1. Mean luminance
2. Luminance contrast
3. Shadow ratio
4. Excess-Green index (2G - R - B)
5. Vegetation-pixel ratio
6. Edge density (Sobel / gradient magnitude)
7. Noise level (Laplacian high-frequency energy)
8. Colour-temperature ratio (R / B)
"""

import numpy as np
import cv2
import torch
from typing import Union, Tuple


def extract_environmental_descriptors(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Extracts the 8-dimensional environmental descriptor vector e_t.
    
    Args:
        image: RGB image as numpy array [H, W, 3] in range [0, 255] or [0, 1],
               or torch.Tensor [C, H, W] or [B, C, H, W].
               
    Returns:
        torch.Tensor of shape [8] or [B, 8] with values normalized in [0, 1] or appropriate ranges.
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            # Batch of images
            results = [extract_environmental_descriptors(img) for img in image]
            return torch.stack(results, dim=0)
        # Single image tensor [C, H, W]
        img_np = image.detach().cpu().permute(1, 2, 0).numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img_np = img_np.clip(0, 255).astype(np.uint8)
    else:
        img_np = image.copy()
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img_np = img_np.clip(0, 255).astype(np.uint8)

    # Convert to float for numerical stability
    img_f = img_np.astype(np.float32) / 255.0
    R = img_f[:, :, 0]
    G = img_f[:, :, 1]
    B = img_f[:, :, 2]

    # 1. Mean luminance (ITU-R BT.601 standard)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    mean_luminance = float(np.mean(Y))

    # 2. Luminance contrast (Standard deviation of luminance)
    luminance_contrast = float(np.std(Y))

    # 3. Shadow ratio (Fraction of pixels below luminance threshold 0.20)
    shadow_ratio = float(np.mean(Y < 0.20))

    # 4. Excess-Green index: ExG = 2*G - R - B, normalized to [0, 1]
    exg = 2.0 * G - R - B
    # ExG typically ranges from -2 to +2; normalize to [0, 1]
    exg_norm = (exg + 2.0) / 4.0
    excess_green_mean = float(np.mean(exg_norm))

    # 5. Vegetation-pixel ratio (Pixels where ExG > 0.05 and G > R and G > B)
    veg_mask = (exg > 0.05) & (G > R) & (G > B)
    vegetation_ratio = float(np.mean(veg_mask))

    # 6. Edge density (Sobel gradient magnitude normalized)
    sobelx = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_density = float(np.mean(grad_mag > 0.15))

    # 7. Noise level (High-frequency Laplacian variance)
    laplacian = cv2.Laplacian(Y, cv2.CV_32F)
    noise_var = float(np.var(laplacian))
    # Normalized noise level
    noise_level = float(np.clip(noise_var * 10.0, 0.0, 1.0))

    # 8. Colour-temperature ratio (Mean R / (Mean B + eps))
    r_mean = np.mean(R) + 1e-6
    b_mean = np.mean(B) + 1e-6
    color_temp_ratio = float(np.clip(r_mean / (r_mean + b_mean), 0.0, 1.0))

    descriptors = np.array([
        mean_luminance,
        luminance_contrast,
        shadow_ratio,
        excess_green_mean,
        vegetation_ratio,
        edge_density,
        noise_level,
        color_temp_ratio
    ], dtype=np.float32)

    return torch.from_numpy(descriptors)
