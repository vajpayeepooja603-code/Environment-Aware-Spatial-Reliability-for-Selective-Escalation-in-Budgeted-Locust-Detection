"""
Spatial Perturbation-Survival Head
Implements Equations (6), (7), (9), (10), (11) from Section 4.2 of the manuscript.
Outputs a spatial survival map R_t(u, v) over cell grid U x V.
Computes box-level survival r_{w_i} and survival-gated score s_i = c_i * r_i^eta.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional


class SpatialSurvivalHead(nn.Module):
    """
    Lightweight convolutional head attached to pyramid feature maps.
    Outputs:
        z(u, v): pre-activation logits
        R_t(u, v) = sigma(z(u, v)): per-cell survival probability in [0, 1]
    """
    def __init__(self, in_channels_list: List[int] = [64, 128, 256], grid_size: Tuple[int, int] = (16, 16)):
        super().__init__()
        self.grid_size = grid_size
        
        # Convolutions to reduce each pyramid level to a common feature dimension
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ) for c in in_channels_list
        ])
        
        # Fusion and spatial survival prediction
        total_channels = 32 * len(in_channels_list)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(grid_size),
            nn.Conv2d(64, 1, kernel_size=1)  # Produces pre-activation z(u, v)
        )

    def forward(self, pyramid_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pyramid_features: Multi-scale feature maps from backbone levels (e.g., P3, P4, P5)
        Returns:
            z: pre-activations [B, 1, U, V]
            R: spatial survival probability map [B, 1, U, V]
        """
        target_size = self.grid_size
        lateral_outs = []
        for feat, conv in zip(pyramid_features, self.lateral_convs):
            reduced = conv(feat)
            resized = F.interpolate(reduced, size=target_size, mode='bilinear', align_corners=False)
            lateral_outs.append(resized)
            
        fused = torch.cat(lateral_outs, dim=1)
        z = self.fusion(fused)  # [B, 1, U, V]
        R = torch.sigmoid(z)     # Eq. (6): R_t(u, v) = sigma(z_t(u, v))
        return z, R

    @staticmethod
    def compute_box_survival(R_map: torch.Tensor, box: torch.Tensor,
                             img_size: Tuple[int, int] = (512, 512)) -> float:
        """
        Eq. (7): r_{w_i} = (1 / |cells(w_i)|) * sum_{(u, v) in cells(w_i)} R_t(u, v)
        
        Args:
            R_map: [U, V] spatial survival probability map
            box: [x1, y1, x2, y2] normalized or pixel coordinates
            img_size: (H, W)
        Returns:
            mean survival probability in the box
        """
        U, V = R_map.shape[-2:]
        H, W = img_size
        
        x1, y1, x2, y2 = box.tolist() if isinstance(box, torch.Tensor) else box
        if x2 <= 1.0 and y2 <= 1.0:
            # Normalized
            x1, y1, x2, y2 = x1 * W, y1 * H, x2 * W, y2 * H

        # Map to cell indices in [0, U-1] x [0, V-1]
        u1 = int(np.clip(np.floor((y1 / H) * U), 0, U - 1))
        u2 = int(np.clip(np.ceil((y2 / H) * U), 1, U))
        v1 = int(np.clip(np.floor((x1 / W) * V), 0, V - 1))
        v2 = int(np.clip(np.ceil((x2 / W) * V), 1, V))

        sub_grid = R_map[u1:u2, v1:v2]
        if sub_grid.numel() == 0:
            return float(R_map.mean().item())
        return float(sub_grid.mean().item())

    @staticmethod
    def compute_survival_loss(pred_box_survivals: torch.Tensor,
                              targets: torch.Tensor) -> torch.Tensor:
        """
        Eq. (9): L_rel = -(1 / K) * sum_{k=1}^K [y_k * log(r_k) + (1 - y_k) * log(1 - r_k)]
        """
        pred_clamped = torch.clamp(pred_box_survivals, 1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(pred_clamped, targets.float())
        return loss

    @staticmethod
    def compute_survival_gated_score(confidences: torch.Tensor,
                                     survivals: torch.Tensor,
                                     eta: float = 1.5) -> torch.Tensor:
        """
        Eq. (11): s_i = c_i * (r_i ** eta)
        """
        return confidences * (survivals ** eta)
