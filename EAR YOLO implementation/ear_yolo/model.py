"""
EAR-YOLO Full Network Architecture
Integrates:
- YOLO11 backbone and pyramid features (P3, P4, P5)
- Environmental Descriptor conditioning with FiLM blocks at each pyramid level
- Detection head (bounding boxes, class scores)
- Spatial Survival Head (per-cell survival map R_t(u, v))
- Support for escalated resolution (640x640) and localized patch zoom
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

from .descriptors import extract_environmental_descriptors
from .film import FiLMBlock, FiLMMLP
from .survival_head import SpatialSurvivalHead


class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class BackboneYOLO(nn.Module):
    """
    Lightweight CSP-style pyramid backbone extracting multi-scale feature maps:
    P3: stride 8 (e.g. 64x64 for 512 input)
    P4: stride 16 (e.g. 32x32)
    P5: stride 32 (e.g. 16x16)
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        # Stem
        self.stem = ConvBNReLU(in_channels, base_channels, k=3, s=2, p=1) # 256
        self.stage1 = ConvBNReLU(base_channels, base_channels * 2, k=3, s=2, p=1) # 128
        
        # P3
        self.stage2 = nn.Sequential(
            ConvBNReLU(base_channels * 2, base_channels * 4, k=3, s=2, p=1), # 64
            ConvBNReLU(base_channels * 4, base_channels * 4, k=3, s=1, p=1)
        )
        # P4
        self.stage3 = nn.Sequential(
            ConvBNReLU(base_channels * 4, base_channels * 8, k=3, s=2, p=1), # 32
            ConvBNReLU(base_channels * 8, base_channels * 8, k=3, s=1, p=1)
        )
        # P5
        self.stage4 = nn.Sequential(
            ConvBNReLU(base_channels * 8, base_channels * 16, k=3, s=2, p=1), # 16
            ConvBNReLU(base_channels * 16, base_channels * 16, k=3, s=1, p=1)
        )
        
        self.out_channels = [base_channels * 4, base_channels * 8, base_channels * 16] # [128, 256, 512]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        p3 = self.stage2(x1)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return [p3, p4, p5]


class DetectionHead(nn.Module):
    """
    Detection head predicting bounding boxes (xywh) and objectness/class confidence.
    """
    def __init__(self, in_channels_list: List[int], num_classes: int = 1):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                ConvBNReLU(c, c // 2, k=3, s=1, p=1),
                nn.Conv2d(c // 2, 4 + num_classes, kernel_size=1)
            ) for c in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [head(f) for head, f in zip(self.heads, features)]


class EARYOLOModel(nn.Module):
    """
    Full EAR-YOLO Architecture:
    - Backbone generating P3, P4, P5
    - FiLM conditioning from 8-D environmental descriptors
    - Detection Head
    - Spatial Perturbation-Survival Head
    """
    def __init__(self, num_classes: int = 1, base_channels: int = 32, grid_size: Tuple[int, int] = (16, 16)):
        super().__init__()
        self.backbone = BackboneYOLO(in_channels=3, base_channels=base_channels)
        c_p3, c_p4, c_p5 = self.backbone.out_channels
        
        # FiLM modules
        self.film_mlp = FiLMMLP(in_features=8, hidden_dim=64, out_dim=64)
        self.film_p3 = FiLMBlock(in_channels=c_p3, embed_dim=64)
        self.film_p4 = FiLMBlock(in_channels=c_p4, embed_dim=64)
        self.film_p5 = FiLMBlock(in_channels=c_p5, embed_dim=64)
        
        # Detection Head
        self.det_head = DetectionHead(in_channels_list=[c_p3, c_p4, c_p5], num_classes=num_classes)
        
        # Spatial Survival Head
        self.survival_head = SpatialSurvivalHead(in_channels_list=[c_p3, c_p4, c_p5], grid_size=grid_size)

    def extract_features(self, x: torch.Tensor,
                         descriptors: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Extracts backbone features and applies FiLM conditioning if descriptors are provided.
        """
        raw_feats = self.backbone(x)
        
        if descriptors is None:
            # Extract on the fly
            descriptors = extract_environmental_descriptors(x).to(x.device)
            
        emb = self.film_mlp(descriptors)
        
        mod_p3 = self.film_p3(raw_feats[0], emb)
        mod_p4 = self.film_p4(raw_feats[1], emb)
        mod_p5 = self.film_p5(raw_feats[2], emb)
        
        return [mod_p3, mod_p4, mod_p5], descriptors

    def forward(self, x: torch.Tensor,
                descriptors: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass producing:
        - raw_detections: multi-scale outputs
        - survival_logits: z(u, v)
        - survival_map: R_t(u, v) in [0, 1]
        - mean_survival: r_t scalar
        - descriptors: e_t
        """
        feats, desc = self.extract_features(x, descriptors)
        det_outputs = self.det_head(feats)
        z, R = self.survival_head(feats)
        
        # R is [B, 1, U, V] -> squeeze channel for convenience [B, U, V]
        R_squeezed = R.squeeze(1)
        r_mean = R_squeezed.mean(dim=[-2, -1]) # [B]
        
        return {
            "det_outputs": det_outputs,
            "survival_logits": z.squeeze(1),
            "survival_map": R_squeezed,
            "mean_survival": r_mean,
            "descriptors": desc
        }
