"""
Feature-wise Linear Modulation (FiLM) Module
Implements Equations (12) and (13) from Section 4.3 of the manuscript.
Encodes 8-D environmental descriptors into modulation parameters (gamma, beta)
for each pyramid level of the detector backbone.
Initialized to zeros so training begins from unmodified YOLO11.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class FiLMMLP(nn.Module):
    """
    Multilayer Perceptron mapping 8-D descriptors e_t to latent embedding e_proj.
    Eq. (12): e_t' = MLP(e_t)
    """
    def __init__(self, in_features: int = 8, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMBlock(nn.Module):
    """
    Applies feature-wise linear modulation to a feature map F_l at pyramid level l:
    Eq. (13): F_l' = (1 + gamma_l) * F_l + beta_l
    where [gamma_l, beta_l] = W_l * e_t'
    Parameters are initialized to zero so training starts from unmodified YOLO11.
    """
    def __init__(self, in_channels: int, embed_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        # Linear layer producing gamma and beta
        self.proj = nn.Linear(embed_dim, in_channels * 2)
        # Initialize gamma and beta to 0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feature_map: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: [B, C, H, W]
            embedding: [B, embed_dim]
        Returns:
            Modulated feature map: [B, C, H, W]
        """
        # [B, 2*C]
        modulation = self.proj(embedding)
        gamma, beta = torch.chunk(modulation, 2, dim=-1)
        # Reshape to [B, C, 1, 1] for broadcasting
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * feature_map + beta
