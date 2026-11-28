"""
Deterministic Selective Escalation Rule (Algorithm 1)
Implements Sections 4.4 and 4.5 of the manuscript:
- Step 1: Compute image-level mean survival r_t (Eq. 1)
- Step 2: Compute window survivals and set W_low = {w : r_w < rho} (Eq. 14, rho = 0.5)
- Step 3: Check triggers:
          T1: r_t < rho (Eq. 15)
          T2: candidate c_i in [c_lo, c_hi] and w(b_i) in W_low (Eq. 16, [0.10, 0.50])
- Step 4: Zoom set Z_t: at most K_max = 4 least-survival windows of W_T2 (Eq. 17)
- Step 5: Legal Action Set A_legal (Eq. 18):
          If not T1 and not T2: {a_def} U A_cheap
          Else: {a_640, a_enh} U {a_zoom_w : w in Z_t}
- Step 6: Select action a* from A_legal
- Step 7: Apply look-ahead Budget Guard (Eq. 19)
- Step 8: Execution & Feedback
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import torch

from .budget_guard import LookAheadLatencyGuard


@dataclass
class WindowBox:
    id: int
    x1: float
    y1: float
    x2: float
    y2: float
    mean_survival: float = 1.0


@dataclass
class CandidateDetection:
    box: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    survival: float = 1.0
    gated_score: float = 0.0


@dataclass
class InspectionAction:
    action_type: str  # "a_def", "a_cheap", "a_640", "a_enh", "azoom"
    zoom_windows: List[WindowBox]
    look_ahead_latency: float
    fell_back: bool = False


class SelectiveEscalationRule:
    """
    Implements Algorithm 1 and handles both spatial mode and scalar mode (Test N1).
    """
    def __init__(self,
                 rho: float = 0.5,
                 c_lo: float = 0.10,
                 c_hi: float = 0.50,
                 K_max: int = 4,
                 budget_guard: Optional[LookAheadLatencyGuard] = None,
                 eta: float = 1.5,
                 grid_divisions: Tuple[int, int] = (4, 4)):
        """
        Args:
            rho: Survival trigger threshold (frozen at 0.5)
            c_lo, c_hi: Candidate confidence interval ([0.10, 0.50])
            K_max: Max zoom windows (frozen at 4)
            budget_guard: Look-ahead latency guard instance
            eta: Score exponent (frozen at 1.5)
            grid_divisions: Window grid decomposition (e.g. 4x4 = 16 windows)
        """
        self.rho = rho
        self.c_lo = c_lo
        self.c_hi = c_hi
        self.K_max = K_max
        self.guard = budget_guard or LookAheadLatencyGuard()
        self.eta = eta
        self.grid_divisions = grid_divisions

    def generate_candidate_windows(self, img_size: Tuple[int, int] = (512, 512)) -> List[WindowBox]:
        """Decomposes image into a regular candidate window grid (e.g., 4x4 = 16 windows)."""
        H, W = img_size
        ny, nx = self.grid_divisions
        win_h = H / ny
        win_w = W / nx
        windows = []
        wid = 0
        for i in range(ny):
            for j in range(nx):
                x1 = j * win_w
                y1 = i * win_h
                x2 = x1 + win_w
                y2 = y1 + win_h
                windows.append(WindowBox(id=wid, x1=x1, y1=y1, x2=x2, y2=y2))
                wid += 1
        return windows

    def compute_window_survivals(self, windows: List[WindowBox],
                                 R_map: np.ndarray,
                                 img_size: Tuple[int, int] = (512, 512),
                                 scalar_mode: bool = False,
                                 scalar_mean: Optional[float] = None) -> List[WindowBox]:
        """
        Computes survival for each window w in W.
        In scalar mode (N1 ablation): all windows receive the uniform image mean r_t.
        """
        if scalar_mode:
            uniform_val = float(np.mean(R_map)) if scalar_mean is None else scalar_mean
            for w in windows:
                w.mean_survival = uniform_val
            return windows

        U, V = R_map.shape
        H, W = img_size
        for w in windows:
            u1 = int(np.clip(np.floor((w.y1 / H) * U), 0, U - 1))
            u2 = int(np.clip(np.ceil((w.y2 / H) * U), 1, U))
            v1 = int(np.clip(np.floor((w.x1 / W) * V), 0, V - 1))
            v2 = int(np.clip(np.ceil((w.x2 / W) * V), 1, V))
            sub = R_map[u1:u2, v1:v2]
            w.mean_survival = float(np.mean(sub)) if sub.size > 0 else float(np.mean(R_map))
        return windows

    def find_window_for_box(self, box: Tuple[float, float, float, float],
                            windows: List[WindowBox]) -> Optional[WindowBox]:
        """Finds the window containing the center of candidate detection box."""
        bx = (box[0] + box[2]) / 2.0
        by = (box[1] + box[3]) / 2.0
        for w in windows:
            if w.x1 <= bx <= w.x2 and w.y1 <= by <= w.y2:
                return w
        return None

    def execute_rule(self,
                     R_map: np.ndarray,
                     candidates: List[CandidateDetection],
                     img_size: Tuple[int, int] = (512, 512),
                     scalar_mode: bool = False,
                     seed: int = 42) -> InspectionAction:
        """
        Full implementation of Algorithm 1.
        
        Args:
            R_map: Spatial survival map [U, V]
            candidates: Detected candidate bounding boxes and confidences
            scalar_mode: If True, evaluates uniform scalar map (Test N1 ablation)
            seed: Pre-declared seed for signal-independent tie-breaking in scalar mode
        """
        # Step 1: Image-level mean reliability r_t (Eq. 1)
        r_t = float(np.mean(R_map))
        
        # Step 2: Low-Reliability Windows W_low (Eq. 14)
        windows = self.generate_candidate_windows(img_size)
        windows = self.compute_window_survivals(windows, R_map, img_size,
                                               scalar_mode=scalar_mode, scalar_mean=r_t)
        W_low = [w for w in windows if w.mean_survival < self.rho]
        W_low_ids = {w.id for w in W_low}
        
        # Step 3: Reliability Triggers (Eqs. 15, 16)
        T1 = (r_t < self.rho)
        
        # Candidate trigger T2: exists candidate with c_i in [c_lo, c_hi] and w(b_i) in W_low
        W_T2_candidates: Set[int] = set()
        for cand in candidates:
            # Update survival-gated score: s_i = c_i * r_i^eta
            w_box = self.find_window_for_box(cand.box, windows)
            if w_box is not None:
                cand.survival = w_box.mean_survival
                cand.gated_score = cand.confidence * (cand.survival ** self.eta)
                if (self.c_lo <= cand.confidence <= self.c_hi) and (w_box.id in W_low_ids):
                    W_T2_candidates.add(w_box.id)

        T2 = len(W_T2_candidates) > 0
        W_T2_windows = [w for w in windows if w.id in W_T2_candidates]

        # Step 4: Zoom Set Z_t (Eq. 17)
        # Select at most K_max least-survival windows of W_T2
        if scalar_mode:
            # In scalar mode, survival scores are identical across windows.
            # Section 5.2 / Section 6.4: Tie-breaking MUST be signal-independent (pre-declared permutation),
            # never based on candidate confidence!
            rng = np.random.RandomState(seed)
            shuffled_indices = rng.permutation(len(W_T2_windows))
            Z_t = [W_T2_windows[idx] for idx in shuffled_indices[:self.K_max]]
        else:
            # Spatial mode: strictly sorted by lowest mean survival
            sorted_windows = sorted(W_T2_windows, key=lambda w: w.mean_survival)
            Z_t = sorted_windows[:self.K_max]

        # Step 5: Legal Action Set A_legal (Eq. 18)
        if (not T1) and (not T2):
            A_legal = ["a_def", "a_cheap"]
        else:
            A_legal = ["a_640", "a_enh"] + [f"azoom_{w.id}" for w in Z_t]

        # Step 6: Action Selection
        # Selective preference order: if specific low-reliability windows found, prefer targeted zoom
        if len(Z_t) > 0 and any(a.startswith("azoom") for a in A_legal):
            preferred_action = "azoom"
            num_zoom = len(Z_t)
        elif T1 and "a_640" in A_legal:
            preferred_action = "a_640"
            num_zoom = 0
        else:
            preferred_action = "a_def"
            num_zoom = 0

        # Step 7: Budget Guard (Eq. 19)
        executed_action, fell_back, look_ahead = self.guard.filter_action(
            preferred_action, num_zoom_windows=num_zoom
        )
        
        # Step 8: Execution & Feedback record
        self.guard.record_execution(executed_action, num_zoom_windows=num_zoom)

        return InspectionAction(
            action_type=executed_action,
            zoom_windows=Z_t if executed_action == "azoom" else [],
            look_ahead_latency=look_ahead,
            fell_back=fell_back
        )
