"""
Look-Ahead Latency Guard
Implements Eq. (5) and Eq. (19) from Section 4.5 and Proposition 2 (M1b Budget Safety):
Maintains a rolling window of W = 30 latency measurements.
Calculates the look-ahead running mean:
  L_{t+1}(a*) = L_t + (ell_hat(a*) - ell_{t - W + 1}) / W
Replaces any action that exceeds the hard budget B = 34.8 ms with the default action a_def.
"""

from collections import deque
from typing import Dict, Tuple, Optional
import numpy as np


class LookAheadLatencyGuard:
    """
    Deterministic latency guard enforcing running-mean budget B over window W.
    """
    def __init__(self,
                 budget: float = 34.8,
                 window_size: int = 30,
                 default_latency: float = 22.5,
                 estimated_latencies: Optional[Dict[str, float]] = None):
        """
        Args:
            budget: Hard latency budget B in milliseconds (34.8 ms = 60% of C2 58.0 ms).
            window_size: Rolling window size W = 30 observations.
            default_latency: Nominal latency of the default action a_def (22.5 ms).
            estimated_latencies: Mapping from action name to estimated execution latency in ms.
        """
        self.budget = budget
        self.window_size = window_size
        self.default_latency = default_latency
        
        # Nominal profile matching NVIDIA Jetson Orin Nano / TensorRT measurements
        self.estimated_latencies = estimated_latencies or {
            "a_def": 22.5,     # 512x512 standard
            "a_cheap": 18.0,   # Cheaper tier / downsampled
            "a_640": 58.0,     # Full 640x640 escalated
            "a_enh": 38.0,     # Enhancement filter + 512
            "a_zoom_1": 26.2,  # 512 + 1 zoom window
            "a_zoom_2": 29.8,  # 512 + 2 zoom windows
            "a_zoom_3": 33.5,  # 512 + 3 zoom windows
            "a_zoom_4": 37.2,  # 512 + 4 zoom windows
        }
        
        # Initialize rolling history with default latency
        self.history = deque([default_latency] * window_size, maxlen=window_size)
        self.current_mean = default_latency
        self.total_observations = 0
        self.budget_violations = 0
        self.guard_fallbacks = 0

    def get_estimated_latency(self, action_name: str, num_zoom_windows: int = 0) -> float:
        if action_name.startswith("azoom"):
            key = f"a_zoom_{min(max(1, num_zoom_windows), 4)}"
            return self.estimated_latencies.get(key, 26.0 + 3.6 * num_zoom_windows)
        return self.estimated_latencies.get(action_name, self.default_latency)

    def calculate_look_ahead_mean(self, action_name: str, num_zoom_windows: int = 0) -> float:
        """
        Eq. (5): L_{t+1}(a*) = L_t + (ell_hat(a*) - ell_{t - W + 1}) / W
        """
        ell_hat = self.get_estimated_latency(action_name, num_zoom_windows)
        oldest_latency = self.history[0]  # ell_{t - W + 1}
        look_ahead = self.current_mean + (ell_hat - oldest_latency) / float(self.window_size)
        return float(look_ahead)

    def filter_action(self, candidate_action: str, num_zoom_windows: int = 0) -> Tuple[str, bool, float]:
        """
        Eq. (19): If look-ahead running mean exceeds B, fallback to a_def.
        Returns:
            (executed_action, fell_back_to_default, look_ahead_mean)
        """
        look_ahead = self.calculate_look_ahead_mean(candidate_action, num_zoom_windows)
        
        if look_ahead > self.budget:
            # Budget violation prevented by fallback!
            self.guard_fallbacks += 1
            executed_action = "a_def"
            fallback = True
        else:
            executed_action = candidate_action
            fallback = False
            
        return executed_action, fallback, look_ahead

    def record_execution(self, executed_action: str, num_zoom_windows: int = 0,
                         measured_latency: Optional[float] = None) -> float:
        """
        Updates rolling window with actual or profiled measured latency.
        """
        if measured_latency is None:
            # Sample realistic execution latency with slight hardware jitter (±0.3 ms)
            nominal = self.get_estimated_latency(executed_action, num_zoom_windows)
            jitter = np.random.normal(0, 0.2)
            measured_latency = max(15.0, nominal + jitter)

        self.history.append(measured_latency)
        self.current_mean = float(np.mean(self.history))
        self.total_observations += 1
        
        if self.current_mean > self.budget:
            self.budget_violations += 1
            
        return measured_latency

    def reset(self):
        self.history = deque([self.default_latency] * self.window_size, maxlen=self.window_size)
        self.current_mean = self.default_latency
        self.total_observations = 0
        self.budget_violations = 0
        self.guard_fallbacks = 0
