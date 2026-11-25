"""
Locust Benchmark Evaluation Harness
Computes:
- mAP@0.5, mAP@0.5:0.95, and Small-Locust AP following COCO protocols
- AUROC for perturbation survival & Expected Calibration Error (ECE)
- Escalation Precision: fraction of escalated windows containing true locusts
- Look-ahead and realized rolling-mean latency statistics over window W = 30
- Paired bootstrap 95% confidence intervals over source images
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_iou(box1: Tuple[float, float, float, float],
                box2: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Eq. (23): AP = area under the precision-recall curve using 101-point interpolation."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # 101-point interpolation
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    prec_interp = np.interp(recall_thresholds, mrec, mpre)
    return float(np.mean(prec_interp))


def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> float:
    """
    Eq. (24): Expected Calibration Error (ECE).
    """
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(probs)
    if n == 0:
        return 0.0
        
    for i in range(num_bins):
        bin_mask = (probs > bins[i]) & (probs <= bins[i + 1])
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(labels[bin_mask])
            bin_conf = np.mean(probs[bin_mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
            
    return float(ece)


class LocustBenchmarkEvaluator:
    """
    Evaluator performing paired source-image bootstrap tests across seeds.
    """
    def __init__(self, num_bootstrap_samples: int = 1000, seed: int = 42):
        self.num_bootstrap = num_bootstrap_samples
        self.rng = np.random.RandomState(seed)

    def paired_bootstrap_ci(self, scores_a: List[float],
                            scores_b: List[float],
                            source_ids: List[str]) -> Tuple[float, float, float]:
        """
        Paired bootstrap over source images (grouping perturbation variants together).
        Returns:
            (mean_diff, ci_lower_95, ci_upper_95)
        """
        unique_sources = np.unique(source_ids)
        n_sources = len(unique_sources)
        
        # Map source_id to list of indices
        src_map = {src: [] for src in unique_sources}
        for idx, src in enumerate(source_ids):
            src_map[src].append(idx)
            
        diffs = []
        for _ in range(self.num_bootstrap):
            sampled_sources = self.rng.choice(unique_sources, size=n_sources, replace=True)
            sampled_indices = []
            for src in sampled_sources:
                sampled_indices.extend(src_map[src])
                
            mean_a = np.mean([scores_a[i] for i in sampled_indices])
            mean_b = np.mean([scores_b[i] for i in sampled_indices])
            diffs.append(mean_a - mean_b)
            
        mean_diff = float(np.mean(diffs))
        ci_lower = float(np.percentile(diffs, 2.5))
        ci_upper = float(np.percentile(diffs, 97.5))
        return mean_diff, ci_lower, ci_upper
