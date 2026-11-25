"""
EAR-YOLO: Environment-Aware Spatial Reliability for Selective Escalation in Budgeted Locust Detection
Package initialization.
"""

from .descriptors import extract_environmental_descriptors
from .film import FiLMBlock, FiLMMLP
from .survival_head import SpatialSurvivalHead
from .perturbations import SyntheticPerturbationEngine
from .model import EARYOLOModel
from .budget_guard import LookAheadLatencyGuard
from .decision_rule import SelectiveEscalationRule, InspectionAction
from .evaluator import LocustBenchmarkEvaluator

__all__ = [
    "extract_environmental_descriptors",
    "FiLMBlock",
    "FiLMMLP",
    "SpatialSurvivalHead",
    "SyntheticPerturbationEngine",
    "EARYOLOModel",
    "LookAheadLatencyGuard",
    "SelectiveEscalationRule",
    "InspectionAction",
    "LocustBenchmarkEvaluator",
]
