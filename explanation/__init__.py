"""
XMH Explanation Module

Contains methods for explaining metaheuristic search processes:
- Operator-SHAP: Shapley value-based operator attribution
- Counterfactual analysis: "What if" explanations
- Natural language generation: Human-readable explanations
"""

from .operator_shap import OperatorSHAP, SHAPExplanation, QuickSHAP, KernelSHAP, ExactOperatorSHAP
from .counterfactual import CounterfactualAnalyzer, Counterfactual
from .attribution import AttributionAnalyzer
from .tracking_attribution import TrackingAttribution, TrackingResult

__all__ = [
    "OperatorSHAP",
    "SHAPExplanation",
    "QuickSHAP",
    "KernelSHAP",
    "ExactOperatorSHAP",
    "CounterfactualAnalyzer",
    "Counterfactual",
    "AttributionAnalyzer",
    "TrackingAttribution",
    "TrackingResult",
]
