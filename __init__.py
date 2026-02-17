"""
XMH: eXplainable MetaHeuristics Framework

A framework for providing causal, counterfactual, and contrastive
explanations of metaheuristic search processes.

Author: [Your Name]
License: MIT
"""

__version__ = "0.1.0"
__author__ = "XMH Research Team"

from .core.instrumentation import InstrumentedAlgorithm
from .core.trace_logger import TraceLogger, SearchEvent, GenerationSnapshot
from .explanation.operator_shap import OperatorSHAP
from .explanation.counterfactual import CounterfactualAnalyzer

__all__ = [
    "InstrumentedAlgorithm",
    "TraceLogger",
    "SearchEvent",
    "GenerationSnapshot",
    "OperatorSHAP",
    "CounterfactualAnalyzer",
]
