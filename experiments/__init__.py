"""
XMH Experiments Module

Scripts for validating the XMH framework.
"""

# Lazy imports to avoid circular dependencies when running scripts directly
def run_shap_validation(*args, **kwargs):
    from .exp1_shap_validation import run_shap_validation as _run
    return _run(*args, **kwargs)

def run_overhead_analysis(*args, **kwargs):
    from .exp2_overhead import run_overhead_analysis as _run
    return _run(*args, **kwargs)

def run_generalization_study(*args, **kwargs):
    from .exp3_generalization import run_generalization_study as _run
    return _run(*args, **kwargs)

def run_exact_shapley(*args, **kwargs):
    from .exp4_exact_shapley import run_experiment as _run
    return _run(*args, **kwargs)

def run_method_comparison(*args, **kwargs):
    from .exp5_method_comparison import run_experiment as _run
    return _run(*args, **kwargs)

def run_detailed_overhead(*args, **kwargs):
    from .exp6_overhead import run_experiment as _run
    return _run(*args, **kwargs)

__all__ = [
    "run_shap_validation",
    "run_overhead_analysis",
    "run_generalization_study",
    "run_exact_shapley",
    "run_method_comparison",
    "run_detailed_overhead",
]
