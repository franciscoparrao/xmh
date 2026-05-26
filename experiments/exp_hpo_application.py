#!/usr/bin/env python3
"""
Real-World Application: Hyperparameter Optimization with XMH

Demonstrates the practical value of explainable metaheuristics by
optimizing XGBoost hyperparameters and analyzing operator contributions.

For IEEE TEVC paper.
"""

import sys
import os
import time
import json
import pickle
import shutil
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime
import numpy as np
from scipy import stats

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from xmh.algorithms.instrumented_de import InstrumentedDE
from xmh.algorithms.instrumented_ga import InstrumentedGA
from xmh.algorithms.instrumented_pso import InstrumentedPSO
from xmh.explanation.operator_shap import QuickSHAP
from xmh.core.trace_logger import InstrumentationLevel

# ML imports
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.datasets import load_breast_cancer, load_wine, fetch_openml
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# ==================== CONFIGURATION ====================

@dataclass
class HPOConfig:
    """Configuration for HPO experiments."""
    # Datasets to test
    datasets: List[str] = field(default_factory=lambda: [
        "breast_cancer",
        "wine",
        "diabetes",
        "vehicle"
    ])

    # XGBoost hyperparameter search space
    # Each tuple: (min, max, is_integer, log_scale)
    hyperparameters: Dict[str, Tuple] = field(default_factory=lambda: {
        'n_estimators': (50, 500, True, False),      # Number of trees
        'max_depth': (3, 12, True, False),           # Tree depth
        'learning_rate': (0.01, 0.3, False, True),   # Learning rate (log scale)
        'subsample': (0.5, 1.0, False, False),       # Row sampling
        'colsample_bytree': (0.5, 1.0, False, False), # Column sampling
        'reg_alpha': (1e-5, 10, False, True),        # L1 regularization
        'reg_lambda': (1e-5, 10, False, True),       # L2 regularization
        'min_child_weight': (1, 10, True, False),    # Minimum child weight
    })

    # Algorithm settings
    pop_size: int = 30
    max_generations: int = 50
    n_runs: int = 10

    # CV settings
    cv_folds: int = 5

    # Output
    output_dir: str = "results_hpo"


# ==================== CHECKPOINT ====================

CHECKPOINT_FILE = "hpo_checkpoint.pkl"


@dataclass
class HPOCheckpoint:
    """Checkpoint for resuming HPO experiments."""
    config: HPOConfig
    completed_configs: Set[Tuple[str, str]]  # (dataset, algorithm)
    results: List[Any]  # HPOResult objects
    start_time: float
    last_update: float

    def get_progress_string(self) -> str:
        total = len(self.config.datasets) * 3  # 3 algorithms
        completed = len(self.completed_configs)
        pct = 100 * completed / total if total > 0 else 0
        return f"{completed}/{total} ({pct:.1f}%)"


def save_hpo_checkpoint(checkpoint: HPOCheckpoint, output_dir: str):
    """Save checkpoint atomically."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, CHECKPOINT_FILE)
    temp_path = checkpoint_path + ".tmp"
    checkpoint.last_update = time.time()
    with open(temp_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    shutil.move(temp_path, checkpoint_path)


def load_hpo_checkpoint(output_dir: str) -> Optional[HPOCheckpoint]:
    """Load checkpoint if exists."""
    checkpoint_path = os.path.join(output_dir, CHECKPOINT_FILE)
    if not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return None


# ==================== DATASET LOADING ====================

def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load a dataset by name."""
    if name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target, "Breast Cancer (569 samples, 30 features)"

    elif name == "wine":
        data = load_wine()
        return data.data, data.target, "Wine (178 samples, 13 features)"

    elif name == "diabetes":
        # Pima Indians Diabetes
        try:
            data = fetch_openml(name='diabetes', version=1, as_frame=False, parser='auto')
            y = (data.target == 'tested_positive').astype(int)
            return data.data, y, "Diabetes (768 samples, 8 features)"
        except:
            # Fallback: use sklearn's diabetes for regression, convert to classification
            from sklearn.datasets import load_diabetes
            data = load_diabetes()
            y = (data.target > data.target.median()).astype(int)
            return data.data, y, "Diabetes (442 samples, 10 features)"

    elif name == "vehicle":
        try:
            data = fetch_openml(name='vehicle', version=1, as_frame=False, parser='auto')
            # Convert string labels to integers
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(data.target)
            return data.data, y, "Vehicle (846 samples, 18 features)"
        except:
            # Fallback to wine if vehicle not available
            data = load_wine()
            return data.data, data.target, "Wine (178 samples, 13 features)"

    else:
        raise ValueError(f"Unknown dataset: {name}")


# ==================== HPO OBJECTIVE FUNCTION ====================

class XGBoostHPO:
    """
    Objective function for XGBoost hyperparameter optimization.

    This class wraps XGBoost training with cross-validation as a
    minimization problem for metaheuristic optimization.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 hyperparams: Dict[str, Tuple],
                 cv_folds: int = 5,
                 random_state: int = 42):
        self.X = X
        self.y = y
        self.hyperparams = hyperparams
        self.param_names = list(hyperparams.keys())
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_classes = len(np.unique(y))

        # Precompute bounds
        self.lower_bounds = np.array([hp[0] for hp in hyperparams.values()])
        self.upper_bounds = np.array([hp[1] for hp in hyperparams.values()])

        # Scale data once
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)

        # Track evaluations
        self.eval_count = 0
        self.best_score = float('inf')
        self.best_params = None

    def decode_solution(self, x: np.ndarray) -> Dict[str, Any]:
        """Convert normalized vector to XGBoost parameters."""
        params = {}
        for i, (name, (min_val, max_val, is_int, log_scale)) in enumerate(self.hyperparams.items()):
            val = x[i]

            if log_scale:
                # Log-scale transformation
                log_min = np.log10(min_val)
                log_max = np.log10(max_val)
                val = 10 ** (log_min + val * (log_max - log_min))
            else:
                # Linear transformation
                val = min_val + val * (max_val - min_val)

            if is_int:
                val = int(round(val))

            params[name] = val

        return params

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate a hyperparameter configuration.

        Args:
            x: Normalized vector [0, 1]^d

        Returns:
            Negative mean CV accuracy (for minimization)
        """
        self.eval_count += 1

        # Clip to valid range
        x = np.clip(x, 0, 1)

        # Decode hyperparameters
        params = self.decode_solution(x)

        # Create XGBoost classifier
        if self.n_classes == 2:
            objective = 'binary:logistic'
        else:
            objective = 'multi:softmax'

        clf = xgb.XGBClassifier(
            **params,
            objective=objective,
            n_jobs=1,
            random_state=self.random_state,
            verbosity=0,
            use_label_encoder=False,
        )

        # Cross-validation
        try:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                                random_state=self.random_state)
            scores = cross_val_score(clf, self.X_scaled, self.y,
                                    cv=cv, scoring='accuracy', n_jobs=1)
            mean_score = scores.mean()

            # Track best
            neg_score = -mean_score  # For minimization
            if neg_score < self.best_score:
                self.best_score = neg_score
                self.best_params = params.copy()

            return neg_score

        except Exception as e:
            # Return penalty for invalid configurations
            return 0.0  # Worst possible accuracy

    def get_bounds(self, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get normalized bounds [0,1] for optimization."""
        return np.zeros(dim), np.ones(dim)

    @property
    def dim(self) -> int:
        return len(self.hyperparams)


# ==================== EXPERIMENT RUNNER ====================

@dataclass
class HPOResult:
    """Result from a single HPO run."""
    algorithm: str
    dataset: str
    run_id: int
    best_accuracy: float
    best_params: Dict[str, Any]
    convergence_curve: List[float]
    shap_values: Dict[str, float]
    operator_stats: Dict[str, Dict]
    execution_time: float


def run_hpo_experiment(
    alg_class,
    alg_name: str,
    hpo_objective: XGBoostHPO,
    dataset_name: str,
    config: HPOConfig,
    run_id: int
) -> HPOResult:
    """Run a single HPO experiment."""

    dim = hpo_objective.dim
    lower, upper = hpo_objective.get_bounds(dim)

    # Algorithm kwargs
    kwargs = {
        'dim': dim,
        'bounds': (lower, upper),
        'pop_size': config.pop_size,
        'max_generations': config.max_generations,
        'instrumentation_level': InstrumentationLevel.STANDARD,
        'seed': 42 + run_id,
    }

    # Algorithm-specific settings
    if alg_name == "DE":
        kwargs['F'] = 0.5
        kwargs['CR'] = 0.9
    elif alg_name == "GA":
        kwargs['mutation_prob'] = 1.0 / dim
        kwargs['crossover_prob'] = 0.9

    # Reset objective counter
    hpo_objective.eval_count = 0
    hpo_objective.best_score = float('inf')
    hpo_objective.best_params = None

    # Run optimization
    start_time = time.time()
    alg = alg_class(**kwargs)
    result = alg.run(hpo_objective, verbose=False)
    execution_time = time.time() - start_time

    # Get trace and SHAP values
    trace = result['trace']
    quick_shap = QuickSHAP()
    explanation = quick_shap.explain_from_trace(trace)

    # Get operator stats
    op_stats = {}
    contributions = trace.get_operator_contributions()
    for op, stats in contributions.items():
        op_stats[op] = {
            'total_used': stats['total_used'],
            'total_success': stats['total_success'],
            'success_rate': stats['success_rate'],
        }

    # Get convergence curve
    convergence = [-s.best_fitness for s in trace.snapshots]  # Convert back to accuracy

    # Best accuracy found
    best_accuracy = -result['best_fitness']

    return HPOResult(
        algorithm=alg_name,
        dataset=dataset_name,
        run_id=run_id,
        best_accuracy=best_accuracy,
        best_params=hpo_objective.best_params,
        convergence_curve=convergence,
        shap_values=explanation.shap_values,
        operator_stats=op_stats,
        execution_time=execution_time,
    )


def run_full_hpo_study(config: HPOConfig, verbose: bool = True, resume: bool = True):
    """Run complete HPO study across all datasets and algorithms with checkpoint support."""

    algorithms = {
        "DE": InstrumentedDE,
        "GA": InstrumentedGA,
        "PSO": InstrumentedPSO,
    }

    os.makedirs(config.output_dir, exist_ok=True)

    # Try to load checkpoint
    checkpoint = None
    if resume:
        checkpoint = load_hpo_checkpoint(config.output_dir)
        if checkpoint:
            print(f"\n*** RESUMING FROM CHECKPOINT ***")
            print(f"Progress: {checkpoint.get_progress_string()}")
            print(f"Last update: {datetime.fromtimestamp(checkpoint.last_update).strftime('%Y-%m-%d %H:%M:%S')}")
            print()

    # Create new checkpoint if not resuming
    if checkpoint is None:
        checkpoint = HPOCheckpoint(
            config=config,
            completed_configs=set(),
            results=[],
            start_time=time.time(),
            last_update=time.time()
        )

    if verbose:
        print("=" * 70)
        print("XMH HYPERPARAMETER OPTIMIZATION STUDY")
        print("=" * 70)
        print(f"Datasets: {config.datasets}")
        print(f"Algorithms: {list(algorithms.keys())}")
        print(f"Hyperparameters: {len(config.hyperparameters)}")
        print(f"Population: {config.pop_size}, Generations: {config.max_generations}")
        print(f"Runs per config: {config.n_runs}")
        print(f"Checkpoint: {config.output_dir}/{CHECKPOINT_FILE}")
        print()

    for dataset_name in config.datasets:
        if verbose:
            print(f"\n{'='*60}")
            print(f"DATASET: {dataset_name}")
            print(f"{'='*60}")

        # Load dataset
        try:
            X, y, description = load_dataset(dataset_name)
            if verbose:
                print(f"  {description}")
                print(f"  Classes: {len(np.unique(y))}")
        except Exception as e:
            print(f"  ERROR loading dataset: {e}")
            continue

        # Create HPO objective
        hpo_objective = XGBoostHPO(
            X, y,
            config.hyperparameters,
            cv_folds=config.cv_folds
        )

        for alg_name, alg_class in algorithms.items():
            # Check if already completed
            if (dataset_name, alg_name) in checkpoint.completed_configs:
                if verbose:
                    print(f"\n  [{alg_name}] SKIPPED (already completed)")
                continue

            if verbose:
                print(f"\n  [{alg_name}]", end=" ")

            alg_results = []
            for run_id in range(config.n_runs):
                if verbose:
                    print(f".", end="", flush=True)

                result = run_hpo_experiment(
                    alg_class, alg_name, hpo_objective,
                    dataset_name, config, run_id
                )
                alg_results.append(result)

            # Summary for this algorithm
            accuracies = [r.best_accuracy for r in alg_results]
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)

            if verbose:
                print(f" Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

            # SHAP analysis
            all_shap = {}
            for r in alg_results:
                for op, val in r.shap_values.items():
                    if op not in all_shap:
                        all_shap[op] = []
                    all_shap[op].append(val)

            if verbose:
                print(f"    SHAP contributions:")
                for op, vals in sorted(all_shap.items(), key=lambda x: -np.mean(x[1])):
                    print(f"      {op}: {np.mean(vals):.2f} ± {np.std(vals):.2f}")

            # Add results and save checkpoint
            checkpoint.results.extend(alg_results)
            checkpoint.completed_configs.add((dataset_name, alg_name))
            save_hpo_checkpoint(checkpoint, config.output_dir)

            if verbose:
                print(f"    [Checkpoint saved: {checkpoint.get_progress_string()}]")

    # Save final results
    if verbose:
        print(f"\n\nSaving results to {config.output_dir}/...")

    save_hpo_results(checkpoint.results, config)

    # Remove checkpoint after successful completion
    checkpoint_path = os.path.join(config.output_dir, CHECKPOINT_FILE)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        if verbose:
            print("Checkpoint removed (experiment completed)")

    # Print summary
    print_hpo_summary(checkpoint.results, config)

    return checkpoint.results


def save_hpo_results(results: List[HPOResult], config: HPOConfig):
    """Save HPO results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON
    results_data = []
    for r in results:
        data = {
            'algorithm': r.algorithm,
            'dataset': r.dataset,
            'run_id': r.run_id,
            'best_accuracy': r.best_accuracy,
            'best_params': r.best_params,
            'shap_values': r.shap_values,
            'operator_stats': r.operator_stats,
            'execution_time': r.execution_time,
        }
        results_data.append(data)

    with open(f"{config.output_dir}/hpo_results_{timestamp}.json", 'w') as f:
        json.dump(results_data, f, indent=2)

    # Save convergence curves
    convergence_data = {}
    for r in results:
        key = f"{r.dataset}_{r.algorithm}_{r.run_id}"
        convergence_data[key] = r.convergence_curve

    with open(f"{config.output_dir}/convergence_{timestamp}.json", 'w') as f:
        json.dump(convergence_data, f)

    print(f"  Results saved to {config.output_dir}/")


def print_hpo_summary(results: List[HPOResult], config: HPOConfig):
    """Print summary of HPO results."""
    print("\n" + "=" * 70)
    print("HPO RESULTS SUMMARY")
    print("=" * 70)

    # Group by dataset
    datasets = list(set(r.dataset for r in results))
    algorithms = list(set(r.algorithm for r in results))

    print("\n1. BEST ACCURACY BY DATASET AND ALGORITHM")
    print("-" * 70)
    print(f"{'Dataset':<15} {'DE':>12} {'GA':>12} {'PSO':>12} {'Best':>10}")
    print("-" * 70)

    for dataset in sorted(datasets):
        row = [dataset]
        best_alg = None
        best_acc = 0

        for alg in ['DE', 'GA', 'PSO']:
            accs = [r.best_accuracy for r in results
                   if r.dataset == dataset and r.algorithm == alg]
            if accs:
                mean_acc = np.mean(accs)
                row.append(f"{mean_acc:.4f}")
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_alg = alg
            else:
                row.append("-")

        row.append(best_alg)
        print(f"{row[0]:<15} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>10}")

    # SHAP analysis across all datasets
    print("\n\n2. OPERATOR CONTRIBUTIONS (SHAP VALUES)")
    print("-" * 70)

    for alg in algorithms:
        print(f"\n{alg}:")
        alg_results = [r for r in results if r.algorithm == alg]

        all_shap = {}
        for r in alg_results:
            for op, val in r.shap_values.items():
                if op not in all_shap:
                    all_shap[op] = []
                all_shap[op].append(val)

        total = sum(np.mean(vals) for vals in all_shap.values())
        for op, vals in sorted(all_shap.items(), key=lambda x: -np.mean(x[1])):
            mean_val = np.mean(vals)
            pct = 100 * mean_val / total if total > 0 else 0
            print(f"  {op:<25} {mean_val:>8.2f} ({pct:>5.1f}%)")

    # Best hyperparameters found
    print("\n\n3. BEST HYPERPARAMETERS FOUND")
    print("-" * 70)

    for dataset in sorted(datasets):
        dataset_results = [r for r in results if r.dataset == dataset]
        best_result = max(dataset_results, key=lambda r: r.best_accuracy)

        print(f"\n{dataset} (Best: {best_result.algorithm}, Acc: {best_result.best_accuracy:.4f}):")
        if best_result.best_params:
            for param, value in best_result.best_params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.6f}")
                else:
                    print(f"  {param}: {value}")


def main():
    """Run HPO study."""
    import argparse

    parser = argparse.ArgumentParser(description="XMH HPO Study")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (2 runs, 2 datasets)")
    parser.add_argument("--output", type=str, default="results_hpo",
                        help="Output directory")

    args = parser.parse_args()

    config = HPOConfig()
    config.output_dir = args.output

    if args.quick:
        config.n_runs = 2
        config.max_generations = 20
        config.datasets = ["breast_cancer", "wine"]
        print("Running in QUICK mode")

    run_full_hpo_study(config)


if __name__ == "__main__":
    main()
