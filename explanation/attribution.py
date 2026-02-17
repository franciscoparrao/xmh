"""
Attribution Analysis Module

Provides simple attribution methods for quick analysis of operator contributions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class AttributionResult:
    """Result of attribution analysis."""
    operator_contributions: Dict[str, float]
    total_improvement: float
    generation_contributions: List[float]

    def get_top_contributors(self, k: int = 3) -> List[Tuple[str, float]]:
        """Get top k contributing operators."""
        sorted_ops = sorted(
            self.operator_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_ops[:k]

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "ATTRIBUTION ANALYSIS",
            "=" * 40,
            f"Total improvement: {self.total_improvement:.6e}",
            "",
            "Operator contributions:"
        ]

        for op, contrib in sorted(
            self.operator_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            pct = (contrib / self.total_improvement * 100
                   if self.total_improvement != 0 else 0)
            lines.append(f"  {op}: {contrib:.6e} ({pct:.1f}%)")

        return "\n".join(lines)


class AttributionAnalyzer:
    """
    Simple attribution analysis based on trace data.

    Provides fast, approximate attribution without requiring
    re-execution of the algorithm.
    """

    def __init__(self):
        pass

    def analyze(self, trace) -> AttributionResult:
        """
        Analyze operator contributions from trace.

        Args:
            trace: TraceLogger with recorded execution

        Returns:
            AttributionResult with contribution analysis
        """
        if not trace.snapshots:
            raise ValueError("Trace has no recorded data")

        # Get contributions
        contributions = trace.get_operator_contributions()

        # Calculate improvement per generation
        gens, best_fitness, _ = trace.get_fitness_history()
        gen_improvements = []
        for i in range(1, len(best_fitness)):
            improvement = best_fitness[i-1] - best_fitness[i]
            gen_improvements.append(max(0, improvement))

        total_improvement = sum(gen_improvements)

        # Distribute improvement based on success rates
        operator_contributions = {}
        total_success = sum(c["total_success"] for c in contributions.values())

        for op, stats in contributions.items():
            if total_success > 0:
                proportion = stats["total_success"] / total_success
            else:
                proportion = 1.0 / len(contributions)

            operator_contributions[op] = total_improvement * proportion

        return AttributionResult(
            operator_contributions=operator_contributions,
            total_improvement=total_improvement,
            generation_contributions=gen_improvements
        )

    def compare_runs(
        self,
        traces: List,
        labels: Optional[List[str]] = None
    ) -> str:
        """
        Compare attribution across multiple runs.

        Args:
            traces: List of TraceLogger instances
            labels: Optional labels for each run

        Returns:
            Comparison report as string
        """
        if labels is None:
            labels = [f"Run {i+1}" for i in range(len(traces))]

        results = [self.analyze(trace) for trace in traces]

        # Collect all operators
        all_operators = set()
        for r in results:
            all_operators.update(r.operator_contributions.keys())

        lines = [
            "ATTRIBUTION COMPARISON",
            "=" * 60,
            ""
        ]

        # Header
        header = f"{'Operator':<25}"
        for label in labels:
            header += f" {label:>12}"
        lines.append(header)
        lines.append("-" * 60)

        # Data rows
        for op in sorted(all_operators):
            row = f"{op:<25}"
            for r in results:
                val = r.operator_contributions.get(op, 0)
                row += f" {val:>12.4e}"
            lines.append(row)

        # Total
        lines.append("-" * 60)
        row = f"{'TOTAL':<25}"
        for r in results:
            row += f" {r.total_improvement:>12.4e}"
        lines.append(row)

        return "\n".join(lines)
