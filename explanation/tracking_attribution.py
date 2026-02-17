"""
Tracking-based operator attribution (EvoMapX-style).

Each fitness improvement is directly attributed to the operator that
produced it, then normalised so values sum to the total improvement.
This provides a lightweight, single-run comparison baseline against
game-theoretic methods like Shapley values.

Reference:
    Abed-alguni (2025). EvoMapX - tracking-based attribution for
    evolutionary algorithms.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.trace_logger import TraceLogger, EventType


@dataclass
class TrackingResult:
    """Result of tracking-based attribution."""

    attributions: Dict[str, float]
    total_improvement: float
    operator_names: List[str]
    raw_improvements: Dict[str, float]

    def get_ranking(self) -> List[Tuple[str, float]]:
        """Operators ranked by attribution (descending)."""
        return sorted(
            self.attributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

    def to_dict(self) -> dict:
        return {
            "attributions": self.attributions,
            "total_improvement": self.total_improvement,
            "ranking": self.get_ranking(),
            "raw_improvements": self.raw_improvements,
        }

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "TRACKING-BASED ATTRIBUTION",
            "=" * 60,
            f"Total improvement: {self.total_improvement:.6e}",
            "",
            "OPERATOR ATTRIBUTIONS (ranked by importance):",
        ]
        for op, value in self.get_ranking():
            pct = (
                abs(value) / abs(self.total_improvement) * 100
                if self.total_improvement != 0
                else 0
            )
            bar = "#" * int(pct / 5)
            lines.append(f"  {op:30s} {value:+12.6e} ({pct:5.1f}%) {bar}")
        lines.append("=" * 60)
        return "\n".join(lines)


class TrackingAttribution:
    """
    Attribute fitness improvements directly to the operator that
    produced each improvement.

    This mirrors the approach of EvoMapX: whenever an operator
    application leads to a fitness improvement, the magnitude
    of that improvement is credited to the operator.  The final
    values are normalised so that they sum to the total
    improvement observed during the run.
    """

    def explain_from_trace(
        self,
        trace: TraceLogger,
        baseline_fitness: Optional[float] = None,
    ) -> TrackingResult:
        """
        Compute tracking-based attributions from an execution trace.

        Args:
            trace: A TraceLogger instance with recorded execution data.
            baseline_fitness: Starting fitness.  Defaults to the best
                fitness of the first generation snapshot.

        Returns:
            TrackingResult with normalised operator attributions.
        """
        if not trace.snapshots:
            raise ValueError("Trace has no recorded data")

        if baseline_fitness is None:
            baseline_fitness = trace.snapshots[0].best_fitness

        final_fitness = trace.snapshots[-1].best_fitness
        total_improvement = baseline_fitness - final_fitness

        # Accumulate raw improvements per operator from events
        raw_improvements: Dict[str, float] = {}

        for event in trace.events:
            if event.event_type != EventType.OPERATOR_APPLICATION:
                continue
            if not event.data.get("success", False):
                continue

            op_name = event.data.get("operator", "")
            improvement = event.data.get("details", {}).get("improvement", 0.0)

            if improvement > 0:
                raw_improvements[op_name] = (
                    raw_improvements.get(op_name, 0.0) + improvement
                )

        # Fall back to snapshot-level data when events lack detail
        if not raw_improvements:
            contributions = trace.get_operator_contributions()
            for op, stats in contributions.items():
                raw_improvements[op] = stats.get("cumulative_improvement", 0.0)

        # Normalise so attributions sum to total_improvement
        raw_total = sum(raw_improvements.values())
        attributions: Dict[str, float] = {}

        if raw_total > 0 and total_improvement != 0:
            for op, raw in raw_improvements.items():
                attributions[op] = (raw / raw_total) * total_improvement
        else:
            # Equal split when no improvement data
            n_ops = max(len(raw_improvements), 1)
            for op in raw_improvements:
                attributions[op] = total_improvement / n_ops

        return TrackingResult(
            attributions=attributions,
            total_improvement=total_improvement,
            operator_names=list(attributions.keys()),
            raw_improvements=raw_improvements,
        )
