"""
Visualization Functions for XMH

Provides plotting utilities for analyzing and explaining search processes.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_convergence(
    trace,
    ax: Optional[Any] = None,
    show_mean: bool = True,
    log_scale: bool = True,
    title: str = "Convergence Plot"
) -> Any:
    """
    Plot fitness convergence over generations.

    Args:
        trace: TraceLogger instance
        ax: Matplotlib axes (creates new if None)
        show_mean: Whether to show mean fitness
        log_scale: Use logarithmic y-axis
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    gens, best, mean = trace.get_fitness_history()

    ax.plot(gens, best, 'b-', linewidth=2, label='Best fitness')

    if show_mean:
        ax.plot(gens, mean, 'r--', linewidth=1, alpha=0.7, label='Mean fitness')
        ax.fill_between(gens, best, mean, alpha=0.2, color='blue')

    if log_scale and all(f > 0 for f in best):
        ax.set_yscale('log')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_diversity(
    trace,
    ax: Optional[Any] = None,
    title: str = "Population Diversity"
) -> Any:
    """
    Plot population diversity over generations.

    Args:
        trace: TraceLogger instance
        ax: Matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    gens, diversity = trace.get_diversity_history()

    ax.plot(gens, diversity, 'g-', linewidth=2)
    ax.fill_between(gens, 0, diversity, alpha=0.3, color='green')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Diversity', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    return ax


def plot_operator_usage(
    trace,
    ax: Optional[Any] = None,
    title: str = "Operator Usage"
) -> Any:
    """
    Plot operator usage as stacked area chart.

    Args:
        trace: TraceLogger instance
        ax: Matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    contributions = trace.get_operator_contributions()

    if not contributions:
        ax.text(0.5, 0.5, 'No operator data available',
                ha='center', va='center', transform=ax.transAxes)
        return ax

    operators = list(contributions.keys())
    values = [contributions[op]['total_used'] for op in operators]

    colors = plt.cm.Set3(np.linspace(0, 1, len(operators)))

    bars = ax.bar(operators, values, color=colors)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val}', ha='center', va='bottom')

    ax.set_xlabel('Operator', fontsize=12)
    ax.set_ylabel('Usage Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')

    return ax


def plot_shap_values(
    explanation,
    ax: Optional[Any] = None,
    title: str = "Operator SHAP Values"
) -> Any:
    """
    Plot SHAP values as horizontal bar chart.

    Args:
        explanation: SHAPExplanation instance
        ax: Matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by absolute value
    sorted_items = sorted(
        explanation.shap_values.items(),
        key=lambda x: abs(x[1])
    )
    operators = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    # Color by positive/negative
    colors = ['green' if v > 0 else 'red' for v in values]

    y_pos = np.arange(len(operators))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(operators)
    ax.axvline(x=0, color='black', linewidth=0.5)

    ax.set_xlabel('SHAP Value (contribution to fitness improvement)', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add legend
    pos_patch = mpatches.Patch(color='green', alpha=0.7, label='Positive contribution')
    neg_patch = mpatches.Patch(color='red', alpha=0.7, label='Negative contribution')
    ax.legend(handles=[pos_patch, neg_patch], loc='lower right')

    return ax


def plot_operator_heatmap(
    trace,
    ax: Optional[Any] = None,
    window_size: int = 10,
    title: str = "Operator Success Rate Over Time"
) -> Any:
    """
    Plot heatmap of operator success rates over generations.

    Args:
        trace: TraceLogger instance
        ax: Matplotlib axes (creates new if None)
        window_size: Window size for rolling average
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    # Build matrix of success rates
    operators = set()
    for snapshot in trace.snapshots:
        operators.update(snapshot.operators_used.keys())

    operators = sorted(operators)
    n_gens = len(trace.snapshots)

    if n_gens == 0 or len(operators) == 0:
        ax.text(0.5, 0.5, 'Insufficient data for heatmap',
                ha='center', va='center', transform=ax.transAxes)
        return ax

    # Create success rate matrix
    matrix = np.zeros((len(operators), n_gens))

    for g, snapshot in enumerate(trace.snapshots):
        for i, op in enumerate(operators):
            used = snapshot.operators_used.get(op, 0)
            success = snapshot.operators_success.get(op, 0)
            if used > 0:
                matrix[i, g] = success / used

    # Apply smoothing
    if window_size > 1 and n_gens > window_size:
        smoothed = np.zeros_like(matrix)
        for i in range(len(operators)):
            smoothed[i] = np.convolve(
                matrix[i],
                np.ones(window_size)/window_size,
                mode='same'
            )
        matrix = smoothed

    # Plot heatmap
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=1, interpolation='nearest')

    ax.set_yticks(np.arange(len(operators)))
    ax.set_yticklabels(operators)
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Operator', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Success Rate', fontsize=10)

    return ax


def create_summary_figure(
    trace,
    explanation=None,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> Any:
    """
    Create a comprehensive summary figure with multiple plots.

    Args:
        trace: TraceLogger instance
        explanation: Optional SHAPExplanation
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure object
    """
    _check_matplotlib()

    fig = plt.figure(figsize=figsize)

    # Determine layout based on available data
    if explanation is not None:
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[2, :])
    else:
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = None

    # Plot 1: Convergence
    plot_convergence(trace, ax=ax1, title="Fitness Convergence")

    # Plot 2: Diversity
    plot_diversity(trace, ax=ax2, title="Population Diversity")

    # Plot 3: Operator Usage
    plot_operator_usage(trace, ax=ax3, title="Operator Usage")

    # Plot 4: Operator Heatmap
    plot_operator_heatmap(trace, ax=ax4, title="Operator Success Over Time")

    # Plot 5: SHAP values (if available)
    if explanation is not None and ax5 is not None:
        plot_shap_values(explanation, ax=ax5, title="Operator SHAP Values")

    # Overall title
    fig.suptitle(
        f"XMH Analysis: {trace.algorithm_name} on {trace.problem_name}",
        fontsize=16, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_counterfactuals(
    counterfactuals: List,
    trace,
    ax: Optional[Any] = None,
    title: str = "Counterfactual Analysis"
) -> Any:
    """
    Visualize counterfactual predictions vs actual trajectory.

    Args:
        counterfactuals: List of Counterfactual objects
        trace: TraceLogger instance
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual trajectory
    gens, best, _ = trace.get_fitness_history()
    ax.plot(gens, best, 'b-', linewidth=2, label='Actual trajectory')

    # Plot counterfactual points
    for i, cf in enumerate(counterfactuals):
        gen = cf.generation
        predicted = cf.predicted_outcome

        # Find actual fitness at that generation
        if gen < len(best):
            actual_at_gen = best[gen]

            # Plot counterfactual point
            ax.scatter([gen], [predicted], s=100, marker='*',
                      c='orange', zorder=5, label=f'CF {i+1}: {cf.cf_type.name}' if i < 3 else '')

            # Draw line from actual to predicted
            ax.plot([gen, gen], [actual_at_gen, predicted],
                   'r--', alpha=0.5)

            # If validated, show actual outcome
            if cf.validated and cf.actual_outcome is not None:
                ax.scatter([gen], [cf.actual_outcome], s=100, marker='o',
                          c='green', alpha=0.7, zorder=5)

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    return ax
