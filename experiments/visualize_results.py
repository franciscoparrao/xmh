#!/usr/bin/env python3
"""
Visualization Script for XMH Experimental Results

Generates publication-quality figures for IEEE TEVC paper.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette for algorithms
COLORS = {
    'DE': '#2ecc71',   # Green
    'GA': '#3498db',   # Blue
    'PSO': '#e74c3c',  # Red
}

MARKERS = {
    'DE': 'o',
    'GA': 's',
    'PSO': '^',
}


def load_results(results_dir: str):
    """Load all result files from directory."""
    results_dir = Path(results_dir)

    # Find latest files
    run_files = list(results_dir.glob("run_results_*.csv"))
    stat_files = list(results_dir.glob("statistical_results_*.json"))
    shap_files = list(results_dir.glob("shap_validation_*.json"))
    ablation_files = list(results_dir.glob("ablation_results_*.csv"))

    if not run_files:
        raise FileNotFoundError(f"No run_results found in {results_dir}")

    # Load data
    run_df = pd.read_csv(sorted(run_files)[-1])

    with open(sorted(stat_files)[-1]) as f:
        stat_results = json.load(f)

    with open(sorted(shap_files)[-1]) as f:
        shap_results = json.load(f)

    ablation_df = pd.read_csv(sorted(ablation_files)[-1])

    return run_df, stat_results, shap_results, ablation_df


def plot_algorithm_wins_heatmap(stat_results, output_dir):
    """Create heatmap showing which algorithm wins each configuration."""
    # Extract data
    functions = []
    dimensions = []
    winners = []

    for result in stat_results:
        functions.append(result['function'])
        dimensions.append(result['dimension'])
        winners.append(result['best_algorithm'])

    # Create DataFrame
    df = pd.DataFrame({
        'Function': functions,
        'Dimension': dimensions,
        'Winner': winners
    })

    # Pivot for heatmap
    pivot = df.pivot(index='Function', columns='Dimension', values='Winner')

    # Map to numeric for coloring
    winner_map = {'DE': 0, 'GA': 1, 'PSO': 2}
    numeric_pivot = pivot.replace(winner_map)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))

    # Custom colormap
    colors = [COLORS['DE'], COLORS['GA'], COLORS['PSO']]
    cmap = LinearSegmentedColormap.from_list('algorithms', colors, N=3)

    # Plot heatmap
    sns.heatmap(numeric_pivot.astype(float), annot=pivot, fmt='',
                cmap=cmap, vmin=-0.5, vmax=2.5,
                cbar=False, ax=ax,
                linewidths=2, linecolor='white')

    ax.set_xlabel('Dimension', fontweight='bold')
    ax.set_ylabel('Function', fontweight='bold')
    ax.set_title('Best Algorithm per Configuration\n(Friedman test, p < 0.0001)',
                 fontweight='bold', pad=15)

    # Legend
    patches = [mpatches.Patch(color=COLORS[alg], label=alg) for alg in ['DE', 'GA', 'PSO']]
    ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_algorithm_wins_heatmap.png')
    plt.savefig(output_dir / 'fig1_algorithm_wins_heatmap.pdf')
    plt.close()
    print("  Generated: fig1_algorithm_wins_heatmap.png/pdf")


def plot_performance_comparison(run_df, output_dir):
    """Create bar charts comparing algorithm performance."""
    functions = run_df['function'].unique()
    dimensions = run_df['dimension'].unique()

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    for idx, func in enumerate(functions):
        ax = axes[idx]

        func_data = run_df[run_df['function'] == func]

        # Prepare data for grouped bar chart
        x = np.arange(len(dimensions))
        width = 0.25

        for i, alg in enumerate(['DE', 'GA', 'PSO']):
            means = []
            stds = []
            for dim in dimensions:
                data = func_data[(func_data['dimension'] == dim) &
                                (func_data['algorithm'] == alg)]['best_fitness']
                means.append(data.mean())
                stds.append(data.std())

            bars = ax.bar(x + i*width, means, width, label=alg,
                         color=COLORS[alg], alpha=0.8,
                         yerr=stds, capsize=3, error_kw={'linewidth': 1})

        ax.set_xlabel('Dimension')
        ax.set_ylabel('Fitness (log scale)')
        ax.set_title(func.capitalize(), fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(dimensions)
        ax.set_yscale('log')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Algorithm Performance Comparison\n(Mean ± Std, 30 runs)',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_performance_comparison.png')
    plt.savefig(output_dir / 'fig2_performance_comparison.pdf')
    plt.close()
    print("  Generated: fig2_performance_comparison.png/pdf")


def plot_boxplots_by_dimension(run_df, output_dir):
    """Create boxplots showing distribution by dimension."""
    dimensions = sorted(run_df['dimension'].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, dim in enumerate(dimensions):
        ax = axes[idx]
        dim_data = run_df[run_df['dimension'] == dim]

        # Normalize fitness for comparison across functions
        normalized_data = []
        labels = []
        colors_list = []

        for func in dim_data['function'].unique():
            func_data = dim_data[dim_data['function'] == func]
            # Normalize by best value in this function
            best = func_data['best_fitness'].min()

            for alg in ['DE', 'GA', 'PSO']:
                alg_data = func_data[func_data['algorithm'] == alg]['best_fitness']
                # Log ratio to best
                if best > 0:
                    normalized = np.log10(alg_data / best + 1)
                else:
                    normalized = alg_data
                normalized_data.append(normalized)
                labels.append(alg)
                colors_list.append(COLORS[alg])

        bp = ax.boxplot(normalized_data, patch_artist=True, labels=labels[:3] * len(dim_data['function'].unique()))

        # Color boxes
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f'D = {dim}', fontweight='bold', fontsize=13)
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Normalized Performance\n(log₁₀ ratio to best)')
        ax.set_xticks([2, 5, 8, 11, 14, 17, 20, 23, 26])
        ax.set_xticklabels(['DE', 'GA', 'PSO'] * 3)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Performance Distribution by Dimension\n(All functions, normalized)',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_boxplots_by_dimension.png')
    plt.savefig(output_dir / 'fig3_boxplots_by_dimension.pdf')
    plt.close()
    print("  Generated: fig3_boxplots_by_dimension.png/pdf")


def plot_ranking_summary(stat_results, output_dir):
    """Create ranking summary visualization."""
    # Extract mean ranks
    algorithms = ['DE', 'GA', 'PSO']

    # Group by function
    functions = list(set(r['function'] for r in stat_results))
    dimensions = list(set(r['dimension'] for r in stat_results))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for idx, dim in enumerate(sorted(dimensions)):
        ax = axes[idx]

        dim_results = [r for r in stat_results if r['dimension'] == dim]

        func_names = []
        de_ranks = []
        ga_ranks = []
        pso_ranks = []

        for r in sorted(dim_results, key=lambda x: x['function']):
            func_names.append(r['function'].capitalize())
            ranks = r['mean_ranks']
            de_ranks.append(ranks.get('DE', 3))
            ga_ranks.append(ranks.get('GA', 3))
            pso_ranks.append(ranks.get('PSO', 3))

        x = np.arange(len(func_names))
        width = 0.25

        ax.barh(x - width, de_ranks, width, label='DE', color=COLORS['DE'], alpha=0.8)
        ax.barh(x, ga_ranks, width, label='GA', color=COLORS['GA'], alpha=0.8)
        ax.barh(x + width, pso_ranks, width, label='PSO', color=COLORS['PSO'], alpha=0.8)

        ax.set_xlabel('Mean Rank (lower is better)')
        ax.set_yticks(x)
        ax.set_yticklabels(func_names)
        ax.set_title(f'D = {dim}', fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim(0.5, 3.5)
        ax.axvline(x=2, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Friedman Mean Ranks by Dimension\n(Lower rank = better performance)',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_ranking_summary.png')
    plt.savefig(output_dir / 'fig4_ranking_summary.pdf')
    plt.close()
    print("  Generated: fig4_ranking_summary.png/pdf")


def plot_win_count_pie(stat_results, output_dir):
    """Create pie chart of total wins."""
    wins = {'DE': 0, 'GA': 0, 'PSO': 0}

    for r in stat_results:
        wins[r['best_algorithm']] += 1

    fig, ax = plt.subplots(figsize=(8, 8))

    labels = list(wins.keys())
    sizes = list(wins.values())
    colors = [COLORS[alg] for alg in labels]
    explode = (0.02, 0.02, 0.1)  # Explode PSO slightly

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                       colors=colors, autopct='%1.1f%%',
                                       shadow=True, startangle=90,
                                       textprops={'fontsize': 14, 'fontweight': 'bold'})

    # Make percentage text white
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('Distribution of Wins\n(27 configurations total)',
                 fontweight='bold', fontsize=14)

    # Add count legend
    legend_labels = [f'{alg}: {wins[alg]} wins' for alg in labels]
    ax.legend(wedges, legend_labels, loc='lower right', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_win_distribution.png')
    plt.savefig(output_dir / 'fig5_win_distribution.pdf')
    plt.close()
    print("  Generated: fig5_win_distribution.png/pdf")


def plot_scalability_analysis(run_df, output_dir):
    """Analyze how algorithms scale with dimension."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Select representative functions
    selected_funcs = ['sphere', 'rastrigin', 'ackley',
                      'rosenbrock', 'schwefel', 'levy']

    for idx, func in enumerate(selected_funcs):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        func_data = run_df[run_df['function'] == func]
        dimensions = sorted(func_data['dimension'].unique())

        for alg in ['DE', 'GA', 'PSO']:
            means = []
            stds = []
            for dim in dimensions:
                data = func_data[(func_data['dimension'] == dim) &
                                (func_data['algorithm'] == alg)]['best_fitness']
                means.append(data.mean())
                stds.append(data.std())

            means = np.array(means)
            stds = np.array(stds)

            ax.plot(dimensions, means, marker=MARKERS[alg],
                   color=COLORS[alg], label=alg, linewidth=2, markersize=8)
            ax.fill_between(dimensions, means - stds, means + stds,
                           color=COLORS[alg], alpha=0.2)

        ax.set_xlabel('Dimension')
        ax.set_ylabel('Fitness')
        ax.set_title(func.capitalize(), fontweight='bold')
        ax.set_yscale('log')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Scalability Analysis: Performance vs Dimension\n(Mean ± Std, 30 runs)',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_scalability_analysis.png')
    plt.savefig(output_dir / 'fig6_scalability_analysis.pdf')
    plt.close()
    print("  Generated: fig6_scalability_analysis.png/pdf")


def plot_ablation_analysis(ablation_df, output_dir):
    """Visualize ablation study results."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    algorithms = ['DE', 'GA', 'PSO']

    for idx, alg in enumerate(algorithms):
        ax = axes[idx]

        alg_data = ablation_df[ablation_df['algorithm'] == alg]

        # Average contribution across all functions and dimensions
        avg_contribution = alg_data.groupby('operator')['contribution'].mean()
        avg_contribution = avg_contribution.sort_values(ascending=True)

        colors = [COLORS[alg] if c > 0 else '#95a5a6' for c in avg_contribution]

        bars = ax.barh(range(len(avg_contribution)), avg_contribution.values,
                      color=colors, alpha=0.8)

        ax.set_yticks(range(len(avg_contribution)))
        ax.set_yticklabels(avg_contribution.index)
        ax.set_xlabel('Contribution (fitness degradation when disabled)')
        ax.set_title(f'{alg} Operators', fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Ablation Study: Operator Contributions\n(Higher = more important)',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_ablation_analysis.png')
    plt.savefig(output_dir / 'fig7_ablation_analysis.pdf')
    plt.close()
    print("  Generated: fig7_ablation_analysis.png/pdf")


def generate_latex_table(stat_results, output_dir):
    """Generate LaTeX table for paper."""
    functions = sorted(list(set(r['function'] for r in stat_results)))
    dimensions = sorted(list(set(r['dimension'] for r in stat_results)))

    # Build table
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Best algorithm per configuration (Friedman test, $p < 0.0001$)}",
        r"\label{tab:best_algorithms}",
        r"\begin{tabular}{l" + "c" * len(dimensions) + "}",
        r"\toprule",
        r"Function & " + " & ".join([f"D={d}" for d in dimensions]) + r" \\",
        r"\midrule",
    ]

    for func in functions:
        row = [func.capitalize()]
        for dim in dimensions:
            result = next((r for r in stat_results
                          if r['function'] == func and r['dimension'] == dim), None)
            if result:
                winner = result['best_algorithm']
                row.append(r"\textbf{" + winner + "}" if winner != 'PSO' else winner)
            else:
                row.append("-")
        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_dir / 'table_best_algorithms.tex', 'w') as f:
        f.write('\n'.join(lines))

    print("  Generated: table_best_algorithms.tex")


def main():
    """Generate all visualizations."""
    results_dir = Path("results_full")
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    print("Loading results...")
    run_df, stat_results, shap_results, ablation_df = load_results(results_dir)

    print(f"  Loaded {len(run_df)} run results")
    print(f"  Loaded {len(stat_results)} statistical results")
    print(f"  Loaded {len(ablation_df)} ablation results")

    print("\nGenerating visualizations...")

    # Generate all figures
    plot_algorithm_wins_heatmap(stat_results, output_dir)
    plot_performance_comparison(run_df, output_dir)
    plot_boxplots_by_dimension(run_df, output_dir)
    plot_ranking_summary(stat_results, output_dir)
    plot_win_count_pie(stat_results, output_dir)
    plot_scalability_analysis(run_df, output_dir)
    plot_ablation_analysis(ablation_df, output_dir)

    # Generate LaTeX table
    generate_latex_table(stat_results, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nFigures generated:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
