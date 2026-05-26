#!/usr/bin/env python3
"""Regenerate publication-quality figures for the XMH paper (Swarm and Evolutionary Computation submission).

Usage:
    python generate_paper_figures.py

Outputs PDFs to ./figures/ in the same directory as this script.
Reads the three result JSONs from ./data/ (shipped alongside this script).
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'figures')
DATA_DIR = os.path.join(HERE, 'data')
os.makedirs(OUTDIR, exist_ok=True)

with open(os.path.join(DATA_DIR, 'exact_shapley_results.json')) as f:
    exact_data = json.load(f)

with open(os.path.join(DATA_DIR, 'method_comparison.json')) as f:
    method_data = json.load(f)

with open(os.path.join(DATA_DIR, 'overhead_results.json')) as f:
    overhead_data = json.load(f)

# Optional: exp8 discriminant data (Information Sciences resub)
exp8_part_a = None
exp8_part_b = None
_p_a = os.path.join(DATA_DIR, 'exp8_part_a.json')
_p_b = os.path.join(DATA_DIR, 'exp8_part_b.json')
if os.path.exists(_p_a) and os.path.exists(_p_b):
    with open(_p_a) as f:
        exp8_part_a = json.load(f)
    with open(_p_b) as f:
        exp8_part_b = json.load(f)

# Optional: exp9 budget sensitivity data
exp9_sweep_a = None
exp9_sweep_b = None
_q_a = os.path.join(DATA_DIR, 'exp9_sweep_a.json')
_q_b = os.path.join(DATA_DIR, 'exp9_sweep_b.json')
if os.path.exists(_q_a) and os.path.exists(_q_b):
    with open(_q_a) as f:
        exp9_sweep_a = json.load(f)
    with open(_q_b) as f:
        exp9_sweep_b = json.load(f)

FUNCTIONS = ['sphere', 'rosenbrock', 'zakharov', 'dixon_price',
             'rastrigin', 'schwefel', 'ackley', 'griewank', 'levy']
FUNC_LABELS = ['Sphere', 'Rosenbrock', 'Zakharov', 'Dixon-P.',
               'Rastrigin', 'Schwefel', 'Ackley', 'Griewank', 'Levy']
DIMS = ['10', '30', '50']


def get_shapley_pct(alg, func, dim):
    """Get Shapley values as percentages for a config."""
    try:
        config = exact_data[alg][func][dim]
        sv = config['shapley_values']
        total = sum(abs(v) for v in sv.values())
        if total < 1e-15:
            return {k: 0.0 for k in sv}
        return {k: 100.0 * v / total for k, v in sv.items()}
    except KeyError:
        return None


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: GA Shapley Heatmap (operator × function × dimension)
# ═══════════════════════════════════════════════════════════════════════
def fig1_ga_heatmap():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.5), sharey=True,
                             gridspec_kw={'wspace': 0.08, 'right': 0.88})

    op_names_display = ['Selection', 'Crossover', 'Mutation']
    op_keys = ['selection_tournament', 'crossover_sbx', 'mutation_polynomial']

    for ax_idx, (dim, ax) in enumerate(zip(DIMS, axes)):
        matrix = np.zeros((3, len(FUNCTIONS)))
        for f_idx, func in enumerate(FUNCTIONS):
            pct = get_shapley_pct('GA', func, dim)
            if pct:
                for o_idx, op_key in enumerate(op_keys):
                    matrix[o_idx, f_idx] = pct.get(op_key, 0)

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=60)

        # Annotate cells
        for i in range(3):
            for j in range(len(FUNCTIONS)):
                val = matrix[i, j]
                color = 'white' if val > 40 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                       fontsize=10, color=color, fontweight='bold')

        ax.set_xticks(range(len(FUNCTIONS)))
        ax.set_xticklabels(FUNC_LABELS, rotation=45, ha='right', fontsize=9)
        ax.set_title(f'D = {dim}', fontweight='bold', fontsize=11)

        if ax_idx == 0:
            ax.set_yticks(range(3))
            ax.set_yticklabels(op_names_display, fontsize=10)

    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])
    fig.colorbar(im, cax=cbar_ax, label='Shapley Value (%)')
    fig.suptitle('GA: Exact Shapley Values by Function and Dimension', fontweight='bold', y=0.98)
    plt.savefig(os.path.join(OUTDIR, 'fig1_ga_shapley_heatmap.pdf'))
    plt.close()
    print("  ✓ fig1_ga_shapley_heatmap.pdf")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: PSO Shapley Heatmap
# ═══════════════════════════════════════════════════════════════════════
def fig2_pso_heatmap():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.2), sharey=True,
                             gridspec_kw={'wspace': 0.08, 'right': 0.88, 'top': 0.82})

    op_names_display = ['Velocity', 'Topology']
    # PSO operators — need to find the actual keys
    # Check what keys PSO uses
    sample = exact_data.get('PSO', {}).get('sphere', {}).get('10', {})
    if sample:
        pso_ops = list(sample.get('shapley_values', {}).keys())
    else:
        pso_ops = ['velocity_constriction', 'topology_gbest']

    # Filter out position_update if present (deterministic)
    pso_ops_filtered = [op for op in pso_ops if 'position' not in op.lower()]
    if len(pso_ops_filtered) < 2:
        pso_ops_filtered = pso_ops[:2]

    for ax_idx, (dim, ax) in enumerate(zip(DIMS, axes)):
        matrix = np.zeros((len(pso_ops_filtered), len(FUNCTIONS)))
        for f_idx, func in enumerate(FUNCTIONS):
            pct = get_shapley_pct('PSO', func, dim)
            if pct:
                for o_idx, op_key in enumerate(pso_ops_filtered):
                    matrix[o_idx, f_idx] = pct.get(op_key, 0)

        im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=55)

        for i in range(len(pso_ops_filtered)):
            for j in range(len(FUNCTIONS)):
                val = matrix[i, j]
                color = 'white' if val > 50 else 'black'
                display_val = abs(val) if abs(val) < 0.5 else val
                ax.text(j, i, f'{display_val:.0f}', ha='center', va='center',
                       fontsize=10, color=color, fontweight='bold')

        ax.set_xticks(range(len(FUNCTIONS)))
        ax.set_xticklabels(FUNC_LABELS, rotation=45, ha='right', fontsize=9)
        ax.set_title(f'D = {dim}', fontweight='bold', fontsize=11)

        if ax_idx == 0:
            ax.set_yticks(range(len(pso_ops_filtered)))
            labels = []
            for op in pso_ops_filtered:
                if 'velocity' in op.lower():
                    labels.append('Velocity')
                elif 'topology' in op.lower():
                    labels.append('Topology')
                else:
                    labels.append(op)
            ax.set_yticklabels(labels)

    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])
    fig.colorbar(im, cax=cbar_ax, label='Shapley Value (%)')
    fig.suptitle('PSO: Exact Shapley Values by Function and Dimension', fontweight='bold', y=0.95)
    plt.savefig(os.path.join(OUTDIR, 'fig2_pso_shapley_heatmap.pdf'))
    plt.close()
    print("  ✓ fig2_pso_shapley_heatmap.pdf")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Dimensionality Effect (Selection & Topology growth)
# ═══════════════════════════════════════════════════════════════════════
def fig3_dimensionality_effect():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))

    # GA Selection by dimension
    ga_sel = {d: [] for d in DIMS}
    ga_cross = {d: [] for d in DIMS}
    ga_mut = {d: [] for d in DIMS}

    for func in FUNCTIONS:
        for dim in DIMS:
            pct = get_shapley_pct('GA', func, dim)
            if pct:
                for k, v in pct.items():
                    if 'selection' in k:
                        ga_sel[dim].append(v)
                    elif 'crossover' in k:
                        ga_cross[dim].append(v)
                    elif 'mutation' in k:
                        ga_mut[dim].append(v)

    dims_int = [10, 30, 50]
    sel_means = [np.mean(ga_sel[d]) for d in DIMS]
    cross_means = [np.mean(ga_cross[d]) for d in DIMS]
    mut_means = [np.mean(ga_mut[d]) for d in DIMS]

    ax1.plot(dims_int, sel_means, 'o-', color='#e74c3c', label='Selection', linewidth=2, markersize=6)
    ax1.plot(dims_int, cross_means, 's-', color='#3498db', label='Crossover', linewidth=2, markersize=6)
    ax1.plot(dims_int, mut_means, '^-', color='#2ecc71', label='Mutation', linewidth=2, markersize=6)
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Shapley Value (%)')
    ax1.set_title('GA: Operator Attribution vs Dimension', fontweight='bold')
    ax1.legend(loc='center right')
    ax1.set_xticks(dims_int)
    ax1.set_ylim(0, 55)
    ax1.grid(True, alpha=0.3)

    # PSO Topology by dimension
    pso_vel = {d: [] for d in DIMS}
    pso_topo = {d: [] for d in DIMS}

    for func in FUNCTIONS:
        for dim in DIMS:
            pct = get_shapley_pct('PSO', func, dim)
            if pct:
                for k, v in pct.items():
                    if 'velocity' in k:
                        pso_vel[dim].append(v)
                    elif 'topology' in k:
                        pso_topo[dim].append(v)

    vel_means = [np.mean(pso_vel[d]) if pso_vel[d] else 0 for d in DIMS]
    topo_means = [np.mean(pso_topo[d]) if pso_topo[d] else 0 for d in DIMS]

    ax2.plot(dims_int, vel_means, 'o-', color='#9b59b6', label='Velocity', linewidth=2, markersize=6)
    ax2.plot(dims_int, topo_means, 's-', color='#f39c12', label='Topology', linewidth=2, markersize=6)
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Shapley Value (%)')
    ax2.set_title('PSO: Operator Attribution vs Dimension', fontweight='bold')
    ax2.legend(loc='center right')
    ax2.set_xticks(dims_int)
    ax2.set_ylim(0, 55)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig3_dimensionality_effect.pdf'))
    plt.close()
    print("  ✓ fig3_dimensionality_effect.pdf")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: Method Comparison (cosine similarity & rank agreement)
# ═══════════════════════════════════════════════════════════════════════
def fig4_method_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))

    methods = ['QuickSHAP', 'KernelSHAP', 'Tracking']
    algorithms = ['DE', 'GA', 'PSO']
    colors = {'DE': '#3498db', 'GA': '#2ecc71', 'PSO': '#e74c3c'}

    # Compute averages by method and algorithm
    cosine_data = {m: {a: [] for a in algorithms} for m in methods}
    rank_data = {m: {a: [] for a in algorithms} for m in methods}

    for row in method_data:
        m = row['method']
        a = row['algorithm']
        if m in methods and a in algorithms:
            cos_val = row.get('cosine', float('nan'))
            rank_val = row.get('rank_agreement', float('nan'))
            if not np.isnan(cos_val):
                cosine_data[m][a].append(cos_val)
            if not np.isnan(rank_val):
                rank_data[m][a].append(rank_val)

    x = np.arange(len(methods))
    width = 0.25

    # Cosine similarity
    for i, alg in enumerate(algorithms):
        vals = [np.mean(cosine_data[m][alg]) if cosine_data[m][alg] else 0 for m in methods]
        ax1.bar(x + i * width, vals, width, label=alg, color=colors[alg], alpha=0.85)

    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Accuracy: Cosine Similarity vs Exact', fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.set_ylim(0, 1.15)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.grid(True, alpha=0.2, axis='y')

    # Rank agreement
    for i, alg in enumerate(algorithms):
        vals = [np.mean(rank_data[m][alg]) if rank_data[m][alg] else 0 for m in methods]
        ax2.bar(x + i * width, vals, width, label=alg, color=colors[alg], alpha=0.85)

    ax2.set_ylabel('Rank Agreement')
    ax2.set_title('Accuracy: Rank Agreement vs Exact', fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.set_ylim(0, 1.15)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig4_method_comparison.pdf'))
    plt.close()
    print("  ✓ fig4_method_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5: Overhead Comparison (log scale)
# ═══════════════════════════════════════════════════════════════════════
def fig5_overhead():
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    components = ['Instrumentation', 'QuickSHAP', 'Tracking', 'KernelSHAP', 'Exact Shapley']

    # Aggregate overhead data
    instr_vals = []
    quick_vals = []
    track_vals = []
    kernel_vals = []
    exact_vals = []

    for row in overhead_data:
        instr_pct = row.get('instrumentation_overhead_pct', 0)
        if instr_pct > 0:  # skip negative (noise)
            instr_vals.append(instr_pct)
        quick_vals.append(row.get('quickshap_overhead_pct', 0))
        track_vals.append(row.get('tracking_overhead_pct', 0))
        kernel_vals.append(row.get('kernelshap_multiplier', 0) * 100)  # as % of single run
        exact_vals.append(row.get('exact_multiplier', 0) * 100)

    means = [
        np.mean(instr_vals) if instr_vals else 15,
        np.mean(quick_vals),
        np.mean(track_vals),
        np.mean(kernel_vals) / 100,  # back to multiplier
        np.mean(exact_vals) / 100
    ]

    # Use multiplier for display: 1.15x, 1.0001x, etc.
    labels = ['Instrum.', 'QuickSHAP', 'Tracking', 'KernelSHAP', 'Exact']
    bar_labels = [
        f'{means[0]:.0f}%',
        f'{means[1]:.3f}%',
        f'{means[2]:.1f}%',
        f'{means[3]:.0f}×',
        f'{means[4]:.0f}×',
    ]

    # Normalize to "times single run" for common axis
    times = [
        1 + means[0]/100,  # instrumentation: 1.15x
        1 + means[1]/100,  # quickshap: 1.00001x
        1 + means[2]/100,  # tracking: 1.003x
        means[3],          # kernelshap: 30x
        means[4],          # exact: 37x
    ]

    colors_oh = ['#3498db', '#2ecc71', '#27ae60', '#e67e22', '#e74c3c']
    bars = ax.barh(range(len(components)), times, color=colors_oh, alpha=0.85)

    for i, (bar, label) in enumerate(zip(bars, bar_labels)):
        xpos = bar.get_width() + 0.3
        if times[i] < 2:
            xpos = max(bar.get_width() + 0.3, 1.5)
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=8, fontweight='bold')

    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Cost (× single run)')
    ax.set_title('Computational Cost by Component', fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(0.9, 80)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig5_overhead.pdf'))
    plt.close()
    print("  ✓ fig5_overhead.pdf")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6: DE invariance — all configs show 50/50/0
# ═══════════════════════════════════════════════════════════════════════
def fig6_de_invariance():
    fig, ax = plt.subplots(figsize=(7.16, 2.2))

    # Collect DE values for ALL configs
    configs = []
    mut_vals = []
    cross_vals = []
    sel_vals = []

    for func in FUNCTIONS:
        for dim in DIMS:
            pct = get_shapley_pct('DE', func, dim)
            if pct:
                configs.append(f'{func[:4]}\nD={dim}')
                for k, v in pct.items():
                    if 'mutation' in k:
                        mut_vals.append(v)
                    elif 'crossover' in k:
                        cross_vals.append(v)
                    elif 'selection' in k:
                        sel_vals.append(v)

    x = np.arange(len(configs))
    width = 0.28

    ax.bar(x - width, mut_vals, width, label='Mutation', color='#e74c3c', alpha=0.85)
    ax.bar(x, cross_vals, width, label='Crossover', color='#3498db', alpha=0.85)
    ax.bar(x + width, sel_vals, width, label='Selection', color='#95a5a6', alpha=0.85)

    ax.set_ylabel('Shapley Value (%)')
    ax.set_title('DE: Invariant 50/50/0 Attribution Across All Configurations', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=5.5)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 65)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig6_de_invariance.pdf'))
    plt.close()
    print("  ✓ fig6_de_invariance.pdf")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 7: Summary — three algorithms side by side (average across funcs)
# ═══════════════════════════════════════════════════════════════════════
def fig7_summary():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.4))

    algorithms = ['DE', 'GA', 'PSO']
    titles = ['DE (Coupled)', 'GA (Landscape-Dependent)', 'PSO (Dimension-Dependent)']

    for ax, alg, title in zip(axes, algorithms, titles):
        # Collect operator names from data
        sample_dim = DIMS[0]
        sample_func = FUNCTIONS[0]
        sample = exact_data.get(alg, {}).get(sample_func, {}).get(sample_dim, {})
        if not sample:
            continue
        ops = list(sample.get('shapley_values', {}).keys())

        # Average across all functions for each dimension
        dim_data = {dim: {op: [] for op in ops} for dim in DIMS}
        for func in FUNCTIONS:
            for dim in DIMS:
                pct = get_shapley_pct(alg, func, dim)
                if pct:
                    for op in ops:
                        dim_data[dim][op].append(pct.get(op, 0))

        # Stacked bar chart
        x = np.arange(len(DIMS))
        bottom = np.zeros(len(DIMS))

        op_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        op_labels = []
        for op in ops:
            short = op.split('_')[0].capitalize()
            if 'selection' in op or 'greedy' in op:
                short = 'Selection'
            elif 'mutation' in op or 'rand' in op:
                short = 'Mutation'
            elif 'crossover' in op or 'binomial' in op:
                short = 'Crossover'
            elif 'velocity' in op:
                short = 'Velocity'
            elif 'topology' in op:
                short = 'Topology'
            elif 'position' in op:
                short = 'Position'
            op_labels.append(short)

        for i, (op, label) in enumerate(zip(ops, op_labels)):
            means = [np.mean(dim_data[d][op]) if dim_data[d][op] else 0 for d in DIMS]
            ax.bar(x, means, 0.5, bottom=bottom, label=label,
                   color=op_colors[i % len(op_colors)], alpha=0.85)
            # Add value labels
            for j, m in enumerate(means):
                if m > 3:
                    ax.text(j, bottom[j] + m/2, f'{m:.0f}%',
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           color='white' if m > 15 else 'black')
            bottom += means

        ax.set_xticks(x)
        ax.set_xticklabels([f'D={d}' for d in DIMS], fontsize=10)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_ylim(0, 110)
        ax.legend(loc='upper right', fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel('Shapley Value (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig7_summary_all_algorithms.pdf'))
    plt.close()
    print("  ✓ fig7_summary_all_algorithms.pdf")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 8: Discriminant — XMH vs fANOVA (operator type vs cx_rate)
# ═══════════════════════════════════════════════════════════════════════
def fig8_discriminant():
    """Two-panel figure showing complementarity of XMH and fANOVA.

    Left:  Part A — vary categorical cx_type. XMH per-type cx-Shapley% with CIs.
           fANOVA-on-hyperparameters has 0% variance to explain; categorical
           fANOVA reports η² for between-type variance.
    Right: Part B — vary continuous cx_rate. XMH cx-Shapley% modulates smoothly;
           fANOVA reports R² for cx_rate sensitivity.
    """
    if exp8_part_a is None or exp8_part_b is None:
        print("  (skip fig8: exp8 data not found)")
        return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.16, 3.4),
                                   gridspec_kw={'wspace': 0.32})

    # ---- Part A (left): per-cx_type Shapley breakdown ----
    cx_order = ['sbx', 'blx_alpha', 'arithmetic', 'uniform', 'two_point', 'single_point']
    cx_labels = ['SBX', r'BLX-$\alpha$', 'Arith.', 'Uniform', 'Two-pt', 'Single-pt']

    sel_pct, cx_pct, mut_pct, cx_ci = [], [], [], []
    for cx in cx_order:
        case = exp8_part_a['per_cx_type'][cx]
        total = sum(abs(v) for v in case['shapley_values'].values())
        per = {k: 100 * abs(v) / total for k, v in case['shapley_values'].items()}
        sel_pct.append(next(v for k, v in per.items() if k.startswith('selection')))
        cx_pct.append(next(v for k, v in per.items() if k.startswith('crossover')))
        mut_pct.append(next(v for k, v in per.items() if k.startswith('mutation')))
        # CI for crossover operator in percentage units
        cx_op = next(op for op in case['operators'] if op.startswith('crossover'))
        sv = case['shapley_values'][cx_op]
        ci_lo, ci_hi = case['confidence_intervals'][cx_op]
        cx_ci.append((100 * (sv - ci_lo) / total, 100 * (ci_hi - sv) / total))

    x = np.arange(len(cx_order))
    width = 0.27
    axL.bar(x - width, sel_pct, width, label='Selection', color='#e74c3c', alpha=0.85)
    axL.bar(x, cx_pct, width, label='Crossover', color='#3498db', alpha=0.85,
            yerr=np.array(cx_ci).T, capsize=3, error_kw={'elinewidth': 0.9, 'ecolor': '#222'})
    axL.bar(x + width, mut_pct, width, label='Mutation', color='#2ecc71', alpha=0.85)

    axL.set_xticks(x)
    axL.set_xticklabels(cx_labels, fontsize=9, rotation=20, ha='right')
    axL.set_ylabel('Shapley value (%)', fontsize=10)
    axL.set_title('Part A: Vary categorical crossover operator', fontsize=10, fontweight='bold')
    axL.set_ylim(0, 65)
    axL.legend(loc='upper right', fontsize=8, framealpha=0.9)
    axL.grid(True, axis='y', alpha=0.25)

    # Annotations: fANOVA-on-hyperparams = 0, categorical η²
    eta2 = exp8_part_a['anova']['eta_squared'] * 100
    axL.text(0.02, 0.98,
             f"fANOVA on hyperparameters: 0%\n"
             f"(nothing to decompose:\nall knobs are fixed)\n\n"
             rf"$\eta^2$ on cx\_type: {eta2:.1f}%"
             "\n(between-type variance only)",
             transform=axL.transAxes, fontsize=8, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', fc='#fffae6', ec='#d4a017', lw=0.8))

    # ---- Part B (right): cx_rate sweep ----
    rates_keys = sorted(exp8_part_b['per_cx_rate'].keys(), key=lambda s: float(s))
    rates = [float(k) for k in rates_keys]
    cx_pct_b, fitness_means = [], []
    for k in rates_keys:
        case = exp8_part_b['per_cx_rate'][k]
        total = sum(abs(v) for v in case['shapley_values'].values())
        cx_op = next(op for op in case['operators'] if op.startswith('crossover'))
        cx_pct_b.append(100 * abs(case['shapley_values'][cx_op]) / total)
        fitness_means.append(float(np.mean(case['final_fitness_per_run'])))

    # Twin axis: cx-Shapley (left) + fitness mean (right)
    axR2 = axR.twinx()
    line1, = axR.plot(rates, cx_pct_b, 'o-', color='#3498db', linewidth=2,
                      markersize=7, label=r'XMH crossover $\phi$ (%)')
    line2, = axR2.plot(rates, fitness_means, 's--', color='#e67e22', linewidth=1.8,
                      markersize=6, label='Final fitness (mean)')

    axR.set_xlabel('Crossover probability (cx_rate)', fontsize=10)
    axR.set_ylabel(r'XMH crossover $\phi$ (%)', color='#3498db', fontsize=10)
    axR2.set_ylabel('Final fitness (Rastrigin D=30)', color='#e67e22', fontsize=10)
    axR.tick_params(axis='y', labelcolor='#3498db')
    axR2.tick_params(axis='y', labelcolor='#e67e22')
    axR.set_title('Part B: Vary continuous cx_rate (SBX fixed)', fontsize=10, fontweight='bold')
    axR.grid(True, alpha=0.2)
    axR.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])

    # Annotations: fANOVA R² and XMH ρ
    r2 = exp8_part_b['fanova_on_cx_rate']['quadratic_r_squared'] * 100
    rho_xmh = exp8_part_b['xmh_cx_shapley_vs_rate']['spearman_rho_rate_vs_cx_pct']
    axR.text(0.02, 0.98,
             rf"fANOVA on cx\_rate: $R^2$ = {r2:.1f}%"
             "\n(detects continuous sensitivity)"
             "\n\n"
             rf"XMH $\rho$(cx\_rate, $\phi_\mathrm{{cx}}$) = {rho_xmh:.2f}"
             "\n(smooth, monotonic modulation)",
             transform=axR.transAxes, fontsize=8, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', fc='#e8f4fd', ec='#3498db', lw=0.8))

    # Combined legend
    axR.legend(handles=[line1, line2], loc='lower right', fontsize=8, framealpha=0.9)

    fig.suptitle(
        'XMH vs fANOVA: Complementary, Not Equivalent (GA, Rastrigin $D{=}30$)',
        fontsize=11, fontweight='bold', y=1.00
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTDIR, 'fig8_discriminant.pdf'))
    plt.close()
    print("  ✓ fig8_discriminant.pdf")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 9: Budget sensitivity — convergence curves + Shapley trends
# ═══════════════════════════════════════════════════════════════════════
def fig9_budget():
    """2×2 figure showing the effect of max_gen and pop_size on operator
    attribution. Top row: convergence curves (Rastrigin). Bottom row:
    Shapley trends vs budget for both functions."""
    if exp9_sweep_a is None or exp9_sweep_b is None:
        print("  (skip fig9: exp9 data not found)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.4),
                              gridspec_kw={'hspace': 0.45, 'wspace': 0.32})

    OP_COLORS = {'sel': '#e74c3c', 'cx': '#3498db', 'mut': '#2ecc71'}
    FUNC_LS = {'sphere': '--', 'rastrigin': '-'}
    FUNC_MK = {'sphere': 'o', 'rastrigin': 's'}

    def get_pct(case):
        total = sum(abs(v) for v in case['shapley_values'].values())
        if total < 1e-15:
            return {}
        out = {}
        for op in case['operators']:
            short = ('sel' if op.startswith('selection')
                    else 'cx' if op.startswith('crossover')
                    else 'mut' if op.startswith('mutation') else op)
            out[short] = 100.0 * abs(case['shapley_values'][op]) / total
        return out

    # ---- Top-left: convergence on Rastrigin under different max_gen ----
    ax = axes[0, 0]
    rast_a = exp9_sweep_a['per_func']['rastrigin']
    # Largest level has the full trace
    longest = str(max(exp9_sweep_a['levels']))
    full_traces = np.array(rast_a[longest]['convergence_traces'])  # n_runs × (gen+1)
    n_runs, n_steps = full_traces.shape
    gens_axis = np.arange(n_steps)
    median = np.median(full_traces, axis=0)
    p25 = np.percentile(full_traces, 25, axis=0)
    p75 = np.percentile(full_traces, 75, axis=0)
    ax.plot(gens_axis, median, color='#34495e', lw=1.5, label='Median (30 runs)')
    ax.fill_between(gens_axis, p25, p75, color='#34495e', alpha=0.2, label='IQR')
    # Mark the budget levels as vertical lines + labels at top (outside data area)
    for gen in exp9_sweep_a['levels']:
        ax.axvline(x=gen, color='#888', ls=':', lw=0.7)
    # Use a secondary x-axis on top to host the level labels cleanly
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(exp9_sweep_a['levels'])
    ax_top.set_xticklabels([f'g={g}' for g in exp9_sweep_a['levels']], fontsize=8)
    ax_top.tick_params(axis='x', which='both', length=0, pad=2)
    ax.set_xlabel('Generation', fontsize=9)
    ax.set_ylabel('Best fitness (Rastrigin)', fontsize=9)
    ax.set_title('(a) Convergence vs max\\_gen (pop=50)', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_yscale('symlog')
    ax.grid(True, alpha=0.25)

    # ---- Top-right: convergence on Rastrigin under different pop_size ----
    ax = axes[0, 1]
    rast_b = exp9_sweep_b['per_func']['rastrigin']
    colors_pop = plt.cm.viridis(np.linspace(0.1, 0.85, len(exp9_sweep_b['levels'])))
    for color, pop in zip(colors_pop, exp9_sweep_b['levels']):
        traces = np.array(rast_b[str(pop)]['convergence_traces'])  # n_runs × (gen+1)
        median = np.median(traces, axis=0)
        gens_axis = np.arange(len(median))
        ax.plot(gens_axis, median, color=color, lw=1.6, label=f'pop={pop}')
    ax.set_xlabel('Generation', fontsize=9)
    ax.set_ylabel('Best fitness (Rastrigin)', fontsize=9)
    ax.set_title('(b) Convergence vs pop\\_size (max\\_gen=200)', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, ncol=1)
    ax.set_yscale('symlog')
    ax.grid(True, alpha=0.25)

    # ---- Bottom-left: Shapley% vs gen for both functions ----
    ax = axes[1, 0]
    gens = exp9_sweep_a['levels']
    for func in exp9_sweep_a['functions']:
        per_lvl = exp9_sweep_a['per_func'][func]
        sel = [get_pct(per_lvl[str(g)]).get('sel', 0) for g in gens]
        cx = [get_pct(per_lvl[str(g)]).get('cx', 0) for g in gens]
        mut = [get_pct(per_lvl[str(g)]).get('mut', 0) for g in gens]
        for label, values, color in [('sel', sel, OP_COLORS['sel']),
                                      ('cx', cx, OP_COLORS['cx']),
                                      ('mut', mut, OP_COLORS['mut'])]:
            ax.plot(gens, values, ls=FUNC_LS[func], marker=FUNC_MK[func],
                    color=color, lw=1.6, markersize=5, alpha=0.85)
    # Custom legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=OP_COLORS['sel'], lw=2, label='Selection'),
        Line2D([0], [0], color=OP_COLORS['cx'], lw=2, label='Crossover'),
        Line2D([0], [0], color=OP_COLORS['mut'], lw=2, label='Mutation'),
        Line2D([0], [0], color='gray', ls='-', marker='s', label='Rastrigin', markersize=5),
        Line2D([0], [0], color='gray', ls='--', marker='o', label='Sphere', markersize=5),
    ]
    ax.legend(handles=legend_handles, loc='center left', fontsize=7, ncol=1,
              bbox_to_anchor=(0.0, 0.55), framealpha=0.85)
    ax.set_xlabel('max\\_generations', fontsize=9)
    ax.set_ylabel('Shapley value (%)', fontsize=9)
    ax.set_title('(c) Shapley vs max\\_gen', fontsize=10, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(gens)
    ax.set_xticklabels([str(g) for g in gens])
    ax.set_ylim(0, 70)
    ax.grid(True, alpha=0.25)

    # ---- Bottom-right: Shapley% vs pop for both functions ----
    ax = axes[1, 1]
    pops = exp9_sweep_b['levels']
    for func in exp9_sweep_b['functions']:
        per_lvl = exp9_sweep_b['per_func'][func]
        sel = [get_pct(per_lvl[str(p)]).get('sel', 0) for p in pops]
        cx = [get_pct(per_lvl[str(p)]).get('cx', 0) for p in pops]
        mut = [get_pct(per_lvl[str(p)]).get('mut', 0) for p in pops]
        for values, color in [(sel, OP_COLORS['sel']),
                              (cx, OP_COLORS['cx']),
                              (mut, OP_COLORS['mut'])]:
            ax.plot(pops, values, ls=FUNC_LS[func], marker=FUNC_MK[func],
                    color=color, lw=1.6, markersize=5, alpha=0.85)
    ax.set_xlabel('pop\\_size', fontsize=9)
    ax.set_ylabel('Shapley value (%)', fontsize=9)
    ax.set_title('(d) Shapley vs pop\\_size', fontsize=10, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(pops)
    ax.set_xticklabels([str(p) for p in pops])
    ax.set_ylim(0, 90)
    ax.grid(True, alpha=0.25)

    fig.suptitle('Budget sensitivity: operator attribution shifts with max\\_gen and pop\\_size',
                 fontsize=11, fontweight='bold', y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(OUTDIR, 'fig9_budget.pdf'))
    plt.close()
    print("  ✓ fig9_budget.pdf")


# ═══════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating figures for XMH paper v2...")
    fig1_ga_heatmap()
    fig2_pso_heatmap()
    fig3_dimensionality_effect()
    fig4_method_comparison()
    fig5_overhead()
    fig6_de_invariance()
    fig7_summary()
    fig8_discriminant()
    fig9_budget()
    print(f"\nAll figures saved to {OUTDIR}/")
