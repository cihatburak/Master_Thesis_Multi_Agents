"""Build the three results figures for the thesis from a finished experiment.

Outputs go to thesis/figures/:
    figure1_dimension_comparison.png
    figure2_loop_vs_clarity.png
    figure3_efficiency_comparison.png

Usage: python generate_thesis_figures.py
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


EXP_DIR = Path("results/experiments/exp_20260403_100301")
OUT_DIR = Path("thesis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Academic style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FLAT_COLOR = "#2196F3"
HIER_COLOR = "#FF7043"


def load_experiment_data():
    """Load all eval JSON files and organize by architecture."""
    flat_data = []
    hier_data = []

    for f in sorted(EXP_DIR.glob("eval_*.json")):
        with open(f) as fh:
            d = json.load(fh)

        arch = d["metadata"]["architecture"]
        entry = {
            "asin": d["metadata"].get("asin") or d["metadata"].get("product_asin", ""),
            "architecture": arch,
            "final_score": d["evaluation"]["final_score"],
            "quality": d["evaluation"]["quality"]["mean_score"],
            "utility": d["evaluation"]["utility"]["mean_score"],
            "accuracy": d["evaluation"]["accuracy"]["score"],
            "structure": d["evaluation"]["quality"]["per_dimension_scores"]["structure"],
            "coherence": d["evaluation"]["quality"]["per_dimension_scores"]["coherence"],
            "conciseness": d["evaluation"]["quality"]["per_dimension_scores"]["conciseness"],
            "actionability": d["evaluation"]["utility"]["per_dimension_scores"]["actionability"],
            "root_cause": d["evaluation"]["utility"]["per_dimension_scores"]["root_cause_analysis"],
            "strategic_depth": d["evaluation"]["utility"]["per_dimension_scores"]["strategic_depth"],
            "loop_count": d["efficiency"]["loop_count"],
            "total_tokens": d["efficiency"]["total_tokens"],
            "total_cost": d["efficiency"]["total_cost_usd"],
            "latency": d["efficiency"]["latency_seconds"],
            "gen_cost": d["efficiency"]["generation_cost_usd"],
        }

        if arch == "flat":
            flat_data.append(entry)
        else:
            hier_data.append(entry)

    return flat_data, hier_data


def figure1_dimension_comparison(flat_data, hier_data):
    """
    Grouped bar chart: Flat vs Hierarchical across all 6 evaluation dimensions.
    Stars indicate statistical significance from paired tests.
    """
    dimensions = [
        ("Structure", "structure"),
        ("Coherence", "coherence"),
        ("Conciseness", "conciseness"),
        ("Actionability", "actionability"),
        ("Root Cause\nAnalysis", "root_cause"),
        ("Strategic\nDepth", "strategic_depth"),
    ]

    flat_means = [np.mean([d[key] for d in flat_data]) for _, key in dimensions]
    hier_means = [np.mean([d[key] for d in hier_data]) for _, key in dimensions]
    flat_stds = [np.std([d[key] for d in flat_data], ddof=1) / np.sqrt(len(flat_data))
                 for _, key in dimensions]
    hier_stds = [np.std([d[key] for d in hier_data], ddof=1) / np.sqrt(len(hier_data))
                 for _, key in dimensions]

    # Significance markers (from statistical_analysis.md Wilcoxon p-values)
    sig_markers = ["", "", "**", "", "", "**"]  # Conciseness p=0.017, Strategic Depth p=0.007

    x = np.arange(len(dimensions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars_flat = ax.bar(x - width/2, flat_means, width, yerr=flat_stds,
                       label="Flat (Control)", color=FLAT_COLOR, alpha=0.85,
                       edgecolor="white", linewidth=0.5, capsize=3)
    bars_hier = ax.bar(x + width/2, hier_means, width, yerr=hier_stds,
                       label="Hierarchical (Treatment)", color=HIER_COLOR, alpha=0.85,
                       edgecolor="white", linewidth=0.5, capsize=3)

    # Add significance stars
    for i, sig in enumerate(sig_markers):
        if sig:
            max_val = max(flat_means[i] + flat_stds[i], hier_means[i] + hier_stds[i])
            ax.text(x[i], max_val + 0.03, sig, ha="center", va="bottom",
                    fontsize=14, fontweight="bold", color="#333333")

    # Category separators
    ax.axvline(x=2.5, color="#999999", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(1.0, 3.65, "Writing Clarity", ha="center", fontsize=11,
            fontweight="bold", fontstyle="italic", color="#444444",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8))
    ax.text(4.0, 3.65, "Utility", ha="center", fontsize=11,
            fontweight="bold", fontstyle="italic", color="#444444",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8))

    ax.set_ylabel("Mean Score (1–5)")
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in dimensions])
    ax.set_ylim(3.6, 5.15)
    ax.legend(loc="upper right")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Note at bottom
    fig.text(0.5, -0.02,
             "Note: y-axis begins at 3.6 to highlight differences. Error bars = SEM. ** p < 0.01 (Wilcoxon signed-rank test, N = 43 paired products).",
             ha="center", fontsize=9, color="#666666")

    plt.tight_layout()
    out_path = OUT_DIR / "figure1_dimension_comparison.png"
    fig.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


def figure2_loop_vs_clarity(hier_data):
    """
    Combined bar + scatter plot: Loop count effect on Writing Clarity.
    Shows both the mean trend and individual data points.
    """
    # Group by loop count
    loop_groups = defaultdict(list)
    for d in hier_data:
        loop_groups[d["loop_count"]].append(d["quality"])

    loop_counts = sorted(loop_groups.keys())
    means = [np.mean(loop_groups[lc]) for lc in loop_counts]
    stds = [np.std(loop_groups[lc], ddof=1) if len(loop_groups[lc]) > 1 else 0
            for lc in loop_counts]
    ns = [len(loop_groups[lc]) for lc in loop_counts]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Bars — all hierarchical (graduated orange tones: darker = more loops)
    bar_alphas = [0.50, 0.65, 0.80, 0.95]
    bars = []
    for idx, lc in enumerate(loop_counts):
        b = ax.bar(lc, means[idx], width=0.6, color=HIER_COLOR,
                   alpha=bar_alphas[idx] if idx < len(bar_alphas) else 0.95,
                   edgecolor="white", linewidth=0.5, yerr=stds[idx], capsize=4)
        bars.append(b)

    # Scatter individual points
    for lc in loop_counts:
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(loop_groups[lc]))
        ax.scatter([lc + j for j in jitter], loop_groups[lc],
                   color="#333333", alpha=0.4, s=25, zorder=5, edgecolors="white", linewidth=0.3)

    # Trend line
    ax.plot(loop_counts, means, color="#D32F2F", marker="D", markersize=7,
            linewidth=2, linestyle="--", zorder=6, label="Mean Writing Clarity")

    # Annotations
    for i, (lc, m, n) in enumerate(zip(loop_counts, means, ns)):
        ax.text(lc, m + stds[i] + 0.05, f"n={n}\n({m:.2f})", ha="center",
                va="bottom", fontsize=9, fontweight="bold", color="#333333")

    # OLS regression annotation (full-controls spec matches §4.5.1 in thesis)
    ax.annotate(
        r"OLS (full controls): $\beta$ = −0.142, p < 0.001"
        "\n"
        r"Drop loop=3:   $\beta$ = −0.074, p = 0.038",
        xy=(0.65, 3.18), fontsize=9, fontstyle="italic", color="#D32F2F",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor="#D32F2F", alpha=0.85),
    )

    ax.set_xlabel("Number of Manager Loop-Backs")
    ax.set_ylabel("Writing Clarity Score (1–5)")
    ax.set_xticks(loop_counts)
    ax.set_xticklabels([str(lc) for lc in loop_counts])
    ax.set_ylim(3.0, 5.0)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.text(0.5, -0.02,
             "Each dot = one product (N = 43). Loop count 3 indicates a blocked third attempt (MAX_LOOPS = 2).",
             ha="center", fontsize=9, color="#666666")

    plt.tight_layout()
    out_path = OUT_DIR / "figure2_loop_vs_clarity.png"
    fig.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


def figure3_efficiency_comparison(flat_data, hier_data):
    """
    Grouped bar chart: Efficiency metrics (cost, latency, tokens) with overhead percentages.
    """
    metrics = [
        ("Generation\nCost (USD)", "gen_cost", "$"),
        ("Total\nCost (USD)", "total_cost", "$"),
        ("Latency\n(seconds)", "latency", "s"),
        ("Total\nTokens (×1000)", "total_tokens", "k"),
    ]

    flat_means = []
    hier_means = []
    flat_stds = []
    hier_stds = []
    overheads = []

    for _, key, unit in metrics:
        fvals = [d[key] for d in flat_data]
        hvals = [d[key] for d in hier_data]

        if unit == "k":
            fvals = [v / 1000 for v in fvals]
            hvals = [v / 1000 for v in hvals]

        fm, hm = np.mean(fvals), np.mean(hvals)
        flat_means.append(fm)
        hier_means.append(hm)
        flat_stds.append(np.std(fvals, ddof=1) / np.sqrt(len(fvals)))
        hier_stds.append(np.std(hvals, ddof=1) / np.sqrt(len(hvals)))
        overheads.append((hm - fm) / fm * 100)

    x = np.arange(len(metrics))
    width = 0.35

    fig, axes = plt.subplots(1, 4, figsize=(14, 5.5), sharey=False)

    for i, (ax, (label, key, unit)) in enumerate(zip(axes, metrics)):
        bars_f = ax.bar(0, flat_means[i], width=0.6, yerr=flat_stds[i],
                        color=FLAT_COLOR, alpha=0.85, capsize=4, edgecolor="white")
        bars_h = ax.bar(1, hier_means[i], width=0.6, yerr=hier_stds[i],
                        color=HIER_COLOR, alpha=0.85, capsize=4, edgecolor="white")

        # Overhead annotation
        overhead_text = f"+{overheads[i]:.0f}%"
        max_val = max(flat_means[i] + flat_stds[i], hier_means[i] + hier_stds[i])
        ax.annotate(
            overhead_text,
            xy=(0.5, max_val * 1.02), fontsize=11, fontweight="bold",
            ha="center", va="bottom", color="#D32F2F",
        )

        # Value labels — positioned inside bars with better precision
        if unit == "$":
            fmt_f = f"${flat_means[i]:.2f}"
            fmt_h = f"${hier_means[i]:.2f}"
        elif unit == "s":
            fmt_f = f"{flat_means[i]:.0f}s"
            fmt_h = f"{hier_means[i]:.0f}s"
        else:
            fmt_f = f"{flat_means[i]:.1f}k"
            fmt_h = f"{hier_means[i]:.1f}k"
        ax.text(0, flat_means[i] * 0.5, fmt_f,
                ha="center", va="center", fontsize=10, color="#333333", fontweight="bold")
        ax.text(1, hier_means[i] * 0.5, fmt_h,
                ha="center", va="center", fontsize=10, color="#333333", fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Flat", "Hier"], fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, max_val * 1.25)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Shared legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=FLAT_COLOR, alpha=0.85, label="Flat (Control)"),
        Patch(facecolor=HIER_COLOR, alpha=0.85, label="Hierarchical (Treatment)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.08))

    fig.text(0.5, -0.14,
             "Error bars = standard error of the mean (N = 43 per architecture). "
             "Red percentages = hierarchical overhead relative to flat.",
             ha="center", fontsize=9, color="#666666")

    plt.tight_layout()
    out_path = OUT_DIR / "figure3_efficiency_comparison.png"
    fig.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    print("Loading experiment data...")
    flat_data, hier_data = load_experiment_data()
    print(f"  Loaded {len(flat_data)} flat + {len(hier_data)} hierarchical runs\n")

    print("Generating figures...")
    figure1_dimension_comparison(flat_data, hier_data)
    figure2_loop_vs_clarity(hier_data)
    figure3_efficiency_comparison(flat_data, hier_data)

    print(f"\nAll figures saved to {OUT_DIR}/")
    print("Reference in thesis:")
    print("  Figure 1 → Section 4.2 (Per-Dimension Comparison)")
    print("  Figure 2 → Section 4.4/4.5 (Loop Count Effect)")
    print("  Figure 3 → Section 4.4 (Efficiency Trade-offs)")
