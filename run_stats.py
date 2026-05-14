"""Statistical analysis for the Flat vs. Hierarchical experiment.

Produces descriptives, paired tests with Cohen's d, per-dimension breakdown,
OLS regression with controls, and an efficiency comparison. Writes a Markdown
report next to the input summary.json by default.

Usage:
    python run_stats.py results/experiments/exp_<timestamp>
    python run_stats.py results/experiments/exp_<timestamp> --output results.md
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load_experiment_data(experiment_dir: str) -> dict:
    """
    Load experiment results and dataset metadata.
    Returns a dict with 'products' mapping ASIN → {flat: {...}, hierarchical: {...}}.
    """
    exp_path = Path(experiment_dir)
    summary_path = exp_path / "summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} does not exist.")

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_path = Path(__file__).parent / "dataset_final.json"
    product_features = {}
    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        for item in dataset:
            asin = item.get("id", "")
            reviews = item.get("reviews", [])
            ratings = [r.get("rating", 0) for r in reviews]
            product_features[asin] = {
                "n_reviews": len(reviews),
                "mean_rating": np.mean(ratings) if ratings else 0.0,
            }

    products = {}
    for r in data.get("results", []):
        asin = r["metadata"]["product_asin"]
        arch = r["metadata"]["architecture"]

        if asin not in products:
            products[asin] = {"features": product_features.get(asin, {})}

        products[asin][arch] = {
            "final_score": r["evaluation"]["final_score"],
            "quality": r["evaluation"]["quality"]["mean_score"],
            "utility": r["evaluation"]["utility"]["mean_score"],
            "accuracy": r["evaluation"]["accuracy"]["score"],
            "loop_count": r["efficiency"].get("loop_count", 0),
            "total_tokens": r["efficiency"].get("total_tokens", 0),
            "latency": r["efficiency"].get("latency_seconds", 0),
            "cost": r["efficiency"].get("total_cost_usd", 0),
            "writer_model": r["metadata"].get("role_assignments", {}).get("Writer", "unknown"),
            "structure": r["evaluation"]["quality"].get("per_dimension_scores", {}).get("structure", 0),
            "coherence": r["evaluation"]["quality"].get("per_dimension_scores", {}).get("coherence", 0),
            "conciseness": r["evaluation"]["quality"].get("per_dimension_scores", {}).get("conciseness", 0),
            "actionability": r["evaluation"]["utility"].get("per_dimension_scores", {}).get("actionability", 0),
            "root_cause_analysis": r["evaluation"]["utility"].get("per_dimension_scores", {}).get("root_cause_analysis", 0),
            "strategic_depth": r["evaluation"]["utility"].get("per_dimension_scores", {}).get("strategic_depth", 0),
        }

    return products


def get_paired_data(products: dict):
    """Filter to products that have both flat and hierarchical results."""
    valid = {
        asin: data
        for asin, data in products.items()
        if "flat" in data and "hierarchical" in data
    }
    return valid


def descriptive_stats(paired: dict, metrics: list) -> str:
    lines = []
    lines.append("## 1. Descriptive Statistics\n")
    lines.append("| Metric | Architecture | Mean | Std | Median | Min | Max |")
    lines.append("|--------|-------------|------|-----|--------|-----|-----|")

    for metric in metrics:
        for arch in ["flat", "hierarchical"]:
            values = [paired[asin][arch][metric] for asin in paired]
            arr = np.array(values)
            lines.append(
                f"| {metric.replace('_', ' ').title()} | {arch.title()} | "
                f"{arr.mean():.3f} | {arr.std():.3f} | {np.median(arr):.3f} | "
                f"{arr.min():.2f} | {arr.max():.2f} |"
            )

    return "\n".join(lines)


def cohens_d_paired(x, y):
    """Cohen's d for paired samples: mean of pairwise differences divided by their std."""
    diff = np.array(x) - np.array(y)
    if diff.std() == 0:
        return 0.0
    return float(diff.mean() / diff.std())


def paired_tests(paired: dict, metrics: list) -> str:
    lines = []
    lines.append("\n## 2. Paired Statistical Tests (Flat vs. Hierarchical)\n")
    lines.append("| Metric | Flat Mean | Hier Mean | Diff | Cohen's d | Wilcoxon p | t-test p | Significant? |")
    lines.append("|--------|-----------|-----------|------|-----------|------------|----------|-------------|")

    for metric in metrics:
        flat_scores = [paired[asin]["flat"][metric] for asin in paired]
        hier_scores = [paired[asin]["hierarchical"][metric] for asin in paired]

        flat_mean = np.mean(flat_scores)
        hier_mean = np.mean(hier_scores)
        diff = flat_mean - hier_mean

        d = cohens_d_paired(flat_scores, hier_scores)

        differences = np.array(flat_scores) - np.array(hier_scores)
        if np.all(differences == 0):
            w_p = 1.0
            t_p = 1.0
        else:
            try:
                _, w_p = stats.wilcoxon(flat_scores, hier_scores)
            except Exception:
                w_p = float("nan")
            try:
                _, t_p = stats.ttest_rel(flat_scores, hier_scores)
            except Exception:
                t_p = float("nan")

        sig = "Yes*" if (w_p < 0.05 or t_p < 0.05) else "No"
        d_label = f"{d:+.3f}"

        lines.append(
            f"| {metric.replace('_', ' ').title()} | {flat_mean:.3f} | {hier_mean:.3f} | "
            f"{diff:+.3f} | {d_label} | {w_p:.4f} | {t_p:.4f} | {sig} |"
        )

    lines.append("\n*Significant at p < 0.05. Cohen's d interpretation: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large.*")
    return "\n".join(lines)


def per_dimension_analysis(paired: dict) -> str:
    dimensions = [
        ("structure", "Writing Clarity"),
        ("coherence", "Writing Clarity"),
        ("conciseness", "Writing Clarity"),
        ("actionability", "Utility"),
        ("root_cause_analysis", "Utility"),
        ("strategic_depth", "Utility"),
    ]

    lines = []
    lines.append("\n## 3. Per-Dimension Analysis\n")
    lines.append("| Dimension | Category | Flat Mean | Hier Mean | Diff | Cohen's d | Wilcoxon p | Hypothesis |")
    lines.append("|-----------|----------|-----------|-----------|------|-----------|------------|------------|")

    for dim, category in dimensions:
        flat_scores = [paired[asin]["flat"].get(dim, 0) for asin in paired]
        hier_scores = [paired[asin]["hierarchical"].get(dim, 0) for asin in paired]

        flat_mean = np.mean(flat_scores)
        hier_mean = np.mean(hier_scores)
        diff = flat_mean - hier_mean
        d = cohens_d_paired(flat_scores, hier_scores)

        differences = np.array(flat_scores) - np.array(hier_scores)
        if np.all(differences == 0):
            w_p = 1.0
        else:
            try:
                _, w_p = stats.wilcoxon(flat_scores, hier_scores)
            except Exception:
                w_p = float("nan")

        # Hypothesis alignment
        if category == "Writing Clarity":
            hypothesis = "H1: Hier > Flat" if hier_mean > flat_mean else "Against H1"
        else:
            hypothesis = "H2: Flat > Hier" if flat_mean > hier_mean else "Against H2"

        dim_name = dim.replace("_", " ").title()
        lines.append(
            f"| {dim_name} | {category} | {flat_mean:.3f} | {hier_mean:.3f} | "
            f"{diff:+.3f} | {d:+.3f} | {w_p:.4f} | {hypothesis} |"
        )

    return "\n".join(lines)


def ols_regression(paired: dict, metrics: list) -> str:
    """OLS for each DV: Score ~ Architecture + Loop + NumReviews + MeanRating + Writer dummies.

    Hand-rolled with numpy.lstsq to avoid pulling in statsmodels for this script.
    """
    lines = []
    lines.append("\n## 4. OLS Regression Analysis\n")
    lines.append("Controls for Writer model, loop count, number of reviews, and mean product rating.\n")

    all_writer_models = set()
    for asin in paired:
        for arch in ["flat", "hierarchical"]:
            all_writer_models.add(paired[asin][arch].get("writer_model", "unknown"))
    writer_model_list = sorted(all_writer_models)
    # First writer alphabetically is the reference category.
    writer_dummies = writer_model_list[1:]

    for metric in metrics:
        lines.append(f"\n### Dependent Variable: {metric.replace('_', ' ').title()}\n")

        y_vals = []
        X_rows = []

        for asin in paired:
            features = paired[asin].get("features", {})
            n_reviews = features.get("n_reviews", 10)
            mean_rating = features.get("mean_rating", 4.0)

            for arch in ["flat", "hierarchical"]:
                score = paired[asin][arch][metric]
                arch_val = 1 if arch == "hierarchical" else 0
                loop_count = paired[asin][arch].get("loop_count", 0)
                writer = paired[asin][arch].get("writer_model", "unknown")
                writer_vals = [1 if writer == wm else 0 for wm in writer_dummies]
                X_rows.append([1, arch_val, loop_count, n_reviews, mean_rating] + writer_vals)
                y_vals.append(score)

        X = np.array(X_rows, dtype=float)
        y = np.array(y_vals, dtype=float)
        n = len(y)
        k = X.shape[1]

        if n <= k:
            lines.append(f"*Insufficient observations ({n}) for {k} parameters. Skipping.*\n")
            continue

        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            lines.append("*Regression failed (singular matrix). Skipping.*\n")
            continue

        y_hat = X @ beta
        resid = y - y_hat
        dof = n - k
        if dof <= 0:
            lines.append("*Zero degrees of freedom. Skipping.*\n")
            continue

        mse = np.sum(resid ** 2) / dof
        try:
            cov_matrix = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov_matrix))
        except np.linalg.LinAlgError:
            lines.append("*Cannot compute standard errors (singular matrix). Skipping.*\n")
            continue

        t_stats = beta / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))

        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / dof) if dof > 0 else 0.0

        feature_names = ["Intercept", "Architecture (Hier=1)", "Loop Count",
                         "Num Reviews", "Mean Rating"]
        for wm in writer_dummies:
            short_name = wm.split("/")[-1] if "/" in wm else wm
            feature_names.append(f"Writer: {short_name}")

        lines.append(f"N = {n}, R² = {r_squared:.4f}, Adj. R² = {adj_r_squared:.4f}\n")
        lines.append("| Variable | Coefficient | Std Error | t-stat | p-value | Sig |")
        lines.append("|----------|------------|-----------|--------|---------|-----|")

        for i, name in enumerate(feature_names):
            sig = "*" if p_values[i] < 0.05 else ("." if p_values[i] < 0.1 else "")
            lines.append(
                f"| {name} | {beta[i]:+.4f} | {se[i]:.4f} | {t_stats[i]:+.3f} | "
                f"{p_values[i]:.4f} | {sig} |"
            )

        arch_coef = beta[1]
        arch_p = p_values[1]
        if arch_p < 0.05:
            direction = "increases" if arch_coef > 0 else "decreases"
            lines.append(
                f"\n**Architecture effect:** Hierarchical {direction} "
                f"{metric.replace('_', ' ')} by {abs(arch_coef):.3f} points (p={arch_p:.4f})."
            )
        else:
            lines.append(
                f"\n**Architecture effect:** No significant effect on "
                f"{metric.replace('_', ' ')} (p={arch_p:.4f})."
            )

    return "\n".join(lines)


def efficiency_comparison(paired: dict) -> str:
    lines = []
    lines.append("\n## 5. Efficiency Comparison\n")
    lines.append("| Metric | Flat Mean | Hier Mean | Ratio (Hier/Flat) |")
    lines.append("|--------|-----------|-----------|-------------------|")

    for metric, label in [
        ("total_tokens", "Total Tokens"),
        ("cost", "Cost (USD)"),
        ("latency", "Latency (sec)"),
        ("loop_count", "Loop Count"),
    ]:
        flat_vals = [paired[asin]["flat"].get(metric, 0) for asin in paired]
        hier_vals = [paired[asin]["hierarchical"].get(metric, 0) for asin in paired]
        flat_mean = np.mean(flat_vals)
        hier_mean = np.mean(hier_vals)
        ratio = hier_mean / flat_mean if flat_mean > 0 else float("inf")

        if metric == "cost":
            lines.append(f"| {label} | ${flat_mean:.4f} | ${hier_mean:.4f} | {ratio:.2f}x |")
        elif metric == "latency":
            lines.append(f"| {label} | {flat_mean:.1f}s | {hier_mean:.1f}s | {ratio:.2f}x |")
        else:
            lines.append(f"| {label} | {flat_mean:,.0f} | {hier_mean:,.0f} | {ratio:.2f}x |")

    return "\n".join(lines)


def run_statistical_analysis(experiment_dir: str, output_path: str = None):
    products = load_experiment_data(experiment_dir)
    paired = get_paired_data(products)

    if not paired:
        print("Error: No paired results found (need both flat and hierarchical for the same product).")
        return

    n = len(paired)
    print(f"Loaded {n} paired products for statistical analysis.\n")

    core_metrics = ["final_score", "quality", "utility", "accuracy"]

    report_lines = []
    report_lines.append(f"# Statistical Analysis Report — Flat vs. Hierarchical\n")
    report_lines.append(f"**Paired Products:** {n}")
    report_lines.append(f"**Experiment:** {experiment_dir}\n")
    report_lines.append("---\n")

    report_lines.append(descriptive_stats(paired, core_metrics))
    report_lines.append(paired_tests(paired, core_metrics))
    report_lines.append(per_dimension_analysis(paired))
    report_lines.append(ols_regression(paired, core_metrics))
    report_lines.append(efficiency_comparison(paired))

    full_report = "\n".join(report_lines)
    print(full_report)

    if output_path:
        out = Path(output_path)
    else:
        out = Path(experiment_dir) / "statistical_analysis.md"
    out.write_text(full_report, encoding="utf-8")
    print(f"\nSaved to: {out}")


def run_statistical_tests(experiment_dir: str):
    """Legacy entry point retained so older callers keep working."""
    run_statistical_analysis(experiment_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run statistical analysis on experimental results."
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Directory containing summary.json to analyze.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for the report (default: <experiment_dir>/statistical_analysis.md)",
    )
    args = parser.parse_args()
    run_statistical_analysis(args.experiment_dir, args.output)
