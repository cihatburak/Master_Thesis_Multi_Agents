"""Supplementary statistical analyses for the Flat vs. Hierarchical experiment.

Produces:
  - TOST equivalence test on Structure (Flat vs. Hierarchical)
  - Dose-response robustness: OLS on writing clarity within the hierarchical
    arm under three loop-handling specs (original, drop loop=3, recode loop=3 -> 2),
    plus a mixed-effects model on the full sample for comparison.
  - Paired-difference inter-judge correlations on Writing Clarity and Utility.
  - Inter-dimension correlation matrix across the six rubric dimensions.

Run:
    python revision_analyses.py results/experiments/exp_<timestamp>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


QUALITY_DIMS = ["structure", "coherence", "conciseness"]
UTILITY_DIMS = ["actionability", "root_cause_analysis", "strategic_depth"]


def load_summary(experiment_dir: str) -> dict:
    p = Path(experiment_dir) / "summary.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_product_features() -> dict:
    p = Path(__file__).parent / "dataset_final.json"
    feats = {}
    if not p.exists():
        return feats
    with open(p, "r", encoding="utf-8") as f:
        ds = json.load(f)
    for item in ds:
        asin = item.get("id", "")
        rev = item.get("reviews", [])
        ratings = [r.get("rating", 0) for r in rev]
        feats[asin] = {
            "n_reviews": len(rev),
            "mean_rating": float(np.mean(ratings)) if ratings else 0.0,
        }
    return feats


def build_long_df(summary: dict, feats: dict) -> pd.DataFrame:
    rows = []
    for r in summary.get("results", []):
        asin = r["metadata"]["product_asin"]
        arch = r["metadata"]["architecture"]
        writer = r["metadata"].get("role_assignments", {}).get("Writer", "unknown")
        ev = r["evaluation"]
        f = feats.get(asin, {})

        row = {
            "product": asin,
            "architecture": 1 if arch == "hierarchical" else 0,
            "arch_name": arch,
            "loop_count": r["efficiency"].get("loop_count", 0),
            "n_reviews": f.get("n_reviews", 10),
            "mean_rating": f.get("mean_rating", 4.0),
            "writer_model": writer.split("/")[-1] if "/" in writer else writer,
            "writing_clarity": ev["quality"]["mean_score"],
            "utility": ev["utility"]["mean_score"],
            "accuracy": ev["accuracy"]["score"],
        }
        for dim in QUALITY_DIMS + UTILITY_DIMS:
            row[dim] = (
                ev["quality"]["per_dimension_scores"].get(dim)
                if dim in QUALITY_DIMS
                else ev["utility"]["per_dimension_scores"].get(dim)
            )

        # Per-judge means: WC averaged across 3 WC dims, Utility across 3 Utility dims.
        wc_per_judge = {}
        ut_per_judge = {}
        for dim in QUALITY_DIMS:
            for entry in ev["quality"]["dimensions"].get(dim, []):
                m = entry["model"]
                wc_per_judge.setdefault(m, []).append(entry["score"])
        for dim in UTILITY_DIMS:
            for entry in ev["utility"]["dimensions"].get(dim, []):
                m = entry["model"]
                ut_per_judge.setdefault(m, []).append(entry["score"])
        for m, scores in wc_per_judge.items():
            row[f"wc_judge::{m}"] = float(np.mean(scores))
        for m, scores in ut_per_judge.items():
            row[f"ut_judge::{m}"] = float(np.mean(scores))

        rows.append(row)
    return pd.DataFrame(rows)


def get_paired(df: pd.DataFrame) -> pd.DataFrame:
    """Return wide-format paired DataFrames (flat, hier) indexed by product."""
    flat = df[df["arch_name"] == "flat"].set_index("product")
    hier = df[df["arch_name"] == "hierarchical"].set_index("product")
    common = flat.index.intersection(hier.index)
    return flat.loc[common], hier.loc[common]


# ---------------------------------------------------------------------
# TOST equivalence test on Structure
# ---------------------------------------------------------------------
def tost_paired(diffs: np.ndarray, low_bound: float, up_bound: float) -> dict:
    """TOST for paired differences. Equivalence concluded if both one-sided
    t-tests reject at alpha. Returns p-values and joint conclusion."""
    n = len(diffs)
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1
    t1 = (mean - low_bound) / se
    p1 = 1 - stats.t.cdf(t1, df)
    t2 = (mean - up_bound) / se
    p2 = stats.t.cdf(t2, df)
    p_tost = max(p1, p2)
    return {
        "n": n,
        "mean_diff": mean,
        "sd_diff": sd,
        "se_diff": se,
        "low_bound": low_bound,
        "up_bound": up_bound,
        "t1": t1, "p1": p1,
        "t2": t2, "p2": p2,
        "p_tost": p_tost,
        "equivalent_at_alpha_0.05": p_tost < 0.05,
    }


def run_tost_structure(flat_w: pd.DataFrame, hier_w: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  TOST equivalence test on Structure (Flat vs Hierarchical)")
    print("=" * 70)

    diffs = (flat_w["structure"] - hier_w["structure"]).values
    n = len(diffs)
    sd_diff = float(np.std(diffs, ddof=1))
    print(f"\nN paired = {n}")
    print(f"Mean(Flat - Hier) on Structure = {np.mean(diffs):.4f}")
    print(f"SD of diffs = {sd_diff:.4f}")

    # Equivalence bounds: d_z = 0.20 is the conventional small-effect bound;
    # 0.35 is reported as a lenient sensitivity check.
    for d_bound in [0.20, 0.35]:
        raw_bound = d_bound * sd_diff
        out = tost_paired(diffs, low_bound=-raw_bound, up_bound=+raw_bound)
        print(f"\n--- Equivalence bound: |d_z| = {d_bound} -> raw |mean diff| < {raw_bound:.4f}")
        print(f"  p_lower (vs -bound) = {out['p1']:.4f}")
        print(f"  p_upper (vs +bound) = {out['p2']:.4f}")
        print(f"  p_TOST = max = {out['p_tost']:.4f}")
        print(f"  Equivalence at alpha=0.05: {out['equivalent_at_alpha_0.05']}")

    t_stat, p_t = stats.ttest_rel(flat_w["structure"], hier_w["structure"])
    cohen_d = float(np.mean(diffs) / sd_diff)
    print(f"\n--- Reference: paired-t difference test")
    print(f"  t({n-1}) = {t_stat:.3f}, p = {p_t:.4f}")
    print(f"  Cohen's d (paired) = {cohen_d:.4f}")

    pct_at_or_above_4_5_flat = float(np.mean(flat_w["structure"] >= 4.5))
    pct_at_or_above_4_5_hier = float(np.mean(hier_w["structure"] >= 4.5))
    print(f"\n--- Ceiling diagnostic on Structure")
    print(f"  Flat:  {pct_at_or_above_4_5_flat * 100:.1f}% of products score >= 4.5")
    print(f"  Hier:  {pct_at_or_above_4_5_hier * 100:.1f}% of products score >= 4.5")
    print(f"  Flat mean = {flat_w['structure'].mean():.3f}, Hier mean = {hier_w['structure'].mean():.3f}")


# ---------------------------------------------------------------------
# Dose-response robustness
# ---------------------------------------------------------------------
def fit_lmm(df: pd.DataFrame, dv: str, label: str) -> dict:
    formula = f"{dv} ~ architecture + loop_count + n_reviews + mean_rating + C(writer_model)"
    model = smf.mixedlm(formula, df, groups=df["product"])
    res = model.fit(reml=True)
    return {
        "label": label,
        "n_obs": len(df),
        "n_groups": df["product"].nunique(),
        "loop_beta": float(res.fe_params.get("loop_count", np.nan)),
        "loop_se": float(res.bse_fe.get("loop_count", np.nan)),
        "loop_p": float(res.pvalues.get("loop_count", np.nan)),
        "arch_beta": float(res.fe_params.get("architecture", np.nan)),
        "arch_p": float(res.pvalues.get("architecture", np.nan)),
    }


def fit_ols_within_hier(df_hier: pd.DataFrame, dv: str, label: str) -> dict:
    formula = f"{dv} ~ loop_count + n_reviews + mean_rating + C(writer_model)"
    model = smf.ols(formula, df_hier).fit()
    return {
        "label": label,
        "n_obs": len(df_hier),
        "loop_beta": float(model.params.get("loop_count", np.nan)),
        "loop_se": float(model.bse.get("loop_count", np.nan)),
        "loop_p": float(model.pvalues.get("loop_count", np.nan)),
        "r2": float(model.rsquared),
    }


def run_dose_response_robustness(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  Dose-response robustness (original, drop loop=3, recode loop=3 -> 2)")
    print("=" * 70)

    df_hier = df[df["architecture"] == 1].copy()
    print(f"\nHierarchical N = {len(df_hier)}")
    print(f"Loop distribution: {df_hier['loop_count'].value_counts().sort_index().to_dict()}")

    print("\n=== Within-hierarchical OLS (Writing Clarity ~ loop_count + controls) ===")
    spec_orig = fit_ols_within_hier(df_hier, "writing_clarity", "ORIG (loop 0-3)")

    df_hier_drop = df_hier[df_hier["loop_count"] != 3].copy()
    spec_drop = fit_ols_within_hier(df_hier_drop, "writing_clarity", "DROP loop=3 (loop in 0,1,2)")

    df_hier_recode = df_hier.copy()
    df_hier_recode["loop_count"] = df_hier_recode["loop_count"].clip(upper=2)
    spec_recode = fit_ols_within_hier(df_hier_recode, "writing_clarity", "RECODE loop=3->2")

    print(f"\n{'Spec':<28} {'N':>4} {'β(loop)':>10} {'SE':>8} {'p':>10} {'R²':>8}")
    print("-" * 70)
    for s in (spec_orig, spec_drop, spec_recode):
        print(f"{s['label']:<28} {s['n_obs']:>4} {s['loop_beta']:>+10.4f} {s['loop_se']:>8.4f} {s['loop_p']:>10.4f} {s['r2']:>8.3f}")

    print("\n=== Mixed-effects model (full N=86, all DVs) ===")
    for dv in ("writing_clarity", "utility", "accuracy"):
        print(f"\n-- DV: {dv}")
        orig = fit_lmm(df, dv, "ORIG (loop 0-3)")
        dropped = fit_lmm(df[df["loop_count"] != 3].copy(), dv, "DROP loop=3")
        df_rec = df.copy()
        df_rec["loop_count"] = df_rec["loop_count"].clip(upper=2)
        recoded = fit_lmm(df_rec, dv, "RECODE loop=3->2")
        for r in (orig, dropped, recoded):
            print(f"  {r['label']:<22} N={r['n_obs']:<3} loop β={r['loop_beta']:+.4f} (p={r['loop_p']:.4f})   arch β={r['arch_beta']:+.4f} (p={r['arch_p']:.4f})")


# ---------------------------------------------------------------------
# Paired-difference inter-judge correlations
# ---------------------------------------------------------------------
def run_paired_diff_correlations(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  Paired-difference inter-judge correlations (WC, Utility)")
    print("=" * 70)

    flat_w, hier_w = get_paired(df)

    for label, prefix in (("Writing Clarity", "wc_judge::"), ("Utility", "ut_judge::")):
        judge_cols = [c for c in flat_w.columns if c.startswith(prefix)]
        judges = sorted({c.replace(prefix, "") for c in judge_cols})
        print(f"\n--- {label}: {len(judges)} judges, N = {len(flat_w)} paired products ---")
        diffs = pd.DataFrame(index=flat_w.index)
        for j in judges:
            col = prefix + j
            if col in flat_w.columns and col in hier_w.columns:
                diffs[j] = flat_w[col] - hier_w[col]
        diffs = diffs.dropna()
        print(f"After dropna: N = {len(diffs)}, judges = {list(diffs.columns)}")

        corr = diffs.corr(method="pearson")
        print(f"\nPearson correlation matrix of (Flat - Hier) per-judge differences:")
        print(corr.round(3).to_string())

        print(f"\nMean (Flat - Hier) per judge (positive = Flat scores higher):")
        for j in diffs.columns:
            mu = diffs[j].mean()
            n = diffs[j].notna().sum()
            t, p = stats.ttest_rel(flat_w.loc[diffs.index, prefix + j],
                                    hier_w.loc[diffs.index, prefix + j])
            print(f"  {j:<22} mean diff = {mu:+.3f}  paired-t p = {p:.4f}  N = {n}")

        n_j = corr.shape[0]
        off = corr.values[~np.eye(n_j, dtype=bool)]
        print(f"\nMean off-diagonal correlation = {off.mean():.3f}  (range: {off.min():.3f} to {off.max():.3f})")


# ---------------------------------------------------------------------
# Inter-dimension correlations
# ---------------------------------------------------------------------
def run_inter_dimension_correlations(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  Inter-dimension correlations across paired products")
    print("=" * 70)

    flat_w, hier_w = get_paired(df)
    all_dims = QUALITY_DIMS + UTILITY_DIMS

    stacked = pd.concat(
        [flat_w[all_dims].reset_index(drop=True), hier_w[all_dims].reset_index(drop=True)],
        axis=0, ignore_index=True
    )
    print(f"\nN observations stacked = {len(stacked)}  (43 paired × 2 architectures)")
    corr = stacked.corr(method="pearson")
    print(f"\nPearson inter-dimension correlation matrix:")
    print(corr.round(3).to_string())

    print(f"\nCross-construct correlations (Writing Clarity dim × Utility dim):")
    for wd in QUALITY_DIMS:
        for ud in UTILITY_DIMS:
            print(f"  {wd:<22} × {ud:<22} r = {corr.loc[wd, ud]:+.3f}")


def main(experiment_dir: str) -> None:
    summary = load_summary(experiment_dir)
    feats = load_product_features()
    df = build_long_df(summary, feats)

    flat_w, hier_w = get_paired(df)
    print(f"Loaded {len(df)} obs ({(df['architecture']==1).sum()} hier, {(df['architecture']==0).sum()} flat)")
    print(f"Paired products = {len(flat_w)}")

    run_tost_structure(flat_w, hier_w)
    run_dose_response_robustness(df)
    run_paired_diff_correlations(df)
    run_inter_dimension_correlations(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=str)
    args = parser.parse_args()
    main(args.experiment_dir)
