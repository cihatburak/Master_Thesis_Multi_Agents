"""Robustness check: re-runs the LMM with loop_count=3 capped at 2.

The third revision attempt is blocked by the experiment, so capping it
removes the artefactual loop_count=3 observations and tests whether the
loop-count coefficient survives.

Usage:
    python robustness_check.py results/experiments/exp_<timestamp>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_data(experiment_dir: str) -> pd.DataFrame:
    """Load experiment data into a long-format DataFrame."""
    exp_path = Path(experiment_dir)
    with open(exp_path / "summary.json", "r", encoding="utf-8") as f:
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

    rows = []
    for r in data.get("results", []):
        asin = r["metadata"]["product_asin"]
        arch = r["metadata"]["architecture"]
        writer = r["metadata"].get("role_assignments", {}).get("Writer", "unknown")
        features = product_features.get(asin, {})

        rows.append({
            "product": asin,
            "architecture": 1 if arch == "hierarchical" else 0,
            "loop_count": r["efficiency"].get("loop_count", 0),
            "n_reviews": features.get("n_reviews", 10),
            "mean_rating": features.get("mean_rating", 4.0),
            "writer_model": writer.split("/")[-1] if "/" in writer else writer,
            "writing_clarity": r["evaluation"]["quality"]["mean_score"],
            "utility": r["evaluation"]["utility"]["mean_score"],
            "accuracy": r["evaluation"]["accuracy"]["score"],
        })

    return pd.DataFrame(rows)


def run_lmm(df: pd.DataFrame, dv: str, label: str) -> dict:
    """Run a single LMM and return results dict."""
    formula = f"{dv} ~ architecture + loop_count + n_reviews + mean_rating + C(writer_model)"
    model = smf.mixedlm(formula, df, groups=df["product"])
    result = model.fit(reml=True)

    params = {}
    for var in ["architecture", "loop_count"]:
        params[var] = {
            "beta": result.fe_params[var],
            "se": result.bse_fe[var],
            "z": result.tvalues[var],
            "p": result.pvalues[var],
        }
    params["random_intercept_var"] = result.cov_re.iloc[0, 0]

    print(f"\n{'='*60}")
    print(f"  {label}: DV = {dv}")
    print(f"{'='*60}")
    print(result.summary())
    return params


def main(experiment_dir: str):
    df = load_data(experiment_dir)
    n_total = len(df)
    n_hier = len(df[df["architecture"] == 1])
    print(f"Loaded {n_total} observations ({n_hier} hierarchical, {n_total - n_hier} flat)")
    print(f"Loop count distribution (hierarchical): {df[df['architecture']==1]['loop_count'].value_counts().sort_index().to_dict()}")

    df_recoded = df.copy()
    n_recoded = (df_recoded["loop_count"] == 3).sum()
    df_recoded["loop_count"] = df_recoded["loop_count"].clip(upper=2)
    print(f"\nRecoded {n_recoded} observations from loop_count=3 to loop_count=2")

    results = {}
    for dv in ["writing_clarity", "utility", "accuracy"]:
        print(f"\n\n{'#'*60}")
        print(f"  DEPENDENT VARIABLE: {dv.upper()}")
        print(f"{'#'*60}")

        orig = run_lmm(df, dv, "ORIGINAL (loop_count 0-3)")
        recoded = run_lmm(df_recoded, dv, "RECODED (loop_count 0-2, capped)")
        results[dv] = {"original": orig, "recoded": recoded}

    print(f"\n\n{'='*70}")
    print("  ROBUSTNESS CHECK SUMMARY")
    print(f"{'='*70}")
    print(f"{'DV':<20} {'Variable':<15} {'Original β':>12} {'Recoded β':>12} {'Orig p':>10} {'Recode p':>10}")
    print("-" * 70)
    for dv in ["writing_clarity", "utility", "accuracy"]:
        for var in ["architecture", "loop_count"]:
            orig = results[dv]["original"][var]
            reco = results[dv]["recoded"][var]
            print(f"{dv:<20} {var:<15} {orig['beta']:>+12.4f} {reco['beta']:>+12.4f} {orig['p']:>10.4f} {reco['p']:>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness check: loop count recoding")
    parser.add_argument("experiment_dir", type=str)
    args = parser.parse_args()
    main(args.experiment_dir)
