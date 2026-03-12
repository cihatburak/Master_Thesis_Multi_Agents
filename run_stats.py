import json
import argparse
from pathlib import Path
from scipy import stats
import numpy as np

def run_statistical_tests(experiment_dir: str):
    """
    Reads the summary.json from an experiment run and performs paired statistical tests
    (Wilcoxon signed-rank test and Paired t-test) comparing Flat vs. Hierarchical architectures.
    """
    exp_path = Path(experiment_dir)
    summary_path = exp_path / "summary.json"
    
    if not summary_path.exists():
        print(f"Error: {summary_path} does not exist.")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    
    # Extract paired scores
    # We need to group by product_asin
    products = {}
    for r in results:
        asin = r["metadata"]["product_asin"]
        arch = r["metadata"]["architecture"]
        
        if asin not in products:
            products[asin] = {}
        
        products[asin][arch] = {
            "final_score": r["evaluation"]["final_score"],
            "quality": r["evaluation"]["quality"]["mean_score"],
            "utility": r["evaluation"]["utility"]["mean_score"],
            "accuracy": r["evaluation"]["accuracy"]["score"]
        }

    # Filter out products that don't have both architectures evaluated
    valid_asins = [asin for asin, archs in products.items() if "flat" in archs and "hierarchical" in archs]
    
    if not valid_asins:
        print("Error: No paired results found (need both flat and hierarchical for the same product).")
        return

    print(f"Loaded {len(valid_asins)} paired products for statistical testing.\n")

    metrics = ["final_score", "quality", "utility", "accuracy"]
    
    print("=" * 60)
    print("STATISTICAL ANALYSIS: FLAT vs HIERARCHICAL")
    print("=" * 60)
    
    for metric in metrics:
        flat_scores = [products[asin]["flat"][metric] for asin in valid_asins]
        hier_scores = [products[asin]["hierarchical"][metric] for asin in valid_asins]
        
        flat_mean = np.mean(flat_scores)
        hier_mean = np.mean(hier_scores)
        
        print(f"\n--- Metric: {metric.upper()} ---")
        print(f"Flat Mean: {flat_mean:.3f}")
        print(f"Hierarchical Mean: {hier_mean:.3f}")
        
        # Check if all differences are zero
        differences = np.array(flat_scores) - np.array(hier_scores)
        if np.all(differences == 0):
            print("  Result: All differences are 0. Tests cannot be calculated (p-value = 1.0).")
            continue
            
        # 1. Wilcoxon Signed-Rank Test (Non-parametric)
        # Robust for small samples and ordinal/non-normal data
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(flat_scores, hier_scores)
            print(f"  Wilcoxon Test: Statistic={wilcoxon_stat:.3f}, p-value={wilcoxon_p:.4f}")
            if wilcoxon_p < 0.05:
                print("    -> STATISTICALLY SIGNIFICANT difference (p < 0.05).")
            else:
                print("    -> No statistically significant difference (p >= 0.05).")
        except Exception as e:
            print(f"  Wilcoxon Test: Could not compute ({e})")
            
        # 2. Paired T-Test (Parametric)
        # Assumes normal distribution of differences
        try:
            ttest_stat, ttest_p = stats.ttest_rel(flat_scores, hier_scores)
            print(f"  Paired t-test: Statistic={ttest_stat:.3f}, p-value={ttest_p:.4f}")
            if ttest_p < 0.05:
                print("    -> STATISTICALLY SIGNIFICANT difference (p < 0.05).")
            else:
                print("    -> No statistically significant difference (p >= 0.05).")
        except Exception as e:
            print(f"  Paired t-test: Could not compute ({e})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run statistical tests on experimental results.")
    parser.add_argument("experiment_dir", type=str, help="Directory containing the summary.json to analyze.")
    args = parser.parse_args()
    run_statistical_tests(args.experiment_dir)
