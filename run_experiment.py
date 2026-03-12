"""
Experiment Runner — Flat vs. Hierarchical Architecture Comparison
Multi-Agent BI Report System — Master's Thesis

Runs Flat and Hierarchical architectures across N products,
evaluates each report with evaluate_v2, and generates thesis-ready summary tables.

Experimental Design:
  - Control (Flat): Manager-interleaved pipeline, Manager has NO loop-back authority
  - Treatment (Hierarchical): Same workers + Manager WITH loop-back authority
  - Treatment variable: Manager's loop-back authority (only difference)

Usage:
    python run_experiment.py                      # 15 random products
    python run_experiment.py --n 5                # 5 random products
    python run_experiment.py --arch flat          # only flat
    python run_experiment.py --dry-run            # show plan, don't run
    python run_experiment.py --resume EXP_DIR     # resume interrupted experiment
"""

import argparse
import json
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Local imports
from flat_graph import run_flat_graph
from hierarchical_graph import run_hierarchical_graph
from evaluate import evaluate_report_v2, JUDGE_MODELS
from efficiency_tracker import EfficiencyTracker
from config import calculate_cost, MODEL_NAME
from tools import get_product_specs

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default number of products to sample
DEFAULT_N_PRODUCTS = 15

# Minimum reviews required (0-review products are excluded)
MIN_REVIEWS = 1

# Judge models for evaluation
def _find_model(name):
    """Find a model by name from JUDGE_MODELS."""
    for m in JUDGE_MODELS:
        if m["name"] == name:
            return m
    return None

EXPERIMENT_MODELS = [m for m in [
    _find_model("GPT-5.4"),
    _find_model("Gemini-3.1-Pro"),
    _find_model("Qwen-3.5-122B"),
    _find_model("GLM-5"),
    _find_model("Mistral-Small"),
] if m is not None]

# Architectures: Flat (control) vs. Hierarchical (treatment)
ARCHITECTURES = [
    "flat",
    "hierarchical",
]

# Delay between API calls (seconds) - Set to 0 for maximum speed with 2026 models
API_DELAY = 0


# =============================================================================
# PRODUCT LOADING
# =============================================================================

def load_products(dataset_path: str = "dataset_final.json") -> list[dict]:
    """Load products from dataset, filtering out those with < MIN_REVIEWS."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    products = []
    for item in data:
        n_reviews = len(item.get("reviews", []))
        if n_reviews >= MIN_REVIEWS:
            products.append({
                "asin": item["id"],
                "title": item.get("metadata", {}).get("title", "Unknown"),
                "n_reviews": n_reviews,
            })

    return products


def select_products(products: list[dict], n: int, seed: Optional[int] = None) -> list[dict]:
    """Randomly select n products. Use seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
    if n >= len(products):
        return products
    return random.sample(products, n)


# =============================================================================
# SINGLE RUN
# =============================================================================

def run_single(
    asin: str,
    architecture: str,
    models: list[dict],
    experiment_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Run a single architecture on a single product and evaluate.
    Returns a complete result dict ready for JSON serialization.
    """
    product_metadata = get_product_specs.invoke({"asin": asin})
    query = (
        f"Analyze the product (ASIN: {asin}). Provide a comprehensive BI report "
        f"covering customer feedback, issues, and recommendations."
    )

    # ── 1. Generate Report ──
    tracker = EfficiencyTracker(architecture, asin=asin, model="gpt-4o")
    tracker.start()

    if architecture == "flat":
        report, _, metrics = run_flat_graph(query, session_id=asin)
    elif architecture == "hierarchical":
        report, _, metrics = run_hierarchical_graph(query, session_id=asin)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    tracker.record_from_metrics(metrics)
    tracker.stop()
    gen_metrics = tracker.get_metrics()

    if verbose:
        print(f"      📝 Report generated ({gen_metrics.total_tokens:,} tokens, "
              f"${gen_metrics.total_cost_usd:.4f}, {gen_metrics.latency_seconds:.1f}s)")

    # ── 2. Evaluate Report ──
    time.sleep(API_DELAY)

    # Pass experiment_dir so CoT analyses are saved alongside results
    cot_output_dir = str(experiment_dir) if experiment_dir else None

    evaluation = evaluate_report_v2(
        report, product_metadata,
        asin=asin,
        models=models,
        verbose=False,
        output_dir=cot_output_dir,
        run_id=f"{asin}_{architecture}",
    )

    if verbose:
        print(f"      ⚖️  Evaluated: Quality={evaluation.quality.mean_score} "
              f"Accuracy={evaluation.accuracy_score} "
              f"Utility={evaluation.utility.mean_score} "
              f"→ Final={evaluation.final_score}")

    # ── 3. Calculate Cost ──
    gen_cost = calculate_cost(
        MODEL_NAME,
        gen_metrics.prompt_tokens,
        gen_metrics.completion_tokens,
    )

    # Estimate evaluation cost based on empirical data from pilot run
    # True evaluation cost was ~2.8x higher than initial formula due to lengthy CoT outputs and large prompts
    n_judges = len(models)
    n_dimensions = 7  # 4 quality + 3 utility
    eval_cost = 0.0
    for m in models:
        prompt_t = n_dimensions * 3200
        completion_t = n_dimensions * 500
        eval_cost += calculate_cost(m["model_id"], prompt_t, completion_t)

    total_cost = round(gen_cost + eval_cost, 6)

    # ── 4. Package Result ──
    return {
        "metadata": {
            "product_asin": asin,
            "architecture": architecture,
            "timestamp": datetime.now().isoformat(),
            "judge_models": [m["name"] for m in models],
        },
        "efficiency": {
            "total_tokens": gen_metrics.total_tokens,
            "prompt_tokens": gen_metrics.prompt_tokens,
            "completion_tokens": gen_metrics.completion_tokens,
            "generation_cost_usd": gen_cost,
            "evaluation_cost_usd": eval_cost,
            "total_cost_usd": total_cost,
            "latency_seconds": gen_metrics.latency_seconds,
            "step_count": gen_metrics.step_count,
            "loop_count": metrics.get("loop_count", 0),
        },
        "evaluation": evaluation.to_dict(),
        "report_text": report,
    }


# =============================================================================
# SUMMARY GENERATION
# =============================================================================

def generate_summary(experiment_dir: Path, products: list[dict]) -> str:
    """Read all result JSONs and generate thesis-ready summary tables."""
    results = []
    for json_file in sorted(experiment_dir.glob("eval_*.json")):
        with open(json_file) as f:
            results.append(json.load(f))

    if not results:
        return "No results found."

    # Group by architecture
    arch_data: dict[str, list[dict]] = {"flat": [], "hierarchical": []}
    for r in results:
        arch = r["metadata"]["architecture"]
        if arch in arch_data:
            arch_data[arch].append(r)

    # ── Table 1: Overall Score Comparison ──
    md = "# Experiment Summary — Flat vs. Hierarchical\n\n"
    md += f"**Products**: {len(products)} | **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    md += "> **Treatment variable**: Manager loop-back authority (present in Hierarchical only)\n\n"
    md += "---\n\n"
    md += "## Table 1: Overall Score Comparison (Mean across all products)\n\n"
    md += "| Architecture | Quality | Accuracy | Utility | **Final** | Cost | Latency | Tokens |\n"
    md += "|---|---|---|---|---|---|---|---|\n"

    for arch_name in ARCHITECTURES:
        data = arch_data.get(arch_name, [])
        if not data:
            continue
        q = sum(r["evaluation"]["quality"]["mean_score"] for r in data) / len(data)
        a = sum(r["evaluation"]["accuracy"]["score"] for r in data) / len(data)
        u = sum(r["evaluation"]["utility"]["mean_score"] for r in data) / len(data)
        f_score = sum(r["evaluation"]["final_score"] for r in data) / len(data)
        cost = sum(r["efficiency"]["total_cost_usd"] for r in data) / len(data)
        lat = sum(r["efficiency"]["latency_seconds"] for r in data) / len(data)
        tokens = sum(r["efficiency"]["total_tokens"] for r in data) / len(data)
        md += (f"| {arch_name.title()} | {q:.2f} | {a:.2f} | {u:.2f} | "
               f"**{f_score:.2f}** | ${cost:.4f} | {lat:.1f}s | {tokens:,.0f} |\n")

    # ── Table 2: Per-Dimension Scores ──
    md += "\n---\n\n## Table 2: Per-Dimension Mean Scores\n\n"
    all_dims = [
        "structure", "coherence", "conciseness",
        "actionability", "root_cause_analysis", "strategic_depth",
    ]
    md += "| Dimension | " + " | ".join(a.title() for a in ARCHITECTURES) + " |\n"
    md += "|---|" + "|".join("---" for _ in ARCHITECTURES) + "|\n"

    for dim in all_dims:
        row = f"| {dim.replace('_', ' ').title()} |"
        for arch_name in ARCHITECTURES:
            data = arch_data.get(arch_name, [])
            if not data:
                row += " — |"
                continue
            dim_scores = []
            for r in data:
                qual_dims = r["evaluation"]["quality"].get("dimensions", {})
                util_dims = r["evaluation"]["utility"].get("dimensions", {})
                if dim in qual_dims:
                    for judge in qual_dims[dim]:
                        dim_scores.append(judge["score"])
                elif dim in util_dims:
                    for judge in util_dims[dim]:
                        dim_scores.append(judge["score"])
            if dim_scores:
                row += f" {sum(dim_scores)/len(dim_scores):.2f} |"
            else:
                row += " — |"
        md += row + "\n"

    md += f"| **Accuracy** |"
    for arch_name in ARCHITECTURES:
        data = arch_data.get(arch_name, [])
        if data:
            a_mean = sum(r["evaluation"]["accuracy"]["score"] for r in data) / len(data)
            md += f" **{a_mean:.2f}** |"
        else:
            md += " — |"
    md += "\n"

    # ── Table 3: Per-Model Scores ──
    md += "\n---\n\n## Table 3: Per-Model Mean Scores\n\n"
    all_models = set()
    for r in results:
        for m in r["metadata"].get("judge_models", []):
            all_models.add(m)
    all_models = sorted(all_models)

    md += "| Model | " + " | ".join(a.title() for a in ARCHITECTURES) + " |\n"
    md += "|---|" + "|".join("---" for _ in ARCHITECTURES) + "|\n"

    for model_name in all_models:
        row = f"| {model_name} |"
        for arch_name in ARCHITECTURES:
            data = arch_data.get(arch_name, [])
            scores = []
            for r in data:
                q_models = r["evaluation"]["quality"].get("per_model_scores", {})
                u_models = r["evaluation"]["utility"].get("per_model_scores", {})
                if model_name in q_models:
                    scores.append(q_models[model_name])
                if model_name in u_models:
                    scores.append(u_models[model_name])
            if scores:
                row += f" {sum(scores)/len(scores):.2f} |"
            else:
                row += " — |"
        md += row + "\n"

    # ── Table 4: Inter-Rater Reliability ──
    md += "\n---\n\n## Table 4: Inter-Rater Reliability\n\n"
    md += "| Metric | " + " | ".join(a.title() for a in ARCHITECTURES) + " |\n"
    md += "|---|" + "|".join("---" for _ in ARCHITECTURES) + "|\n"

    for metric in ["quality", "utility"]:
        # Krippendorff's α
        row = f"| {metric.title()} Krippendorff α |"
        for arch_name in ARCHITECTURES:
            data = arch_data.get(arch_name, [])
            if data:
                alphas = [r["evaluation"][metric].get("inter_rater_alpha", 0) for r in data]
                row += f" {sum(alphas)/len(alphas):.4f} |"
            else:
                row += " — |"
        md += row + "\n"

        # Cronbach's α
        row = f"| {metric.title()} Cronbach α |"
        for arch_name in ARCHITECTURES:
            data = arch_data.get(arch_name, [])
            if data:
                cronbachs = [r["evaluation"][metric].get("inter_rater_cronbach", 0) for r in data]
                row += f" {sum(cronbachs)/len(cronbachs):.4f} |"
            else:
                row += " — |"
        md += row + "\n"

    # ── Table 5: Efficiency ──
    md += "\n---\n\n## Table 5: Efficiency Trade-off\n\n"
    md += "| Architecture | Tokens (mean) | Cost (mean) | Latency (mean) | Steps (mean) | Loops (mean) |\n"
    md += "|---|---|---|---|---|---|\n"

    for arch_name in ARCHITECTURES:
        data = arch_data.get(arch_name, [])
        if not data:
            continue
        t = sum(r["efficiency"]["total_tokens"] for r in data) / len(data)
        c = sum(r["efficiency"]["total_cost_usd"] for r in data) / len(data)
        l = sum(r["efficiency"]["latency_seconds"] for r in data) / len(data)
        s = sum(r["efficiency"]["step_count"] for r in data) / len(data)
        loops = sum(r["efficiency"].get("loop_count", 0) for r in data) / len(data)
        md += f"| {arch_name.title()} | {t:,.0f} | ${c:.4f} | {l:.1f}s | {s:.1f} | {loops:.1f} |\n"

    # ── Per-Product Detail ──
    md += "\n---\n\n## Per-Product Scores\n\n"
    md += "| ASIN | Architecture | Quality | Accuracy | Utility | Final |\n"
    md += "|---|---|---|---|---|---|\n"
    for r in sorted(results, key=lambda x: (x["metadata"]["product_asin"], x["metadata"]["architecture"])):
        asin = r["metadata"]["product_asin"]
        arch = r["metadata"]["architecture"]
        q = r["evaluation"]["quality"]["mean_score"]
        a = r["evaluation"]["accuracy"]["score"]
        u = r["evaluation"]["utility"]["mean_score"]
        f = r["evaluation"]["final_score"]
        md += f"| {asin} | {arch.title()} | {q:.2f} | {a:.2f} | {u:.2f} | {f:.2f} |\n"

    return md


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

def run_experiment(
    n_products: int = DEFAULT_N_PRODUCTS,
    architectures: list[str] = None,
    models: list[dict] = None,
    specific_asins: list[str] = None,
    seed: int = 42,
    dry_run: bool = False,
    resume_dir: Optional[str] = None,
):
    """Run the full experiment."""
    if architectures is None:
        architectures = ARCHITECTURES
    if models is None:
        models = EXPERIMENT_MODELS

    print("=" * 70)
    print("  EXPERIMENT RUNNER — Flat vs. Hierarchical Comparison")
    print("  Treatment: Manager loop-back authority")
    print("=" * 70)

    # ── 1. Load & Select Products ──
    all_products = load_products()
    print(f"\n📦 Total products in dataset: {len(all_products)} (with ≥{MIN_REVIEWS} reviews)")

    if specific_asins:
        selected = [p for p in all_products if p["asin"] in specific_asins]
        print(f"   Using specified ASINs: {specific_asins}")
    else:
        selected = select_products(all_products, n_products, seed=seed)
        print(f"   Randomly selected {len(selected)} products (seed={seed})")

    print(f"\n   Selected products:")
    for i, p in enumerate(selected, 1):
        print(f"   {i:2d}. {p['asin']} — {p['title'][:55]}... ({p['n_reviews']} reviews)")

    # ── 2. Setup Experiment Directory ──
    if resume_dir:
        experiment_dir = Path(resume_dir)
        print(f"\n📂 Resuming experiment: {experiment_dir}")
    else:
        exp_id = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        experiment_dir = Path("results/experiments") / exp_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📂 Experiment directory: {experiment_dir}")

    # Save config
    exp_config = {
        "experiment_type": "flat_vs_hierarchical",
        "treatment_variable": "manager_loop_back_authority",
        "n_products": len(selected),
        "architectures": architectures,
        "judge_models": [m["name"] for m in models],
        "seed": seed,
        "min_reviews": MIN_REVIEWS,
        "products": [p["asin"] for p in selected],
        "started_at": datetime.now().isoformat(),
    }
    config_path = experiment_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(exp_config, f, indent=2)

    # ── 3. Estimate Cost ──
    n_runs = len(selected) * len(architectures)
    est_cost = n_runs * 0.28
    print(f"\n💰 Estimated cost: ~${est_cost:.2f} ({n_runs} runs × ~$0.28/run)")
    print(f"   Models: {[m['name'] for m in models]}")
    print(f"   Architectures: {architectures}")

    if dry_run:
        print("\n🏁 DRY RUN — stopping before execution.")
        return

    # ── 4. Check for completed runs (resume support) ──
    completed = set()
    for json_file in experiment_dir.glob("eval_*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            key = f"{data['metadata']['product_asin']}_{data['metadata']['architecture']}"
            completed.add(key)
        except Exception:
            pass

    if completed:
        print(f"\n♻️  Found {len(completed)} completed runs — skipping those.")

    # ── 5. Run Experiment ──
    print(f"\n{'='*70}")
    print("  STARTING EXPERIMENT")
    print(f"{'='*70}\n")

    total_runs = len(selected) * len(architectures)
    current_run = 0
    total_cost = 0.0
    errors = []
    start_time = time.time()

    for i, product in enumerate(selected, 1):
        asin = product["asin"]
        print(f"\n{'─'*70}")
        print(f"  [{i}/{len(selected)}] Product: {asin}")
        print(f"  {product['title'][:65]}")
        print(f"{'─'*70}")

        for arch in architectures:
            current_run += 1
            run_key = f"{asin}_{arch}"

            if run_key in completed:
                print(f"   ⏭️  {arch.title():15s} — already completed, skipping")
                continue

            print(f"   🔄 {arch.title():15s} — running... ", end="", flush=True)

            try:
                result = run_single(
                    asin, arch, models,
                    experiment_dir=experiment_dir,
                    verbose=False,
                )

                # Save immediately (checkpoint)
                result_path = experiment_dir / f"eval_{asin}_{arch}.json"
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                run_cost = result["efficiency"]["total_cost_usd"]
                total_cost += run_cost
                latency = result["efficiency"]["latency_seconds"]
                final = result["evaluation"]["final_score"]

                elapsed = time.time() - start_time
                remaining_runs = total_runs - current_run
                est_remaining = (elapsed / max(current_run, 1)) * remaining_runs

                print(f"✅ Final={final:.2f} | ${run_cost:.4f} | {latency:.0f}s "
                      f"| [{current_run}/{total_runs}] ~{est_remaining/60:.0f}min left")

                time.sleep(API_DELAY)

            except KeyboardInterrupt:
                print(f"\n\n⚠️  Interrupted by user at run {current_run}/{total_runs}")
                print(f"   Resume with: python run_experiment.py --resume {experiment_dir}")
                sys.exit(0)

            except Exception as e:
                print(f"❌ Error: {e}")
                errors.append({"asin": asin, "architecture": arch, "error": str(e)})
                traceback.print_exc()
                continue

    # ── 6. Generate Summary ──
    elapsed_total = time.time() - start_time
    print(f"\n{'='*70}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {elapsed_total/60:.1f} minutes")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Errors: {len(errors)}")

    if errors:
        print(f"\n  Failed runs:")
        for e in errors:
            print(f"    ❌ {e['asin']} / {e['architecture']}: {e['error'][:80]}")

        errors_path = experiment_dir / "errors.json"
        with open(errors_path, "w") as f:
            json.dump(errors, f, indent=2)

    print("\n📊 Generating summary tables...")
    summary_md = generate_summary(experiment_dir, selected)

    summary_md_path = experiment_dir / "summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"   ✅ Saved: {summary_md_path}")

    all_results = []
    for json_file in sorted(experiment_dir.glob("eval_*.json")):
        with open(json_file) as f:
            all_results.append(json.load(f))

    summary_json_path = experiment_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": exp_config,
            "total_cost_usd": total_cost,
            "total_time_seconds": elapsed_total,
            "n_errors": len(errors),
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Saved: {summary_json_path}")

    print(f"\n{summary_md}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Flat vs. Hierarchical experiment"
    )
    parser.add_argument(
        "--n", type=int, default=DEFAULT_N_PRODUCTS,
        help=f"Number of products to sample (default: {DEFAULT_N_PRODUCTS})",
    )
    parser.add_argument(
        "--arch", type=str, default=None,
        choices=["flat", "hierarchical"],
        help="Run only this architecture",
    )
    parser.add_argument(
        "--asins", type=str, default=None,
        help="Comma-separated ASINs to use instead of random selection",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without running",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume experiment from directory path",
    )

    args = parser.parse_args()

    archs = ARCHITECTURES
    if args.arch:
        archs = [args.arch]

    specific_asins = None
    if args.asins:
        specific_asins = [a.strip() for a in args.asins.split(",")]

    run_experiment(
        n_products=args.n,
        architectures=archs,
        models=EXPERIMENT_MODELS,
        specific_asins=specific_asins,
        seed=args.seed,
        dry_run=args.dry_run,
        resume_dir=args.resume,
    )
