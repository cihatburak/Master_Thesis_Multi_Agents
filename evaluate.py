"""Multi-model judge panel for Writing Clarity, Utility, and (script-based) Accuracy.

Each rubric dimension is scored independently by every available judge model
using a chain-of-thought protocol (qualitative analysis first, then numeric
score). Inter-rater reliability is reported via Krippendorff's and Cronbach's
alpha. Accuracy is computed deterministically against ground-truth specs.

Protocol follows Choudhury, Vanneste & Zohrehvand (2025).
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

# Avoid HuggingFace tokenizers deadlocking under ThreadPoolExecutor.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config
import prompts


load_dotenv()


JUDGE_MODELS = config.JUDGE_MODELS


class DimensionJudgment(BaseModel):
    """Chain-of-thought judgment for a single evaluation dimension."""
    qualitative_analysis: str = Field(
        description="Detailed qualitative analysis of this dimension. "
                    "Explain what you observe, provide specific examples from the report, "
                    "and reason about strengths and weaknesses BEFORE scoring."
    )
    score: float = Field(
        description="Quantitative score from 1.0 to 5.0 based on your qualitative analysis",
        ge=1.0,
        le=5.0,
    )


@dataclass
class DimensionResult:
    """Result of evaluating a single dimension with a single model."""
    dimension: str
    model_name: str
    qualitative_analysis: str
    score: float
    latency_seconds: float
    tokens_used: int = 0


@dataclass
class MetricResult:
    metric_name: str
    dimensions: dict[str, list[DimensionResult]] = field(default_factory=dict)
    mean_score: float = 0.0
    per_model_scores: dict[str, float] = field(default_factory=dict)
    per_dimension_scores: dict[str, float] = field(default_factory=dict)
    inter_rater_alpha: float = 0.0
    inter_rater_cronbach: float = 0.0


@dataclass
class EvaluationResultV2:
    quality: MetricResult = field(default_factory=lambda: MetricResult(metric_name="writing_clarity"))
    utility: MetricResult = field(default_factory=lambda: MetricResult(metric_name="utility"))
    accuracy_score: float = 0.0
    accuracy_reasoning: str = ""
    total_eval_tokens: int = 0
    total_eval_cost_usd: float = 0.0
    total_eval_latency_seconds: float = 0.0
    final_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "quality": {
                "mean_score": self.quality.mean_score,
                "per_model_scores": self.quality.per_model_scores,
                "per_dimension_scores": self.quality.per_dimension_scores,
                "inter_rater_alpha": self.quality.inter_rater_alpha,
                "inter_rater_cronbach": self.quality.inter_rater_cronbach,
                "dimensions": {
                    dim: [
                        {
                            "model": r.model_name,
                            "score": r.score,
                            "analysis": r.qualitative_analysis,
                            "latency": r.latency_seconds,
                        }
                        for r in results
                    ]
                    for dim, results in self.quality.dimensions.items()
                },
            },
            "utility": {
                "mean_score": self.utility.mean_score,
                "per_model_scores": self.utility.per_model_scores,
                "per_dimension_scores": self.utility.per_dimension_scores,
                "inter_rater_alpha": self.utility.inter_rater_alpha,
                "inter_rater_cronbach": self.utility.inter_rater_cronbach,
                "dimensions": {
                    dim: [
                        {
                            "model": r.model_name,
                            "score": r.score,
                            "analysis": r.qualitative_analysis,
                            "latency": r.latency_seconds,
                        }
                        for r in results
                    ]
                    for dim, results in self.utility.dimensions.items()
                },
            },
            "accuracy": {
                "score": self.accuracy_score,
                "reasoning": self.accuracy_reasoning,
            },
            "efficiency": {
                "total_tokens": self.total_eval_tokens,
                "cost_usd": self.total_eval_cost_usd,
                "latency_seconds": self.total_eval_latency_seconds,
            },
            "final_score": self.final_score,
        }


WRITING_CLARITY_DIMENSIONS = prompts.EVAL_QUALITY_DIMENSIONS
UTILITY_DIMENSIONS = prompts.EVAL_UTILITY_DIMENSIONS


def load_ground_truth(asin: str, dataset_path: str = "dataset_final.json") -> Optional[dict]:
    """Load ground truth metadata for a given ASIN from the dataset."""
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        for item in data:
            if item.get("id") == asin:
                return item.get("metadata", {})
        return None
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return None


def extract_specs_from_text(text: str) -> dict:
    specs = {}

    gpu_patterns = [
        r'RTX\s*(\d{4})\s*(Ti)?', r'GTX\s*(\d{4})\s*(Ti)?',
        r'GeForce\s+RTX\s*(\d{4})', r'GeForce\s+GTX\s*(\d{4})',
    ]
    for pattern in gpu_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            specs['gpu'] = match.group(0).strip()
            break

    ram_patterns = [r'(\d+)\s*GB\s*(DDR\d)?(\s*RAM)?', r'(\d+)GB\s+DDR\d']
    for pattern in ram_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            specs['ram'] = match.group(0).strip()
            break

    cpu_patterns = [
        r'Ryzen\s*\d+[\s\-]*\d*\w*', r'Core\s*i\d[\s\-]*\d*\w*',
        r'Intel\s+Core\s+i\d', r'AMD\s+Ryzen\s+\d+',
    ]
    for pattern in cpu_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            specs['cpu'] = match.group(0).strip()
            break

    storage_patterns = [r'(\d+)\s*(TB|GB)\s*(SSD|NVMe|HDD)', r'(\d+)(TB|GB)\s+SSD']
    for pattern in storage_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            specs['storage'] = match.group(0).strip()
            break

    display_patterns = [r'(\d+\.?\d*)[""\s]*(inch|")', r'(\d+)\s*Hz', r'FHD|QHD|4K|1080p|1440p']
    for pattern in display_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if 'display' not in specs:
                specs['display'] = match.group(0).strip()
            break

    price_patterns = [r'\$[\d,]+\.?\d*', r'[\d,]+\.?\d*\s*(?:USD|dollars)']
    for pattern in price_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            specs['price'] = match.group(0).strip()
            break

    return specs


def normalize_spec(spec: str) -> str:
    return re.sub(r'[\s\-_]+', '', spec.lower())


def compare_spec(key: str, report_val: str, gt_val: str) -> bool:
    """Semantic comparison per spec category (handles RAM/GPU/CPU/storage forms)."""
    report_lower = report_val.lower()
    gt_lower = gt_val.lower()

    if key == 'ram':
        r_num = re.search(r'(\d+)\s*gb', report_lower)
        g_num = re.search(r'(\d+)\s*gb', gt_lower)
        if r_num and g_num:
            return r_num.group(1) == g_num.group(1)

    elif key == 'gpu':
        r_model = re.search(r'(rtx|gtx)\s*(\d{4})\s*(ti)?', report_lower)
        g_model = re.search(r'(rtx|gtx)\s*(\d{4})\s*(ti)?', gt_lower)
        if r_model and g_model:
            return (r_model.group(1) == g_model.group(1) and
                    r_model.group(2) == g_model.group(2) and
                    r_model.group(3) == g_model.group(3))

    elif key == 'cpu':
        r_cpu = re.search(r'(ryzen\s*\d|core\s*i\d)', report_lower)
        g_cpu = re.search(r'(ryzen\s*\d|core\s*i\d)', gt_lower)
        if r_cpu and g_cpu:
            return re.sub(r'\s+', '', r_cpu.group(1)) == re.sub(r'\s+', '', g_cpu.group(1))

    elif key == 'storage':
        r_size = re.search(r'(\d+)\s*(tb|gb)', report_lower)
        g_size = re.search(r'(\d+)\s*(tb|gb)', gt_lower)
        if r_size and g_size:
            return (r_size.group(1) == g_size.group(1) and
                    r_size.group(2) == g_size.group(2))

    return normalize_spec(report_val) in normalize_spec(gt_val) or \
           normalize_spec(gt_val) in normalize_spec(report_val)


def calculate_script_accuracy(
    report_text: str,
    ground_truth: dict,
    verbose: bool = True,
) -> tuple[float, str]:
    """Calculate accuracy score by comparing extracted specs with ground truth."""
    report_specs = extract_specs_from_text(report_text)
    gt_text = f"{ground_truth.get('title', '')} {ground_truth.get('description', '')}"
    gt_specs = extract_specs_from_text(gt_text)

    if verbose:
        print(f"\nScript-based accuracy check")
        print(f"   Report specs: {report_specs}")
        print(f"   Ground truth specs: {gt_specs}")

    matches = 0
    mismatches = []
    total_checks = 0

    for key in ['gpu', 'ram', 'cpu', 'storage']:
        if key in report_specs:
            total_checks += 1
            if key in gt_specs:
                if compare_spec(key, report_specs[key], gt_specs[key]):
                    matches += 1
                else:
                    mismatches.append(f"{key}: '{report_specs[key]}' vs '{gt_specs[key]}'")
            else:
                mismatches.append(f"{key}: '{report_specs[key]}' (not in ground truth)")

    if total_checks == 0:
        score = 3.0
        reasoning = "No verifiable technical specifications found in report."
    else:
        accuracy_ratio = matches / total_checks
        score = 1.0 + (accuracy_ratio * 4.0)
        if mismatches:
            reasoning = f"Matched {matches}/{total_checks} specs. Mismatches: {'; '.join(mismatches)}"
        else:
            reasoning = f"All {matches} verifiable specs matched ground truth perfectly."

    return round(score, 2), reasoning


def _get_llm_client(model_config: dict) -> ChatOpenAI:
    kwargs = {
        "model": model_config["model_id"],
        "temperature": 0,
        # 120s rather than 60s: Qwen/GLM judges occasionally exceed the shorter timeout.
        "request_timeout": 120,
    }

    if model_config["provider"] == "openrouter":
        api_key = os.getenv(model_config["api_key_env"])
        if not api_key:
            raise ValueError(
                f"API key not found for {model_config['name']}. "
                f"Set {model_config['api_key_env']} environment variable."
            )
        kwargs["openai_api_key"] = api_key
        kwargs["openai_api_base"] = model_config["api_base"]
    else:
        api_key = os.getenv(model_config["api_key_env"])
        if api_key:
            kwargs["openai_api_key"] = api_key

    return ChatOpenAI(**kwargs)


def _parse_json_fallback(text: str) -> dict:
    """Extract a judgment dict from a free-text LLM response, tolerating code fences."""
    def _clean_and_parse(raw: str) -> dict:
        # Strip non-whitespace control chars before parsing; some judges emit them.
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
        return json.loads(cleaned, strict=False)

    json_match = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return _clean_and_parse(json_match.group(1))
    json_match = re.search(r'\{[^{}]*"qualitative_analysis"[^{}]*"score"[^{}]*\}', text, re.DOTALL)
    if json_match:
        return _clean_and_parse(json_match.group(0))
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return _clean_and_parse(json_match.group(0))
    raise ValueError("No JSON found in response")


def _evaluate_single_dimension(
    dimension_key: str,
    dimension_config: dict,
    model_config: dict,
    report_text: str,
    product_metadata: str,
) -> DimensionResult:
    """Score one dimension with one judge: qualitative analysis first, then a numeric score.

    Uses native structured output when the model supports it, falls back to
    free-text JSON parsing otherwise. Retries with backoff on transient errors.
    """
    start_time = time.time()

    user_content = f"""=== PRODUCT METADATA (For Context) ===
{product_metadata}

=== REPORT TO EVALUATE ===
{report_text}

INSTRUCTIONS:
1. First, provide your detailed QUALITATIVE ANALYSIS of the {dimension_config['name']} dimension.
   Be specific — cite examples from the report, explain your observations.
2. Then, based ONLY on your qualitative analysis, provide your QUANTITATIVE SCORE (1-5).

Remember: analyze first, score second. Your score must be justified by your analysis."""

    messages = [
        SystemMessage(content=dimension_config["prompt"]),
        HumanMessage(content=user_content),
    ]

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            llm = _get_llm_client(model_config)
            use_structured = model_config.get("structured_output", True)

            if use_structured:
                try:
                    judge_llm = llm.with_structured_output(DimensionJudgment)
                    result = judge_llm.invoke(messages)
                    latency = time.time() - start_time

                    if result is None:
                        raise ValueError(f"Judge {model_config['name']} returned None for structured output")

                    return DimensionResult(
                        dimension=dimension_key,
                        model_name=model_config["name"],
                        qualitative_analysis=result.qualitative_analysis,
                        score=result.score,
                        latency_seconds=round(latency, 2),
                    )

                except Exception as structured_err:
                    # Re-raise on timeout so the retry loop can back off; otherwise
                    # silently fall through to the free-text JSON fallback path.
                    if "timeout" in str(structured_err).lower() and attempt < max_retries - 1:
                        raise structured_err
                    print(f"   {model_config['name']}: structured output failed, using JSON fallback")

            json_instruction = (
                "\n\nRESPOND ONLY with a JSON object in this exact format:\n"
                '{"qualitative_analysis": "<your detailed analysis>", "score": <number 1.0-5.0>}'
            )
            fallback_messages = [
                SystemMessage(content=dimension_config["prompt"]),
                HumanMessage(content=user_content + json_instruction),
            ]

            response = llm.invoke(fallback_messages)
            parsed = _parse_json_fallback(response.content)
            latency = time.time() - start_time

            score = float(parsed.get("score", 3.0))
            score = max(1.0, min(5.0, score))

            return DimensionResult(
                dimension=dimension_key,
                model_name=model_config["name"],
                qualitative_analysis=parsed.get("qualitative_analysis", "Fallback parse"),
                score=score,
                latency_seconds=round(latency, 2),
            )

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"   Retry {attempt+1}/{max_retries} for {dimension_key}/{model_config['name']} due to: {str(e)[:50]}...")
                time.sleep(retry_delay * (attempt + 1))
                continue

            latency = time.time() - start_time
            print(f"   Final error evaluating {dimension_config['name']} with {model_config['name']}: {e}")
            # Neutral fallback so the run still produces a complete matrix.
            return DimensionResult(
                dimension=dimension_key,
                model_name=model_config["name"],
                qualitative_analysis=f"ERROR: {str(e)}",
                score=3.0,
                latency_seconds=round(latency, 2),
            )


def calculate_krippendorff_alpha(scores_matrix: list[list[float]]) -> float:
    """Krippendorff's alpha (interval-level) for an [n_items x n_raters] score matrix."""
    if not scores_matrix or len(scores_matrix) < 2:
        return 0.0

    n_raters = len(scores_matrix[0])
    if n_raters < 2:
        return 0.0

    # Flatten all valid values
    all_values = []
    for row in scores_matrix:
        all_values.extend([v for v in row if v is not None])

    if len(all_values) < 2:
        return 0.0

    # Calculate observed disagreement (Do)
    n_items = len(scores_matrix)
    do_sum = 0.0
    do_count = 0

    for row in scores_matrix:
        valid = [v for v in row if v is not None]
        m = len(valid)
        if m < 2:
            continue
        for i in range(m):
            for j in range(i + 1, m):
                do_sum += (valid[i] - valid[j]) ** 2
                do_count += 1

    if do_count == 0:
        return 1.0  # perfect agreement

    Do = do_sum / do_count

    # Calculate expected disagreement (De)
    n = len(all_values)
    de_sum = 0.0
    de_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            de_sum += (all_values[i] - all_values[j]) ** 2
            de_count += 1

    if de_count == 0:
        return 1.0

    De = de_sum / de_count

    if De == 0:
        return 1.0
    alpha = 1.0 - (Do / De) if De > 0 else 1.0
    return round(alpha, 4)


def calculate_cronbachs_alpha(scores_matrix: list[list[float]]) -> float:
    """Cronbach's alpha for the same matrix; tolerant to absolute-score differences."""
    if not scores_matrix or len(scores_matrix) < 2:
        return 0.0

    n_items = len(scores_matrix)
    n_raters = len(scores_matrix[0])
    if n_raters < 2:
        return 0.0

    # Drop any rater (column) with missing values so all raters score every item.
    valid_cols = []
    for col_idx in range(n_raters):
        col = [scores_matrix[row_idx][col_idx] for row_idx in range(n_items)]
        if all(v is not None for v in col):
            valid_cols.append(col)

    k = len(valid_cols)
    if k < 2:
        return 0.0

    rater_variances = []
    for col in valid_cols:
        mean = sum(col) / len(col)
        var = sum((x - mean) ** 2 for x in col) / len(col)
        rater_variances.append(var)

    totals = [sum(valid_cols[c][i] for c in range(k)) for i in range(n_items)]
    total_mean = sum(totals) / len(totals)
    total_var = sum((t - total_mean) ** 2 for t in totals) / len(totals)

    sum_rater_var = sum(rater_variances)

    if total_var == 0:
        return 1.0 if sum_rater_var == 0 else 0.0

    alpha = (k / (k - 1)) * (1 - sum_rater_var / total_var)
    return round(alpha, 4)


def _evaluate_metric(
    metric_name: str,
    dimensions: dict,
    report_text: str,
    product_metadata: str,
    models: list[dict],
    verbose: bool = True,
) -> MetricResult:
    """Score every (dimension x model) cell in parallel and aggregate."""
    result = MetricResult(metric_name=metric_name)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {metric_name.upper()} EVALUATION")
        print(f"  Dimensions: {', '.join(d['name'] for d in dimensions.values())}")
        print(f"  Models: {', '.join(m['name'] for m in models)}")
        print(f"{'='*60}")

    tasks = []
    with ThreadPoolExecutor(max_workers=config.EVAL_MAX_WORKERS) as executor:
        for dim_key, dim_config in dimensions.items():
            result.dimensions[dim_key] = []
            for model_config in models:
                future = executor.submit(
                    _evaluate_single_dimension,
                    dim_key, dim_config, model_config,
                    report_text, product_metadata,
                )
                tasks.append((dim_key, model_config["name"], future))

        for dim_key, model_name, future in tasks:
            try:
                dim_result = future.result(timeout=config.EVAL_TIMEOUT)
                result.dimensions[dim_key].append(dim_result)
                if verbose:
                    print(f"   {dim_result.dimension}/{dim_result.model_name}: "
                          f"{dim_result.score}/5.0 ({dim_result.latency_seconds}s)")
            except Exception as e:
                print(f"   FAIL {dim_key}/{model_name}: {e}")

    model_scores: dict[str, list[float]] = {}
    for dim_key, dim_results in result.dimensions.items():
        for dr in dim_results:
            model_scores.setdefault(dr.model_name, []).append(dr.score)

    for model_name, scores in model_scores.items():
        result.per_model_scores[model_name] = round(sum(scores) / len(scores), 2)

    for dim_key, dim_results in result.dimensions.items():
        scores = [dr.score for dr in dim_results]
        result.per_dimension_scores[dim_key] = round(sum(scores) / len(scores), 2) if scores else 0.0

    all_scores = [dr.score for dim_results in result.dimensions.values() for dr in dim_results]
    result.mean_score = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0

    # Inter-rater matrix: rows = dimensions, columns = models (raters).
    model_names = [m["name"] for m in models]
    scores_matrix = []
    for dim_key in dimensions:
        row = []
        for mname in model_names:
            dim_results = result.dimensions.get(dim_key, [])
            model_score = next((dr.score for dr in dim_results if dr.model_name == mname), None)
            row.append(model_score)
        scores_matrix.append(row)

    result.inter_rater_alpha = calculate_krippendorff_alpha(scores_matrix)
    result.inter_rater_cronbach = calculate_cronbachs_alpha(scores_matrix)

    if verbose:
        print(f"\n   {metric_name.upper()} SUMMARY:")
        print(f"   Mean score:      {result.mean_score}/5.0")
        print(f"   Per-model:       {result.per_model_scores}")
        print(f"   Per-dimension:   {result.per_dimension_scores}")
        print(f"   Krippendorff a:  {result.inter_rater_alpha}")
        print(f"   Cronbach a:      {result.inter_rater_cronbach}")

    return result


def save_cot_analyses(
    result: "EvaluationResultV2",
    output_dir: str,
    run_id: str,
) -> str:
    """Dump every judge's qualitative analysis to a readable markdown file."""
    from pathlib import Path
    
    out_path = Path(output_dir) / f"cot_{run_id}.md"
    
    lines = []
    lines.append(f"# Chain-of-Thought Analyses: {run_id}")
    lines.append(f"")
    lines.append(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"")
    lines.append(f"---")
    
    # Quality dimensions
    lines.append(f"")
    lines.append(f"## Writing Clarity")
    for dim_key, dim_results in result.quality.dimensions.items():
        dim_name = dim_key.replace('_', ' ').title()
        lines.append(f"")
        lines.append(f"### {dim_name}")
        for dr in dim_results:
            lines.append(f"")
            lines.append(f"#### {dr.model_name} - Score: {dr.score}/5.0")
            lines.append(f"")
            lines.append(f"{dr.qualitative_analysis}")
            lines.append(f"")
            lines.append(f"*Latency: {dr.latency_seconds}s*")
    
    lines.append(f"")
    lines.append(f"---")
    
    # Utility dimensions
    lines.append(f"")
    lines.append(f"## Utility")
    for dim_key, dim_results in result.utility.dimensions.items():
        dim_name = dim_key.replace('_', ' ').title()
        lines.append(f"")
        lines.append(f"### {dim_name}")
        for dr in dim_results:
            lines.append(f"")
            lines.append(f"#### {dr.model_name} - Score: {dr.score}/5.0")
            lines.append(f"")
            lines.append(f"{dr.qualitative_analysis}")
            lines.append(f"")
            lines.append(f"*Latency: {dr.latency_seconds}s*")
    
    lines.append(f"")
    lines.append(f"---")
    
    # Accuracy
    lines.append(f"")
    lines.append(f"## Accuracy (Deterministic)")
    lines.append(f"")
    lines.append(f"**Score:** {result.accuracy_score}/5.0")
    lines.append(f"")
    lines.append(f"**Reasoning:** {result.accuracy_reasoning}")
    
    # Summary
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Summary")
    lines.append(f"")
    lines.append(f"| Metric | Mean Score | Krippendorff α | Cronbach α |")
    lines.append(f"|---|---|---|---|")
    lines.append(f"| Writing Clarity | {result.quality.mean_score}/5.0 | {result.quality.inter_rater_alpha} | {result.quality.inter_rater_cronbach} |")
    lines.append(f"| Utility | {result.utility.mean_score}/5.0 | {result.utility.inter_rater_alpha} | {result.utility.inter_rater_cronbach} |")
    lines.append(f"| Accuracy | {result.accuracy_score}/5.0 | — |")
    lines.append(f"| **Final** | **{result.final_score}/5.0** | — |")
    
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)


def evaluate_report_v2(
    report_text: str,
    product_metadata: str,
    asin: Optional[str] = None,
    models: Optional[list[dict]] = None,
    verbose: bool = True,
    output_dir: Optional[str] = None,
    run_id: Optional[str] = None,
) -> EvaluationResultV2:
    """Evaluate a single BI report. Returns an EvaluationResultV2 with all judge cells."""
    if models is None:
        models = JUDGE_MODELS

    # Skip judges whose API key is not set so the panel still runs partial.
    available_models = []
    for m in models:
        key = os.getenv(m["api_key_env"])
        if key:
            available_models.append(m)
        elif verbose:
            print(f"   Skipping {m['name']} - {m['api_key_env']} not set")

    if not available_models:
        raise ValueError("No models available. Set at least OPENAI_API_KEY.")

    eval_start = time.time()
    result = EvaluationResultV2()

    if verbose:
        print("\n" + "=" * 60)
        print("  Judge panel evaluation")
        print("=" * 60)
        print(f"  Available models: {[m['name'] for m in available_models]}")

    result.quality = _evaluate_metric(
        "writing_clarity",
        WRITING_CLARITY_DIMENSIONS,
        report_text,
        product_metadata,
        available_models,
        verbose,
    )

    result.utility = _evaluate_metric(
        "utility",
        UTILITY_DIMENSIONS,
        report_text,
        product_metadata,
        available_models,
        verbose,
    )

    if verbose:
        print(f"\n{'='*60}")
        print("  Accuracy (deterministic)")
        print(f"{'='*60}")

    ground_truth = None
    if asin:
        ground_truth = load_ground_truth(asin)

    if ground_truth:
        result.accuracy_score, result.accuracy_reasoning = calculate_script_accuracy(
            report_text, ground_truth, verbose
        )
    else:
        gt_dict = {"title": product_metadata, "description": ""}
        result.accuracy_score, result.accuracy_reasoning = calculate_script_accuracy(
            report_text, gt_dict, verbose
        )

    if verbose:
        print(f"   Score: {result.accuracy_score}/5.0")
        print(f"   Reasoning: {result.accuracy_reasoning}")

    result.total_eval_latency_seconds = round(time.time() - eval_start, 2)
    result.final_score = round(
        (result.quality.mean_score + result.utility.mean_score + result.accuracy_score) / 3, 2
    )

    if verbose:
        print("\n" + "=" * 60)
        print("  Final aggregated results")
        print("=" * 60)
        print(f"   Writing Clarity:  {result.quality.mean_score}/5.0 (Krippendorff a={result.quality.inter_rater_alpha}, Cronbach a={result.quality.inter_rater_cronbach})")
        print(f"   Utility:          {result.utility.mean_score}/5.0 (Krippendorff a={result.utility.inter_rater_alpha}, Cronbach a={result.utility.inter_rater_cronbach})")
        print(f"   Accuracy:         {result.accuracy_score}/5.0 (deterministic)")
        print(f"\n   FINAL SCORE: {result.final_score}/5.0")
        print(f"   Total latency: {result.total_eval_latency_seconds}s")
        print("=" * 60)

    if output_dir and run_id:
        try:
            cot_path = save_cot_analyses(result, output_dir, run_id)
            if verbose:
                print(f"   CoT analyses saved: {cot_path}")
        except Exception as e:
            if verbose:
                print(f"   Failed to save CoT analyses: {e}")

    return result


if __name__ == "__main__":
    DUMMY_METADATA = """
Title: MSI Katana A15 AI 15.6" 144Hz FHD Gaming Laptop
Price: $1334.66
Specifications:
- CPU: AMD Ryzen 7-8845HS
- GPU: NVIDIA GeForce RTX 4060 (8GB GDDR6)
- RAM: 32GB DDR5
- Storage: 1TB NVMe SSD
- Display: 15.6" FHD 144Hz
"""

    DUMMY_REPORT = """
## Business Intelligence Report: MSI Katana A15 AI Gaming Laptop

### Executive Summary
This report analyzes customer feedback for the MSI Katana A15 AI gaming laptop, focusing on
blue screen (BSOD) issues reported by users.

### Findings

#### Hardware Specifications
- The laptop features RTX 4060 graphics and Ryzen 7-8845HS processor
- 32GB DDR5 RAM and 1TB NVMe SSD storage
- 15.6" 144Hz display

#### Blue Screen Issues
Multiple customers reported experiencing Blue Screen of Death crashes:
- One user stated: "Out of the box, the new laptop suffered BSOD crashes every 10 minutes"
- Issues appear more common with the RTX 4060 variant

### Recommendations

1. **Quality Control**: Improve pre-shipment testing to catch BSOD issues before delivery
2. **Driver Updates**: Work with NVIDIA on RTX 4060 driver stability
3. **Warranty Process**: Streamline warranty claims for affected customers

### Conclusion
While the MSI Katana offers strong gaming performance, the recurring BSOD issues significantly
impact the user experience and brand reputation. Immediate action is recommended.
"""

    print("Evaluation engine standalone test")
    print("Only models with available API keys will be used.\n")

    result = evaluate_report_v2(DUMMY_REPORT, DUMMY_METADATA, asin="B0CXVGSY2H")

    print("\n\nFull result (JSON):\n")
    print(json.dumps(result.to_dict(), indent=2))
