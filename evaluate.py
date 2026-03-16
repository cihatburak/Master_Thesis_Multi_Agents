"""
Evaluation Engine v2 — Dimension-Separated, Chain-of-Thought, Multi-Model
Multi-Agent BI Report System

Redesigned based on supervisor feedback and reference paper
("The Wade Test" — Choudhury, Vanneste & Zohrehvand, 2025):

Key changes from v1:
  - Each quality/utility sub-dimension is evaluated SEPARATELY
  - Chain-of-Thought: qualitative reasoning FIRST, then quantitative score
  - Multi-model judging: runs across 3-4 LLM models for inter-rater reliability
  - Inter-rater reliability metrics (Krippendorff's alpha)
  - Efficiency metrics tracking (tokens, cost, latency)
  - Accuracy (script-based) remains unchanged
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

# Prevent tokenizers fork deadlock with ThreadPoolExecutor
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config
import prompts

load_dotenv()

# =============================================================================
# CONFIGURATION — MULTI-MODEL SETUP
# =============================================================================

# Judge models are now centralized in config.py
JUDGE_MODELS = config.JUDGE_MODELS

# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

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
    """Aggregated result for a metric (quality/utility) across all dimensions and models."""
    metric_name: str
    dimensions: dict[str, list[DimensionResult]] = field(default_factory=dict)
    # Aggregated scores
    mean_score: float = 0.0
    per_model_scores: dict[str, float] = field(default_factory=dict)
    per_dimension_scores: dict[str, float] = field(default_factory=dict)
    inter_rater_alpha: float = 0.0
    inter_rater_cronbach: float = 0.0


@dataclass
class EvaluationResultV2:
    """Complete evaluation result with all metrics, dimensions, and models."""
    quality: MetricResult = field(default_factory=lambda: MetricResult(metric_name="quality"))
    utility: MetricResult = field(default_factory=lambda: MetricResult(metric_name="utility"))
    accuracy_score: float = 0.0
    accuracy_reasoning: str = ""
    # Efficiency
    total_eval_tokens: int = 0
    total_eval_cost_usd: float = 0.0
    total_eval_latency_seconds: float = 0.0
    # Final
    final_score: float = 0.0

    def to_dict(self) -> dict:
        """Convert to serializable dict for JSON export."""
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


# =============================================================================
# DIMENSION-SEPARATED PROMPTS — QUALITY
# =============================================================================
# Each dimension is evaluated independently with its own prompt.
# Chain-of-thought: qualitative analysis FIRST, then score.

QUALITY_DIMENSIONS = prompts.EVAL_QUALITY_DIMENSIONS

# =============================================================================
# DIMENSION-SEPARATED PROMPTS — UTILITY
# =============================================================================

UTILITY_DIMENSIONS = prompts.EVAL_UTILITY_DIMENSIONS


# =============================================================================
# SCRIPT-BASED ACCURACY (unchanged from v1)
# =============================================================================

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
    """Extract technical specifications from text using regex patterns."""
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
    """Normalize a spec string for comparison."""
    return re.sub(r'[\s\-_]+', '', spec.lower())


def compare_spec(key: str, report_val: str, gt_val: str) -> bool:
    """Semantic comparison per spec category."""
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
        print(f"\n📊 Script-Based Accuracy Check")
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


# =============================================================================
# LLM JUDGE — DIMENSION-LEVEL EVALUATION
# =============================================================================

def _get_llm_client(model_config: dict) -> ChatOpenAI:
    """Create an LLM client based on model configuration."""
    kwargs = {
        "model": model_config["model_id"],
        "temperature": 0,
        "request_timeout": 120,  # Increased from 60 to handle slow judge models (Qwen/GLM)
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
    """Extract JSON from LLM response text, handling markdown code fences."""
    def _clean_and_parse(raw: str) -> dict:
        """Remove invalid control characters before parsing JSON."""
        # Remove control chars (0x00-0x1F) except valid JSON whitespace (\t, \n, \r)
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
        # Also fix unescaped newlines/tabs INSIDE JSON string values
        return json.loads(cleaned, strict=False)

    # Try to find JSON in code fences first
    json_match = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return _clean_and_parse(json_match.group(1))
    # Try to parse the whole text as JSON
    json_match = re.search(r'\{[^{}]*"qualitative_analysis"[^{}]*"score"[^{}]*\}', text, re.DOTALL)
    if json_match:
        return _clean_and_parse(json_match.group(0))
    # Last resort: find any JSON object
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
    """
    Evaluate a single dimension with a single model.
    Following the reference paper protocol:
      1. Qualitative analysis first (chain-of-thought reasoning)
      2. Then quantitative score based on the analysis

    Uses structured output when available, falls back to JSON parsing otherwise.
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
                # ── Path A: Structured output (native support) ──
                try:
                    judge_llm = llm.with_structured_output(DimensionJudgment)
                    result = judge_llm.invoke(messages)
                    latency = time.time() - start_time

                    return DimensionResult(
                        dimension=dimension_key,
                        model_name=model_config["name"],
                        qualitative_analysis=result.qualitative_analysis,
                        score=result.score,
                        latency_seconds=round(latency, 2),
                    )

                except Exception as structured_err:
                    # If structured failed, fall back to JSON but check if it's a timeout
                    if "timeout" in str(structured_err).lower() and attempt < max_retries - 1:
                        raise structured_err # Trigger retry
                    
                    print(f"   ↩️  {model_config['name']}: structured output failed, using JSON fallback")
                    # Fall through to JSON mode below

            # ── Path B: JSON-mode (direct or fallback) ──
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
            score = max(1.0, min(5.0, score))  # clamp to valid range

            return DimensionResult(
                dimension=dimension_key,
                model_name=model_config["name"],
                qualitative_analysis=parsed.get("qualitative_analysis", "Fallback parse"),
                score=score,
                latency_seconds=round(latency, 2),
            )

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"   🔄  Retry {attempt+1}/{max_retries} for {dimension_key}/{model_config['name']} due to: {str(e)[:50]}...")
                time.sleep(retry_delay * (attempt + 1))
                continue
            
            latency = time.time() - start_time
            print(f"   ⚠️  Final Error evaluating {dimension_config['name']} with {model_config['name']}: {e}")
            return DimensionResult(
                dimension=dimension_key,
                model_name=model_config["name"],
                qualitative_analysis=f"ERROR: {str(e)}",
                score=3.0,  # neutral fallback
                latency_seconds=round(latency, 2),
            )


# =============================================================================
# INTER-RATER RELIABILITY
# =============================================================================

def calculate_krippendorff_alpha(scores_matrix: list[list[float]]) -> float:
    """
    Calculate Krippendorff's alpha for inter-rater reliability.

    Args:
        scores_matrix: List of lists where each inner list contains scores
                       from different raters for the same item.
                       Shape: [n_items x n_raters]

    Returns:
        Alpha value between -1 and 1. >0.8 is good, >0.67 is acceptable.
    """
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
    """
    Calculate Cronbach's alpha for internal consistency.

    Measures whether raters RANK items consistently, regardless of
    absolute score differences. More tolerant than Krippendorff's alpha.

    Args:
        scores_matrix: [n_items x n_raters] — each row is an item,
                       each column is a rater's score.

    Returns:
        Alpha value. >0.8 = excellent, >0.7 = acceptable, >0.6 = questionable.
    """
    if not scores_matrix or len(scores_matrix) < 2:
        return 0.0

    n_items = len(scores_matrix)
    n_raters = len(scores_matrix[0])
    if n_raters < 2:
        return 0.0

    # Filter out columns (raters) that have None values
    valid_cols = []
    for col_idx in range(n_raters):
        col = [scores_matrix[row_idx][col_idx] for row_idx in range(n_items)]
        if all(v is not None for v in col):
            valid_cols.append(col)

    k = len(valid_cols)  # number of valid raters
    if k < 2:
        return 0.0

    # Calculate variance of each rater's scores
    rater_variances = []
    for col in valid_cols:
        mean = sum(col) / len(col)
        var = sum((x - mean) ** 2 for x in col) / len(col)
        rater_variances.append(var)

    # Calculate variance of total scores (sum across raters for each item)
    totals = [sum(valid_cols[c][i] for c in range(k)) for i in range(n_items)]
    total_mean = sum(totals) / len(totals)
    total_var = sum((t - total_mean) ** 2 for t in totals) / len(totals)

    sum_rater_var = sum(rater_variances)

    if total_var == 0:
        return 1.0 if sum_rater_var == 0 else 0.0

    alpha = (k / (k - 1)) * (1 - sum_rater_var / total_var)
    return round(alpha, 4)


# =============================================================================
# METRIC-LEVEL EVALUATION (Quality / Utility)
# =============================================================================

def _evaluate_metric(
    metric_name: str,
    dimensions: dict,
    report_text: str,
    product_metadata: str,
    models: list[dict],
    verbose: bool = True,
) -> MetricResult:
    """
    Evaluate all dimensions of a metric across all models.
    Each dimension is evaluated separately by each model.
    """
    result = MetricResult(metric_name=metric_name)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {metric_name.upper()} EVALUATION")
        print(f"  Dimensions: {', '.join(d['name'] for d in dimensions.values())}")
        print(f"  Models: {', '.join(m['name'] for m in models)}")
        print(f"{'='*60}")

    # Run all dimension×model combinations
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
                    print(f"   ✅ {dim_result.dimension}/{dim_result.model_name}: "
                          f"{dim_result.score}/5.0 ({dim_result.latency_seconds}s)")
            except Exception as e:
                print(f"   ❌ {dim_key}/{model_name}: {e}")

    # Aggregate per-model scores (average across dimensions)
    model_scores: dict[str, list[float]] = {}
    for dim_key, dim_results in result.dimensions.items():
        for dr in dim_results:
            model_scores.setdefault(dr.model_name, []).append(dr.score)

    for model_name, scores in model_scores.items():
        result.per_model_scores[model_name] = round(sum(scores) / len(scores), 2)

    # Aggregate per-dimension scores (average across models)
    for dim_key, dim_results in result.dimensions.items():
        scores = [dr.score for dr in dim_results]
        result.per_dimension_scores[dim_key] = round(sum(scores) / len(scores), 2) if scores else 0.0

    # Overall mean score
    all_scores = [dr.score for dim_results in result.dimensions.values() for dr in dim_results]
    result.mean_score = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0

    # Inter-rater reliability (Krippendorff's alpha)
    # Build matrix: rows = dimensions, columns = models (raters)
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
        print(f"\n   📊 {metric_name.upper()} SUMMARY:")
        print(f"   Mean Score: {result.mean_score}/5.0")
        print(f"   Per-Model: {result.per_model_scores}")
        print(f"   Per-Dimension: {result.per_dimension_scores}")
        print(f"   Krippendorff's α: {result.inter_rater_alpha}")
        print(f"   Cronbach's α:     {result.inter_rater_cronbach}")

    return result


# =============================================================================
# COT ANALYSIS PERSISTENCE
# =============================================================================

def save_cot_analyses(
    result: "EvaluationResultV2",
    output_dir: str,
    run_id: str,
) -> str:
    """
    Save all qualitative (Chain-of-Thought) analyses to a readable markdown file.
    
    Creates a file at {output_dir}/cot_{run_id}.md with all judge analyses
    organized by metric > dimension > model for easy inspection.
    
    Returns the path to the saved file.
    """
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
    lines.append(f"## Quality")
    for dim_key, dim_results in result.quality.dimensions.items():
        dim_name = dim_key.replace('_', ' ').title()
        lines.append(f"")
        lines.append(f"### {dim_name}")
        for dr in dim_results:
            lines.append(f"")
            lines.append(f"#### 🤖 {dr.model_name} — Score: {dr.score}/5.0")
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
            lines.append(f"#### 🤖 {dr.model_name} — Score: {dr.score}/5.0")
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
    lines.append(f"| Quality | {result.quality.mean_score}/5.0 | {result.quality.inter_rater_alpha} | {result.quality.inter_rater_cronbach} |")
    lines.append(f"| Utility | {result.utility.mean_score}/5.0 | {result.utility.inter_rater_alpha} | {result.utility.inter_rater_cronbach} |")
    lines.append(f"| Accuracy | {result.accuracy_score}/5.0 | — |")
    lines.append(f"| **Final** | **{result.final_score}/5.0** | — |")
    
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate_report_v2(
    report_text: str,
    product_metadata: str,
    asin: Optional[str] = None,
    models: Optional[list[dict]] = None,
    verbose: bool = True,
    output_dir: Optional[str] = None,
    run_id: Optional[str] = None,
) -> EvaluationResultV2:
    """
    Evaluate a BI report using the v2 evaluation engine.

    Protocol (based on Choudhury, Vanneste & Zohrehvand, 2025):
      1. Quality: 4 dimensions evaluated separately, CoT, multi-model
      2. Utility: 3 dimensions evaluated separately, CoT, multi-model
      3. Accuracy: deterministic script-based (unchanged)

    Args:
        report_text: The generated BI report to evaluate
        product_metadata: Product specifications string
        asin: Product ASIN for ground truth lookup
        models: List of model configs to use (defaults to JUDGE_MODELS)
        verbose: Whether to print detailed results

    Returns:
        EvaluationResultV2 with all metrics, dimensions, and cross-model results
    """
    if models is None:
        models = JUDGE_MODELS

    # Filter models to only those with available API keys
    available_models = []
    for m in models:
        key = os.getenv(m["api_key_env"])
        if key:
            available_models.append(m)
        elif verbose:
            print(f"   ⚠️  Skipping {m['name']} — {m['api_key_env']} not set")

    if not available_models:
        raise ValueError("No models available. Set at least OPENAI_API_KEY.")

    eval_start = time.time()
    result = EvaluationResultV2()

    if verbose:
        print("\n" + "=" * 60)
        print("  EVALUATION ENGINE v2")
        print("  Dimension-Separated | Chain-of-Thought | Multi-Model")
        print("=" * 60)
        print(f"  Available models: {[m['name'] for m in available_models]}")

    # ─────────────────────────────────────
    # 1. QUALITY (LLM Judges — 4 dimensions)
    # ─────────────────────────────────────
    result.quality = _evaluate_metric(
        "quality",
        QUALITY_DIMENSIONS,
        report_text,
        product_metadata,
        available_models,
        verbose,
    )

    # ─────────────────────────────────────
    # 2. UTILITY (LLM Judges — 3 dimensions)
    # ─────────────────────────────────────
    result.utility = _evaluate_metric(
        "utility",
        UTILITY_DIMENSIONS,
        report_text,
        product_metadata,
        available_models,
        verbose,
    )

    # ─────────────────────────────────────
    # 3. ACCURACY (Script-Based)
    # ─────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print("  ACCURACY (Deterministic)")
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

    # ─────────────────────────────────────
    # AGGREGATE
    # ─────────────────────────────────────
    result.total_eval_latency_seconds = round(time.time() - eval_start, 2)
    result.final_score = round(
        (result.quality.mean_score + result.utility.mean_score + result.accuracy_score) / 3, 2
    )

    if verbose:
        print("\n" + "=" * 60)
        print("  FINAL AGGREGATED RESULTS")
        print("=" * 60)
        print(f"   Quality:  {result.quality.mean_score}/5.0 (Krippendorff α={result.quality.inter_rater_alpha}, Cronbach α={result.quality.inter_rater_cronbach})")
        print(f"   Utility:  {result.utility.mean_score}/5.0 (Krippendorff α={result.utility.inter_rater_alpha}, Cronbach α={result.utility.inter_rater_cronbach})")
        print(f"   Accuracy: {result.accuracy_score}/5.0 (Deterministic)")
        print(f"\n   ⭐ FINAL SCORE: {result.final_score}/5.0")
        print(f"   ⏱  Total Latency: {result.total_eval_latency_seconds}s")
        print("=" * 60)

    # Save CoT analyses to file if output_dir is provided
    if output_dir and run_id:
        try:
            cot_path = save_cot_analyses(result, output_dir, run_id)
            if verbose:
                print(f"   📝 CoT analyses saved: {cot_path}")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Failed to save CoT analyses: {e}")

    return result


# =============================================================================
# BACKWARD COMPATIBILITY — simple interface matching v1
# =============================================================================

def evaluate_report(
    report_text: str,
    product_metadata: str,
    asin: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Backward-compatible wrapper. Returns a dict matching v1 EvaluationResult fields.
    Uses only the first available model for speed (single-model mode).
    """
    # Use only GPT-4o for backward compatibility
    single_model = [JUDGE_MODELS[0]]

    result = evaluate_report_v2(
        report_text, product_metadata, asin,
        models=single_model, verbose=verbose,
    )

    return {
        "quality_score": result.quality.mean_score,
        "quality_reasoning": "; ".join(
            dr.qualitative_analysis[:100]
            for dim_results in result.quality.dimensions.values()
            for dr in dim_results
        ),
        "accuracy_score": result.accuracy_score,
        "accuracy_reasoning": result.accuracy_reasoning,
        "utility_score": result.utility.mean_score,
        "utility_reasoning": "; ".join(
            dr.qualitative_analysis[:100]
            for dim_results in result.utility.dimensions.values()
            for dr in dim_results
        ),
        "final_score": result.final_score,
    }


# =============================================================================
# EXECUTION
# =============================================================================

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

    print("=" * 60)
    print("  EVALUATION ENGINE v2 — TEST RUN")
    print("=" * 60)
    print("\n📄 Testing with dummy report and metadata...")
    print("   (Only models with available API keys will be used)\n")

    result = evaluate_report_v2(DUMMY_REPORT, DUMMY_METADATA, asin="B0CXVGSY2H")

    print("\n\n" + "=" * 60)
    print("  FULL RESULT (JSON)")
    print("=" * 60)
    print(json.dumps(result.to_dict(), indent=2))
