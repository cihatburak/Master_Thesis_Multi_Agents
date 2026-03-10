"""
Efficiency Tracker for Multi-Agent BI Report System

Provides timing, token tracking, and cost analysis across all three architectures.
Designed for the comparative analysis thesis — enables quality-efficiency trade-off analysis.

Usage:
    tracker = EfficiencyTracker("baseline")
    tracker.start()
    # ... run architecture ...
    tracker.record_from_callback(cb)  # from get_openai_callback
    tracker.stop()
    metrics = tracker.get_metrics()
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ArchitectureMetrics:
    """Complete efficiency metrics for a single architecture run."""
    architecture: str
    product_asin: str = ""

    # Token usage
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Cost
    total_cost_usd: float = 0.0

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    latency_seconds: float = 0.0

    # Architecture-specific
    step_count: int = 0
    verification_attempts: int = 0  # hierarchical only

    # Metadata
    model_name: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComparisonReport:
    """Side-by-side comparison of efficiency across architectures."""
    baseline: Optional[ArchitectureMetrics] = None
    flat: Optional[ArchitectureMetrics] = None
    hierarchical: Optional[ArchitectureMetrics] = None
    product_asin: str = ""
    generated_at: str = ""

    def to_dict(self) -> dict:
        result = {
            "product_asin": self.product_asin,
            "generated_at": self.generated_at,
        }
        for arch_name in ["baseline", "flat", "hierarchical"]:
            metrics = getattr(self, arch_name)
            if metrics:
                result[arch_name] = metrics.to_dict()
        return result


# =============================================================================
# EFFICIENCY TRACKER
# =============================================================================

class EfficiencyTracker:
    """
    Tracks efficiency metrics for a single architecture execution.

    Usage:
        tracker = EfficiencyTracker("flat", asin="B0CXVGSY2H", model="gpt-4o")
        tracker.start()
        with get_openai_callback() as cb:
            # ... run graph ...
            tracker.record_from_callback(cb)
        tracker.stop()
        metrics = tracker.get_metrics()
    """

    def __init__(
        self,
        architecture: str,
        asin: str = "",
        model: str = "",
    ):
        self.architecture = architecture
        self.asin = asin
        self.model = model
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._total_tokens: int = 0
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._total_cost: float = 0.0
        self._step_count: int = 0
        self._verification_attempts: int = 0

    def start(self):
        """Mark the start of execution."""
        self._start_time = time.time()

    def stop(self):
        """Mark the end of execution."""
        self._end_time = time.time()

    def record_from_callback(self, cb):
        """
        Record token usage from LangChain's get_openai_callback.

        Args:
            cb: The callback object from `with get_openai_callback() as cb:`
        """
        self._total_tokens = cb.total_tokens
        self._prompt_tokens = cb.prompt_tokens
        self._completion_tokens = cb.completion_tokens
        self._total_cost = cb.total_cost

    def record_from_metrics(self, metrics: dict):
        """
        Record from an existing metrics dict (backward compatibility
        with current run_baseline/run_flat_graph/run_hierarchical_graph).
        """
        self._total_tokens = metrics.get("total_tokens", 0)
        self._prompt_tokens = metrics.get("prompt_tokens", 0)
        self._completion_tokens = metrics.get("completion_tokens", 0)
        self._total_cost = metrics.get("total_cost", 0.0)
        self._step_count = metrics.get("step_count", 0)
        self._verification_attempts = metrics.get("verification_attempts", 0)

    def set_step_count(self, count: int):
        self._step_count = count

    def set_verification_attempts(self, count: int):
        self._verification_attempts = count

    def get_metrics(self) -> ArchitectureMetrics:
        """Get the complete metrics for this run."""
        latency = self._end_time - self._start_time if self._end_time > 0 else 0.0

        return ArchitectureMetrics(
            architecture=self.architecture,
            product_asin=self.asin,
            total_tokens=self._total_tokens,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_cost_usd=round(self._total_cost, 6),
            start_time=self._start_time,
            end_time=self._end_time,
            latency_seconds=round(latency, 2),
            step_count=self._step_count,
            verification_attempts=self._verification_attempts,
            model_name=self.model,
            timestamp=datetime.now().isoformat(),
        )


# =============================================================================
# COMPARISON & EXPORT UTILITIES
# =============================================================================

def create_comparison(
    baseline_metrics: Optional[ArchitectureMetrics] = None,
    flat_metrics: Optional[ArchitectureMetrics] = None,
    hierarchical_metrics: Optional[ArchitectureMetrics] = None,
    asin: str = "",
) -> ComparisonReport:
    """Create a side-by-side comparison report."""
    return ComparisonReport(
        baseline=baseline_metrics,
        flat=flat_metrics,
        hierarchical=hierarchical_metrics,
        product_asin=asin,
        generated_at=datetime.now().isoformat(),
    )


def print_comparison(report: ComparisonReport):
    """Print a formatted comparison table to console."""
    print("\n" + "=" * 75)
    print("  EFFICIENCY COMPARISON ACROSS ARCHITECTURES")
    print(f"  Product: {report.product_asin}")
    print("=" * 75)

    headers = ["Metric", "Baseline", "Flat", "Hierarchical"]
    rows = []

    def _val(metrics, attr, fmt=None):
        if metrics is None:
            return "—"
        val = getattr(metrics, attr, "—")
        if fmt and val != "—":
            return fmt(val)
        return str(val)

    rows.append([
        "Total Tokens",
        _val(report.baseline, "total_tokens", lambda v: f"{v:,}"),
        _val(report.flat, "total_tokens", lambda v: f"{v:,}"),
        _val(report.hierarchical, "total_tokens", lambda v: f"{v:,}"),
    ])
    rows.append([
        "Prompt Tokens",
        _val(report.baseline, "prompt_tokens", lambda v: f"{v:,}"),
        _val(report.flat, "prompt_tokens", lambda v: f"{v:,}"),
        _val(report.hierarchical, "prompt_tokens", lambda v: f"{v:,}"),
    ])
    rows.append([
        "Completion Tokens",
        _val(report.baseline, "completion_tokens", lambda v: f"{v:,}"),
        _val(report.flat, "completion_tokens", lambda v: f"{v:,}"),
        _val(report.hierarchical, "completion_tokens", lambda v: f"{v:,}"),
    ])
    rows.append([
        "Cost (USD)",
        _val(report.baseline, "total_cost_usd", lambda v: f"${v:.4f}"),
        _val(report.flat, "total_cost_usd", lambda v: f"${v:.4f}"),
        _val(report.hierarchical, "total_cost_usd", lambda v: f"${v:.4f}"),
    ])
    rows.append([
        "Latency (sec)",
        _val(report.baseline, "latency_seconds", lambda v: f"{v:.1f}s"),
        _val(report.flat, "latency_seconds", lambda v: f"{v:.1f}s"),
        _val(report.hierarchical, "latency_seconds", lambda v: f"{v:.1f}s"),
    ])
    rows.append([
        "Steps",
        _val(report.baseline, "step_count"),
        _val(report.flat, "step_count"),
        _val(report.hierarchical, "step_count"),
    ])
    rows.append([
        "Verifications",
        "—",
        "—",
        _val(report.hierarchical, "verification_attempts"),
    ])

    # Print table
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(4)]
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"\n  {header_line}")
    print(f"  {'─' * sum(col_widths)}")
    for row in rows:
        line = "".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        print(f"  {line}")
    print("=" * 75)

    # Quality-efficiency insight
    if report.baseline and report.hierarchical:
        token_ratio = report.hierarchical.total_tokens / max(report.baseline.total_tokens, 1)
        cost_ratio = report.hierarchical.total_cost_usd / max(report.baseline.total_cost_usd, 0.0001)
        print(f"\n  📊 Hierarchical uses {token_ratio:.1f}x tokens and {cost_ratio:.1f}x cost vs Baseline")

    if report.flat and report.hierarchical:
        token_ratio = report.hierarchical.total_tokens / max(report.flat.total_tokens, 1)
        cost_ratio = report.hierarchical.total_cost_usd / max(report.flat.total_cost_usd, 0.0001)
        print(f"  📊 Hierarchical uses {token_ratio:.1f}x tokens and {cost_ratio:.1f}x cost vs Flat")

    print()


def save_comparison(report: ComparisonReport, output_dir: str = "logs") -> str:
    """Save comparison report to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"efficiency_comparison_{report.product_asin}_{timestamp}.json"
    filepath = output_path / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"  💾 Comparison saved to: {filepath}")
    return str(filepath)


# =============================================================================
# QUICK HELPER — wrap existing run functions
# =============================================================================

def run_with_tracking(run_func, architecture: str, query: str, asin: str = "",
                      model: str = "gpt-4o", session_id: str = "session"):
    """
    Wrapper that adds efficiency tracking to any architecture's run function.

    Args:
        run_func: The run function (run_baseline, run_flat_graph, run_hierarchical_graph)
        architecture: "baseline", "flat", or "hierarchical"
        query: The analysis query
        asin: Product ASIN
        model: Model name for tracking
        session_id: Session ID

    Returns:
        tuple: (report_text, messages_or_none, efficiency_metrics)
    """
    tracker = EfficiencyTracker(architecture, asin=asin, model=model)
    tracker.start()

    if architecture == "baseline":
        report, metrics = run_func(query, asin, session_id)
        messages = None
    else:
        report, messages, metrics = run_func(query, session_id)

    tracker.record_from_metrics(metrics)
    tracker.stop()

    return report, messages, tracker.get_metrics()
