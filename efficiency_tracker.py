"""Per-run efficiency metrics: tokens, cost, latency, step counts.

Used by both architectures to record costs from inside a
`with get_openai_callback() as cb:` block, then summarised side by side.
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ArchitectureMetrics:
    architecture: str
    product_asin: str = ""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    total_cost_usd: float = 0.0

    start_time: float = 0.0
    end_time: float = 0.0
    latency_seconds: float = 0.0

    step_count: int = 0
    verification_attempts: int = 0  # hierarchical only

    model_name: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComparisonReport:
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


class EfficiencyTracker:
    def __init__(self, architecture: str, asin: str = "", model: str = ""):
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
        self._start_time = time.time()

    def stop(self):
        self._end_time = time.time()

    def record_from_callback(self, cb):
        self._total_tokens = cb.total_tokens
        self._prompt_tokens = cb.prompt_tokens
        self._completion_tokens = cb.completion_tokens
        self._total_cost = cb.total_cost

    def record_from_metrics(self, metrics: dict):
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


def create_comparison(
    baseline_metrics: Optional[ArchitectureMetrics] = None,
    flat_metrics: Optional[ArchitectureMetrics] = None,
    hierarchical_metrics: Optional[ArchitectureMetrics] = None,
    asin: str = "",
) -> ComparisonReport:
    return ComparisonReport(
        baseline=baseline_metrics,
        flat=flat_metrics,
        hierarchical=hierarchical_metrics,
        product_asin=asin,
        generated_at=datetime.now().isoformat(),
    )


def print_comparison(report: ComparisonReport):
    print("\n" + "=" * 75)
    print("  EFFICIENCY COMPARISON ACROSS ARCHITECTURES")
    print(f"  Product: {report.product_asin}")
    print("=" * 75)

    headers = ["Metric", "Baseline", "Flat", "Hierarchical"]
    rows = []

    def _val(metrics, attr, fmt=None):
        if metrics is None:
            return "-"
        val = getattr(metrics, attr, "-")
        if fmt and val != "-":
            return fmt(val)
        return str(val)

    rows.append(["Total Tokens",
                 _val(report.baseline, "total_tokens", lambda v: f"{v:,}"),
                 _val(report.flat, "total_tokens", lambda v: f"{v:,}"),
                 _val(report.hierarchical, "total_tokens", lambda v: f"{v:,}")])
    rows.append(["Prompt Tokens",
                 _val(report.baseline, "prompt_tokens", lambda v: f"{v:,}"),
                 _val(report.flat, "prompt_tokens", lambda v: f"{v:,}"),
                 _val(report.hierarchical, "prompt_tokens", lambda v: f"{v:,}")])
    rows.append(["Completion Tokens",
                 _val(report.baseline, "completion_tokens", lambda v: f"{v:,}"),
                 _val(report.flat, "completion_tokens", lambda v: f"{v:,}"),
                 _val(report.hierarchical, "completion_tokens", lambda v: f"{v:,}")])
    rows.append(["Cost (USD)",
                 _val(report.baseline, "total_cost_usd", lambda v: f"${v:.4f}"),
                 _val(report.flat, "total_cost_usd", lambda v: f"${v:.4f}"),
                 _val(report.hierarchical, "total_cost_usd", lambda v: f"${v:.4f}")])
    rows.append(["Latency (sec)",
                 _val(report.baseline, "latency_seconds", lambda v: f"{v:.1f}s"),
                 _val(report.flat, "latency_seconds", lambda v: f"{v:.1f}s"),
                 _val(report.hierarchical, "latency_seconds", lambda v: f"{v:.1f}s")])
    rows.append(["Steps",
                 _val(report.baseline, "step_count"),
                 _val(report.flat, "step_count"),
                 _val(report.hierarchical, "step_count")])
    rows.append(["Verifications", "-", "-",
                 _val(report.hierarchical, "verification_attempts")])

    col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(4)]
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"\n  {header_line}")
    print(f"  {'-' * sum(col_widths)}")
    for row in rows:
        line = "".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        print(f"  {line}")
    print("=" * 75)

    if report.baseline and report.hierarchical:
        token_ratio = report.hierarchical.total_tokens / max(report.baseline.total_tokens, 1)
        cost_ratio = report.hierarchical.total_cost_usd / max(report.baseline.total_cost_usd, 0.0001)
        print(f"\n  Hierarchical uses {token_ratio:.1f}x tokens and {cost_ratio:.1f}x cost vs Baseline")

    if report.flat and report.hierarchical:
        token_ratio = report.hierarchical.total_tokens / max(report.flat.total_tokens, 1)
        cost_ratio = report.hierarchical.total_cost_usd / max(report.flat.total_cost_usd, 0.0001)
        print(f"  Hierarchical uses {token_ratio:.1f}x tokens and {cost_ratio:.1f}x cost vs Flat")

    print()


def save_comparison(report: ComparisonReport, output_dir: str = "logs") -> str:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"efficiency_comparison_{report.product_asin}_{timestamp}.json"
    filepath = output_path / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"  Comparison saved to: {filepath}")
    return str(filepath)
