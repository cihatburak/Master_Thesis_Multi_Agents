"""
Evaluation Engine (3-Metric System)
Multi-Agent BI Report System

Implements a simplified 3-metric evaluation:
- Quality (LLM Judge): Coherence + Conciseness + Structure
- Accuracy (Script-Based): Ground truth comparison using regex extraction
- Utility (LLM Judge): Actionability + Criticality combined
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config

# Load environment variables
load_dotenv()

# =============================================================================
# OUTPUT SCHEMAS
# =============================================================================

class JudgeScore(BaseModel):
    """Schema for individual judge's evaluation output."""
    score: float = Field(description="Score from 1.0 to 5.0", ge=1.0, le=5.0)
    reasoning: str = Field(description="Detailed reasoning for the score")


class EvaluationResult(BaseModel):
    """Schema for the complete evaluation result with 3 metrics."""
    quality_score: float
    quality_reasoning: str
    accuracy_score: float
    accuracy_reasoning: str
    utility_score: float
    utility_reasoning: str
    final_score: float


# =============================================================================
# JUDGE PROMPTS (2 LLM Judges)
# =============================================================================

QUALITY_PROMPT = """You are a Senior Editor evaluating Business Intelligence reports.

Your task is to evaluate the overall QUALITY of the report including:
- Coherence: Logical flow, clear structure, smooth transitions
- Conciseness: No redundancy, efficient communication, executive-ready
- Completeness: All sections present, nothing important missing
- Professionalism: Proper formatting, grammar, and tone

EVALUATION CRITERIA for 'Quality' (1-5 scale):
- 5: Exceptional quality - well-structured, concise, professional, comprehensive
- 4: Good quality - minor issues but overall effective communication
- 3: Adequate - some structural or clarity issues, could be improved
- 2: Poor quality - confusing, verbose, or incomplete
- 1: Unacceptable - incoherent, unprofessional, or severely lacking

FOCUS ON:
- Clear section headers and organization (Executive Summary, Findings, Recommendations)
- Logical progression from findings to recommendations
- Economy of words - every sentence adds value
- Professional markdown formatting
- No redundancy or unnecessary verbosity

Evaluate the report and provide your score with detailed reasoning."""


UTILITY_PROMPT = """You are a Business Strategist evaluating BI reports for practical value.

Your task is to evaluate if the report provides USEFUL insights for business decision-making.
This combines two aspects:
1. ACTIONABILITY: Are recommendations specific, measurable, and implementable?
2. CRITICALITY: Does the analysis identify root causes and strategic implications?

EVALUATION CRITERIA for 'Utility' (1-5 scale):
- 5: Exceptional utility - specific actionable recommendations with root cause analysis
- 4: Good utility - mostly actionable with meaningful insights
- 3: Adequate - some useful recommendations but lacking depth or specificity
- 2: Limited utility - vague recommendations or shallow analysis
- 1: No utility - purely descriptive with no actionable guidance

FOCUS ON:
- Specificity of recommendations (WHO should do WHAT by WHEN?)
- Root cause analysis (does it explain WHY problems occur?)
- Strategic implications (connects findings to broader business context)
- Prioritization (which actions are most critical?)
- Implementation feasibility (are steps and resources defined?)

Evaluate the report and provide your score with detailed reasoning."""


# =============================================================================
# SCRIPT-BASED ACCURACY (No LLM - Ground Truth Comparison)
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
    """
    Extract technical specifications from text using regex patterns.
    Returns a dict with extracted specs.
    """
    specs = {}
    
    # GPU patterns
    gpu_patterns = [
        r'RTX\s*(\d{4})\s*(Ti)?',
        r'GTX\s*(\d{4})\s*(Ti)?',
        r'GeForce\s+RTX\s*(\d{4})',
        r'GeForce\s+GTX\s*(\d{4})',
    ]
    for pattern in gpu_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            specs['gpu'] = match.group(0).strip()
            break
    
    # RAM patterns
    ram_patterns = [
        r'(\d+)\s*GB\s*(DDR\d)?(\s*RAM)?',
        r'(\d+)GB\s+DDR\d',
    ]
    for pattern in ram_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            specs['ram'] = match.group(0).strip()
            break
    
    # CPU patterns
    cpu_patterns = [
        r'Ryzen\s*\d+[\s\-]*\d*\w*',
        r'Core\s*i\d[\s\-]*\d*\w*',
        r'Intel\s+Core\s+i\d',
        r'AMD\s+Ryzen\s+\d+',
    ]
    for pattern in cpu_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            specs['cpu'] = match.group(0).strip()
            break
    
    # Storage patterns
    storage_patterns = [
        r'(\d+)\s*(TB|GB)\s*(SSD|NVMe|HDD)',
        r'(\d+)(TB|GB)\s+SSD',
    ]
    for pattern in storage_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            specs['storage'] = match.group(0).strip()
            break
    
    # Display patterns
    display_patterns = [
        r'(\d+\.?\d*)["\s]*(inch|")',
        r'(\d+)\s*Hz',
        r'FHD|QHD|4K|1080p|1440p',
    ]
    for pattern in display_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if 'display' not in specs:
                specs['display'] = match.group(0).strip()
            break
    
    # Price patterns
    price_patterns = [
        r'\$[\d,]+\.?\d*',
        r'[\d,]+\.?\d*\s*(?:USD|dollars)',
    ]
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
    """
    Semantic comparison per spec category.
    Extracts the core value (numeric amount, model number) and compares,
    ignoring format differences like 'DDR5' suffix or 'GeForce' prefix.
    """
    report_lower = report_val.lower()
    gt_lower = gt_val.lower()
    
    if key == 'ram':
        # Extract just the GB number — ignore DDR generation, "RAM" suffix
        r_num = re.search(r'(\d+)\s*gb', report_lower)
        g_num = re.search(r'(\d+)\s*gb', gt_lower)
        if r_num and g_num:
            return r_num.group(1) == g_num.group(1)
    
    elif key == 'gpu':
        # Extract model number (e.g., "RTX 4060", "GTX 1650 Ti")
        r_model = re.search(r'(rtx|gtx)\s*(\d{4})\s*(ti)?', report_lower)
        g_model = re.search(r'(rtx|gtx)\s*(\d{4})\s*(ti)?', gt_lower)
        if r_model and g_model:
            return (r_model.group(1) == g_model.group(1) and 
                    r_model.group(2) == g_model.group(2) and
                    r_model.group(3) == g_model.group(3))
    
    elif key == 'cpu':
        # Extract brand + series (e.g., "Ryzen 7", "Core i7")
        r_cpu = re.search(r'(ryzen\s*\d|core\s*i\d)', report_lower)
        g_cpu = re.search(r'(ryzen\s*\d|core\s*i\d)', gt_lower)
        if r_cpu and g_cpu:
            return re.sub(r'\s+', '', r_cpu.group(1)) == re.sub(r'\s+', '', g_cpu.group(1))
    
    elif key == 'storage':
        # Extract size + type (e.g., "1TB SSD") — treat NVMe as SSD
        r_size = re.search(r'(\d+)\s*(tb|gb)', report_lower)
        g_size = re.search(r'(\d+)\s*(tb|gb)', gt_lower)
        if r_size and g_size:
            return (r_size.group(1) == g_size.group(1) and 
                    r_size.group(2) == g_size.group(2))
    
    # Fallback: normalized substring match
    return normalize_spec(report_val) in normalize_spec(gt_val) or \
           normalize_spec(gt_val) in normalize_spec(report_val)


def calculate_script_accuracy(
    report_text: str,
    ground_truth: dict,
    verbose: bool = True
) -> tuple[float, str]:
    """
    Calculate accuracy score by comparing extracted specs with ground truth.
    Returns (score, reasoning).
    """
    # Extract specs from report
    report_specs = extract_specs_from_text(report_text)
    
    # Extract specs from ground truth (title + description)
    gt_text = f"{ground_truth.get('title', '')} {ground_truth.get('description', '')}"
    gt_specs = extract_specs_from_text(gt_text)
    
    if verbose:
        print(f"\n📊 Script-Based Accuracy Check")
        print(f"   Report specs: {report_specs}")
        print(f"   Ground truth specs: {gt_specs}")
    
    # Compare specs
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
                # Report claims something not in ground truth - potential hallucination
                mismatches.append(f"{key}: '{report_specs[key]}' (not in ground truth)")
    
    # Calculate score
    if total_checks == 0:
        # No specs to verify - give neutral score
        score = 3.0
        reasoning = "No verifiable technical specifications found in report."
    else:
        accuracy_ratio = matches / total_checks
        score = 1.0 + (accuracy_ratio * 4.0)  # Scale to 1-5
        
        if mismatches:
            reasoning = f"Matched {matches}/{total_checks} specs. Mismatches: {'; '.join(mismatches)}"
        else:
            reasoning = f"All {matches} verifiable specs matched ground truth perfectly."
    
    return round(score, 2), reasoning


# =============================================================================
# LLM JUDGE EVALUATION
# =============================================================================

def _run_judge(
    judge_name: str,
    system_prompt: str,
    report_text: str,
    product_metadata: str
) -> JudgeScore:
    """Run a single LLM judge evaluation."""
    llm = ChatOpenAI(model=config.MODEL_NAME, temperature=config.TEMPERATURE)
    judge_llm = llm.with_structured_output(JudgeScore)
    
    user_content = f"""
=== PRODUCT METADATA (For Context) ===
{product_metadata}

=== REPORT TO EVALUATE ===
{report_text}

Please evaluate this report according to your criteria and provide a score (1-5) with reasoning.
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]
    
    result = judge_llm.invoke(messages)
    return result


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate_report(
    report_text: str,
    product_metadata: str,
    asin: Optional[str] = None,
    verbose: bool = True
) -> EvaluationResult:
    """
    Evaluate a BI report using 3 metrics:
    - Quality (LLM Judge)
    - Accuracy (Script-Based)
    - Utility (LLM Judge)
    
    Args:
        report_text: The generated BI report to evaluate
        product_metadata: Product specifications string (for LLM context)
        asin: Product ASIN for ground truth lookup (for script accuracy)
        verbose: Whether to print detailed results
    
    Returns:
        EvaluationResult with 3 metric scores and final aggregated score
    """
    
    if verbose:
        print("\n" + "=" * 60)
        print("3-METRIC EVALUATION SYSTEM")
        print("=" * 60)
    
    # ===================
    # 1. QUALITY (LLM)
    # ===================
    if verbose:
        print("\n🔍 Running Quality Judge (LLM)...")
    
    quality_result = _run_judge(
        "Quality Judge",
        QUALITY_PROMPT,
        report_text,
        product_metadata
    )
    
    if verbose:
        print(f"   Score: {quality_result.score}/5.0")
        print(f"   Reasoning: {quality_result.reasoning[:150]}...")
    
    # ===================
    # 2. ACCURACY (Script)
    # ===================
    if verbose:
        print("\n🔍 Running Accuracy Check (Script-Based)...")
    
    # Try to load ground truth from JSON
    ground_truth = None
    if asin:
        ground_truth = load_ground_truth(asin)
    
    if ground_truth:
        accuracy_score, accuracy_reasoning = calculate_script_accuracy(
            report_text, ground_truth, verbose
        )
    else:
        # Fallback: extract from provided metadata string
        gt_dict = {"title": product_metadata, "description": ""}
        accuracy_score, accuracy_reasoning = calculate_script_accuracy(
            report_text, gt_dict, verbose
        )
    
    if verbose:
        print(f"   Score: {accuracy_score}/5.0")
        print(f"   Reasoning: {accuracy_reasoning}")
    
    # ===================
    # 3. UTILITY (LLM)
    # ===================
    if verbose:
        print("\n🔍 Running Utility Judge (LLM)...")
    
    utility_result = _run_judge(
        "Utility Judge",
        UTILITY_PROMPT,
        report_text,
        product_metadata
    )
    
    if verbose:
        print(f"   Score: {utility_result.score}/5.0")
        print(f"   Reasoning: {utility_result.reasoning[:150]}...")
    
    # ===================
    # AGGREGATE
    # ===================
    final_score = round((quality_result.score + accuracy_score + utility_result.score) / 3, 2)
    
    evaluation = EvaluationResult(
        quality_score=quality_result.score,
        quality_reasoning=quality_result.reasoning,
        accuracy_score=accuracy_score,
        accuracy_reasoning=accuracy_reasoning,
        utility_score=utility_result.score,
        utility_reasoning=utility_result.reasoning,
        final_score=final_score
    )
    
    if verbose:
        print("\n" + "-" * 60)
        print("AGGREGATED RESULTS (3 Metrics)")
        print("-" * 60)
        print(f"   Quality:  {quality_result.score}/5.0 (LLM)")
        print(f"   Accuracy: {accuracy_score}/5.0 (Script)")
        print(f"   Utility:  {utility_result.score}/5.0 (LLM)")
        print(f"\n   ⭐ FINAL SCORE: {final_score}/5.0")
        print("=" * 60)
    
    return evaluation


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Test with dummy data
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
    print("3-METRIC EVALUATION ENGINE TEST")
    print("=" * 60)
    print("\n📄 Testing with dummy report and metadata...")
    
    result = evaluate_report(DUMMY_REPORT, DUMMY_METADATA, asin="B0CXVGSY2H")
    
    print("\n" + "=" * 60)
    print("FULL EVALUATION OBJECT")
    print("=" * 60)
    print(result.model_dump_json(indent=2))
