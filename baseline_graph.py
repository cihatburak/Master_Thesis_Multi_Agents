"""
Baseline Architecture (Control Group - Zero-Shot)
Multi-Agent BI Report System

This implements a single-agent zero-shot baseline where GPT-4 directly generates
a BI report from raw product data without any agent coordination.

Purpose: To demonstrate the added value of multi-agent systems (Flat and Hierarchical)
compared to a simple single-model approach.
"""

from typing import Optional

import os
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tools import get_product_specs, search_reviews
from logger import save_logs
import config

# Load environment variables
load_dotenv()

# =============================================================================
# LLM INITIALIZATION (via OpenRouter)
# =============================================================================

llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=config.TEMPERATURE,
    openai_api_key=os.getenv(config.OPENROUTER_API_KEY_ENV_NAME),
    openai_api_base=config.OPENROUTER_API_BASE,
)


# =============================================================================
# BASELINE PROMPT (Minimal - True Zero-Shot)
# =============================================================================

BASELINE_PROMPT = """You are a Business Intelligence Report Writer.
Based on the provided product data and customer reviews, write a comprehensive BI report.

REPORT STRUCTURE (use these exact sections):
1. **Executive Summary** — 2-3 sentence overview of key findings
2. **Product Overview & Technical Specifications** — Include all key specs:
   CPU, GPU, RAM, Storage, Display (size, resolution, refresh rate), Price.
   Present as a clear specifications table or list.
3. **Customer Feedback Analysis** — Categorize findings by theme with evidence.
   Include the overall rating distribution (e.g., "60% positive, 25% negative").
4. **Key Issues & Root Causes** — Top problems with underlying causes.
   For each issue: WHAT is it, WHY does it occur, HOW MANY mention it.
5. **Actionable Recommendations** — Each recommendation MUST specify:
   - WHO should act (e.g., "Product team", "Marketing", "Customer support")
   - WHAT specifically to do (concrete action, not vague suggestion)
   - WHEN / PRIORITY (immediate, short-term, long-term)
   - EXPECTED IMPACT (what improvement this would bring)
6. **Conclusion** — Strategic summary for decision-makers

CRITICAL RULES:
- ALWAYS include the product's technical specifications in Section 2
- ALWAYS cite specific customer quotes as evidence
- Recommendations must be SPECIFIC and ACTIONABLE, not generic
- Format the report in Markdown for readability

PRODUCT DATA:
{product_specs}

CUSTOMER REVIEWS:
{reviews}"""


# =============================================================================
# BASELINE EXECUTION
# =============================================================================

def run_baseline(query: str, asin: str, session_id: str = "session") -> tuple[str, dict]:
    """
    Run zero-shot baseline: Single API call to generate BI report.
    
    Args:
        query: The analysis query (for context)
        asin: Product ASIN to analyze
        session_id: Session identifier for logging
    
    Returns:
        tuple: (report_text, metrics_dict)
    """
    # Step 1: Gather raw data from ChromaDB (using existing tools)
    product_specs = get_product_specs.invoke({"asin": asin})
    
    # Get both positive and negative reviews
    positive_reviews = search_reviews.invoke({
        "query": f"product quality performance {asin}",
        "asin": asin,
        "sentiment_type": "positive"
    })
    
    negative_reviews = search_reviews.invoke({
        "query": f"problems issues complaints {asin}",
        "asin": asin,
        "sentiment_type": "negative"
    })
    
    reviews_combined = f"POSITIVE REVIEWS:\n{positive_reviews}\n\nNEGATIVE REVIEWS:\n{negative_reviews}"
    
    # Step 2: Format the prompt
    formatted_prompt = BASELINE_PROMPT.format(
        product_specs=product_specs,
        reviews=reviews_combined
    )
    
    # Step 3: Single API call with token tracking
    messages = [
        SystemMessage(content="You are a professional BI report writer."),
        HumanMessage(content=formatted_prompt)
    ]
    
    with get_openai_callback() as cb:
        response = llm.invoke(messages)
        
        metrics = {
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_cost": cb.total_cost,
            "step_count": 1,  # Always 1 for baseline (single API call)
            "architecture": "baseline"
        }
    
    report_text = response.content
    
    # Save logs
    save_logs(session_id, "baseline", [messages[0], messages[1], response], {
        "query": query,
        "asin": asin,
        "metrics": metrics
    })
    
    return report_text, metrics


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BASELINE ARCHITECTURE (Zero-Shot GPT-4) - BI Report Generation")
    print("=" * 70)
    
    query = "Analyze the MSI Katana (B0CXVGSY2H) focusing on customer feedback."
    asin = "B0CXVGSY2H"
    
    print(f"\n📋 Query: {query}")
    print(f"📦 ASIN: {asin}\n")
    print("-" * 70)
    print("Running Baseline (Single API call, no agents)...\n")
    
    report, metrics = run_baseline(query, asin)
    
    print("\n" + "=" * 70)
    print("FINAL REPORT:")
    print("=" * 70)
    print(report)
    
    print("\n" + "=" * 70)
    print("EFFICIENCY METRICS:")
    print("=" * 70)
    print(f"Total Tokens: {metrics['total_tokens']}")
    print(f"  - Prompt Tokens: {metrics['prompt_tokens']}")
    print(f"  - Completion Tokens: {metrics['completion_tokens']}")
    print(f"Total Cost: ${metrics['total_cost']:.4f}")
    print(f"Step Count: {metrics['step_count']}")
