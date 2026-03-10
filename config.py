"""
Configuration file for Multi-Agent BI Report System
Centralizes all configurable parameters.
"""

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Default model for report generation via OpenRouter
MODEL_NAME = "openai/gpt-4o"

# Temperature setting for LLM (0 = deterministic)
TEMPERATURE = 0

# =============================================================================
# OPENROUTER CONFIGURATION (ALL API calls go through OpenRouter)
# =============================================================================

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV_NAME = "OPENROUTER_API_KEY"

# =============================================================================
# PATHS
# =============================================================================

# Directory for storing conversation logs
LOG_DIR = "logs"

# ChromaDB persistence directory
CHROMA_DB_DIR = "./chroma_db"

# =============================================================================
# GRAPH SETTINGS
# =============================================================================

# Maximum recursion limit for LangGraph
RECURSION_LIMIT = 100

# Flat graph: exactly 4 steps (R → A → W → C), deterministic
FLAT_MAX_STEPS = 4

# Hierarchical graph: 4 workers + Manager decisions + up to 2 loop-backs
HIERARCHICAL_MAX_STEPS = 10

# =============================================================================
# EVALUATION SETTINGS
# =============================================================================

# Number of parallel workers for evaluation judges
EVAL_MAX_WORKERS = 5

# Timeout for each judge evaluation (seconds)
EVAL_TIMEOUT = 90

# =============================================================================
# EVALUATION v2 SETTINGS (Multi-Model)
# =============================================================================

# OpenRouter API key environment variable name
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

# =============================================================================
# MODEL PRICING (OpenRouter, per 1M tokens, as of March 2026)
# =============================================================================

MODEL_PRICING = {
    # Generation model
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    # Judge models
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.52, "output": 0.75},
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "mistralai/mistral-large-2411": {"input": 2.00, "output": 6.00},
}


def calculate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate cost in USD based on token counts and model pricing.

    Args:
        model_id: OpenRouter model ID (e.g. 'openai/gpt-4o')
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    pricing = MODEL_PRICING.get(model_id, {"input": 2.50, "output": 10.00})
    cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000
    return round(cost, 6)
