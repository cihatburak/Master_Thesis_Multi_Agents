"""
Configuration file for Multi-Agent BI Report System
Centralizes all configurable parameters.
"""

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Default model for report generation via OpenRouter
MODEL_NAME = "openai/gpt-5.4"

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

# Flat graph: 5 Manager decisions + 4 worker steps = 9 total steps
FLAT_MAX_STEPS = 9

# Hierarchical graph: 4 workers + Manager decisions + up to 2 loop-backs
HIERARCHICAL_MAX_STEPS = 10

# =============================================================================
# EVALUATION SETTINGS
# =============================================================================

# Number of parallel workers for evaluation judges (Increased for Multi-Model speed)
EVAL_MAX_WORKERS = 20

# Timeout for each judge evaluation (seconds)
EVAL_TIMEOUT = 90

# =============================================================================
# EVALUATION v2 SETTINGS (Multi-Model)
# =============================================================================

# OpenRouter API key environment variable name
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

# Judge models — each report is evaluated by ALL of these
# Uses OpenRouter for access to multiple models via single API
JUDGE_MODELS = [
    {
        "name": "GPT-5.4",
        "model_id": "openai/gpt-5.4",
        "provider": "openrouter",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "structured_output": True,
    },
    {
        "name": "Gemini-3.1-Pro",
        "model_id": "google/gemini-3.1-pro-preview",
        "provider": "openrouter",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "structured_output": False,
    },
    {
        "name": "Qwen-3.5-122B",
        "model_id": "qwen/qwen3.5-122b-a10b",
        "provider": "openrouter",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "structured_output": False,
    },
    {
        "name": "GLM-5",
        "model_id": "z-ai/glm-5",
        "provider": "openrouter",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "structured_output": False,
    },
    {
        "name": "Mistral-Small",
        "model_id": "mistralai/mistral-small-creative",
        "provider": "openrouter",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "structured_output": False,
    },
]

# =============================================================================
# MODEL PRICING (OpenRouter, per 1M tokens, as of March 2026)
# =============================================================================

MODEL_PRICING = {
    # Generation model
    "openai/gpt-5.4": {"input": 2.50, "output": 15.00},
    # Judge models
    "google/gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "qwen/qwen3.5-122b-a10b": {"input": 0.26, "output": 2.08},
    "z-ai/glm-5": {"input": 0.80, "output": 2.56},
    "mistralai/mistral-small-creative": {"input": 0.10, "output": 0.30},
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
