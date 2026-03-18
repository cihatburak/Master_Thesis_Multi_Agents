"""
Configuration file for Multi-Agent BI Report System
Centralizes all configurable parameters.
"""

import random

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Default model for report generation (Base model before random allocation)
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
HIERARCHICAL_MAX_STEPS = 15

# =============================================================================
# EVALUATION SETTINGS
# =============================================================================

# Number of parallel workers for evaluation judges (Increased for Multi-Model speed)
EVAL_MAX_WORKERS = 20

# Timeout for each judge evaluation (seconds)
EVAL_TIMEOUT = 180

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
        "name": "Mistral-Large-3",
        "model_id": "mistralai/mistral-large-2512",
        "provider": "openrouter",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "structured_output": False,
    },
]

# Pool for Random Model Allocation (Agentic Experiment)
# Using 2026-specific models as defined in config
MODEL_POOL = [
    "openai/gpt-5.4",
    "google/gemini-3.1-pro-preview",
    "qwen/qwen3.5-122b-a10b",
    "z-ai/glm-5",
    "mistralai/mistral-large-2512"
]

# Manager-specific pool: Qwen excluded because its Thinking Mode is incompatible
# with LangChain's with_structured_output() (tool_choice="required" is rejected).
# Workers can still be Qwen — only Manager requires strict JSON structured output.
MANAGER_POOL = [
    "openai/gpt-5.4",
    "google/gemini-3.1-pro-preview",
    "z-ai/glm-5",
    "mistralai/mistral-large-2512"
]


def select_models_for_run() -> dict[str, str]:
    """
    Randomly assign a model from the appropriate pool to each agent role.
    
    - Workers (Researcher, Analyst, Writer, Critic): drawn from full MODEL_POOL
    - Manager: drawn from MANAGER_POOL (Qwen excluded — Thinking Mode incompatibility)
    
    Called once per product run to ensure diversity of intelligence
    across experiments (random with replacement).
    
    Returns:
        dict mapping role name to model_id, e.g.:
        {"Researcher": "openai/gpt-5.4", "Analyst": "z-ai/glm-5", ...}
    """
    worker_roles = ["Researcher", "Analyst", "Writer", "Critic"]
    result = {role: random.choice(MODEL_POOL) for role in worker_roles}
    result["Manager"] = random.choice(MANAGER_POOL)
    return result

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
    "mistralai/mistral-large-2512": {"input": 0.50, "output": 1.50},
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
