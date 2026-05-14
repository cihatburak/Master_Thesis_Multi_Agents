import random


MODEL_NAME = "openai/gpt-5.4"
TEMPERATURE = 0

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV_NAME = "OPENROUTER_API_KEY"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

LOG_DIR = "logs"
CHROMA_DB_DIR = "./chroma_db"

RECURSION_LIMIT = 100
FLAT_MAX_STEPS = 9
HIERARCHICAL_MAX_STEPS = 15

EVAL_MAX_WORKERS = 20
EVAL_TIMEOUT = 180


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

MODEL_POOL = [
    "openai/gpt-5.4",
    "google/gemini-3.1-pro-preview",
    "qwen/qwen3.5-122b-a10b",
    "z-ai/glm-5",
    "mistralai/mistral-large-2512",
]

# Qwen excluded from Manager pool: Thinking Mode is incompatible with
# LangChain's with_structured_output (tool_choice="required" rejected).
MANAGER_POOL = [
    "openai/gpt-5.4",
    "google/gemini-3.1-pro-preview",
    "z-ai/glm-5",
    "mistralai/mistral-large-2512",
]


def select_models_for_run() -> dict[str, str]:
    """Randomly assign a model to each agent role, drawing once per product run."""
    worker_roles = ["Researcher", "Analyst", "Writer", "Critic"]
    result = {role: random.choice(MODEL_POOL) for role in worker_roles}
    result["Manager"] = random.choice(MANAGER_POOL)
    return result


# OpenRouter pricing per 1M tokens, snapshot taken March 2026.
MODEL_PRICING = {
    "openai/gpt-5.4": {"input": 2.50, "output": 15.00},
    "google/gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "qwen/qwen3.5-122b-a10b": {"input": 0.26, "output": 2.08},
    "z-ai/glm-5": {"input": 0.80, "output": 2.56},
    "mistralai/mistral-large-2512": {"input": 0.50, "output": 1.50},
}


def calculate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return USD cost for a call given token counts. Falls back to GPT-class pricing for unknown models."""
    pricing = MODEL_PRICING.get(model_id, {"input": 2.50, "output": 10.00})
    cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000
    return round(cost, 6)
