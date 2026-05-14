import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage

import config


LOGS_DIR = Path(__file__).parent / config.LOG_DIR
LOGS_DIR.mkdir(exist_ok=True)


def message_to_dict(msg) -> dict:
    """Convert a LangChain message into a JSON-serializable dict."""
    msg_type = type(msg).__name__
    content = msg.content if hasattr(msg, "content") else ""

    result = {
        "type": msg_type,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    }

    if hasattr(msg, "name"):
        result["name"] = msg.name
    if hasattr(msg, "tool_call_id"):
        result["tool_call_id"] = msg.tool_call_id
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        result["tool_calls"] = [
            {"name": tc.get("name", ""), "args": tc.get("args", {})}
            for tc in msg.tool_calls
        ]

    return result


def save_logs(
    session_id: str,
    architecture: str,
    messages: List[Any],
    metadata: Optional[dict] = None,
) -> str:
    """Persist the full message trace of a run to logs/ as JSON. Returns the file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"log_{architecture}_{session_id}_{timestamp}.json"
    filepath = LOGS_DIR / filename

    serialized_messages = []
    for msg in messages:
        try:
            serialized_messages.append(message_to_dict(msg))
        except Exception as e:
            serialized_messages.append({
                "type": "Unknown",
                "content": str(msg),
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })

    log_data = {
        "session_id": session_id,
        "architecture": architecture,
        "timestamp": datetime.now().isoformat(),
        "message_count": len(messages),
        "metadata": metadata or {},
        "messages": serialized_messages,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    return str(filepath)


def format_messages_for_display(messages: List[Any]) -> List[dict]:
    """Format messages for the Streamlit UI. Truncates content over 1000 characters."""
    formatted = []

    for msg in messages:
        msg_type = type(msg).__name__
        content = msg.content if hasattr(msg, "content") else str(msg)

        if isinstance(msg, HumanMessage):
            role = "Human"
        elif isinstance(msg, AIMessage):
            role = "AI Agent"
        elif isinstance(msg, ToolMessage):
            role = "Tool"
        elif isinstance(msg, SystemMessage):
            role = "System"
        else:
            role = msg_type

        display_content = content
        if len(content) > 1000:
            display_content = content[:1000] + "... [truncated]"

        formatted.append({
            "role": role,
            "type": msg_type,
            "content": display_content,
            "full_content": content,
        })

    return formatted


def get_recent_logs(limit: int = 10) -> List[dict]:
    """Return metadata for the most recent log files in logs/."""
    logs = []

    for filepath in sorted(LOGS_DIR.glob("log_*.json"), reverse=True)[:limit]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                logs.append({
                    "filename": filepath.name,
                    "session_id": data.get("session_id", ""),
                    "architecture": data.get("architecture", ""),
                    "timestamp": data.get("timestamp", ""),
                    "message_count": data.get("message_count", 0),
                })
        except Exception:
            pass

    return logs
