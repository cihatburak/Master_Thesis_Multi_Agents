"""
Logger Utility for Multi-Agent BI Report System
Captures and saves agent conversation history for debugging and analysis.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage

import config

# Logs directory
LOGS_DIR = Path(__file__).parent / config.LOG_DIR
LOGS_DIR.mkdir(exist_ok=True)


def message_to_dict(msg) -> dict:
    """Convert a LangChain message to a serializable dictionary."""
    msg_type = type(msg).__name__
    
    content = ""
    if hasattr(msg, 'content'):
        content = msg.content
    
    result = {
        "type": msg_type,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add additional fields based on message type
    if hasattr(msg, 'name'):
        result["name"] = msg.name
    if hasattr(msg, 'tool_call_id'):
        result["tool_call_id"] = msg.tool_call_id
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        result["tool_calls"] = [
            {"name": tc.get("name", ""), "args": tc.get("args", {})}
            for tc in msg.tool_calls
        ]
    
    return result


def save_logs(
    session_id: str,
    architecture: str,
    messages: List[Any],
    metadata: Optional[dict] = None
) -> str:
    """
    Save the entire conversation history to a JSON file.
    
    Args:
        session_id: Unique identifier for the session (e.g., ASIN or timestamp)
        architecture: "flat" or "hierarchical"
        messages: List of LangChain message objects from State['messages']
        metadata: Optional additional metadata to include
    
    Returns:
        Path to the saved log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"log_{architecture}_{session_id}_{timestamp}.json"
    filepath = LOGS_DIR / filename
    
    # Convert messages to serializable format
    serialized_messages = []
    for msg in messages:
        try:
            serialized_messages.append(message_to_dict(msg))
        except Exception as e:
            serialized_messages.append({
                "type": "Unknown",
                "content": str(msg),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    log_data = {
        "session_id": session_id,
        "architecture": architecture,
        "timestamp": datetime.now().isoformat(),
        "message_count": len(messages),
        "metadata": metadata or {},
        "messages": serialized_messages
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    return str(filepath)


def format_messages_for_display(messages: List[Any]) -> List[dict]:
    """
    Format messages for UI display in Streamlit.
    
    Returns a list of dicts with 'role', 'content', and 'type' keys.
    """
    formatted = []
    
    for msg in messages:
        msg_type = type(msg).__name__
        content = msg.content if hasattr(msg, 'content') else str(msg)
        
        # Determine role for display
        if isinstance(msg, HumanMessage):
            role = "🧑 Human"
        elif isinstance(msg, AIMessage):
            role = "🤖 AI Agent"
        elif isinstance(msg, ToolMessage):
            role = "🔧 Tool"
        elif isinstance(msg, SystemMessage):
            role = "⚙️ System"
        else:
            role = f"📋 {msg_type}"
        
        # Truncate very long content for display
        display_content = content
        if len(content) > 1000:
            display_content = content[:1000] + "... [truncated]"
        
        formatted.append({
            "role": role,
            "type": msg_type,
            "content": display_content,
            "full_content": content
        })
    
    return formatted


def get_recent_logs(limit: int = 10) -> List[dict]:
    """
    Get the most recent log files.
    
    Returns list of log metadata (filename, timestamp, architecture, etc.)
    """
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
                    "message_count": data.get("message_count", 0)
                })
        except Exception:
            pass
    
    return logs
