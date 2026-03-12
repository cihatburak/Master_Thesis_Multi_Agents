"""
Flat Agent Architecture (Control Group)
Multi-Agent BI Report System using LangGraph

Implements a flat/peer architecture where all 5 agents (4 Workers + 1 Manager)
operate with SYMMETRIC influence. The Manager participates as a peer:
it reads the Blackboard and provides commentary, but has ZERO authority
to reject outputs or force loop-backs.

This is the CONTROL condition — the only difference from the Hierarchical
(treatment) group is the ABSENCE of the Manager's loop-back authority.

5 Agents: Manager (peer) + Researcher + Analyst + Writer + Critic
Coordination: Shared Blackboard (symmetric influence, zero hierarchy)
Flow: Manager → R → Manager → A → Manager → W → Manager → C → Manager → END
"""

import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

import prompts
from tools import get_product_specs, search_reviews, verify_claim
from logger import save_logs
import config

# Load environment variables
load_dotenv()


# =============================================================================
# STATE DEFINITION (The Shared Blackboard)
# =============================================================================

class AgentState(TypedDict):
    """
    State shared across all agents in the flat architecture.
    'messages' holds the strict LangChain message history.
    'blackboard' holds the continuously appended collaborative text.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    blackboard: str  # The shared workspace where agents append their findings
    next: str        # The next agent to route to (always forward in flat)
    instructions: str  # Manager's commentary (non-authoritative)
    step_count: int
    loop_count: int  # Always 0 in flat (no loop-backs allowed)


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
# AGENT INITIALIZATION
# =============================================================================

researcher_agent = create_react_agent(
    llm,
    tools=[get_product_specs, search_reviews],
    prompt=prompts.RESEARCHER_PROMPT
)

analyst_agent = create_react_agent(
    llm,
    tools=[],  # Works with data already on the blackboard
    prompt=prompts.ANALYST_PROMPT
)

writer_agent = create_react_agent(
    llm,
    tools=[],  # Works with analysis already on the blackboard
    prompt=prompts.WRITER_PROMPT
)

critic_agent = create_react_agent(
    llm,
    tools=[verify_claim],
    prompt=prompts.CRITIC_PROMPT
)


# =============================================================================
# AGENT NODE WRAPPERS (Blackboard Interaction Logic)
# =============================================================================
# Each node reads the current blackboard, executes its agent, and appends the result.

def _get_last_ai_message(messages: list[BaseMessage]) -> str:
    """Extract the last string response given by the AI agent."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip():
            return message.content
    return ""


def researcher_node(state: AgentState) -> dict:
    """Execute the Researcher agent."""
    input_message = f"User Request: {state['messages'][0].content}\n\nCurrent Blackboard:\n{state.get('blackboard', '')}"
    
    # Manager may have provided non-authoritative commentary
    if state.get("instructions") and state.get("instructions") != "Proceeding forward.":
        input_message += f"\n\n[MANAGER COMMENTARY (for reference only)]:\n{state['instructions']}"

    result = researcher_agent.invoke({"messages": [HumanMessage(content=input_message)]})
    new_insight = _get_last_ai_message(result["messages"])
    
    # Append findings to the shared blackboard
    updated_blackboard = state.get("blackboard", "") + f"\n\n--- RESEARCHER FINDINGS ---\n{new_insight}"
    
    return {
        "messages": result["messages"], 
        "blackboard": updated_blackboard,
        "step_count": state.get("step_count", 0) + 1
    }


def analyst_node(state: AgentState) -> dict:
    """Execute the Analyst agent."""
    input_message = f"Here is the current Shared Blackboard. Analyze the research:\n{state.get('blackboard', '')}"
    
    if state.get("instructions") and state.get("instructions") != "Proceeding forward.":
        input_message += f"\n\n[MANAGER COMMENTARY (for reference only)]:\n{state['instructions']}"

    result = analyst_agent.invoke({"messages": [HumanMessage(content=input_message)]})
    new_insight = _get_last_ai_message(result["messages"])
    
    updated_blackboard = state.get("blackboard", "") + f"\n\n--- ANALYST INSIGHTS ---\n{new_insight}"
    
    return {
        "messages": result["messages"], 
        "blackboard": updated_blackboard,
        "step_count": state.get("step_count", 0) + 1
    }


def writer_node(state: AgentState) -> dict:
    """Execute the Writer agent."""
    input_message = f"Here is the current Shared Blackboard. Write the BI Report based on this:\n{state.get('blackboard', '')}"
    
    if state.get("instructions") and state.get("instructions") != "Proceeding forward.":
        input_message += f"\n\n[MANAGER COMMENTARY (for reference only)]:\n{state['instructions']}"

    result = writer_agent.invoke({"messages": [HumanMessage(content=input_message)]})
    new_insight = _get_last_ai_message(result["messages"])
    
    updated_blackboard = state.get("blackboard", "") + f"\n\n--- DRAFT REPORT (WRITER) ---\n{new_insight}"
    
    return {
        "messages": result["messages"], 
        "blackboard": updated_blackboard,
        "step_count": state.get("step_count", 0) + 1
    }


def critic_node(state: AgentState) -> dict:
    """Execute the Critic agent."""
    input_message = f"Here is the current Shared Blackboard. Review the Draft Report and verify findings:\n{state.get('blackboard', '')}"
    
    if state.get("instructions") and state.get("instructions") != "Proceeding forward.":
        input_message += f"\n\n[MANAGER COMMENTARY (for reference only)]:\n{state['instructions']}"

    result = critic_agent.invoke({"messages": [HumanMessage(content=input_message)]})
    new_insight = _get_last_ai_message(result["messages"])
    
    updated_blackboard = state.get("blackboard", "") + f"\n\n--- CRITIC REVIEW ---\n{new_insight}"
    
    return {
        "messages": result["messages"], 
        "blackboard": updated_blackboard,
        "step_count": state.get("step_count", 0) + 1
    }


# =============================================================================
# MANAGER NODE (FLAT — Peer Participant, No Authority)
# =============================================================================

class ManagerDecision(BaseModel):
    """Schema for the Manager's routing decision (reused from hierarchical)."""
    next_worker: Literal["Researcher", "Analyst", "Writer", "Critic", "FINISH"]
    instructions: str = Field(description="Commentary or suggestions for the next worker")
    reasoning: str = Field(description="Brief reasoning for this observation")


# Fixed sequence that the Manager must follow (no deviation allowed)
FLAT_WORKER_SEQUENCE = ["Researcher", "Analyst", "Writer", "Critic", "FINISH"]


def flat_manager_node(state: AgentState) -> dict:
    """
    Manager node (Flat / Symmetric) — reviews worker output on the Blackboard
    and provides commentary, but ALWAYS proceeds to the next worker in sequence.
    
    This is the CONTROL version: the Manager has the same observational
    capabilities as in the Hierarchical architecture, but ZERO loop-back
    authority. The routing is deterministically forced forward.
    """
    step_count = state.get("step_count", 0) + 1
    blackboard = state.get("blackboard", "")

    # Determine which workers have already contributed via Blackboard tags
    workers_done = []
    if "--- RESEARCHER FINDINGS ---" in blackboard:
        workers_done.append("Researcher")
    if "--- ANALYST INSIGHTS ---" in blackboard:
        workers_done.append("Analyst")
    if "--- DRAFT REPORT (WRITER) ---" in blackboard:
        workers_done.append("Writer")
    if "--- CRITIC REVIEW ---" in blackboard:
        workers_done.append("Critic")

    # Determine the next worker in the fixed sequence
    next_worker = "FINISH"
    for worker in FLAT_WORKER_SEQUENCE:
        if worker not in workers_done and worker != "FINISH":
            next_worker = worker
            break
    else:
        next_worker = "FINISH"

    # Manager still reads the Blackboard and provides commentary (symmetric influence)
    # This ensures the Manager LLM call happens, matching token usage patterns
    manager_llm = llm.with_structured_output(ManagerDecision)

    formatted_prompt = prompts.FLAT_MANAGER_PROMPT.format(
        step_count=step_count,
    )

    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(
            content=f"Step {step_count}. Review the latest Shared Blackboard and provide "
                    f"your observations. You MUST proceed to: {next_worker}.\n\n"
                    f"CURRENT BLACKBOARD:\n{blackboard}"
        ),
    ]

    result = manager_llm.invoke(messages)

    # Append Manager's commentary to the Blackboard (symmetric — all agents can read it)
    commentary = result.reasoning if result.reasoning else "No additional observations."
    updated_blackboard = blackboard + f"\n\n--- MANAGER COMMENT ---\n{commentary}"

    # FORCE the next worker to be the deterministic one (override any LLM suggestion)
    return {
        "next": next_worker,
        "instructions": result.instructions if result.instructions else "Proceeding forward.",
        "blackboard": updated_blackboard,
        "step_count": step_count,
        "loop_count": 0,  # Always 0 in flat — no loop-backs
    }


# =============================================================================
# ROUTING FUNCTION
# =============================================================================

def route_from_manager(state: AgentState) -> str:
    """Route from Manager to the chosen worker or END."""
    next_step = state.get("next", "")
    if next_step == "FINISH":
        return "end"
    return next_step


# =============================================================================
# GRAPH CONSTRUCTION — MANAGER-INTERLEAVED PIPELINE
# =============================================================================

def build_flat_graph() -> StateGraph:
    """
    Build the flat agent graph with Manager-interleaved sequencing.
    
    Same 5 agents as Hierarchical (4 Workers + Manager), but the Manager
    has ZERO loop-back authority. The pipeline is fixed:
    Manager → R → Manager → A → Manager → W → Manager → C → Manager → END
    
    Treatment variable: Manager's loop-back authority is ABSENT here.
    """
    workflow = StateGraph(AgentState)

    # Add nodes — same workers as Hierarchical + Manager (peer role)
    workflow.add_node("Manager", flat_manager_node)
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Analyst", analyst_node)
    workflow.add_node("Writer", writer_node)
    workflow.add_node("Critic", critic_node)

    # Entry point: Manager observes and routes to first worker
    workflow.set_entry_point("Manager")

    # Manager routes to workers or END (always forward, never loops back)
    workflow.add_conditional_edges(
        "Manager",
        route_from_manager,
        {
            "Researcher": "Researcher",
            "Analyst": "Analyst",
            "Writer": "Writer",
            "Critic": "Critic",
            "end": END,
        },
    )

    # All workers report back to Manager after appending to the Blackboard
    workflow.add_edge("Researcher", "Manager")
    workflow.add_edge("Analyst", "Manager")
    workflow.add_edge("Writer", "Manager")
    workflow.add_edge("Critic", "Manager")

    return workflow.compile()


# =============================================================================
# EXECUTION
# =============================================================================

def run_flat_graph(query: str, session_id: str = "session") -> tuple[str, list, dict]:
    """Run the flat graph and return the final report, messages, and metrics."""
    graph = build_flat_graph()

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "blackboard": f"Initial Request: {query}",
        "next": "",
        "instructions": "",
        "step_count": 0,
        "loop_count": 0,
    }

    # Run with token tracking
    with get_openai_callback() as cb:
        final_state = graph.invoke(
            initial_state,
            config={"recursion_limit": config.RECURSION_LIMIT},
        )

        metrics = {
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_cost": cb.total_cost,
            "step_count": final_state.get("step_count", 0),
            "loop_count": final_state.get("loop_count", 0),
            "architecture": "flat",
        }

    # Save logs (Now includes Blackboard)
    save_logs(
        session_id, "flat",
        final_state["messages"],
        {
            "query": query, 
            "metrics": metrics,
            "final_blackboard": final_state.get("blackboard", "")
        },
    )

    # The final report is typically found in the Blackboard under Writer's section
    report = ""
    blackboard_content = final_state.get("blackboard", "")
    
    if "--- DRAFT REPORT (WRITER) ---" in blackboard_content:
        # Split by Writer's section, and take everything before the Critic's review
        parts = blackboard_content.split("--- DRAFT REPORT (WRITER) ---")
        writer_part = parts[1] if len(parts) > 1 else parts[0]
        report = writer_part.split("--- CRITIC REVIEW ---")[0].strip()
        # Also strip out any Manager comments that follow the Writer's draft
        report = report.split("--- MANAGER COMMENT ---")[0].strip()

    # Fallback if markdown parsing fails
    if not report:
        report = _get_last_ai_message(final_state["messages"])

    return report, final_state["messages"], metrics


# =============================================================================
# MAIN (standalone test)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FLAT AGENT ARCHITECTURE — Shared Blackboard & Manager (No Authority)")
    print("Manager → R → Manager → A → Manager → W → Manager → C → Manager → END")
    print("=" * 70)

    # Using a dummy ASIN that exists in the dataset for testing
    query = "Analyze the MSI Katana (B0CXVGSY2H) focusing on customer feedback."

    print(f"\n📋 Query: {query}\n")
    print("-" * 70)
    print("Running Flat Graph...\n")

    result, messages, metrics = run_flat_graph(query)

    print("\n" + "=" * 70)
    print("FINAL EXTRACTED REPORT:")
    print("=" * 70)
    print(result[:1500] + "\n\n...[TRUNCATED]...")

    print("\n" + "=" * 70)
    print("EFFICIENCY METRICS:")
    print("=" * 70)
    print(f"Total Tokens: {metrics['total_tokens']}")
    print(f"  - Prompt Tokens: {metrics['prompt_tokens']}")
    print(f"  - Completion Tokens: {metrics['completion_tokens']}")
    print(f"Total Cost: ${metrics['total_cost']:.4f}")
    print(f"Step Count: {metrics['step_count']}")
    print(f"Loop Count: {metrics['loop_count']}")
