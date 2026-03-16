"""
Hierarchical Agent Architecture (Treatment Group)
Multi-Agent BI Report System using LangGraph

Implements a Manager/Worker topology where a Manager coordinates
workers and can LOOP BACK to previous agents if output quality
is insufficient.

This is the TREATMENT condition — it tests Asymmetric Influence.
Workers read and write to the Shared Blackboard, but the Manager
has the authority to review the Blackboard and force agents to revise
their work, creating a hierarchical power dynamic.
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
# STATE DEFINITION (The Shared Blackboard + Routing State)
# =============================================================================

class AgentState(TypedDict):
    """State shared across all agents in the hierarchical architecture."""
    messages: Annotated[list[BaseMessage], add_messages]
    blackboard: str  # The shared workspace
    next: str        # The next agent to route to, determined by Manager
    instructions: str # Manager's specific instructions for the next agent
    step_count: int  # Total graph steps
    loop_count: int  # Number of times Manager forced a loop-back


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
    tools=[], 
    prompt=prompts.ANALYST_PROMPT
)

writer_agent = create_react_agent(
    llm,
    tools=[], 
    prompt=prompts.WRITER_PROMPT
)

critic_agent = create_react_agent(
    llm,
    tools=[verify_claim], 
    prompt=prompts.CRITIC_PROMPT
)


# =============================================================================
# WORKER NODE WRAPPERS (Blackboard Interaction Logic)
# =============================================================================

def _get_last_ai_message(messages: list[BaseMessage]) -> str:
    """Extract the last string response given by the AI agent."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip():
            return message.content
    return ""


def researcher_node(state: AgentState) -> dict:
    """Execute the Researcher worker."""
    input_text = f"User Request: {state['messages'][0].content}\n\nCurrent Blackboard:\n{state.get('blackboard', '')}"
    
    # Manager can pass specific instructions/corrections
    if state.get("instructions") and state.get("instructions") != "Proceeding forward.":
        input_text += f"\n\n[MANAGER FEEDBACK / LOOP-BACK INSTRUCTIONS]:\n{state['instructions']}"
        
    result = researcher_agent.invoke({"messages": [HumanMessage(content=input_text)]})
    new_insight = _get_last_ai_message(result["messages"])
    
    updated_blackboard = state.get("blackboard", "") + f"\n\n--- RESEARCHER FINDINGS ---\n{new_insight}"
    
    return {"messages": result["messages"], "blackboard": updated_blackboard}


def analyst_node(state: AgentState) -> dict:
    """Execute the Analyst worker."""
    input_text = f"Here is the current Shared Blackboard. Analyze the research:\n{state.get('blackboard', '')}"
    
    if state.get("instructions") and state.get("instructions") != "Proceeding forward.":
        input_text += f"\n\n[MANAGER FEEDBACK / LOOP-BACK INSTRUCTIONS]:\n{state['instructions']}"
        
    result = analyst_agent.invoke({"messages": [HumanMessage(content=input_text)]})
    new_insight = _get_last_ai_message(result["messages"])
    
    updated_blackboard = state.get("blackboard", "") + f"\n\n--- ANALYST INSIGHTS ---\n{new_insight}"
    
    return {"messages": result["messages"], "blackboard": updated_blackboard}


def writer_node(state: AgentState) -> dict:
    """Execute the Writer worker."""
    input_text = f"Here is the current Shared Blackboard. Write the BI Report based on this:\n{state.get('blackboard', '')}"
    
    if state.get("instructions") and state.get("instructions") != "Proceeding forward.":
        input_text += f"\n\n[MANAGER FEEDBACK / LOOP-BACK INSTRUCTIONS]:\n{state['instructions']}"
        
    result = writer_agent.invoke({"messages": [HumanMessage(content=input_text)]})
    new_insight = _get_last_ai_message(result["messages"])
    
    updated_blackboard = state.get("blackboard", "") + f"\n\n--- DRAFT REPORT (WRITER) ---\n{new_insight}"
    
    return {"messages": result["messages"], "blackboard": updated_blackboard}


def critic_node(state: AgentState) -> dict:
    """Execute the Critic worker."""
    input_text = f"Here is the current Shared Blackboard. Review the Draft Report and verify findings:\n{state.get('blackboard', '')}"
    
    if state.get("instructions") and state.get("instructions") != "Proceeding forward.":
        input_text += f"\n\n[MANAGER FEEDBACK / LOOP-BACK INSTRUCTIONS]:\n{state['instructions']}"
        
    result = critic_agent.invoke({"messages": [HumanMessage(content=input_text)]})
    new_insight = _get_last_ai_message(result["messages"])
    
    updated_blackboard = state.get("blackboard", "") + f"\n\n--- CRITIC REVIEW ---\n{new_insight}"
    
    return {"messages": result["messages"], "blackboard": updated_blackboard}


# =============================================================================
# MANAGER NODE (TREATMENT — only present in Hierarchical)
# =============================================================================

class ManagerDecision(BaseModel):
    """Schema for the Manager's routing decision."""
    next_worker: Literal["Researcher", "Analyst", "Writer", "Critic", "FINISH"]
    instructions: str = Field(description="Instructions for the next worker")
    reasoning: str = Field(description="Brief reasoning for this decision")


MAX_LOOPS = 2  # Maximum loop-backs allowed (increased from 1 based on pilot)


def manager_node(state: AgentState) -> dict:
    """
    Manager node — reviews worker output on the Blackboard and decides next step.
    
    This is the ONLY component that differs from the Flat architecture.
    The Manager can loop back to previous workers if needed based on authority.
    """
    step_count = state.get("step_count", 0) + 1
    loop_count = state.get("loop_count", 0)
    blackboard = state.get("blackboard", "")

    # Safety: force finish after too many steps
    if step_count > config.HIERARCHICAL_MAX_STEPS:
        return {
            "next": "FINISH",
            "instructions": "Maximum steps reached — completing workflow.",
            "step_count": step_count,
        }

    # Check if Critic already approved in the Blackboard
    if "APPROVED:" in blackboard and "--- CRITIC REVIEW ---" in blackboard:
        # Get purely the Critic's section to ensure they are the one who approved it
        critic_section = blackboard.split("--- CRITIC REVIEW ---")[-1]
        if "APPROVED:" in critic_section:
            return {
                "next": "FINISH",
                "instructions": "Report approved by Critic.",
                "step_count": step_count,
            }

    # Manager makes a routing decision based on the Blackboard
    manager_llm = llm.with_structured_output(ManagerDecision)

    formatted_prompt = prompts.MANAGER_PROMPT.format(
        step_count=step_count,
        loop_count=loop_count,
        max_loops=MAX_LOOPS,
    )

    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(
            content=f"Step {step_count}. Review the latest Shared Blackboard and decide: "
                    f"proceed to next worker or loop back? "
                    f"(Loops used: {loop_count}/{MAX_LOOPS})\n\n"
                    f"CURRENT BLACKBOARD:\n{blackboard}"
        ),
    ]

    result = manager_llm.invoke(messages)

    # Determine which workers have already contributed via Blackboard tags
    workers_done = set()
    if "--- RESEARCHER FINDINGS ---" in blackboard:
        workers_done.add("Researcher")
    if "--- ANALYST INSIGHTS ---" in blackboard:
        workers_done.add("Analyst")
    if "--- DRAFT REPORT (WRITER) ---" in blackboard:
        workers_done.add("Writer")
    if "--- CRITIC REVIEW ---" in blackboard:
        workers_done.add("Critic")

    new_loop_count = loop_count
    # A loop-back occurs if Manager selects a worker that already went, and it's not FINISH
    if result.next_worker in workers_done and result.next_worker != "FINISH":
        new_loop_count = loop_count + 1
        # If we've exceeded max loops, override to move forward robustly
        if new_loop_count > MAX_LOOPS:
            workers_in_order = ["Researcher", "Analyst", "Writer", "Critic"]
            # Find next worker in sequence that hasn't been done
            for w in workers_in_order:
                if w not in workers_done:
                    result.next_worker = w
                    result.instructions = "Proceeding forward (max loops reached)."
                    break
            else:
                result.next_worker = "FINISH"
                result.instructions = "All workers complete, finishing."

    # Provide default instructions if proceeding naturally so we don't pollute context
    if result.next_worker not in workers_done and result.next_worker != "FINISH":
        result.instructions = "Proceeding forward."

    return {
        "next": result.next_worker,
        "instructions": result.instructions,
        "step_count": step_count,
        "loop_count": new_loop_count,
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
# GRAPH CONSTRUCTION
# =============================================================================

def build_hierarchical_graph() -> StateGraph:
    """
    Build the hierarchical agent graph.

    Same 4 workers as Flat, but with a Manager node that:
    - Routes workers in sequence (R → A → W → C)
    - Can loop back to previous workers (up to MAX_LOOPS times)
    
    Treatment variable: Manager's loop-back authority.
    """
    workflow = StateGraph(AgentState)

    # Add nodes — same workers as Flat + Manager
    workflow.add_node("Manager", manager_node)
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Analyst", analyst_node)
    workflow.add_node("Writer", writer_node)
    workflow.add_node("Critic", critic_node)

    # Entry point: Manager decides first action
    workflow.set_entry_point("Manager")

    # Manager routes to workers or END
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

    # All workers report back to Manager once they append to the Blackboard
    workflow.add_edge("Researcher", "Manager")
    workflow.add_edge("Analyst", "Manager")
    workflow.add_edge("Writer", "Manager")
    workflow.add_edge("Critic", "Manager")

    return workflow.compile()


# =============================================================================
# EXECUTION
# =============================================================================

def run_hierarchical_graph(query: str, session_id: str = "session") -> tuple[str, list, dict]:
    """Run the hierarchical graph and return the final report, messages, and metrics."""
    graph = build_hierarchical_graph()

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
            "architecture": "hierarchical",
        }

    # Save logs
    save_logs(
        session_id, "hierarchical",
        final_state["messages"],
        {
            "query": query, 
            "metrics": metrics,
            "final_blackboard": final_state.get("blackboard", "")
        },
    )

    # Extract the Writer's report from the Blackboard
    report = ""
    blackboard_content = final_state.get("blackboard", "")
    
    if "--- DRAFT REPORT (WRITER) ---" in blackboard_content:
        parts = blackboard_content.split("--- DRAFT REPORT (WRITER) ---")
        writer_part = parts[-1]  # Take the last one in case of loop-backs
        report = writer_part.split("--- CRITIC REVIEW ---")[0].strip()

    # Fallback if parsing fails
    if not report:
        report = _get_last_ai_message(final_state["messages"])

    return report, final_state["messages"], metrics


# =============================================================================
# MAIN (standalone test)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HIERARCHICAL AGENT ARCHITECTURE — Manager with Loop-Back Authority")
    print("Manager evaluates Blackboard and coordinates: R → A → W → C → END")
    print("=" * 70)

    query = "Analyze the MSI Katana (B0CXVGSY2H) focusing on customer feedback."

    print(f"\n📋 Query: {query}\n")
    print("-" * 70)
    print("Running Hierarchical Graph...\n")

    result, messages, metrics = run_hierarchical_graph(query)

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
