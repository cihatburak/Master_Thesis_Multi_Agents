"""
Hierarchical Agent Architecture (Treatment Group)
Multi-Agent BI Report System using LangGraph

Implements a Manager/Worker topology where a Manager coordinates
workers and can LOOP BACK to previous agents if output quality
is insufficient.

This is the TREATMENT condition — the only difference from the Flat
(control) group is the PRESENCE of a Manager node that can:
  1. Review each worker's output
  2. Decide to proceed forward OR loop back to a previous worker
  3. Provide corrective instructions when looping back

4 Agents: Researcher, Analyst, Writer, Critic (IDENTICAL to Flat)
Coordination: Manager node with loop-back authority
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

from tools import get_product_specs, search_reviews, verify_claim
from logger import save_logs
import config

# Load environment variables
load_dotenv()


# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """State shared across all agents in the hierarchical architecture."""
    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    instructions: str
    step_count: int
    loop_count: int


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
# WORKER PROMPTS (IDENTICAL to Flat Architecture)
# =============================================================================

RESEARCHER_PROMPT = """You are a Research Analyst specializing in e-commerce product data.
Your role is to gather factual information about products and customer reviews.

When given a task:
1. Use get_product_specs to retrieve product details
2. Use search_reviews to find relevant customer feedback
3. Provide raw data and findings WITHOUT interpretation

Be thorough but concise. Present facts clearly with specific quotes and numbers."""

ANALYST_PROMPT = """You are a Business Intelligence Analyst.
Your role is to analyze data provided by the Researcher and extract actionable insights.

When given research data:
1. Summarize the RATING DISTRIBUTION (e.g., "8 positive, 4 negative, 3 mixed out of 15 reviews")
2. Categorize findings into themes: Performance, Build Quality, Value for Money, 
   Battery Life, Display, Customer Service, etc.
3. For each negative theme, perform ROOT CAUSE ANALYSIS:
   - WHAT is the issue? (symptom)
   - WHY does it occur? (underlying cause if identifiable)
   - HOW MANY customers mention it? (frequency)
   - WHAT IS THE BUSINESS IMPACT? (returns, brand damage, competitive disadvantage)
4. Identify COMPETITIVE CONTEXT where possible (comparisons customers make)
5. Distinguish SYSTEMIC issues (design/manufacturing) from ONE-OFF incidents (shipping damage)

Do NOT make up data. Only analyze what has been provided in the conversation.
Always cite specific review quotes as evidence for your claims."""

WRITER_PROMPT = """You are a BI Report Writer.
Your role is to compile findings into a professional Business Intelligence report.

REPORT STRUCTURE (use these exact sections):
1. **Executive Summary** — 2-3 sentence overview of key findings
2. **Product Overview & Technical Specifications** — Include key specs from research:
   CPU, GPU, RAM, Storage, Display (size, resolution, refresh rate), Price.
   Present as a clear specifications table or list.
3. **Customer Feedback Analysis** — Categorized findings with evidence
   Include the overall rating distribution (e.g., "60% positive, 25% negative")
4. **Key Issues & Root Causes** — Top problems with underlying causes
5. **Actionable Recommendations** — Each recommendation MUST follow this format:
   - WHO should act (e.g., "Product team", "Marketing", "Customer support")
   - WHAT specifically to do (concrete action, not vague suggestion)
   - WHEN / PRIORITY (immediate, short-term, long-term)
   - EXPECTED IMPACT (what improvement this would bring)
6. **Conclusion** — Strategic summary for decision-makers

CRITICAL RULES:
- ALWAYS include the product's technical specifications in Section 2
- ALWAYS cite specific customer quotes as evidence
- Recommendations must be SPECIFIC and ACTIONABLE, not generic
- Format the report in Markdown for readability"""

CRITIC_PROMPT = """You are a Quality Control Analyst.
Your role is to review the Writer's report for accuracy and completeness.

When reviewing:
1. Check if claims are supported by evidence from the research
2. Identify any unsupported statements or potential hallucinations
3. Verify that recommendations are actionable and specific
4. Assess overall report quality and professionalism

Use verify_claim tool to check specific claims against the database.
Provide specific feedback on what needs improvement.

If the report is satisfactory, confirm with "APPROVED: [brief summary of quality]".
If issues found, list them clearly for the record."""


# =============================================================================
# WORKER AGENT DEFINITIONS (IDENTICAL to Flat Architecture)
# =============================================================================

researcher_agent = create_react_agent(
    llm,
    tools=[get_product_specs, search_reviews],
    prompt=RESEARCHER_PROMPT
)

analyst_agent = create_react_agent(
    llm,
    tools=[],  # Analyst works with data already in conversation
    prompt=ANALYST_PROMPT
)

writer_agent = create_react_agent(
    llm,
    tools=[],  # Writer works with analysis already in conversation
    prompt=WRITER_PROMPT
)

critic_agent = create_react_agent(
    llm,
    tools=[verify_claim],  # Critic can verify claims against DB
    prompt=CRITIC_PROMPT
)


# =============================================================================
# WORKER NODE WRAPPERS
# =============================================================================

def researcher_node(state: AgentState) -> dict:
    """Execute the Researcher worker."""
    messages = state["messages"]
    if state.get("instructions"):
        messages = messages + [
            HumanMessage(content=f"[Manager Feedback]: {state['instructions']}")
        ]
    result = researcher_agent.invoke({"messages": messages})
    return {"messages": result["messages"]}


def analyst_node(state: AgentState) -> dict:
    """Execute the Analyst worker."""
    messages = state["messages"]
    if state.get("instructions"):
        messages = messages + [
            HumanMessage(content=f"[Manager Feedback]: {state['instructions']}")
        ]
    result = analyst_agent.invoke({"messages": messages})
    return {"messages": result["messages"]}


def writer_node(state: AgentState) -> dict:
    """Execute the Writer worker."""
    messages = state["messages"]
    if state.get("instructions"):
        messages = messages + [
            HumanMessage(content=f"[Manager Feedback]: {state['instructions']}")
        ]
    result = writer_agent.invoke({"messages": messages})
    return {"messages": result["messages"]}


def critic_node(state: AgentState) -> dict:
    """Execute the Critic worker."""
    messages = state["messages"]
    if state.get("instructions"):
        messages = messages + [
            HumanMessage(content=f"[Manager Feedback]: {state['instructions']}")
        ]
    result = critic_agent.invoke({"messages": messages})
    return {"messages": result["messages"]}


# =============================================================================
# MANAGER NODE (TREATMENT — only present in Hierarchical)
# =============================================================================

class ManagerDecision(BaseModel):
    """Schema for the Manager's routing decision."""
    next_worker: Literal["Researcher", "Analyst", "Writer", "Critic", "FINISH"]
    instructions: str = Field(description="Instructions for the next worker")
    reasoning: str = Field(description="Brief reasoning for this decision")


MANAGER_PROMPT = """You coordinate a BI report generation team of 4 workers:
- Researcher: Gathers product specs and customer reviews from the database
- Analyst: Analyzes gathered data to find patterns and insights
- Writer: Compiles research into a professional BI report
- Critic: Reviews the report for accuracy and quality

WORKFLOW:
The standard sequence is: Researcher → Analyst → Writer → Critic → FINISH.
After each worker completes their task, you review the output and decide:

1. PROCEED to the next worker in sequence (if output is sufficient), OR
2. LOOP BACK to a previous worker (if output needs improvement)

LOOP-BACK RULES:
- You may loop back at most {max_loops} times total across the entire workflow
- Current loop count: {loop_count} / {max_loops}
- When looping back, provide SPECIFIC instructions on what needs improvement
- If max loops reached, always proceed forward

DECISION GUIDE:
- After Researcher: Does the data include product specs AND customer reviews with quotes?
- After Analyst: Are themes identified with evidence? Is root cause analysis present?
- After Writer: Does the report follow the required 6-section structure?
- After Critic: Has the Critic verified claims? Choose FINISH.

Current step: {step_count}
Analyze the conversation and decide the next step."""

MAX_LOOPS = 1  # Maximum loop-backs allowed (kept minimal to reduce context pollution)


def manager_node(state: AgentState) -> dict:
    """
    Manager node — reviews worker output and decides next step.
    
    This is the ONLY component that differs from the Flat architecture.
    The Manager can loop back to previous workers if needed.
    """
    step_count = state.get("step_count", 0) + 1
    loop_count = state.get("loop_count", 0)

    # Safety: force finish after too many steps
    if step_count > config.HIERARCHICAL_MAX_STEPS:
        return {
            "next": "FINISH",
            "instructions": "Maximum steps reached — completing workflow.",
            "step_count": step_count,
        }

    # Check if Critic already approved
    for msg in reversed(state["messages"]):
        if hasattr(msg, "content") and isinstance(msg, AIMessage):
            if "APPROVED:" in msg.content:
                return {
                    "next": "FINISH",
                    "instructions": "Report approved by Critic.",
                    "step_count": step_count,
                }
            break  # Only check last AI message

    # Manager makes a routing decision
    manager_llm = llm.with_structured_output(ManagerDecision)

    formatted_prompt = MANAGER_PROMPT.format(
        step_count=step_count,
        loop_count=loop_count,
        max_loops=MAX_LOOPS,
    )

    messages = [
        SystemMessage(content=formatted_prompt),
        *state["messages"],
        HumanMessage(
            content=f"Step {step_count}. Review the latest output and decide: "
                    f"proceed to next worker or loop back? "
                    f"(Loops used: {loop_count}/{MAX_LOOPS})"
        ),
    ]

    result = manager_llm.invoke(messages)

    # Track if this is a loop-back (going to a worker that already spoke)
    workers_in_order = ["Researcher", "Analyst", "Writer", "Critic"]
    # Determine which workers have already contributed
    workers_done = set()
    for msg in state["messages"]:
        if hasattr(msg, "content") and isinstance(msg, AIMessage):
            content = msg.content
            # Heuristic: detect which worker produced this message
            if any(s in content for s in ["product specifications", "ASIN", "search_reviews"]):
                workers_done.add("Researcher")
            elif any(s in content for s in ["ROOT CAUSE", "rating distribution", "systemic"]):
                workers_done.add("Analyst")
            elif any(s in content for s in ["Executive Summary", "## ", "Actionable Recommendations"]):
                workers_done.add("Writer")
            elif any(s in content for s in ["APPROVED", "verify_claim", "Quality Control"]):
                workers_done.add("Critic")

    new_loop_count = loop_count
    if result.next_worker in workers_done and result.next_worker != "FINISH":
        new_loop_count = loop_count + 1
        # If we've exceeded max loops, override to move forward
        if new_loop_count > MAX_LOOPS:
            # Find next worker in sequence that hasn't been done
            for w in workers_in_order:
                if w not in workers_done:
                    result.next_worker = w
                    result.instructions = "Proceeding forward (max loops reached)."
                    break
            else:
                result.next_worker = "FINISH"
                result.instructions = "All workers complete, finishing."

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

    # All workers report back to Manager
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
        {"query": query, "metrics": metrics},
    )

    # Extract the Writer's report (NOT the Critic's review)
    for message in reversed(final_state["messages"]):
        if hasattr(message, "content") and isinstance(message, AIMessage):
            content = message.content
            # Skip non-report messages
            if "APPROVED" in content:
                continue
            if "[Manager" in content:
                continue
            if content.strip().startswith("### Review"):
                continue
            # Look for actual BI report (has markdown headers + BI sections)
            if "##" in content or "# " in content:
                if any(s in content for s in [
                    "Executive Summary", "Findings",
                    "Recommendations", "Summary", "Analysis",
                ]):
                    return content, final_state["messages"], metrics

    # Fallback to last message
    return final_state["messages"][-1].content, final_state["messages"], metrics


# =============================================================================
# MAIN (standalone test)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HIERARCHICAL AGENT ARCHITECTURE — Manager with Loop-back")
    print("Manager → Researcher → Analyst → Writer → Critic (+ loop-backs)")
    print("=" * 70)

    query = "Analyze the MSI Katana (B0CXVGSY2H) focusing on customer feedback."

    print(f"\n📋 Query: {query}\n")
    print("-" * 70)
    print("Running Hierarchical Graph...\n")

    result, messages, metrics = run_hierarchical_graph(query)

    print("\n" + "=" * 70)
    print("FINAL REPORT:")
    print("=" * 70)
    print(result)

    print("\n" + "=" * 70)
    print("EFFICIENCY METRICS:")
    print("=" * 70)
    print(f"Total Tokens: {metrics['total_tokens']}")
    print(f"  - Prompt Tokens: {metrics['prompt_tokens']}")
    print(f"  - Completion Tokens: {metrics['completion_tokens']}")
    print(f"Total Cost: ${metrics['total_cost']:.4f}")
    print(f"Step Count: {metrics['step_count']}")
    print(f"Loop Count: {metrics['loop_count']}")
