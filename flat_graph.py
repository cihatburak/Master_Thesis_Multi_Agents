"""
Flat Agent Architecture (Control Group)
Multi-Agent BI Report System using LangGraph

Implements a flat/peer architecture with DETERMINISTIC sequencing.
All agents operate at the same level with NO coordinating intelligence.
The pipeline is fixed: Researcher → Analyst → Writer → Critic → END

This is the CONTROL condition — the only difference from the Hierarchical
(treatment) group is the ABSENCE of a Manager node that can review outputs
and loop back to previous agents.

4 Agents: Researcher → Analyst → Writer → Critic
Coordination: None (deterministic forward-only pipeline)
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
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
    """State shared across all agents in the flat architecture."""
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int


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
# AGENT PROMPTS (IDENTICAL to Hierarchical Architecture)
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
# AGENT DEFINITIONS
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
# AGENT NODE WRAPPERS
# =============================================================================

def researcher_node(state: AgentState) -> dict:
    """Execute the Researcher agent."""
    result = researcher_agent.invoke({"messages": state["messages"]})
    step_count = state.get("step_count", 0) + 1
    return {"messages": result["messages"], "step_count": step_count}


def analyst_node(state: AgentState) -> dict:
    """Execute the Analyst agent."""
    result = analyst_agent.invoke({"messages": state["messages"]})
    step_count = state.get("step_count", 0) + 1
    return {"messages": result["messages"], "step_count": step_count}


def writer_node(state: AgentState) -> dict:
    """Execute the Writer agent."""
    result = writer_agent.invoke({"messages": state["messages"]})
    step_count = state.get("step_count", 0) + 1
    return {"messages": result["messages"], "step_count": step_count}


def critic_node(state: AgentState) -> dict:
    """Execute the Critic agent."""
    result = critic_agent.invoke({"messages": state["messages"]})
    step_count = state.get("step_count", 0) + 1
    return {"messages": result["messages"], "step_count": step_count}


# =============================================================================
# GRAPH CONSTRUCTION — DETERMINISTIC PIPELINE
# =============================================================================

def build_flat_graph() -> StateGraph:
    """
    Build the flat agent graph with deterministic sequencing.
    
    No LLM-based routing — the pipeline is fixed:
    Researcher → Analyst → Writer → Critic → END
    
    This ensures zero coordination intelligence in the control group.
    """
    workflow = StateGraph(AgentState)

    # Add worker nodes (identical to hierarchical)
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Analyst", analyst_node)
    workflow.add_node("Writer", writer_node)
    workflow.add_node("Critic", critic_node)

    # Deterministic forward-only edges (NO routing decisions)
    workflow.set_entry_point("Researcher")
    workflow.add_edge("Researcher", "Analyst")
    workflow.add_edge("Analyst", "Writer")
    workflow.add_edge("Writer", "Critic")
    workflow.add_edge("Critic", END)

    return workflow.compile()


# =============================================================================
# EXECUTION
# =============================================================================

def run_flat_graph(query: str, session_id: str = "session") -> tuple[str, list, dict]:
    """Run the flat graph and return the final report, messages, and metrics."""
    graph = build_flat_graph()

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "step_count": 0,
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
            "step_count": final_state.get("step_count", 4),
            "architecture": "flat",
        }

    # Save logs
    save_logs(
        session_id, "flat",
        final_state["messages"],
        {"query": query, "metrics": metrics},
    )

    # Extract the Writer's report (last substantial AI message with markdown)
    for message in reversed(final_state["messages"]):
        if hasattr(message, "content") and isinstance(message, AIMessage):
            content = message.content
            # Skip Critic's review messages
            if "APPROVED" in content:
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
    print("FLAT AGENT ARCHITECTURE — Deterministic Pipeline")
    print("Researcher → Analyst → Writer → Critic → END")
    print("=" * 70)

    query = "Analyze the MSI Katana (B0CXVGSY2H) focusing on customer feedback."

    print(f"\n📋 Query: {query}\n")
    print("-" * 70)
    print("Running Flat Graph...\n")

    result, messages, metrics = run_flat_graph(query)

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
