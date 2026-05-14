"""Flat (control) architecture: shared Blackboard, Manager as a peer.

The Manager observes the Blackboard and may add commentary, but has no
authority to reject worker output or trigger a loop-back. The pipeline
is forced forward: Manager -> Researcher -> Manager -> Analyst ->
Manager -> Writer -> Manager -> Critic -> Manager -> END.

This is the control condition for the Flat vs. Hierarchical comparison;
the only structural difference from hierarchical_graph.py is the absence
of the Manager's loop-back authority. All five agents (4 workers + 1
Manager) draw their LLM from MODEL_POOL via random allocation per run.
"""

import os
from typing import Annotated, Literal, Optional

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


load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    blackboard: str
    next: str
    instructions: str
    step_count: int
    loop_count: int  # always 0 in flat


def _create_llm(model_id: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_id,
        temperature=config.TEMPERATURE,
        openai_api_key=os.getenv(config.OPENROUTER_API_KEY_ENV_NAME),
        openai_api_base=config.OPENROUTER_API_BASE,
    )


def _get_last_ai_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip():
            return message.content
    return ""


def _make_researcher_node(agent):
    def researcher_node(state: AgentState) -> dict:
        input_message = f"User Request: {state['messages'][0].content}\n\nCurrent Blackboard:\n{state.get('blackboard', '')}"
        if state.get("instructions") and state.get("instructions") != "Proceeding forward.":
            input_message += f"\n\n[MANAGER COMMENTARY (for reference only)]:\n{state['instructions']}"

        result = agent.invoke({"messages": [HumanMessage(content=input_message)]})
        new_insight = _get_last_ai_message(result["messages"])
        updated_blackboard = state.get("blackboard", "") + f"\n\n--- RESEARCHER FINDINGS ---\n{new_insight}"
        return {
            "messages": result["messages"],
            "blackboard": updated_blackboard,
            "step_count": state.get("step_count", 0) + 1,
        }
    return researcher_node


def _make_analyst_node(agent):
    def analyst_node(state: AgentState) -> dict:
        input_message = f"Here is the current Shared Blackboard. Analyze the research:\n{state.get('blackboard', '')}"
        if state.get("instructions") and state.get("instructions") != "Proceeding forward.":
            input_message += f"\n\n[MANAGER COMMENTARY (for reference only)]:\n{state['instructions']}"

        result = agent.invoke({"messages": [HumanMessage(content=input_message)]})
        new_insight = _get_last_ai_message(result["messages"])
        updated_blackboard = state.get("blackboard", "") + f"\n\n--- ANALYST INSIGHTS ---\n{new_insight}"
        return {
            "messages": result["messages"],
            "blackboard": updated_blackboard,
            "step_count": state.get("step_count", 0) + 1,
        }
    return analyst_node


def _make_writer_node(agent):
    def writer_node(state: AgentState) -> dict:
        input_message = f"Here is the current Shared Blackboard. Write the BI Report based on this:\n{state.get('blackboard', '')}"
        if state.get("instructions") and state.get("instructions") != "Proceeding forward.":
            input_message += f"\n\n[MANAGER COMMENTARY (for reference only)]:\n{state['instructions']}"

        result = agent.invoke({"messages": [HumanMessage(content=input_message)]})
        new_insight = _get_last_ai_message(result["messages"])
        updated_blackboard = state.get("blackboard", "") + f"\n\n--- DRAFT REPORT (WRITER) ---\n{new_insight}"
        return {
            "messages": result["messages"],
            "blackboard": updated_blackboard,
            "step_count": state.get("step_count", 0) + 1,
        }
    return writer_node


def _make_critic_node(agent):
    def critic_node(state: AgentState) -> dict:
        input_message = f"Here is the current Shared Blackboard. Review the Draft Report and verify findings:\n{state.get('blackboard', '')}"
        if state.get("instructions") and state.get("instructions") != "Proceeding forward.":
            input_message += f"\n\n[MANAGER COMMENTARY (for reference only)]:\n{state['instructions']}"

        result = agent.invoke({"messages": [HumanMessage(content=input_message)]})
        new_insight = _get_last_ai_message(result["messages"])
        updated_blackboard = state.get("blackboard", "") + f"\n\n--- CRITIC REVIEW ---\n{new_insight}"
        return {
            "messages": result["messages"],
            "blackboard": updated_blackboard,
            "step_count": state.get("step_count", 0) + 1,
        }
    return critic_node


class ManagerDecision(BaseModel):
    next_worker: Literal["Researcher", "Analyst", "Writer", "Critic", "FINISH"]
    instructions: str = Field(description="Commentary or suggestions for the next worker")
    reasoning: str = Field(description="Brief reasoning for this observation")


FLAT_WORKER_SEQUENCE = ["Researcher", "Analyst", "Writer", "Critic", "FINISH"]


def _make_flat_manager_node(manager_llm):
    def flat_manager_node(state: AgentState) -> dict:
        step_count = state.get("step_count", 0) + 1
        blackboard = state.get("blackboard", "")

        # Replay worker progress from Blackboard tags so this node is stateless.
        workers_done = []
        if "--- RESEARCHER FINDINGS ---" in blackboard:
            workers_done.append("Researcher")
        if "--- ANALYST INSIGHTS ---" in blackboard:
            workers_done.append("Analyst")
        if "--- DRAFT REPORT (WRITER) ---" in blackboard:
            workers_done.append("Writer")
        if "--- CRITIC REVIEW ---" in blackboard:
            workers_done.append("Critic")

        next_worker = "FINISH"
        for worker in FLAT_WORKER_SEQUENCE:
            if worker not in workers_done and worker != "FINISH":
                next_worker = worker
                break

        # The Manager LLM is still invoked so token-usage patterns stay
        # comparable with the Hierarchical condition; its routing choice
        # is then overridden by the deterministic next_worker below.
        structured_llm = manager_llm.with_structured_output(ManagerDecision)
        formatted_prompt = prompts.FLAT_MANAGER_PROMPT.format(step_count=step_count)
        messages = [
            SystemMessage(content=formatted_prompt),
            HumanMessage(
                content=f"Step {step_count}. Review the latest Shared Blackboard and provide "
                        f"your observations. You MUST proceed to: {next_worker}.\n\n"
                        f"CURRENT BLACKBOARD:\n{blackboard}"
            ),
        ]

        try:
            result = structured_llm.invoke(messages)
            if result is None or not hasattr(result, "next_worker"):
                raise ValueError("Model returned None or invalid structured output")
        except Exception:
            result = ManagerDecision(
                next_worker=next_worker,
                instructions="Proceeding forward (fallback).",
                reasoning="Model failed to provide structured output; using system fallback.",
            )

        commentary = result.reasoning if result.reasoning else "No additional observations."
        updated_blackboard = blackboard + f"\n\n--- MANAGER COMMENT ---\n{commentary}"

        return {
            "next": next_worker,
            "instructions": result.instructions if result.instructions else "Proceeding forward.",
            "blackboard": updated_blackboard,
            "step_count": step_count,
            "loop_count": 0,
        }
    return flat_manager_node


def route_from_manager(state: AgentState) -> str:
    next_step = state.get("next", "")
    if next_step == "FINISH":
        return "end"
    return next_step


def build_flat_graph(role_assignments: dict[str, str]) -> StateGraph:
    """Compile the flat-architecture LangGraph for the given per-role model assignments."""
    researcher_llm = _create_llm(role_assignments["Researcher"])
    analyst_llm = _create_llm(role_assignments["Analyst"])
    writer_llm = _create_llm(role_assignments["Writer"])
    critic_llm = _create_llm(role_assignments["Critic"])
    manager_llm = _create_llm(role_assignments["Manager"])

    researcher_agent = create_react_agent(
        researcher_llm,
        tools=[get_product_specs, search_reviews],
        messages_modifier=prompts.RESEARCHER_PROMPT,
    )
    analyst_agent = create_react_agent(
        analyst_llm,
        tools=[],
        messages_modifier=prompts.ANALYST_PROMPT,
    )
    writer_agent = create_react_agent(
        writer_llm,
        tools=[],
        messages_modifier=prompts.WRITER_PROMPT,
    )
    critic_agent = create_react_agent(
        critic_llm,
        tools=[verify_claim],
        messages_modifier=prompts.CRITIC_PROMPT,
    )

    workflow = StateGraph(AgentState)
    workflow.add_node("Manager", _make_flat_manager_node(manager_llm))
    workflow.add_node("Researcher", _make_researcher_node(researcher_agent))
    workflow.add_node("Analyst", _make_analyst_node(analyst_agent))
    workflow.add_node("Writer", _make_writer_node(writer_agent))
    workflow.add_node("Critic", _make_critic_node(critic_agent))

    workflow.set_entry_point("Manager")
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
    workflow.add_edge("Researcher", "Manager")
    workflow.add_edge("Analyst", "Manager")
    workflow.add_edge("Writer", "Manager")
    workflow.add_edge("Critic", "Manager")

    return workflow.compile()


def run_flat_graph(
    query: str,
    session_id: str = "session",
    role_assignments: Optional[dict] = None,
) -> tuple:
    """Run the flat graph end-to-end. Returns (report, messages, metrics).

    Pass role_assignments explicitly for paired experimental runs so the
    same product is given identical model assignments under both conditions.
    """
    if role_assignments is None:
        role_assignments = config.select_models_for_run()
    print(f"   Flat model assignments: { {k: v.split('/')[-1] for k, v in role_assignments.items()} }")

    graph = build_flat_graph(role_assignments)

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "blackboard": f"Initial Request: {query}",
        "next": "",
        "instructions": "",
        "step_count": 0,
        "loop_count": 0,
    }

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
            "role_assignments": role_assignments,
        }

    save_logs(
        session_id, "flat",
        final_state["messages"],
        {
            "query": query,
            "metrics": metrics,
            "final_blackboard": final_state.get("blackboard", ""),
        },
    )

    # Pull the report from the Writer's Blackboard section, stopping before
    # any subsequent Critic review or Manager comment block.
    report = ""
    blackboard_content = final_state.get("blackboard", "")
    if "--- DRAFT REPORT (WRITER) ---" in blackboard_content:
        parts = blackboard_content.split("--- DRAFT REPORT (WRITER) ---")
        writer_part = parts[1] if len(parts) > 1 else parts[0]
        report = writer_part.split("--- CRITIC REVIEW ---")[0].strip()
        report = report.split("--- MANAGER COMMENT ---")[0].strip()

    if not report:
        report = _get_last_ai_message(final_state["messages"])

    return report, final_state["messages"], metrics


if __name__ == "__main__":
    print("Flat architecture standalone test")
    query = "Analyze the MSI Katana (B0CXVGSY2H) focusing on customer feedback."
    print(f"Query: {query}\n")

    result, messages, metrics = run_flat_graph(query)

    print("\nFinal extracted report:")
    print(result[:1500] + "\n\n...[truncated]...")

    print("\nEfficiency metrics:")
    print(f"  Total tokens:      {metrics['total_tokens']}")
    print(f"    Prompt tokens:   {metrics['prompt_tokens']}")
    print(f"    Completion:      {metrics['completion_tokens']}")
    print(f"  Total cost:        ${metrics['total_cost']:.4f}")
    print(f"  Step count:        {metrics['step_count']}")
    print(f"  Loop count:        {metrics['loop_count']}")
    print(f"  Role assignments:  {metrics['role_assignments']}")
