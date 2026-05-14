"""Hierarchical (treatment) architecture: Manager with loop-back authority.

Workers (Researcher, Analyst, Writer, Critic) read and write to a shared
Blackboard. After each worker step, control returns to a Manager node that
inspects the Blackboard and either advances to the next worker or sends the
work back to a previous worker for revision (Socratic loop-back). Loops are
bounded by MAX_LOOPS.

This is the treatment condition for the Flat vs. Hierarchical comparison;
the only structural difference from flat_graph.py is the presence of the
Manager's loop-back authority. Random model allocation per role per run.
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
    loop_count: int


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
        input_text = f"User Request: {state['messages'][0].content}\n\nCurrent Blackboard:\n{state.get('blackboard', '')}"
        if state.get("instructions") and state.get("instructions") != "Proceeding forward.":
            input_text += f"\n\n[MANAGER FEEDBACK / LOOP-BACK INSTRUCTIONS]:\n{state['instructions']}"

        result = agent.invoke({"messages": [HumanMessage(content=input_text)]})
        new_insight = _get_last_ai_message(result["messages"])
        updated_blackboard = state.get("blackboard", "") + f"\n\n--- RESEARCHER FINDINGS ---\n{new_insight}"
        return {"messages": result["messages"], "blackboard": updated_blackboard}
    return researcher_node


def _make_analyst_node(agent):
    def analyst_node(state: AgentState) -> dict:
        input_text = f"Here is the current Shared Blackboard. Analyze the research:\n{state.get('blackboard', '')}"
        if state.get("instructions") and state.get("instructions") != "Proceeding forward.":
            input_text += f"\n\n[MANAGER FEEDBACK / LOOP-BACK INSTRUCTIONS]:\n{state['instructions']}"

        result = agent.invoke({"messages": [HumanMessage(content=input_text)]})
        new_insight = _get_last_ai_message(result["messages"])
        updated_blackboard = state.get("blackboard", "") + f"\n\n--- ANALYST INSIGHTS ---\n{new_insight}"
        return {"messages": result["messages"], "blackboard": updated_blackboard}
    return analyst_node


def _make_writer_node(agent):
    def writer_node(state: AgentState) -> dict:
        input_text = f"Here is the current Shared Blackboard. Write the BI Report based on this:\n{state.get('blackboard', '')}"
        if state.get("instructions") and state.get("instructions") != "Proceeding forward.":
            input_text += f"\n\n[MANAGER FEEDBACK / LOOP-BACK INSTRUCTIONS]:\n{state['instructions']}"

        result = agent.invoke({"messages": [HumanMessage(content=input_text)]})
        new_insight = _get_last_ai_message(result["messages"])
        updated_blackboard = state.get("blackboard", "") + f"\n\n--- DRAFT REPORT (WRITER) ---\n{new_insight}"
        return {"messages": result["messages"], "blackboard": updated_blackboard}
    return writer_node


def _make_critic_node(agent):
    def critic_node(state: AgentState) -> dict:
        input_text = f"Here is the current Shared Blackboard. Review the Draft Report and verify findings:\n{state.get('blackboard', '')}"
        if state.get("instructions") and state.get("instructions") != "Proceeding forward.":
            input_text += f"\n\n[MANAGER FEEDBACK / LOOP-BACK INSTRUCTIONS]:\n{state['instructions']}"

        result = agent.invoke({"messages": [HumanMessage(content=input_text)]})
        new_insight = _get_last_ai_message(result["messages"])
        updated_blackboard = state.get("blackboard", "") + f"\n\n--- CRITIC REVIEW ---\n{new_insight}"
        return {"messages": result["messages"], "blackboard": updated_blackboard}
    return critic_node


class ManagerDecision(BaseModel):
    next_worker: Literal["Researcher", "Analyst", "Writer", "Critic", "FINISH"]
    instructions: str = Field(description="Instructions for the next worker")
    reasoning: str = Field(description="Brief reasoning for this decision")


MAX_LOOPS = 2


def _make_manager_node(manager_llm):
    def manager_node(state: AgentState) -> dict:
        step_count = state.get("step_count", 0) + 1
        loop_count = state.get("loop_count", 0)
        blackboard = state.get("blackboard", "")

        # Hard stop in case the recursion limit alone isn't enough.
        if step_count > config.HIERARCHICAL_MAX_STEPS:
            return {
                "next": "FINISH",
                "instructions": "Maximum steps reached - completing workflow.",
                "step_count": step_count,
            }

        # Critic explicitly approved the draft -> short-circuit to FINISH.
        if "APPROVED:" in blackboard and "--- CRITIC REVIEW ---" in blackboard:
            critic_section = blackboard.split("--- CRITIC REVIEW ---")[-1]
            if "APPROVED:" in critic_section:
                return {
                    "next": "FINISH",
                    "instructions": "Report approved by Critic.",
                    "step_count": step_count,
                }

        workers_done = set()
        if "--- RESEARCHER FINDINGS ---" in blackboard:
            workers_done.add("Researcher")
        if "--- ANALYST INSIGHTS ---" in blackboard:
            workers_done.add("Analyst")
        if "--- DRAFT REPORT (WRITER) ---" in blackboard:
            workers_done.add("Writer")
        if "--- CRITIC REVIEW ---" in blackboard:
            workers_done.add("Critic")

        structured_llm = manager_llm.with_structured_output(ManagerDecision)
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

        try:
            result = structured_llm.invoke(messages)
            if result is None or not hasattr(result, "next_worker"):
                raise ValueError("Model returned None or invalid structured output")
        except Exception:
            workers_in_order = ["Researcher", "Analyst", "Writer", "Critic", "FINISH"]
            fallback_worker = "FINISH"
            for w in workers_in_order:
                if w not in workers_done:
                    fallback_worker = w
                    break
            result = ManagerDecision(
                next_worker=fallback_worker,
                instructions="Proceeding to next logical step (fallback).",
                reasoning="Model failed to provide structured output; using system fallback.",
            )

        new_loop_count = loop_count
        # Picking a worker that already produced output counts as a loop-back.
        if result.next_worker in workers_done and result.next_worker != "FINISH":
            new_loop_count = loop_count + 1
            # Override past MAX_LOOPS so the graph keeps moving forward.
            if new_loop_count > MAX_LOOPS:
                workers_in_order = ["Researcher", "Analyst", "Writer", "Critic"]
                for w in workers_in_order:
                    if w not in workers_done:
                        result.next_worker = w
                        result.instructions = "Proceeding forward (max loops reached)."
                        break
                else:
                    result.next_worker = "FINISH"
                    result.instructions = "All workers complete, finishing."

        # Strip stale instructions when advancing naturally to a new worker.
        if result.next_worker not in workers_done and result.next_worker != "FINISH":
            result.instructions = "Proceeding forward."

        return {
            "next": result.next_worker,
            "instructions": result.instructions,
            "step_count": step_count,
            "loop_count": new_loop_count,
        }
    return manager_node


def route_from_manager(state: AgentState) -> str:
    next_step = state.get("next", "")
    if next_step == "FINISH":
        return "end"
    return next_step


def build_hierarchical_graph(role_assignments: dict[str, str]) -> StateGraph:
    """Compile the hierarchical-architecture LangGraph for the given per-role model assignments."""
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
    workflow.add_node("Manager", _make_manager_node(manager_llm))
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


def run_hierarchical_graph(
    query: str,
    session_id: str = "session",
    role_assignments: Optional[dict] = None,
) -> tuple:
    """Run the hierarchical graph end-to-end. Returns (report, messages, metrics).

    Pass role_assignments explicitly for paired experimental runs so the
    same product is given identical model assignments under both conditions.
    """
    if role_assignments is None:
        role_assignments = config.select_models_for_run()
    print(f"   Hierarchical model assignments: { {k: v.split('/')[-1] for k, v in role_assignments.items()} }")

    graph = build_hierarchical_graph(role_assignments)

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
            "architecture": "hierarchical",
            "role_assignments": role_assignments,
        }

    save_logs(
        session_id, "hierarchical",
        final_state["messages"],
        {
            "query": query,
            "metrics": metrics,
            "final_blackboard": final_state.get("blackboard", ""),
        },
    )

    # When the Writer was looped back, take the last revision instead of the first.
    report = ""
    blackboard_content = final_state.get("blackboard", "")
    if "--- DRAFT REPORT (WRITER) ---" in blackboard_content:
        parts = blackboard_content.split("--- DRAFT REPORT (WRITER) ---")
        writer_part = parts[-1]
        report = writer_part.split("--- CRITIC REVIEW ---")[0].strip()

    if not report:
        report = _get_last_ai_message(final_state["messages"])

    return report, final_state["messages"], metrics


if __name__ == "__main__":
    print("Hierarchical architecture standalone test")
    query = "Analyze the MSI Katana (B0CXVGSY2H) focusing on customer feedback."
    print(f"Query: {query}\n")

    result, messages, metrics = run_hierarchical_graph(query)

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
