"""Streamlit demo for the BI report system: pick a product, pick an architecture, see the result.

Single-product playground for inspecting traces, judge scores, and the
generated report side by side. The full experiment uses run_experiment.py;
this app is for interactive exploration and portfolio demos.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from evaluate import evaluate_report_v2, JUDGE_MODELS
from flat_graph import run_flat_graph
from hierarchical_graph import run_hierarchical_graph
from efficiency_tracker import EfficiencyTracker
from tools import get_product_list, get_product_specs
from logger import format_messages_for_display


st.set_page_config(
    page_title="AI Agent Thesis: BI Report Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .judge-score {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
    }
    .dim-score {
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_products():
    """Parse get_product_list output into an {asin: title} dict."""
    result = get_product_list.invoke({})
    products = {}
    for line in result.split("\n"):
        if line.startswith("- ASIN:"):
            parts = line.split(" | ")
            if len(parts) >= 2:
                asin = parts[0].replace("- ASIN: ", "").strip()
                title = parts[1].strip()
                products[asin] = title
    return products


def get_product_metadata(asin: str) -> str:
    return get_product_specs.invoke({"asin": asin})


def score_color(score: float) -> str:
    if score >= 4.5:
        return "🟢"
    elif score >= 3.5:
        return "🟡"
    return "🔴"


with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("⚙️ Configuration")
    st.markdown("---")

    products = load_products()

    if not products:
        st.error("No products found in database. Please run ingest.py first.")
        st.stop()

    st.subheader("📦 Select Product")
    product_options = {
        f"{asin}: {title[:40]}..." if len(title) > 40 else f"{asin}: {title}": asin
        for asin, title in products.items()
    }
    selected_display = st.selectbox(
        "Choose a product to analyze:",
        options=list(product_options.keys()),
        help="Select a product from the database for BI report generation",
    )
    selected_asin = product_options[selected_display]

    st.markdown("---")

    st.subheader("🏗️ Agent Architecture")
    architecture = st.radio(
        "Select team structure:",
        options=["Flat Team", "Hierarchical Team"],
        help="Flat: Manager as peer router. Hierarchical: Manager supervises with loop-back authority.",
    )

    if architecture == "Flat Team":
        st.info("**Flat Team**: Manager (peer) -> R -> A -> W -> C, no loop-back authority.")
    else:
        st.success("**Hierarchical Team**: Manager supervises R -> A -> W -> C with loop-back authority.")

    st.markdown("---")

    st.subheader("📝 Analysis Focus")
    custom_focus = st.text_input(
        "Focus area (optional):",
        placeholder="e.g., battery life issues",
        help="Specify what aspect to focus on in the analysis",
    )

    st.markdown("---")

    st.subheader("🧪 Evaluation Models")
    available_model_names = [m["name"] for m in JUDGE_MODELS]
    selected_model_names = st.multiselect(
        "Select judge models:",
        options=available_model_names,
        default=available_model_names,
        help="Choose which LLM models evaluate the report. More models = higher cost but better reliability.",
    )

    st.markdown("---")

    generate_btn = st.button(
        "🚀 Generate Report",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")
    st.caption("Master's Thesis Project: Flat vs Hierarchical Agent Architectures")


st.markdown('<h1 class="main-title">🤖 AI Agent Thesis: BI Report Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Comparing Flat vs Hierarchical Multi-Agent Architectures for Business Intelligence</p>', unsafe_allow_html=True)

with st.expander("📋 Selected Product Details", expanded=False):
    metadata = get_product_metadata(selected_asin)
    st.code(metadata, language="text")

st.markdown("---")


if generate_btn:
    product_title = products[selected_asin]
    if custom_focus:
        query = f"Analyze the product '{product_title}' (ASIN: {selected_asin}) focusing on {custom_focus}."
    else:
        query = (
            f"Analyze the product '{product_title}' (ASIN: {selected_asin}). "
            "Provide a comprehensive BI report covering customer feedback, issues, and recommendations."
        )

    selected_models = [m for m in JUDGE_MODELS if m["name"] in selected_model_names]
    if not selected_models:
        # Fall back to the first configured judge if the user deselected everything.
        selected_models = [JUDGE_MODELS[0]]

    with st.status("🤖 Agents are working...", expanded=True) as status:

        metrics = None
        agent_messages = None
        arch_key = "flat" if architecture == "Flat Team" else "hierarchical"

        # The model label is informational only; flat_graph / hierarchical_graph
        # do their own per-role random allocation when role_assignments=None.
        tracker = EfficiencyTracker(arch_key, asin=selected_asin, model="multi-model")
        tracker.start()

        if architecture == "Flat Team":
            st.write("📊 **Flat Architecture Selected**")
            st.write("👔 Manager (peer) coordinates the team without loop-back authority.")
            st.write("Manager -> 🔍 Researcher -> 📈 Analyst -> ✍️ Writer -> 🔬 Critic -> END")

            try:
                report, agent_messages, metrics = run_flat_graph(query, session_id=selected_asin)
                st.write("✅ Report generated successfully!")
            except Exception as e:
                st.error(f"Error generating report: {e}")
                st.stop()
        else:
            st.write("🏗️ **Hierarchical Architecture Selected**")
            st.write("👔 Manager supervises workers and may loop back for revision (max 2 loops).")
            st.write("Manager -> 🔍 Researcher -> 📈 Analyst -> ✍️ Writer -> 🔬 Critic -> END (with loop-back)")

            try:
                report, agent_messages, metrics = run_hierarchical_graph(query, session_id=selected_asin)
                st.write("✅ Report generated and verified!")
            except Exception as e:
                st.error(f"Error generating report: {e}")
                st.stop()

        if metrics:
            tracker.record_from_metrics(metrics)
        tracker.stop()
        gen_metrics = tracker.get_metrics()

        st.write("---")
        st.write(f"⚖️ **Evaluating with {len(selected_models)} model(s): "
                 f"{', '.join(m['name'] for m in selected_models)}**")
        st.write("📊 Writing Clarity: Structure, Coherence, Conciseness")
        st.write("🚀 Utility: Actionability, Root Cause Analysis, Strategic Depth")
        st.write("🎯 Accuracy: deterministic ground-truth spec verification")

        try:
            metadata = get_product_metadata(selected_asin)
            evaluation = evaluate_report_v2(
                report, metadata,
                asin=selected_asin,
                models=selected_models,
                verbose=False,
            )
            st.write("✅ Evaluation complete!")
        except Exception as e:
            st.error(f"Error evaluating report: {e}")
            evaluation = None

        status.update(label="✅ Generation Complete!", state="complete", expanded=False)

    st.markdown("---")
    st.subheader("📊 Results")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📄 Generated Report")
        st.markdown(report)

    with col2:
        st.markdown("### ⚖️ Evaluation")

        if evaluation:
            st.metric(
                label="Final Score",
                value=f"{evaluation.final_score}/5.0",
                delta=f"{'Excellent' if evaluation.final_score >= 4.5 else 'Good' if evaluation.final_score >= 3.5 else 'Needs Improvement'}",
            )

            st.markdown("---")

            st.markdown(f"#### 📊 Writing Clarity - {evaluation.quality.mean_score}/5.0")
            st.caption(f"Inter-rater alpha: {evaluation.quality.inter_rater_alpha}")

            for dim_key, dim_score in evaluation.quality.per_dimension_scores.items():
                dim_name = dim_key.replace("_", " ").title()
                st.markdown(f"{score_color(dim_score)} **{dim_name}**: {dim_score}/5.0")

            with st.expander("Writing Clarity Details (per model)"):
                for dim_key, results in evaluation.quality.dimensions.items():
                    st.markdown(f"**{dim_key.replace('_', ' ').title()}**")
                    for dr in results:
                        st.markdown(f"- *{dr.model_name}*: {dr.score}/5.0 ({dr.latency_seconds}s)")
                        st.caption(dr.qualitative_analysis[:200] + "...")

            st.markdown("---")

            st.markdown(f"#### 🎯 Accuracy - {evaluation.accuracy_score}/5.0")
            st.caption("Deterministic (regex-based)")
            with st.expander("Accuracy Details"):
                st.write(evaluation.accuracy_reasoning)

            st.markdown("---")

            st.markdown(f"#### 🚀 Utility - {evaluation.utility.mean_score}/5.0")
            st.caption(f"Inter-rater alpha: {evaluation.utility.inter_rater_alpha}")

            for dim_key, dim_score in evaluation.utility.per_dimension_scores.items():
                dim_name = dim_key.replace("_", " ").title()
                st.markdown(f"{score_color(dim_score)} **{dim_name}**: {dim_score}/5.0")

            with st.expander("Utility Details (per model)"):
                for dim_key, results in evaluation.utility.dimensions.items():
                    st.markdown(f"**{dim_key.replace('_', ' ').title()}**")
                    for dr in results:
                        st.markdown(f"- *{dr.model_name}*: {dr.score}/5.0 ({dr.latency_seconds}s)")
                        st.caption(dr.qualitative_analysis[:200] + "...")

            st.markdown("---")

            st.markdown("#### 🕸️ Dimension Radar Chart")

            categories = (
                list(evaluation.quality.per_dimension_scores.keys())
                + list(evaluation.utility.per_dimension_scores.keys())
                + ["accuracy"]
            )
            scores = (
                list(evaluation.quality.per_dimension_scores.values())
                + list(evaluation.utility.per_dimension_scores.values())
                + [evaluation.accuracy_score]
            )
            labels = [c.replace("_", " ").title() for c in categories]
            labels_closed = labels + [labels[0]]
            scores_closed = scores + [scores[0]]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=scores_closed,
                theta=labels_closed,
                fill='toself',
                name='Report Quality',
                line_color='#667eea',
                fillcolor='rgba(102, 126, 234, 0.3)',
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5],
                        tickvals=[1, 2, 3, 4, 5],
                        ticktext=['1', '2', '3', '4', '5'],
                    )
                ),
                showlegend=False,
                height=400,
                margin=dict(l=60, r=60, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("#### ⚡ Efficiency Metrics")

            eff_col1, eff_col2 = st.columns(2)
            with eff_col1:
                st.metric("📊 Total Tokens", f"{gen_metrics.total_tokens:,}")
                st.metric("📤 Prompt Tokens", f"{gen_metrics.prompt_tokens:,}")
            with eff_col2:
                st.metric("💵 Total Cost", f"${gen_metrics.total_cost_usd:.4f}")
                st.metric("📥 Completion Tokens", f"{gen_metrics.completion_tokens:,}")

            st.metric("⏱ Latency", f"{gen_metrics.latency_seconds:.1f}s")
            st.metric("🔢 Step Count", gen_metrics.step_count)

            if gen_metrics.verification_attempts > 0:
                st.metric("🔍 Verification Attempts", gen_metrics.verification_attempts)

            st.markdown("---")
            st.markdown("#### 📥 Export Results")

            arch_name = "Flat" if architecture == "Flat Team" else "Hierarchical"

            per_model_rows = ""
            for dim_key, results in evaluation.quality.dimensions.items():
                dim_name = f"Writing Clarity: {dim_key.replace('_', ' ').title()}"
                model_scores = " | ".join(f"{dr.score}" for dr in results)
                mean = evaluation.quality.per_dimension_scores.get(dim_key, 0)
                per_model_rows += f"| {dim_name} | {model_scores} | **{mean}** |\n"
            q_model_means = " | ".join(
                str(evaluation.quality.per_model_scores.get(m["name"], "-"))
                for m in selected_models
            )
            per_model_rows += f"| **Writing Clarity (Mean)** | {q_model_means} | **{evaluation.quality.mean_score}** |\n"
            per_model_rows += f"| Accuracy (Script) | - | **{evaluation.accuracy_score}** |\n"

            for dim_key, results in evaluation.utility.dimensions.items():
                dim_name = f"Utility: {dim_key.replace('_', ' ').title()}"
                model_scores = " | ".join(f"{dr.score}" for dr in results)
                mean = evaluation.utility.per_dimension_scores.get(dim_key, 0)
                per_model_rows += f"| {dim_name} | {model_scores} | **{mean}** |\n"
            u_model_means = " | ".join(
                str(evaluation.utility.per_model_scores.get(m["name"], "-"))
                for m in selected_models
            )
            per_model_rows += f"| **Utility (Mean)** | {u_model_means} | **{evaluation.utility.mean_score}** |\n"

            model_headers = " | ".join(m["name"] for m in selected_models)
            model_separators = " | ".join("---" for _ in selected_models)

            download_content = f"""# Business Intelligence Report

## Report Metadata
- **Product ASIN:** {selected_asin}
- **Product Name:** {products[selected_asin]}
- **Architecture Used:** {arch_name} Agent Architecture
- **Generated On:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Analysis Focus:** {custom_focus if custom_focus else 'General Analysis'}
- **Judge Models:** {', '.join(m['name'] for m in selected_models)}

---

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Total Tokens | {gen_metrics.total_tokens:,} |
| Prompt Tokens | {gen_metrics.prompt_tokens:,} |
| Completion Tokens | {gen_metrics.completion_tokens:,} |
| Total Cost | ${gen_metrics.total_cost_usd:.4f} |
| Latency | {gen_metrics.latency_seconds:.1f}s |
| Step Count | {gen_metrics.step_count} |

---

## Generated Report

{report}

---

## Evaluation Results - Final Score: {evaluation.final_score}/5.0

### Per-Model Dimension Scores

| Dimension | {model_headers} | **Mean** |
|-----------|{model_separators}|------|
{per_model_rows}

### Inter-Rater Reliability
- Writing Clarity alpha: {evaluation.quality.inter_rater_alpha}
- Utility alpha: {evaluation.utility.inter_rater_alpha}

### Accuracy Reasoning
{evaluation.accuracy_reasoning}

### Per-Model Qualitative Analysis

"""
            for metric_name, metric_obj in [("Writing Clarity", evaluation.quality), ("Utility", evaluation.utility)]:
                for dim_key, results in metric_obj.dimensions.items():
                    dim_name = dim_key.replace('_', ' ').title()
                    download_content += f"#### {metric_name}: {dim_name}\n\n"
                    for dr in results:
                        download_content += f"**{dr.model_name}** (Score: {dr.score}/5.0, Latency: {dr.latency_seconds}s)\n\n"
                        download_content += f"> {dr.qualitative_analysis}\n\n"

            download_content += "\n---\n\n*Report generated by Multi-Agent BI Report System - Master's Thesis Project*\n"

            filename = f"BI_Report_{selected_asin}_{arch_name}.md"

            st.download_button(
                label="📥 Download Report (Markdown)",
                data=download_content,
                file_name=filename,
                mime="text/markdown",
                use_container_width=True,
            )

            json_data = {
                "metadata": {
                    "product_asin": selected_asin,
                    "product_name": products[selected_asin],
                    "architecture": arch_name.lower(),
                    "generated_on": datetime.now().isoformat(),
                    "analysis_focus": custom_focus if custom_focus else "general",
                    "judge_models": [m["name"] for m in selected_models],
                },
                "efficiency": {
                    "total_tokens": gen_metrics.total_tokens,
                    "prompt_tokens": gen_metrics.prompt_tokens,
                    "completion_tokens": gen_metrics.completion_tokens,
                    "total_cost_usd": gen_metrics.total_cost_usd,
                    "latency_seconds": gen_metrics.latency_seconds,
                    "step_count": gen_metrics.step_count,
                    "verification_attempts": gen_metrics.verification_attempts,
                },
                "evaluation": evaluation.to_dict(),
                "report_text": report,
            }

            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)

            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"eval_{selected_asin}_{arch_name.lower()}_{timestamp_str}.json"
            json_filepath = results_dir / json_filename

            with open(json_filepath, "w", encoding="utf-8") as f:
                f.write(json_str)

            st.success(f"💾 Auto-saved to: `results/{json_filename}`")

            st.download_button(
                label="📥 Download Raw Data (JSON)",
                data=json_str,
                file_name=json_filename,
                mime="application/json",
                use_container_width=True,
                key="json_download",
            )

    st.markdown("---")
    with st.expander("🔍 View Agent Conversation Logs", expanded=False):
        st.markdown("### Agent Dialogue History")
        st.caption("This shows the internal communication between agents during report generation.")

        if agent_messages:
            formatted_logs = format_messages_for_display(agent_messages)

            for i, msg in enumerate(formatted_logs):
                with st.container():
                    st.markdown(f"**{msg['role']}** ({msg['type']})")

                    if "Human" in msg['role']:
                        st.info(msg['content'])
                    elif "AI" in msg['role']:
                        st.success(msg['content'])
                    elif "Tool" in msg['role']:
                        st.warning(msg['content'])
                    else:
                        st.text(msg['content'])

                    st.markdown("---")

            st.caption(f"Total messages: {len(formatted_logs)}")
        else:
            st.warning("No conversation logs available.")

else:
    st.info("👈 **Configure your analysis** in the sidebar and click **Generate Report** to start.")

    st.subheader("🔍 Architecture Comparison")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔄 Flat Architecture")
        st.markdown("""
        - **Structure**: 5 agents (4 Workers + Manager as peer)
        - **Coordination**: Symmetric influence
        - **Workflow**: Manager -> R -> Manager -> A -> Manager -> W -> Manager -> C -> END
        - **Manager Authority**: None (commentary only)
        """)
    with col2:
        st.markdown("#### 🏗️ Hierarchical Architecture")
        st.markdown("""
        - **Structure**: 5 agents (4 Workers + Manager as supervisor)
        - **Coordination**: Asymmetric influence
        - **Workflow**: Manager -> R -> A -> W -> C -> END (with loop-back, max 2 loops)
        - **Manager Authority**: Can reject and force loop-back
        """)

    st.markdown("---")
    st.subheader("📚 How It Works")
    st.markdown("""
    1. **Select a Product** from the database (e-commerce laptops)
    2. **Choose an Architecture** to generate the report
    3. **Select Judge Models** (5-model panel by default)
    4. **Generate Report** - AI agents collaborate to create a BI report
    5. **Evaluation** - 7 dimensions evaluated separately with chain-of-thought
    6. **Compare Results** between architectures
    """)

    st.markdown("---")
    st.subheader("🧪 Evaluation Dimensions")

    ev_col1, ev_col2, ev_col3 = st.columns(3)
    with ev_col1:
        st.markdown("**📊 Writing Clarity (3 dims)**")
        st.markdown("- Structure\n- Coherence\n- Conciseness")
    with ev_col2:
        st.markdown("**🚀 Utility (3 dims)**")
        st.markdown("- Actionability\n- Root Cause Analysis\n- Strategic Depth")
    with ev_col3:
        st.markdown("**🎯 Accuracy (1 dim)**")
        st.markdown("- Deterministic regex\n- Ground truth comparison\n- No LLM bias")
