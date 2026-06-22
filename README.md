# Do LLM Agents Need a Manager?

This repository contains the code for my master's thesis on coordination structures in LLM-based multi-agent systems. The experiment compares two architectures, *Flat* and *Hierarchical*, on a Business Intelligence report-generation task using Amazon laptop reviews. Both architectures use the same five agents and the same prompts; the only structural difference is whether the Manager has loop-back authority to send work back for revision.

The full thesis and conference paper live outside this repository; this README covers what the code does and how to run it.

## What the system does

Given an Amazon ASIN, both architectures produce a Business Intelligence report by routing through four worker agents (Researcher, Analyst, Writer, Critic) plus a Manager. The Researcher pulls product specs and customer reviews from a ChromaDB vector store. The Analyst extracts patterns, the Writer drafts a structured report, and the Critic verifies factual claims against the database. In the Hierarchical condition, the Manager can reject the report and trigger a revision loop (max 2 loops). In the Flat condition, the Manager is a pass-through router with no rejection authority. The same product is run through both architectures with the same model assignment, giving a paired observation per product.

After generation, each report is scored by a 5-model LLM judge panel on Writing Clarity and Utility, plus a deterministic accuracy check against the verified specs. Token usage, cost and latency are tracked per run.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── config.py                    # Model pool, cost tables, run-level model selection
├── prompts.py                   # System prompts for each agent role
├── tools.py                     # LangChain tools: specs lookup, semantic review search, claim verification
├── logger.py                    # Structured logging for agent message traces
├── efficiency_tracker.py        # Per-run token, cost and timing collection
├── flat_graph.py                # Flat architecture (LangGraph state machine)
├── hierarchical_graph.py        # Hierarchical architecture (LangGraph state machine)
├── evaluate.py                  # 5-judge LLM panel and accuracy scoring
├── run_experiment.py            # Main experiment driver: runs both architectures across all products
├── ingest.py                    # Loads dataset_final.json into ChromaDB with sentence-bert embeddings
├── app.py                       # Streamlit UI for interactive single-product runs
├── run_stats.py                 # Paired t-tests, Cohen's d, descriptives
├── robustness_check.py          # Wilcoxon signed-rank and Holm-Bonferroni correction
├── revision_analyses.py         # TOST equivalence, dose-response robustness, inter-judge and inter-dimension correlations
└── generate_thesis_figures.py   # Matplotlib figures used in the thesis and paper
```

Generated artefacts (vector DB, logs, results, generated reports, dataset JSON) are not tracked in git. They are produced by running the scripts.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file at the project root with the API keys you intend to use:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...
DASHSCOPE_API_KEY=...
ZHIPUAI_API_KEY=...
```

You only need the keys for the model providers actually present in the pool you plan to use (see `config.MODEL_POOL`).

## How to run

The expected workflow is:

1. **Build the vector store.** Run `python ingest.py` once. This reads `dataset_final.json` and writes embeddings to `chroma_db/`. The dataset and the database are gitignored, so you will need your own copy of the dataset to reproduce.

2. **Run the experiment.** `python run_experiment.py` walks through every product in the dataset, runs both architectures with the same model assignment per product, evaluates the resulting reports with the judge panel, and writes results to `general_analysis/`. This is the long-running step; on the full 43-product set it takes several hours depending on rate limits.

3. **Run the statistics.** `python run_stats.py` produces the paired t-tests and Cohen's d values reported in the thesis. `python robustness_check.py` adds Wilcoxon and Holm-Bonferroni corrections. `python revision_analyses.py results/experiments/exp_<timestamp>` runs the TOST equivalence test on Structure, the dose-response robustness regressions, and the inter-judge and inter-dimension correlation matrices.

4. **Regenerate figures.** `python generate_thesis_figures.py` produces the matplotlib figures used in the thesis.

The Streamlit app (`streamlit run app.py`) provides an interactive one-product demo of both architectures side by side. It is meant for inspecting traces and the generated reports rather than for running the full experiment.

## A note on reproducibility

Temperature is fixed at 0 for all model calls and the same model is assigned to a given role for both architectures of a given product (paired design). Even so, frontier-model APIs are not deterministic at the token level, so exact-text reproduction is unrealistic; quantitative results should reproduce within sampling noise.
