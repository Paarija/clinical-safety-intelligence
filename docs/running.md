# Running Locally

This project is intended to run as a local Python pipeline plus a Streamlit dashboard.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
python3 -m clinical_safety.init_env
```

Add a Gemini key to `.env` before running real synthesis:

```text
GEMINI_API_KEY=...
```

`NCBI_API_KEY` is optional and only increases PubMed request allowance. Do not commit `.env`.

## Data Refresh

Minimal FAERS-based run:

```bash
python3 -m clinical_safety.acquisition.faers_source
python3 -m clinical_safety.analytics.signal_ranking
python3 run_evidence_workflow.py
```

Optional ClinicalTrials.gov enrichment:

```bash
python3 -m clinical_safety.acquisition.clinicaltrials_source
python3 -m clinical_safety.modeling.trial_comparator
python3 -m clinical_safety.analytics.signal_ranking
```

Dry-run mode avoids Gemini calls and is useful for checking pipeline wiring:

```bash
python3 run_evidence_workflow.py --dry-run
```

Resume a partially completed real LLM run without rerunning existing real evidence packets:

```bash
python3 run_evidence_workflow.py --resume
```

Use a different Gemini model for one run:

```bash
python3 run_evidence_workflow.py --model gemini-2.5-flash
```

## Dashboard

Start Streamlit from the repository root:

```bash
streamlit run src/clinical_safety/app/streamlit_app.py --server.address 127.0.0.1
```

Health endpoint:

```text
http://127.0.0.1:8501/_stcore/health
```

The health endpoint only confirms that Streamlit is running. It does not prove that generated signal files or evidence packets are fresh.

## Local Artifacts

Generated files live under `data/processed/` and are ignored by Git. For a clean local run, use a fresh `DATA_DIR` or delete stale generated outputs before rerunning the pipeline.
