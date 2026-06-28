"""Minimal smoke test — import all key modules and report results."""
import sys
import os
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

modules = [
    "clinical_safety.acquisition.clinicaltrials_source",
    "clinical_safety.modeling.trial_comparator",
    "clinical_safety.llm.gemini_provider",
    "clinical_safety.llm.guardrails",
    "clinical_safety.llm.prompts.synthesis_prompt",
    "clinical_safety.evidence.retrievers.pubmed_retriever",
    "clinical_safety.evidence.retrievers.fda_retriever",
    "clinical_safety.orchestration.grader",
    "clinical_safety.orchestration.graph",
    "clinical_safety.reporting.report_generator",
    "clinical_safety.app.components",
]

errors = []
results = []
for mod in modules:
    try:
        __import__(mod)
        results.append(f"OK   {mod}")
    except Exception as exc:
        tb = traceback.format_exc()
        results.append(f"ERR  {mod}: {exc}\n{tb}")
        errors.append(mod)

results.append("")
results.append("ALL IMPORTS OK" if not errors else f"FAILED: {', '.join(errors)}")

text = "\n".join(results)

# Write to project root
out_path = Path(__file__).parent / "smoke_results.txt"
try:
    out_path.write_text(text, encoding="utf-8")
    print(f"Written to {out_path}")
except Exception as e:
    print(f"Could not write to {out_path}: {e}")

# Also print to stdout
print(text)
sys.exit(1 if errors else 0)
