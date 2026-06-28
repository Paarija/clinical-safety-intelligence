#!/usr/bin/env python3
"""
Run the LangGraph evidence workflow over all candidate signals.

Usage:
    python3 run_evidence_workflow.py              # real LLM calls (requires GEMINI_API_KEY)
    python3 run_evidence_workflow.py --dry-run     # placeholder synthesis, no API calls
"""
import argparse
import json
import os
import sys
import pandas as pd
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from clinical_safety.common.mock_data import is_mock_pipeline_data
from clinical_safety.common.config import get_config
from clinical_safety.common.paths import Paths
from clinical_safety.common.types import SignalMetrics, MappingConfidence
from clinical_safety.orchestration.graph import build_graph, run_signal

_DRY_RUN_WARNING = """
+--------------------------------------------------------------+
| DRY RUN MODE - outputs are placeholders, not real evidence.  |
| Remove --dry-run to run with real Gemini API calls.          |
+--------------------------------------------------------------+
"""

_REAL_RUN_NOTICE = "Running with real Gemini API calls (CLINICAL_SAFETY_DRY_RUN=0)."


def _packet_has_real_synthesis(packet_path: Path) -> bool:
    if not packet_path.exists():
        return False
    try:
        payload = json.loads(packet_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    text = packet_path.read_text(encoding="utf-8")
    if "Dry-run mode was used" in text or "Dry-run placeholder" in text:
        return False
    return bool(payload.get("synthesis_summary")) or bool(payload.get("evidence_grade"))


def _row_to_metrics(row: pd.Series) -> SignalMetrics:
    return SignalMetrics(
        drug_id=row["drug_id"],
        event_id=row["event_id"],
        evidence_window=str(row.get("evidence_window", "unknown")),
        case_count=int(row.get("case_count", 0)),
        ror=float(row.get("ror")) if pd.notna(row.get("ror")) else None,
        ror_lower_ci=float(row.get("ror_lower_ci")) if pd.notna(row.get("ror_lower_ci")) else None,
        ror_upper_ci=float(row.get("ror_upper_ci")) if pd.notna(row.get("ror_upper_ci")) else None,
        prr=float(row.get("prr")) if pd.notna(row.get("prr")) else None,
        chi2_p_value=float(row.get("chi2_p_value")) if pd.notna(row.get("chi2_p_value")) else None,
        seriousness_rate=(
            float(row.get("seriousness_rate"))
            if pd.notna(row.get("seriousness_rate"))
            else None
        ),
        death_count=int(row.get("death_count", 0)),
        hospitalization_count=int(row.get("hospitalization_count", 0)),
        trend_slope=float(row.get("trend_slope")) if pd.notna(row.get("trend_slope")) else None,
        potential_publicity_spike=bool(row.get("potential_publicity_spike", False)),
        drug_mapping_confidence=MappingConfidence(row.get("drug_mapping_confidence", "unmatched")),
        event_mapping_confidence=MappingConfidence(row.get("event_mapping_confidence", "unmatched")),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the LangGraph evidence workflow over candidate signals."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run with placeholder LLM synthesis (no API key required).",
    )
    parser.add_argument(
        "--allow-mock-data",
        action="store_true",
        default=False,
        help="Allow execution against synthetic evaluation outputs marked by _IS_MOCK_DATA.",
    )
    parser.add_argument(
        "--resume",
        "--missing-only",
        dest="resume",
        action="store_true",
        default=False,
        help="Skip signals that already have a non-dry-run evidence packet.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the configured Gemini model for this run, e.g. gemini-2.5-flash.",
    )
    args = parser.parse_args()

    os.environ["CLINICAL_SAFETY_DRY_RUN"] = "1" if args.dry_run else "0"
    if args.model:
        cfg = get_config()
        cfg.model_providers.llm.gemini.model = args.model
        print(f"Using Gemini model override: {args.model}")
    if args.dry_run:
        print(_DRY_RUN_WARNING)
    else:
        print(_REAL_RUN_NOTICE)

    paths = Paths()
    if is_mock_pipeline_data(paths) and not args.allow_mock_data:
        print(
            "Error: synthetic evaluation outputs are present. "
            "Delete or regenerate mock-marked pipeline outputs before running evidence workflow, "
            "or pass --allow-mock-data for local wiring checks only.",
            file=sys.stderr,
        )
        return 1

    signals_file = paths.processed_signals / "candidate_signals.parquet"
    if not signals_file.exists():
        print(f"Error: Candidate signals file not found at {signals_file}")
        sys.exit(1)

    df = pd.read_parquet(signals_file)
    print(f"Loaded {len(df)} candidate signals from candidate_signals.parquet")

    # Build the LangGraph
    print("Building LangGraph evidence workflow...")
    graph = build_graph()

    # Output directory for evidence packets
    out_dir = paths.processed_evidence
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    skipped = []

    for idx, row in df.iterrows():
        drug_id = row['drug_id']
        event_id = row['event_id']
        signal_id = f"{drug_id}__{event_id}"
        packet_path = out_dir / f"{signal_id}.json"
        if args.resume and _packet_has_real_synthesis(packet_path):
            skipped.append(signal_id)
            print(f"[{idx+1}/{len(df)}] Skipping existing real packet for {signal_id}.")
            continue

        print(f"[{idx+1}/{len(df)}] Running evidence workflow for {signal_id}...")

        try:
            metrics = _row_to_metrics(row)

            packet = run_signal(
                drug_id=drug_id,
                event_id=event_id,
                signal_metrics=metrics,
                evidence_window=metrics.evidence_window,
                paths=paths,
                graph=graph
            )

            packet_path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
            print(f"  Saved evidence packet to {packet_path.name}")
            print(f"  Evidence Grade: {packet.evidence_grade}, Triage Status: {packet.triage_status}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            failures.append((signal_id, str(e)))
            print(f"  Error running workflow for {signal_id}: {e}")

    completed = len(df) - len(failures) - len(skipped)
    if failures:
        print("\nEvidence workflow completed with failures.")
        print(
            f"Completed {completed}/{len(df)} selected signals successfully "
            f"({len(skipped)} skipped)."
        )
        print("Failed signals:")
        for signal_id, error in failures:
            print(f"  - {signal_id}: {error}")
        return 1

    if skipped:
        print(f"\nSkipped {len(skipped)} existing real evidence packet(s).")
    runnable_count = len(df) - len(skipped)
    if runnable_count == 0:
        print("No signals needed rerun.")
    else:
        print(f"All {runnable_count} selected evidence workflows completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
