"""Run the expanded fake-data trial for labeler_v1, labeler_v2, and labeler_v3."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from evaluation.metrics import bootstrap_metric_ci, failed_cases, metrics_by_concept_type, summarize_judgements
from judge.io_utils import ensure_dir, load_gold_labels, load_labeler_outputs, load_notes, write_json
from judge.ontology_matcher import load_synonyms
from judge.rule_judge import evaluate_dataset


def evaluate_one(notes_df, gold_df, output_path: str, synonyms, output_dir: Path) -> dict:
    preds_df = load_labeler_outputs(output_path)
    judgements = evaluate_dataset(notes_df, gold_df, preds_df, synonyms)
    summary = summarize_judgements(judgements)
    model_version = summary.get("model_version", "unknown_model")

    judgements.to_csv(output_dir / f"evaluation_report_{model_version}.csv", index=False)
    metrics_by_concept_type(judgements).to_csv(output_dir / f"metrics_by_concept_type_{model_version}.csv", index=False)
    failed_cases(judgements).to_csv(output_dir / f"failed_cases_{model_version}.csv", index=False)
    write_json(output_dir / f"summary_metrics_{model_version}.json", summary)
    write_json(output_dir / f"bootstrap_ci_{model_version}.json", bootstrap_metric_ci(judgements, metric_name="soft_f1"))
    return summary


if __name__ == "__main__":
    out_dir = ensure_dir("reports/expanded_trial")
    notes_df = load_notes("data/notes.csv")
    gold_df = load_gold_labels("data/gold_labels.csv")
    synonyms = load_synonyms("configs/concept_synonyms.json")

    output_files = [
        "data/labeler_outputs_v1.jsonl",
        "data/labeler_outputs_v2.jsonl",
        "data/labeler_outputs_v3.jsonl",
    ]

    summaries = [evaluate_one(notes_df, gold_df, path, synonyms, out_dir) for path in output_files]
    comparison = pd.DataFrame(summaries).sort_values("model_version")
    comparison.to_csv(out_dir / "model_version_comparison.csv", index=False)

    print("\nExpanded Fake Data Trial Complete")
    print("=" * 45)
    print(comparison.to_string(index=False))
    print(f"\nSaved detailed reports in: {out_dir}")
