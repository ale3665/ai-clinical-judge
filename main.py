"""Run the AI Clinical Judge evaluation pipeline.

Example:
    python main.py \
        --notes data/notes.csv \
        --gold data/gold_labels.csv \
        --outputs data/labeler_outputs_v1.jsonl \
        --synonyms configs/concept_synonyms.json \
        --output-dir reports
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluation.metrics import bootstrap_metric_ci, failed_cases, metrics_by_concept_type, summarize_judgements
from judge.io_utils import ensure_dir, load_gold_labels, load_labeler_outputs, load_notes, write_json
from judge.ontology_matcher import load_synonyms
from judge.rule_judge import evaluate_dataset


def maybe_log_mlflow(summary: dict, report_path: Path, failed_cases_path: Path) -> None:
    """Optionally log metrics and artifacts to MLflow."""
    try:
        import mlflow
    except ImportError:
        print("MLflow is not installed; skipping MLflow logging.")
        return

    run_name = f"judge_eval_{summary.get('model_version', 'unknown_model')}"
    with mlflow.start_run(run_name=run_name):
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))
            else:
                mlflow.log_param(key, value)
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(failed_cases_path))


def run_pipeline(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.output_dir)
    notes_df = load_notes(args.notes)
    gold_df = load_gold_labels(args.gold)
    predictions_df = load_labeler_outputs(args.outputs)
    synonyms = load_synonyms(args.synonyms)

    judgements = evaluate_dataset(notes_df, gold_df, predictions_df, synonyms)
    summary = summarize_judgements(judgements)
    by_type = metrics_by_concept_type(judgements)
    failed = failed_cases(judgements)
    ci = bootstrap_metric_ci(judgements, metric_name="soft_f1")

    model_version = str(summary.get("model_version", "unknown_model"))
    report_path = out_dir / f"evaluation_report_{model_version}.csv"
    summary_path = out_dir / f"summary_metrics_{model_version}.json"
    by_type_path = out_dir / f"metrics_by_concept_type_{model_version}.csv"
    failed_path = out_dir / f"failed_cases_{model_version}.csv"
    ci_path = out_dir / f"bootstrap_ci_{model_version}.json"

    judgements.to_csv(report_path, index=False)
    by_type.to_csv(by_type_path, index=False)
    failed.to_csv(failed_path, index=False)
    write_json(summary_path, summary)
    write_json(ci_path, ci)

    print("\nAI Clinical Judge Evaluation Complete")
    print("=" * 45)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nSaved files:")
    print(f"- {report_path}")
    print(f"- {summary_path}")
    print(f"- {by_type_path}")
    print(f"- {failed_path}")
    print(f"- {ci_path}")

    if args.track_mlflow:
        maybe_log_mlflow(summary, report_path, failed_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate AI Labeler outputs with an AI Clinical Judge.")
    parser.add_argument("--notes", default="data/notes.csv", help="Path to notes CSV.")
    parser.add_argument("--gold", default="data/gold_labels.csv", help="Path to gold labels CSV.")
    parser.add_argument("--outputs", default="data/labeler_outputs_v1.jsonl", help="Path to AI Labeler outputs JSONL.")
    parser.add_argument("--synonyms", default="configs/concept_synonyms.json", help="Path to synonym config JSON.")
    parser.add_argument("--output-dir", default="reports", help="Directory for reports.")
    parser.add_argument("--track-mlflow", action="store_true", help="Log run to MLflow.")
    return parser


if __name__ == "__main__":
    run_pipeline(build_arg_parser().parse_args())
