"""Utilities for comparing multiple AI Labeler versions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from evaluation.metrics import summarize_judgements
from judge.io_utils import ensure_dir, load_gold_labels, load_labeler_outputs, load_notes, write_json
from judge.ontology_matcher import load_synonyms
from judge.rule_judge import evaluate_dataset


def evaluate_multiple_versions(
    notes_path: str,
    gold_path: str,
    output_paths: Iterable[str],
    synonyms_path: str | None,
    output_dir: str,
) -> pd.DataFrame:
    """Evaluate multiple labeler output files and save reports."""
    out_dir = ensure_dir(output_dir)
    notes_df = load_notes(notes_path)
    gold_df = load_gold_labels(gold_path)
    synonyms = load_synonyms(synonyms_path)

    summaries: Dict[str, Dict] = {}
    for output_path in output_paths:
        preds_df = load_labeler_outputs(output_path)
        judgements = evaluate_dataset(notes_df, gold_df, preds_df, synonyms)
        model_version = judgements["model_version"].mode().iloc[0]
        report_path = out_dir / f"evaluation_report_{model_version}.csv"
        summary_path = out_dir / f"summary_metrics_{model_version}.json"
        judgements.to_csv(report_path, index=False)
        summary = summarize_judgements(judgements)
        write_json(summary_path, summary)
        summaries[model_version] = summary

    comparison = pd.DataFrame(summaries.values()).sort_values("model_version")
    comparison.to_csv(out_dir / "model_version_comparison.csv", index=False)
    return comparison
