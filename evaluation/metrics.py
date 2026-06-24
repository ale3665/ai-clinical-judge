"""Metrics for AI Labeler evaluation using AI Judge outputs."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


PREDICTION_VERDICTS = {"correct", "partially_correct", "incorrect", "hallucinated"}
GOLD_VERDICTS = {"correct", "partially_correct", "incorrect", "missing"}


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_judgements(judgements: pd.DataFrame) -> Dict[str, float | int | str]:
    """Calculate model-level metrics from judge decisions.

    Strict metrics count only fully correct concepts as true positives.
    Soft metrics give partial credit to partially correct concepts.
    """
    if judgements.empty:
        return {
            "model_version": "unknown_model",
            "total_judge_rows": 0,
            "strict_precision": 0.0,
            "strict_recall": 0.0,
            "strict_f1": 0.0,
            "soft_precision": 0.0,
            "soft_recall": 0.0,
            "soft_f1": 0.0,
            "hallucination_rate": 0.0,
            "missing_rate": 0.0,
            "unsafe_or_high_risk_rate": 0.0,
        }

    verdicts = judgements["verdict"].fillna("").astype(str)
    model_version = judgements["model_version"].mode().iloc[0] if "model_version" in judgements.columns else "unknown_model"

    correct = int((verdicts == "correct").sum())
    partial = int((verdicts == "partially_correct").sum())
    incorrect = int((verdicts == "incorrect").sum())
    hallucinated = int((verdicts == "hallucinated").sum())
    missing = int((verdicts == "missing").sum())

    predicted_count = correct + partial + incorrect + hallucinated
    gold_count = correct + partial + incorrect + missing

    strict_precision = _safe_divide(correct, predicted_count)
    strict_recall = _safe_divide(correct, gold_count)
    strict_f1 = _safe_divide(2 * strict_precision * strict_recall, strict_precision + strict_recall)

    soft_tp = correct + 0.5 * partial
    soft_precision = _safe_divide(soft_tp, predicted_count)
    soft_recall = _safe_divide(soft_tp, gold_count)
    soft_f1 = _safe_divide(2 * soft_precision * soft_recall, soft_precision + soft_recall)

    clinical_risk = judgements.get("clinical_risk", pd.Series(dtype=str)).fillna("").astype(str)
    high_risk = int((clinical_risk == "high").sum())

    return {
        "model_version": model_version,
        "total_judge_rows": int(len(judgements)),
        "predicted_concepts": int(predicted_count),
        "gold_concepts": int(gold_count),
        "correct": correct,
        "partially_correct": partial,
        "incorrect": incorrect,
        "hallucinated": hallucinated,
        "missing": missing,
        "high_risk_errors": high_risk,
        "strict_precision": round(strict_precision, 4),
        "strict_recall": round(strict_recall, 4),
        "strict_f1": round(strict_f1, 4),
        "soft_precision": round(soft_precision, 4),
        "soft_recall": round(soft_recall, 4),
        "soft_f1": round(soft_f1, 4),
        "hallucination_rate": round(_safe_divide(hallucinated, predicted_count), 4),
        "missing_rate": round(_safe_divide(missing, gold_count), 4),
        "unsafe_or_high_risk_rate": round(_safe_divide(high_risk, len(judgements)), 4),
        "average_judge_score": round(float(pd.to_numeric(judgements["score"], errors="coerce").fillna(0).mean()), 4),
    }


def metrics_by_concept_type(judgements: pd.DataFrame) -> pd.DataFrame:
    """Return metrics grouped by concept type."""
    if judgements.empty:
        return pd.DataFrame()

    df = judgements.copy()
    df["concept_type"] = df["gold_type"].where(df["gold_type"].astype(str) != "", df["pred_type"])
    rows = []
    for concept_type, group in df.groupby("concept_type"):
        row = summarize_judgements(group)
        row["concept_type"] = concept_type
        rows.append(row)
    return pd.DataFrame(rows).sort_values("soft_f1", ascending=False)


def failed_cases(judgements: pd.DataFrame) -> pd.DataFrame:
    """Return cases that need review."""
    if judgements.empty:
        return judgements
    mask = (
        judgements["verdict"].isin(["partially_correct", "incorrect", "hallucinated", "missing"])
        | (judgements["clinical_risk"] == "high")
    )
    cols = [
        "note_id",
        "model_version",
        "verdict",
        "score",
        "clinical_risk",
        "error_type",
        "pred_span",
        "gold_span",
        "pred_type",
        "gold_type",
        "pred_concept",
        "gold_concept",
        "explanation",
    ]
    available = [c for c in cols if c in judgements.columns]
    return judgements.loc[mask, available].sort_values(["clinical_risk", "score"], ascending=[False, True])


def compare_model_summaries(summary_frames: Dict[str, Dict[str, float | int | str]]) -> pd.DataFrame:
    """Create a comparison DataFrame from multiple summary dictionaries."""
    return pd.DataFrame(summary_frames.values()).sort_values("model_version")


def bootstrap_metric_ci(
    judgements: pd.DataFrame,
    metric_name: str = "soft_f1",
    n_bootstrap: int = 200,
    random_seed: int = 42,
) -> Dict[str, float]:
    """Simple bootstrap confidence interval by note_id.

    This is helpful when reporting whether one model version appears better.
    """
    if judgements.empty or "note_id" not in judgements.columns:
        return {"metric": metric_name, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = np.random.default_rng(random_seed)
    note_ids = judgements["note_id"].dropna().unique()
    values = []
    for _ in range(n_bootstrap):
        sampled_notes = rng.choice(note_ids, size=len(note_ids), replace=True)
        sampled = pd.concat([judgements[judgements["note_id"] == nid] for nid in sampled_notes], ignore_index=True)
        values.append(float(summarize_judgements(sampled).get(metric_name, 0.0)))

    return {
        "metric": metric_name,
        "mean": round(float(np.mean(values)), 4),
        "ci_low": round(float(np.percentile(values, 2.5)), 4),
        "ci_high": round(float(np.percentile(values, 97.5)), 4),
    }
