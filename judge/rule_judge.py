"""Rule-based judge for clinical concept extraction outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from judge.ontology_matcher import (
    SynonymMap,
    any_negation_before_span,
    concepts_equivalent,
    normalize_text,
    phrase_supported_by_note,
    similarity,
)


@dataclass
class MatchScore:
    total: float
    span_score: float
    type_score: float
    concept_score: float
    evidence_score: float


def _safe_value(row: pd.Series, key: str, default: str = "") -> str:
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return str(value)


def _risk_from_error(verdict: str, gold_risk: str, concept_type: str, negated: bool) -> str:
    """Assign clinical risk for the judge output."""
    gold_risk_norm = normalize_text(gold_risk)
    concept_type_norm = normalize_text(concept_type)

    if verdict in {"correct"}:
        return "low"
    if negated:
        return "medium"
    if gold_risk_norm == "high":
        return "high"
    if verdict in {"hallucinated", "missing"} and concept_type_norm in {
        "diagnosis",
        "medication",
        "procedure",
        "lab",
    }:
        return "high"
    if verdict in {"incorrect", "missing", "hallucinated"}:
        return "medium"
    return "low"


def score_prediction_against_gold(
    pred: pd.Series,
    gold: pd.Series,
    note_text: str,
    synonym_map: SynonymMap,
) -> MatchScore:
    """Score one prediction against one gold label.

    We give separate scores for span, type, normalization, and evidence support.
    The weights can be changed in this function depending on the project goal.
    """
    pred_span = _safe_value(pred, "span")
    gold_span = _safe_value(gold, "span")
    pred_type = normalize_text(_safe_value(pred, "concept_type"))
    gold_type = normalize_text(_safe_value(gold, "concept_type"))
    pred_concept = _safe_value(pred, "normalized_concept")
    gold_concept = _safe_value(gold, "normalized_concept")
    pred_cui = _safe_value(pred, "cui")
    gold_cui = _safe_value(gold, "cui")

    span_score = similarity(pred_span, gold_span)
    type_score = 1.0 if pred_type and pred_type == gold_type else 0.0
    concept_score = 1.0 if concepts_equivalent(
        predicted=pred_concept,
        gold=gold_concept,
        predicted_cui=pred_cui,
        gold_cui=gold_cui,
        synonym_map=synonym_map,
    ) else similarity(pred_concept, gold_concept)

    evidence_text = _safe_value(pred, "evidence") or pred_span
    supported = phrase_supported_by_note(pred_span, note_text) or phrase_supported_by_note(evidence_text, note_text)
    evidence_score = 1.0 if supported else 0.0

    # The concept score is weighted slightly higher because the project is concept extraction.
    total = 0.30 * span_score + 0.25 * type_score + 0.35 * concept_score + 0.10 * evidence_score
    return MatchScore(total, span_score, type_score, concept_score, evidence_score)


def classify_match(
    pred: pd.Series,
    gold: pd.Series,
    note_text: str,
    score: MatchScore,
    synonym_map: SynonymMap,
) -> Dict[str, str | float]:
    """Classify a matched prediction/gold pair into a judge verdict."""
    pred_span = _safe_value(pred, "span")
    gold_span = _safe_value(gold, "span")
    pred_type = normalize_text(_safe_value(pred, "concept_type"))
    gold_type = normalize_text(_safe_value(gold, "concept_type"))
    pred_concept = _safe_value(pred, "normalized_concept")
    gold_concept = _safe_value(gold, "normalized_concept")

    pred_negated = any_negation_before_span(note_text, pred_span)
    gold_risk = _safe_value(gold, "risk_level", "medium")

    errors: List[str] = []
    if score.span_score < 0.90:
        errors.append("span_error")
    if pred_type != gold_type:
        errors.append("concept_type_error")
    if score.concept_score < 0.90:
        errors.append("normalization_error")
    if score.evidence_score < 1.0:
        errors.append("evidence_error")
    if pred_negated:
        errors.append("negation_error")

    if score.span_score >= 0.90 and pred_type == gold_type and score.concept_score >= 0.90 and not pred_negated:
        verdict = "correct"
        judge_score = 1.0
        error_type = "none"
        explanation = "The extracted span, concept type, and normalized concept match the gold label."
    elif score.total >= 0.62:
        verdict = "partially_correct"
        judge_score = round(score.total, 3)
        error_type = "+".join(errors) if errors else "partial_match"
        explanation = (
            "The prediction is clinically related to the gold label, but it has one or more issues: "
            f"{error_type}."
        )
    else:
        verdict = "incorrect"
        judge_score = round(score.total, 3)
        error_type = "+".join(errors) if errors else "incorrect_match"
        explanation = (
            "The prediction does not sufficiently match the gold label based on span, type, "
            "normalization, and evidence support."
        )

    return {
        "verdict": verdict,
        "score": judge_score,
        "error_type": error_type,
        "clinical_risk": _risk_from_error(verdict, gold_risk, gold_type, pred_negated),
        "explanation": explanation,
        "span_score": round(score.span_score, 3),
        "type_score": round(score.type_score, 3),
        "concept_score": round(score.concept_score, 3),
        "evidence_score": round(score.evidence_score, 3),
        "match_score": round(score.total, 3),
    }


def _best_greedy_matches(
    predictions: pd.DataFrame,
    golds: pd.DataFrame,
    note_text: str,
    synonym_map: SynonymMap,
    min_match_score: float = 0.35,
) -> Tuple[List[Tuple[int, int, MatchScore]], List[int], List[int]]:
    """Greedily match predictions to gold labels by best score.

    Returns:
        matches: list of (prediction_index, gold_index, MatchScore)
        unmatched_pred_indices
        unmatched_gold_indices
    """
    candidate_pairs: List[Tuple[float, int, int, MatchScore]] = []

    for pred_idx, pred in predictions.iterrows():
        for gold_idx, gold in golds.iterrows():
            score = score_prediction_against_gold(pred, gold, note_text, synonym_map)
            if score.total >= min_match_score:
                candidate_pairs.append((score.total, pred_idx, gold_idx, score))

    candidate_pairs.sort(key=lambda x: x[0], reverse=True)

    used_preds = set()
    used_golds = set()
    matches: List[Tuple[int, int, MatchScore]] = []

    for _, pred_idx, gold_idx, score in candidate_pairs:
        if pred_idx in used_preds or gold_idx in used_golds:
            continue
        matches.append((pred_idx, gold_idx, score))
        used_preds.add(pred_idx)
        used_golds.add(gold_idx)

    unmatched_preds = [idx for idx in predictions.index if idx not in used_preds]
    unmatched_golds = [idx for idx in golds.index if idx not in used_golds]
    return matches, unmatched_preds, unmatched_golds


def evaluate_note(
    note_id: str,
    note_text: str,
    predictions: pd.DataFrame,
    golds: pd.DataFrame,
    synonym_map: Optional[SynonymMap] = None,
) -> List[Dict[str, object]]:
    """Evaluate all predictions for one note."""
    synonym_map = synonym_map or {}
    results: List[Dict[str, object]] = []

    matches, unmatched_preds, unmatched_golds = _best_greedy_matches(
        predictions=predictions,
        golds=golds,
        note_text=note_text,
        synonym_map=synonym_map,
    )

    for pred_idx, gold_idx, match_score in matches:
        pred = predictions.loc[pred_idx]
        gold = golds.loc[gold_idx]
        classification = classify_match(pred, gold, note_text, match_score, synonym_map)
        results.append(
            {
                "note_id": note_id,
                "model_version": _safe_value(pred, "model_version", "unknown_model"),
                "pred_id": _safe_value(pred, "pred_id"),
                "gold_id": _safe_value(gold, "gold_id"),
                "pred_span": _safe_value(pred, "span"),
                "gold_span": _safe_value(gold, "span"),
                "pred_type": _safe_value(pred, "concept_type"),
                "gold_type": _safe_value(gold, "concept_type"),
                "pred_concept": _safe_value(pred, "normalized_concept"),
                "gold_concept": _safe_value(gold, "normalized_concept"),
                "evidence": _safe_value(pred, "evidence"),
                "confidence": _safe_value(pred, "confidence"),
                **classification,
            }
        )

    for pred_idx in unmatched_preds:
        pred = predictions.loc[pred_idx]
        pred_span = _safe_value(pred, "span")
        pred_type = _safe_value(pred, "concept_type")
        pred_negated = any_negation_before_span(note_text, pred_span)
        supported = phrase_supported_by_note(pred_span, note_text) or phrase_supported_by_note(_safe_value(pred, "evidence"), note_text)
        error_type = "unsupported_hallucination" if not supported else "unmatched_prediction"
        if pred_negated:
            error_type += "+negation_error"

        results.append(
            {
                "note_id": note_id,
                "model_version": _safe_value(pred, "model_version", "unknown_model"),
                "pred_id": _safe_value(pred, "pred_id"),
                "gold_id": "",
                "pred_span": pred_span,
                "gold_span": "",
                "pred_type": pred_type,
                "gold_type": "",
                "pred_concept": _safe_value(pred, "normalized_concept"),
                "gold_concept": "",
                "evidence": _safe_value(pred, "evidence"),
                "confidence": _safe_value(pred, "confidence"),
                "verdict": "hallucinated",
                "score": 0.0,
                "error_type": error_type,
                "clinical_risk": _risk_from_error("hallucinated", "medium", pred_type, pred_negated),
                "explanation": "The prediction was not matched to a gold-standard concept for this note.",
                "span_score": 0.0,
                "type_score": 0.0,
                "concept_score": 0.0,
                "evidence_score": 1.0 if supported else 0.0,
                "match_score": 0.0,
            }
        )

    for gold_idx in unmatched_golds:
        gold = golds.loc[gold_idx]
        gold_risk = _safe_value(gold, "risk_level", "medium")
        gold_type = _safe_value(gold, "concept_type")
        results.append(
            {
                "note_id": note_id,
                "model_version": predictions["model_version"].iloc[0] if not predictions.empty else "unknown_model",
                "pred_id": "",
                "gold_id": _safe_value(gold, "gold_id"),
                "pred_span": "",
                "gold_span": _safe_value(gold, "span"),
                "pred_type": "",
                "gold_type": gold_type,
                "pred_concept": "",
                "gold_concept": _safe_value(gold, "normalized_concept"),
                "evidence": "",
                "confidence": "",
                "verdict": "missing",
                "score": 0.0,
                "error_type": "missing_concept",
                "clinical_risk": _risk_from_error("missing", gold_risk, gold_type, False),
                "explanation": "The gold-standard concept was not extracted by the AI Labeler.",
                "span_score": 0.0,
                "type_score": 0.0,
                "concept_score": 0.0,
                "evidence_score": 0.0,
                "match_score": 0.0,
            }
        )

    return results


def evaluate_dataset(
    notes_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    synonym_map: Optional[SynonymMap] = None,
) -> pd.DataFrame:
    """Evaluate a full dataset and return one row per judge decision."""
    synonym_map = synonym_map or {}
    all_results: List[Dict[str, object]] = []

    note_lookup = dict(zip(notes_df["note_id"], notes_df["note_text"]))
    note_ids = sorted(set(gold_df["note_id"]) | set(predictions_df["note_id"]) | set(notes_df["note_id"]))

    for note_id in note_ids:
        note_text = note_lookup.get(note_id, "")
        note_predictions = predictions_df[predictions_df["note_id"] == note_id].copy()
        note_golds = gold_df[gold_df["note_id"] == note_id].copy()
        all_results.extend(evaluate_note(note_id, note_text, note_predictions, note_golds, synonym_map))

    return pd.DataFrame(all_results)
