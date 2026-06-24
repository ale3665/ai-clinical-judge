"""Lightweight ontology/synonym matching utilities.

This is intentionally simple for an MVP. In a real clinical system, this module
can be replaced or extended with UMLS, SNOMED CT, RxNorm, LOINC, ICD-10, etc.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - fallback for minimal installs
    fuzz = None
    import difflib


SynonymMap = Dict[str, Set[str]]


def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    text = str(text or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity(a: str, b: str) -> float:
    """Return a 0-1 fuzzy similarity score."""
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    if fuzz is not None:
        return fuzz.token_set_ratio(a_norm, b_norm) / 100.0
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


def load_synonyms(path: str | Path | None) -> SynonymMap:
    """Load a concept synonym dictionary and make it bidirectional."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}

    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)

    synonym_map: SynonymMap = {}
    for canonical, synonyms in raw.items():
        canonical_norm = normalize_text(canonical)
        values = {canonical_norm} | {normalize_text(s) for s in synonyms}
        for value in values:
            synonym_map.setdefault(value, set()).update(values)
    return synonym_map


def are_synonyms(a: str, b: str, synonym_map: SynonymMap) -> bool:
    """Check whether two concepts appear in the local synonym map."""
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    return b_norm in synonym_map.get(a_norm, set()) or a_norm in synonym_map.get(b_norm, set())


def concepts_equivalent(
    predicted: str,
    gold: str,
    predicted_cui: str = "",
    gold_cui: str = "",
    synonym_map: SynonymMap | None = None,
    fuzzy_threshold: float = 0.92,
) -> bool:
    """Return True if concepts are equivalent by CUI, exact, synonym, or fuzzy match."""
    synonym_map = synonym_map or {}
    pred_cui = str(predicted_cui or "").strip()
    gold_cui = str(gold_cui or "").strip()

    if pred_cui and gold_cui and pred_cui == gold_cui:
        return True
    if normalize_text(predicted) == normalize_text(gold):
        return True
    if are_synonyms(predicted, gold, synonym_map):
        return True
    return similarity(predicted, gold) >= fuzzy_threshold


def phrase_supported_by_note(phrase: str, note_text: str, fuzzy_threshold: float = 0.90) -> bool:
    """Check whether a predicted phrase is directly or approximately supported in the note."""
    phrase_norm = normalize_text(phrase)
    note_norm = normalize_text(note_text)
    if not phrase_norm:
        return False
    if phrase_norm in note_norm:
        return True

    tokens = phrase_norm.split()
    if len(tokens) == 1:
        return phrase_norm in set(note_norm.split())

    # Compare against windows of the same token length +/- 2.
    note_tokens = note_norm.split()
    best = 0.0
    for window_size in range(max(1, len(tokens) - 2), len(tokens) + 3):
        for i in range(0, max(0, len(note_tokens) - window_size + 1)):
            window = " ".join(note_tokens[i : i + window_size])
            best = max(best, similarity(phrase_norm, window))
            if best >= fuzzy_threshold:
                return True
    return False


def any_negation_before_span(note_text: str, span: str, window_chars: int = 60) -> bool:
    """Very small negation check for demo purposes.

    This catches simple examples like "denies fever". Replace with a clinical
    negation detector such as NegEx/ConText in a production system.
    """
    note_lower = str(note_text or "").lower()
    span_lower = str(span or "").lower()
    idx = note_lower.find(span_lower)
    if idx == -1:
        return False
    before = note_lower[max(0, idx - window_chars) : idx]
    negation_terms = ["denies", "denied", "no ", "without", "negative for", "not"]
    return any(term in before for term in negation_terms)
