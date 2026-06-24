"""Input/output helpers for the AI Clinical Judge project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


REQUIRED_GOLD_COLUMNS = {
    "note_id",
    "gold_id",
    "span",
    "concept_type",
    "normalized_concept",
}

REQUIRED_NOTE_COLUMNS = {"note_id", "note_text"}


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_notes(path: str | Path) -> pd.DataFrame:
    """Load clinical notes from CSV."""
    df = pd.read_csv(path)
    missing = REQUIRED_NOTE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"notes file is missing columns: {sorted(missing)}")
    return df


def load_gold_labels(path: str | Path) -> pd.DataFrame:
    """Load gold-standard labels from CSV."""
    df = pd.read_csv(path)
    missing = REQUIRED_GOLD_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"gold labels file is missing columns: {sorted(missing)}")
    if "risk_level" not in df.columns:
        df["risk_level"] = "medium"
    if "cui" not in df.columns:
        df["cui"] = ""
    return df.fillna("")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSONL file as a list of dictionaries."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc
    return rows


def load_labeler_outputs(path: str | Path) -> pd.DataFrame:
    """Load labeler outputs JSONL and flatten extractions into one row per prediction."""
    records = load_jsonl(path)
    rows: List[Dict[str, Any]] = []

    for record in records:
        note_id = record.get("note_id")
        model_version = record.get("model_version", "unknown_model")
        extractions = record.get("extractions", [])
        if not note_id:
            raise ValueError("Every JSONL record must include note_id")
        if not isinstance(extractions, list):
            raise ValueError(f"extractions must be a list for note_id={note_id}")

        for idx, ext in enumerate(extractions):
            rows.append(
                {
                    "model_version": model_version,
                    "note_id": note_id,
                    "pred_id": f"{note_id}_P{idx + 1:03d}",
                    "span": ext.get("span", ""),
                    "concept_type": ext.get("concept_type", ""),
                    "normalized_concept": ext.get("normalized_concept", ""),
                    "evidence": ext.get("evidence", ""),
                    "confidence": ext.get("confidence", ""),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "model_version",
                "note_id",
                "pred_id",
                "span",
                "concept_type",
                "normalized_concept",
                "evidence",
                "confidence",
            ]
        )
    return pd.DataFrame(rows).fillna("")


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    """Write JSON with indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write a sequence of dictionaries as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
