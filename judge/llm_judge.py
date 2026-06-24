"""Optional LLM judge for difficult/borderline cases.

Important: Do not send identifiable patient data to an external LLM API. Use only
synthetic/de-identified data, or run the model in an approved secure environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


DEFAULT_ALLOWED_VERDICTS = {"correct", "partially_correct", "incorrect", "missing", "hallucinated", "unsafe"}


def load_prompt_template(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_judge_prompt(
    note_text: str,
    gold: Dict[str, Any],
    prediction: Dict[str, Any],
    template: str,
) -> str:
    """Fill the LLM judge prompt template."""
    return template.format(
        note_text=note_text,
        gold=json.dumps(gold, ensure_ascii=False, indent=2),
        prediction=json.dumps(prediction, ensure_ascii=False, indent=2),
    )


def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON from an LLM response, with a small cleanup for fenced blocks."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response was not valid JSON: {text}") from exc

    verdict = str(data.get("verdict", "")).strip()
    if verdict not in DEFAULT_ALLOWED_VERDICTS:
        data["verdict"] = "incorrect"
        data["explanation"] = "LLM returned an invalid verdict; defaulted to incorrect."
    return data


def mock_llm_judge(note_text: str, gold: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    """A deterministic fallback that mimics an LLM output for demos/tests."""
    pred_concept = str(prediction.get("normalized_concept", "")).lower()
    gold_concept = str(gold.get("normalized_concept", "")).lower()
    pred_span = str(prediction.get("span", "")).lower()
    gold_span = str(gold.get("span", "")).lower()

    if pred_concept == gold_concept and pred_span == gold_span:
        return {
            "verdict": "correct",
            "score": 1.0,
            "error_type": "none",
            "clinical_risk": "low",
            "explanation": "Mock judge: exact span and concept match.",
        }
    if pred_concept in gold_concept or gold_concept in pred_concept or pred_span in gold_span or gold_span in pred_span:
        return {
            "verdict": "partially_correct",
            "score": 0.65,
            "error_type": "partial_semantic_match",
            "clinical_risk": "low",
            "explanation": "Mock judge: prediction appears related but incomplete or not exact.",
        }
    return {
        "verdict": "incorrect",
        "score": 0.0,
        "error_type": "semantic_mismatch",
        "clinical_risk": "medium",
        "explanation": "Mock judge: prediction does not match the gold concept.",
    }


def openai_llm_judge(prompt: str) -> Dict[str, Any]:
    """Call an OpenAI-compatible LLM endpoint and parse the JSON response.

    This function is optional. The rest of the project works without it.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the optional openai package to use openai_llm_judge.") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    model = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful clinical AI evaluator. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    return parse_json_response(content)


def judge_with_llm(
    note_text: str,
    gold: Dict[str, Any],
    prediction: Dict[str, Any],
    prompt_template: str,
    provider: str = "mock",
) -> Dict[str, Any]:
    """Run the optional LLM judge.

    provider="mock" is safe for local demos.
    provider="openai" calls the configured API.
    """
    if provider == "mock":
        return mock_llm_judge(note_text, gold, prediction)

    prompt = build_judge_prompt(note_text, gold, prediction, prompt_template)
    if provider == "openai":
        return openai_llm_judge(prompt)

    raise ValueError(f"Unsupported LLM provider: {provider}")
