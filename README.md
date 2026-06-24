# AI Clinical Judge for Healthcare Concept Extraction

This project is a complete starter system for evaluating an existing **AI Labeler** that extracts clinical concepts from healthcare notes.

The AI Labeler is assumed to already exist. This project only builds the **AI Judge** that evaluates the Labeler output, tracks model performance, flags unsafe cases, and helps compare training/model versions.

---

## What this project does

Given:

1. Original de-identified clinical note
2. Gold-standard labels, when available
3. AI Labeler output

The AI Judge returns:

- `correct`
- `partially_correct`
- `incorrect`
- `hallucinated`
- `missing`
- clinical risk level
- error type
- explanation
- model-level metrics

Example judge output:

```json
{
  "note_id": "N001",
  "pred_span": "chest pain",
  "gold_span": "chest pain",
  "verdict": "correct",
  "score": 1.0,
  "error_type": "none",
  "clinical_risk": "low",
  "explanation": "The extracted span, concept type, and normalized concept match the gold label."
}
```

---

## Project structure

```text
ai_clinical_judge/
│
├── data/
│   ├── notes.csv
│   ├── gold_labels.csv
│   ├── labeler_outputs_v1.jsonl
│   └── labeler_outputs_v2.jsonl
│
├── configs/
│   └── concept_synonyms.json
│
├── judge/
│   ├── io_utils.py
│   ├── ontology_matcher.py
│   ├── rule_judge.py
│   └── llm_judge.py
│
├── evaluation/
│   ├── metrics.py
│   └── compare_versions.py
│
├── dashboard/
│   └── app.py
│
├── prompts/
│   └── judge_prompt.txt
│
├── reports/
│
├── scripts/
│   └── run_demo.py
│
├── tests/
│   └── test_rule_judge.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Step 1: Create the environment

From the project folder:

```bash
python -m venv .venv
```

Activate it.

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

---

## Understand the input files

### `data/notes.csv`

Contains the clinical notes.

```csv
note_id,note_text
N001,"Patient is a 58-year-old female with type 2 diabetes mellitus..."
```

### `data/gold_labels.csv`

Contains clinician/reviewer labels.

```csv
note_id,gold_id,span,concept_type,normalized_concept,cui,risk_level
N001,G001,type 2 diabetes mellitus,diagnosis,Type 2 diabetes mellitus,C0011860,medium
```

### `data/labeler_outputs_v1.jsonl`

Contains the AI Labeler output.

```json
{"model_version":"labeler_v1","note_id":"N001","extractions":[{"span":"diabetes","concept_type":"diagnosis","normalized_concept":"Diabetes mellitus","evidence":"with type 2 diabetes mellitus","confidence":0.81}]}
```

Use this format for your own Labeler outputs.

---

## Step 2: Run the judge on one model version

```bash
python main.py \
  --notes data/notes.csv \
  --gold data/gold_labels.csv \
  --outputs data/labeler_outputs_v1.jsonl \
  --synonyms configs/concept_synonyms.json \
  --output-dir reports
```

This creates:

```text
reports/evaluation_report_labeler_v1.csv
reports/summary_metrics_labeler_v1.json
reports/metrics_by_concept_type_labeler_v1.csv
reports/failed_cases_labeler_v1.csv
reports/bootstrap_ci_labeler_v1.json
```

---

## Step 3: Compare model/training versions

Run:

```bash
python scripts/run_demo.py
```

This evaluates both sample versions:

```text
labeler_v1
labeler_v2
```

and creates:

```text
reports/model_version_comparison.csv
```

This helps answer:

- Is the Labeler improving?
- Did F1 improve?
- Did hallucination rate decrease?
- Did missing concept rate decrease?
- Did high-risk error rate decrease?

---

## Step 4: Open the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard shows:

- Strict F1
- Soft F1
- hallucination rate
- missing rate
- high-risk error rate
- verdict distribution
- review queue
- model version comparison

---

## How the judge works

The AI Judge has three layers.

### 1. Rule-based judge

File:

```text
judge/rule_judge.py
```

It checks:

- span similarity
- concept type match
- normalized concept match
- synonym match
- evidence support
- missing concepts
- hallucinated predictions
- high-risk errors

### 2. Ontology/synonym matcher

File:

```text
judge/ontology_matcher.py
```

The MVP uses a local synonym dictionary:

```text
configs/concept_synonyms.json
```

Later, this can be replaced with UMLS, SNOMED CT, RxNorm, ICD-10, or LOINC matching.

### 3. Optional LLM judge

File:

```text
judge/llm_judge.py
```

Use the LLM judge only for harder cases, such as:

- partial matches
- ambiguous normalization
- possible hallucinations
- high-risk clinical concepts
- disagreement between rule judge and human reviewer

Do not send identifiable patient information to an external LLM API.

---

## Metrics explained

The project calculates:

### Strict metrics

Only `correct` counts as a true positive.

### Soft metrics

`correct` counts as 1.0 and `partially_correct` counts as 0.5.

### Safety metrics

The project also tracks:

- hallucination rate
- missing concept rate
- high-risk error rate
- average judge score

---

## How to use this with your real AI Labeler

Replace the sample files with your real files.

### Your real notes

```text
data/notes.csv
```

Required columns:

```text
note_id,note_text
```

### Your gold labels

```text
data/gold_labels.csv
```

Required columns:

```text
note_id,gold_id,span,concept_type,normalized_concept
```

Recommended columns:

```text
cui,risk_level
```

### Your Labeler outputs

```text
data/labeler_outputs_your_model.jsonl
```

Format:

```json
{
  "model_version": "your_labeler_v1",
  "note_id": "N001",
  "extractions": [
    {
      "span": "metformin",
      "concept_type": "medication",
      "normalized_concept": "Metformin",
      "evidence": "Patient is taking metformin daily",
      "confidence": 0.92
    }
  ]
}
```

Then run:

```bash
python main.py --outputs data/labeler_outputs_your_model.jsonl
```

---

## Recommended development workflow

Use this order:

```text
1. Run the demo project.
2. Replace notes.csv with your approved/de-identified notes.
3. Replace gold_labels.csv with clinician/reviewer labels.
4. Convert AI Labeler output to JSONL.
5. Run main.py.
6. Review failed_cases.csv.
7. Adjust the synonym/ontology matcher.
8. Compare model versions.
9. Add LLM judge for uncertain cases.
10. Build dashboard for your advisor/team.
```

---

## Human review queue

The judge flags cases for review when:

```text
verdict = hallucinated
verdict = missing
verdict = incorrect
clinical_risk = high
score is low
```

These appear in:

```text
reports/failed_cases_<model_version>.csv
```

This is the file you can give to a clinician/reviewer.

---

## Safety notes

For healthcare use:

- Use only de-identified or approved clinical data.
- Keep a human reviewer in the loop.
- Do not treat the AI Judge as a clinician.
- Do not send PHI to public APIs.
- Track dataset version, Labeler version, Judge version, prompt version, and reviewer decisions.
- Keep failed cases for error analysis.

---

## Disclaimer

This project is for research and educational purposes only. It is not intended for clinical diagnosis, treatment decisions, or direct patient care. The sample data included in this repository is synthetic and does not contain real patient information.
