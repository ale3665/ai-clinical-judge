from judge.ontology_matcher import load_synonyms
from judge.io_utils import load_gold_labels, load_labeler_outputs, load_notes
from judge.rule_judge import evaluate_dataset
from evaluation.metrics import summarize_judgements


def test_demo_v2_has_better_soft_f1_than_v1():
    notes = load_notes("data/notes.csv")
    gold = load_gold_labels("data/gold_labels.csv")
    synonyms = load_synonyms("configs/concept_synonyms.json")

    v1 = evaluate_dataset(notes, gold, load_labeler_outputs("data/labeler_outputs_v1.jsonl"), synonyms)
    v2 = evaluate_dataset(notes, gold, load_labeler_outputs("data/labeler_outputs_v2.jsonl"), synonyms)

    assert summarize_judgements(v2)["soft_f1"] >= summarize_judgements(v1)["soft_f1"]


def test_evaluate_dataset_outputs_expected_columns():
    notes = load_notes("data/notes.csv")
    gold = load_gold_labels("data/gold_labels.csv")
    preds = load_labeler_outputs("data/labeler_outputs_v2.jsonl")
    synonyms = load_synonyms("configs/concept_synonyms.json")

    results = evaluate_dataset(notes, gold, preds, synonyms)
    required = {"note_id", "verdict", "score", "error_type", "clinical_risk"}
    assert required.issubset(set(results.columns))
