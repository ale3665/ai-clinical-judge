"""Streamlit dashboard for AI Clinical Judge reports.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="AI Clinical Judge Dashboard", layout="wide")
st.title("AI Clinical Judge Dashboard")
st.caption("Evaluate AI Labeler outputs for clinical concept extraction.")

REPORTS_DIR = Path("reports")
report_files = sorted(REPORTS_DIR.glob("evaluation_report_*.csv"))
summary_files = sorted(REPORTS_DIR.glob("summary_metrics_*.json"))
comparison_path = REPORTS_DIR / "model_version_comparison.csv"

if not report_files:
    st.warning("No evaluation reports found. Run `python scripts/run_demo.py` first.")
    st.stop()

selected_report = st.sidebar.selectbox("Choose evaluation report", report_files, format_func=lambda p: p.name)
df = pd.read_csv(selected_report)

summary_file = REPORTS_DIR / selected_report.name.replace("evaluation_report_", "summary_metrics_").replace(".csv", ".json")
summary = {}
if summary_file.exists():
    summary = json.loads(summary_file.read_text(encoding="utf-8"))

st.subheader("Model Summary")
cols = st.columns(5)
cols[0].metric("Strict F1", summary.get("strict_f1", "N/A"))
cols[1].metric("Soft F1", summary.get("soft_f1", "N/A"))
cols[2].metric("Hallucination Rate", summary.get("hallucination_rate", "N/A"))
cols[3].metric("Missing Rate", summary.get("missing_rate", "N/A"))
cols[4].metric("High-Risk Rate", summary.get("unsafe_or_high_risk_rate", "N/A"))

st.subheader("Verdict Distribution")
verdict_counts = df["verdict"].value_counts().rename_axis("verdict").reset_index(name="count")
st.bar_chart(verdict_counts.set_index("verdict"))

st.subheader("Filters")
left, middle, right = st.columns(3)
verdict_options = sorted(df["verdict"].dropna().unique().tolist())
risk_options = sorted(df["clinical_risk"].dropna().unique().tolist())
type_options = sorted(set(df.get("gold_type", pd.Series(dtype=str)).dropna().tolist() + df.get("pred_type", pd.Series(dtype=str)).dropna().tolist()))

selected_verdicts = left.multiselect("Verdict", verdict_options, default=verdict_options)
selected_risks = middle.multiselect("Clinical risk", risk_options, default=risk_options)
selected_types = right.multiselect("Concept type", type_options, default=type_options)

filtered = df[df["verdict"].isin(selected_verdicts) & df["clinical_risk"].isin(selected_risks)].copy()
if selected_types:
    concept_type = filtered["gold_type"].where(filtered["gold_type"].astype(str) != "", filtered["pred_type"])
    filtered = filtered[concept_type.isin(selected_types)]

st.subheader("Judge Decisions")
st.dataframe(filtered, use_container_width=True)

st.subheader("High-Risk / Review Queue")
review_queue = df[(df["clinical_risk"] == "high") | (df["verdict"].isin(["hallucinated", "missing", "incorrect"]))]
st.dataframe(review_queue, use_container_width=True)

if comparison_path.exists():
    st.subheader("Model Version Comparison")
    comparison = pd.read_csv(comparison_path)
    st.dataframe(comparison, use_container_width=True)
    if "model_version" in comparison.columns and "soft_f1" in comparison.columns:
        st.line_chart(comparison.set_index("model_version")[["soft_f1", "strict_f1"]])
