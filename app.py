from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from climate_risk_tool import run_pipeline

st.set_page_config(page_title="Climate Risk Microtool", page_icon="🌍", layout="centered")
st.title("🌍 Climate Risk Microtool")
st.caption("Transparent climate risk scoring with uncertainty quantification")

if st.button("Generate / Refresh Artifacts"):
    run_pipeline()
    st.success("Artifacts rebuilt: risk_scores.csv, uncertainty_intervals.json, sources.json")

scores_path = Path("risk_scores.csv")
intervals_path = Path("uncertainty_intervals.json")
sources_path = Path("sources.json")

if scores_path.exists():
    scores = pd.read_csv(scores_path)
    st.subheader("Risk Scores")
    st.dataframe(scores, use_container_width=True)
else:
    st.info("No risk_scores.csv found yet. Click the button above.")

if intervals_path.exists():
    st.subheader("Uncertainty Intervals")
    st.json(json.loads(intervals_path.read_text()))

if sources_path.exists():
    st.subheader("Framework Sources")
    st.json(json.loads(sources_path.read_text()))
