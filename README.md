# climate-risk-microtool

Transparent climate risk scoring with uncertainty quantification.

## What it does
- Trains an interpretable climate-risk model (calibrated gradient boosting).
- Calibrates a risk threshold against validation recall/F1 targets.
- Estimates uncertainty using bootstrap confidence intervals.
- Retrieves framework sources using Tavily (fallback built-in if key missing).
- Generates explanation/disclaimer/ethical-risk text using Gemini (fallback built-in if key missing).
- Exposes everything in a Streamlit demo app.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Optional API Keys
```bash
export TAVILY_API_KEY="..."
export GEMINI_API_KEY="..."
```

## Build artifacts
```bash
python climate_risk_tool.py
```

Produces:
- `risk_scores.csv`
- `uncertainty_intervals.json`
- `sources.json`

## Run demo
```bash
streamlit run app.py
```
