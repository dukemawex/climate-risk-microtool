from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


ARTIFACT_RISK_SCORES = Path("risk_scores.csv")
ARTIFACT_INTERVALS = Path("uncertainty_intervals.json")
ARTIFACT_SOURCES = Path("sources.json")


@dataclass
class ModelBundle:
    model: CalibratedClassifierCV
    feature_names: list[str]


def _build_synthetic_dataset(n: int = 1400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "flood_exposure": rng.uniform(0, 1, n),
            "heat_days_per_year": rng.normal(40, 15, n).clip(0, 120),
            "drought_index": rng.uniform(0, 1, n),
            "insurance_gap": rng.uniform(0, 1, n),
            "adaptive_capacity": rng.uniform(0, 1, n),
            "income_stability": rng.uniform(0, 1, n),
        }
    )

    linear_score = (
        2.6 * df["flood_exposure"]
        + 0.015 * df["heat_days_per_year"]
        + 1.7 * df["drought_index"]
        + 1.5 * df["insurance_gap"]
        - 2.0 * df["adaptive_capacity"]
        - 1.2 * df["income_stability"]
        - 0.8
    )
    risk_prob = 1.0 / (1.0 + np.exp(-linear_score))
    df["high_risk"] = rng.binomial(1, risk_prob)
    return df


def train_model(seed: int = 7) -> ModelBundle:
    df = _build_synthetic_dataset()
    feature_names = [c for c in df.columns if c != "high_risk"]
    X_train, _, y_train, _ = train_test_split(
        df[feature_names], df["high_risk"], test_size=0.25, random_state=seed, stratify=df["high_risk"]
    )

    base_model = GradientBoostingClassifier(random_state=seed)
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=4)
    model.fit(X_train, y_train)
    return ModelBundle(model=model, feature_names=feature_names)


def bootstrap_confidence_intervals(
    training_df: pd.DataFrame,
    scenarios: pd.DataFrame,
    n_bootstrap: int = 120,
    seed: int = 11,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    feature_names = [c for c in training_df.columns if c != "high_risk"]

    probabilities = []
    for _ in range(n_bootstrap):
        idx = rng.choice(training_df.index, size=len(training_df), replace=True)
        sample = training_df.loc[idx]
        X = sample[feature_names]
        y = sample["high_risk"]

        model = GradientBoostingClassifier(random_state=int(rng.integers(0, 1_000_000)))
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv=3)
        calibrated.fit(X, y)
        probabilities.append(calibrated.predict_proba(scenarios[feature_names])[:, 1])

    stacked = np.vstack(probabilities)
    result: dict[str, dict[str, float]] = {}
    for i, sid in enumerate(scenarios["scenario_id"].tolist()):
        series = stacked[:, i]
        result[sid] = {
            "mean_probability": float(np.mean(series)),
            "ci_5": float(np.quantile(series, 0.05)),
            "ci_50": float(np.quantile(series, 0.50)),
            "ci_95": float(np.quantile(series, 0.95)),
        }
    return result


def calibrate_threshold(
    model: CalibratedClassifierCV, X_valid: pd.DataFrame, y_valid: pd.Series, minimum_recall: float = 0.80
) -> tuple[float, dict[str, float]]:
    probs = model.predict_proba(X_valid)[:, 1]
    thresholds = np.linspace(0.10, 0.90, 81)
    best = {"threshold": 0.5, "f1": -1.0, "recall": 0.0, "precision": 0.0}

    for thr in thresholds:
        pred = (probs >= thr).astype(int)
        tp = int(((pred == 1) & (y_valid == 1)).sum())
        fp = int(((pred == 1) & (y_valid == 0)).sum())
        fn = int(((pred == 0) & (y_valid == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        if recall >= minimum_recall and f1 > best["f1"]:
            best = {"threshold": float(thr), "f1": float(f1), "recall": float(recall), "precision": float(precision)}

    return best["threshold"], best


def fetch_climate_framework_sources(max_results: int = 5) -> dict[str, Any]:
    api_key = os.getenv("TAVILY_API_KEY")
    query = "climate risk assessment framework TCFD IPCC NGFS methodology"

    if not api_key:
        return {
            "query": query,
            "retrieval": "fallback",
            "results": [
                {
                    "title": "TCFD Recommendations",
                    "url": "https://www.fsb-tcfd.org/recommendations/",
                    "snippet": "Task Force guidance for governance, strategy, risk management, metrics and targets.",
                },
                {
                    "title": "IPCC AR6 Synthesis Report",
                    "url": "https://www.ipcc.ch/report/ar6/syr/",
                    "snippet": "Scientific assessment of climate hazards, vulnerabilities, and adaptation pathways.",
                },
                {
                    "title": "NGFS Climate Scenarios",
                    "url": "https://www.ngfs.net/ngfs-scenarios-portal/",
                    "snippet": "Scenario set for assessing transition and physical climate risks.",
                },
            ],
        }

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": max_results},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        return {
            "query": query,
            "retrieval": "tavily",
            "results": [
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("content", "")[:280],
                }
                for item in payload.get("results", [])
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {"query": query, "retrieval": "fallback_with_error", "error": str(exc), "results": []}


def generate_gemini_text(scenario: dict[str, float], probability: float) -> dict[str, str]:
    api_key = os.getenv("GEMINI_API_KEY")
    fallback = {
        "risk_explanation": (
            f"Estimated climate risk is {probability:.1%}. The dominant drivers are flood exposure, heat burden, and drought pressure "
            "offset by adaptive capacity and income stability."
        ),
        "policy_disclaimer": (
            "This output is a decision-support estimate, not regulatory or legal advice. "
            "Validate with local hazard maps, sector rules, and expert judgment before action."
        ),
        "ethical_risks": (
            "Potential harms include proxy discrimination, over-confidence in uncertain data, and resource allocation bias "
            "against already vulnerable communities."
        ),
    }

    if not api_key:
        return fallback

    prompt = (
        "You are assisting a transparent climate risk scoring tool. "
        f"Scenario attributes: {json.dumps(scenario)}. Predicted risk={probability:.3f}. "
        "Return concise JSON with keys risk_explanation, policy_disclaimer, ethical_risks."
    )
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={api_key}"
    )

    try:
        response = requests.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=25,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        maybe_json = json.loads(text)
        return {
            "risk_explanation": maybe_json.get("risk_explanation", fallback["risk_explanation"]),
            "policy_disclaimer": maybe_json.get("policy_disclaimer", fallback["policy_disclaimer"]),
            "ethical_risks": maybe_json.get("ethical_risks", fallback["ethical_risks"]),
        }
    except Exception:
        return fallback


def run_pipeline() -> None:
    training_df = _build_synthetic_dataset(n=1600, seed=12)
    feature_names = [c for c in training_df.columns if c != "high_risk"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        training_df[feature_names],
        training_df["high_risk"],
        test_size=0.25,
        random_state=9,
        stratify=training_df["high_risk"],
    )

    base_model = GradientBoostingClassifier(random_state=9)
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=4)
    model.fit(X_train, y_train)

    threshold, threshold_metrics = calibrate_threshold(model, X_valid, y_valid)

    scenarios = pd.DataFrame(
        [
            {
                "scenario_id": "coastal_city_a",
                "flood_exposure": 0.82,
                "heat_days_per_year": 68,
                "drought_index": 0.40,
                "insurance_gap": 0.55,
                "adaptive_capacity": 0.45,
                "income_stability": 0.62,
            },
            {
                "scenario_id": "agrarian_region_b",
                "flood_exposure": 0.25,
                "heat_days_per_year": 55,
                "drought_index": 0.78,
                "insurance_gap": 0.71,
                "adaptive_capacity": 0.32,
                "income_stability": 0.41,
            },
            {
                "scenario_id": "industrial_hub_c",
                "flood_exposure": 0.61,
                "heat_days_per_year": 43,
                "drought_index": 0.34,
                "insurance_gap": 0.37,
                "adaptive_capacity": 0.65,
                "income_stability": 0.72,
            },
        ]
    )

    probs = model.predict_proba(scenarios[feature_names])[:, 1]
    scenarios["risk_probability"] = probs
    scenarios["risk_label"] = np.where(scenarios["risk_probability"] >= threshold, "high", "moderate")

    llm_rows = []
    for row in scenarios.to_dict(orient="records"):
        llm_rows.append(generate_gemini_text(row, row["risk_probability"]))

    scenarios["risk_explanation"] = [x["risk_explanation"] for x in llm_rows]
    scenarios["policy_disclaimer"] = [x["policy_disclaimer"] for x in llm_rows]
    scenarios["ethical_risks"] = [x["ethical_risks"] for x in llm_rows]
    scenarios.to_csv(ARTIFACT_RISK_SCORES, index=False)

    intervals = {
        "threshold": threshold,
        "threshold_metrics": threshold_metrics,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap": bootstrap_confidence_intervals(training_df, scenarios[["scenario_id", *feature_names]]),
    }
    ARTIFACT_INTERVALS.write_text(json.dumps(intervals, indent=2))

    sources = fetch_climate_framework_sources()
    sources["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    ARTIFACT_SOURCES.write_text(json.dumps(sources, indent=2))


if __name__ == "__main__":
    run_pipeline()
