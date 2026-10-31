"""Package actual notebook results for the backend and dashboard."""

from pathlib import Path
import json
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "backend/artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def evaluate(y, decision, probability):
    """Calculate classification metrics from actual predictions."""
    return dict(
        accuracy=float(accuracy_score(y, decision)),
        precision=float(precision_score(y, decision, zero_division=0)),
        recall=float(recall_score(y, decision, zero_division=0)),
        f1=float(f1_score(y, decision, zero_division=0)),
        roc_auc=float(roc_auc_score(y, probability)),
    )


def records(frame):
    """Convert a small summary table into JSON-compatible records."""
    return json.loads(frame.to_json(orient="records"))


def export_results(
    raw, df, parts, volume_threshold, model, tuning, results, coefficients
):
    """Save the fitted pipeline and genuine dashboard results for FastAPI."""
    train, valid, test = parts["train"], parts["validation"], parts["test"]
    split_summary = [
        dict(
            partition=name,
            hours=len(block),
            start=str(block.date_time.min()),
            end=str(block.date_time.max()),
            high_fraction=float(block.high_traffic.mean()),
        )
        for name, block in parts.items()
    ]
    test_scores = results["test_scores"]
    baseline = results["baseline"]
    cm = results["cm"]
    report = results["report"]
    comparison = results["comparison"]
    evaluation = results["evaluation"]
    fpr = results["fpr"]
    tpr = results["tpr"]
    probability = results["probability"]
    X_test = test[model.feature_names_in_]
    weather_categories = sorted(df.weather_main.dropna().unique().tolist())
    overview = dict(
        raw_observations=len(raw),
        unique_hours=len(df),
        training_hours=len(train),
        validation_hours=len(valid),
        test_hours=len(test),
        start=str(df.date_time.min()),
        end=str(df.date_time.max()),
        high_percent=float(df.high_traffic.mean() * 100),
        normal_percent=float((1 - df.high_traffic.mean()) * 100),
        volume_threshold=volume_threshold,
        mean_volume=float(df.traffic_volume.mean()),
        source="UCI Metro Interstate Traffic Volume (492)",
        source_url="https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume",
        scope="Historical hourly westbound I-94 counts; not congestion, crash risk or live traffic.",
        analysis_scope="Full cleaned historical dataset; predictive EDA and model choices use training only.",
    )
    hourly = (
        df.groupby("hour")
        .traffic_volume.agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "volume"})
    )
    daily = (
        df.groupby("day_of_week")
        .traffic_volume.agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "volume"})
    )
    daily["day"] = daily.day_of_week.map(
        dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))
    )
    weather = (
        df.groupby("weather_main")
        .traffic_volume.agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "volume", "weather_main": "weather"})
    )
    weekend = df.groupby("is_weekend").traffic_volume.mean().reset_index(name="volume")
    weekend["period"] = weekend.is_weekend.map({0: "Weekday", 1: "Weekend"})
    rain_bins = pd.cut(
        df.rain_1h,
        [-0.01, 0, 1, 5, 20, np.inf],
        labels=["Dry", "0–1 mm", "1–5 mm", "5–20 mm", "20+ mm"],
    )
    rain_summary = (
        df.groupby(rain_bins, observed=True)
        .traffic_volume.agg(["mean", "count"])
        .reset_index()
        .rename(columns={"rain_1h": "rain", "mean": "volume"})
    )
    monthly = df.groupby("month").traffic_volume.mean().reset_index(name="volume")
    analytics = dict(
        hourly=records(hourly),
        daily=records(daily),
        weather=records(weather),
        weekend=records(weekend),
        rain=records(rain_summary),
        monthly=records(monthly),
    )
    metrics = dict(
        test=test_scores,
        baseline=baseline,
        confusion_matrix=cm.tolist(),
        classification_report=report,
        comparison=records(comparison),
        tuning=records(tuning),
        evaluation=records(evaluation),
        coefficients=records(coefficients),
        roc=[dict(fpr=float(a), tpr=float(b)) for a, b in zip(fpr, tpr)],
        split=split_summary,
        decision_threshold=0.5,
        volume_threshold=volume_threshold,
        model_name="Binary Logistic Regression · time + weather",
        features=model.feature_names_in_.tolist(),
        weather_categories=weather_categories,
        selected_C=float(model.named_steps["classifier"].C),
        versions={
            name: importlib.metadata.version(name)
            for name in ["scikit-learn", "pandas", "numpy", "joblib"]
        },
    )
    parity = []
    for _, row in test.iloc[[0, len(test) // 2, -1]].iterrows():
        inputs = dict(
            date_time=row.date_time.isoformat(),
            is_holiday=bool(row.is_holiday),
            temp_c=float(row.temp_c) if pd.notna(row.temp_c) else 10.0,
            rain_1h=float(row.rain_1h),
            snow_1h=float(row.snow_1h),
            clouds_all=float(row.clouds_all),
            weather_main=row.weather_main,
        )
        frame = pd.DataFrame(
            [
                dict(
                    temp_c=inputs["temp_c"],
                    rain_1h=inputs["rain_1h"],
                    snow_1h=inputs["snow_1h"],
                    clouds_all=inputs["clouds_all"],
                    weather_main=inputs["weather_main"],
                    hour=row.hour,
                    day_of_week=row.day_of_week,
                    month=row.month,
                    is_holiday=int(row.is_holiday),
                )
            ]
        )
        parity.append(
            dict(
                input=inputs,
                high_probability=float(model.predict_proba(frame)[:, 1][0]),
                prediction=int(model.predict(frame)[0]),
            )
        )
    joblib.dump(model, ARTIFACTS / "traffic_model.joblib")
    exports = dict(
        overview=overview,
        analytics=analytics,
        metrics=metrics,
        inference_examples=parity,
    )
    PUBLIC = ROOT / "frontend/public/data"
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for name, value in exports.items():
        (ARTIFACTS / (name + ".json")).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )
        if name != "inference_examples":
            (PUBLIC / (name + ".json")).write_text(
                json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
            )
    assert np.allclose(
        joblib.load(ARTIFACTS / "traffic_model.joblib").predict_proba(X_test)[:, 1],
        probability,
    )
    print("Exported fitted pipeline, actual metrics, summaries and parity cases.")
