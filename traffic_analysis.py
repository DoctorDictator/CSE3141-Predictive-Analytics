"""TrafficRush analysis: straightforward functions shared by script and notebook."""

from pathlib import Path
import json, zipfile, io, hashlib, platform, importlib.metadata
import requests, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/raw"
ARTIFACTS = ROOT / "backend/artifacts"
DATA.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette=["#0d9488", "#f59e0b", "#334155"])
plt.rcParams.update({"figure.figsize": (11, 4), "figure.dpi": 110})
thread_limit = threadpool_limits(limits=1)
SEED = 42


def load_data():
    """Download/cache the official UCI data and show its schema."""
    SOURCE = "https://archive.ics.uci.edu/static/public/492/metro%2Binterstate%2Btraffic%2Bvolume.zip"
    archive = DATA / "metro_interstate_traffic_volume.zip"
    if not archive.exists():
        try:
            response = requests.get(SOURCE, timeout=(15, 120))
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as check:
                assert any(name.endswith(".csv.gz") for name in check.namelist())
            archive.write_bytes(response.content)
        except Exception as error:
            raise RuntimeError(
                "UCI download failed. Download the ZIP from "
                + SOURCE
                + " into data/raw/metro_interstate_traffic_volume.zip and rerun. "
                + str(error)
            ) from error
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.endswith(".csv.gz"))
        raw = pd.read_csv(io.BytesIO(bundle.read(member)), compression="gzip")
    print("Official source:", SOURCE)
    print("Local archive SHA-256:", hashlib.sha256(archive.read_bytes()).hexdigest())
    print("Dataset shape:", raw.shape)
    display(raw.head(), raw.dtypes.to_frame("dtype"))
    print("Columns:", raw.columns.tolist())
    raw.info()
    return raw


def clean_data(raw):
    """Keep one valid record per hour and report data-quality decisions."""
    display(raw.isna().sum().to_frame("missing"))
    print("Exact duplicate rows:", int(raw.duplicated().sum()))
    df = raw.drop_duplicates().copy()
    df["date_time"] = pd.to_datetime(df.date_time, errors="coerce")
    print("Extra records sharing an hour:", int(df.date_time.duplicated().sum()))
    conflicting_hours = df.groupby("date_time").traffic_volume.nunique()
    conflicting_hours = conflicting_hours[conflicting_hours > 1].index
    print("Hours with conflicting volume:", len(conflicting_hours))
    df = df[~df.date_time.isin(conflicting_hours)].dropna(
        subset=["date_time", "traffic_volume"]
    )
    # Holiday is a known calendar property; propagate the source's midnight marker
    # within its calendar day, not from future traffic measurements.
    df["is_holiday"] = (
        df.holiday.ne("None").groupby(df.date_time.dt.date).transform("max").astype(int)
    )
    df = (
        df.drop_duplicates("date_time", keep="first")
        .sort_values("date_time")
        .reset_index(drop=True)
    )
    for column in ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df.traffic_volume.ge(0)].copy()
    print("Zero/nonpositive Kelvin readings:", int(df.temp.le(0).sum()))
    df.loc[df.temp.le(0), "temp"] = np.nan
    for column in ["rain_1h", "snow_1h"]:
        df.loc[df[column].lt(0), column] = np.nan
    df.loc[~df.clouds_all.between(0, 100), "clouds_all"] = np.nan
    df["temp_c"] = df.temp - 273.15
    display(
        df[["temp_c", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]].describe()
    )
    q1 = df[["temp_c", "rain_1h", "snow_1h", "traffic_volume"]].quantile(0.25)
    q3 = df[q1.index].quantile(0.75)
    flags = (
        (df[q1.index] < q1 - 1.5 * (q3 - q1)) | (df[q1.index] > q3 + 1.5 * (q3 - q1))
    ).sum()
    display(flags.to_frame("IQR flags retained"))
    print("Largest rain measurements (mm):", df.rain_1h.nlargest(3).tolist())
    assert df.date_time.is_unique
    print(
        "Usable unique hours:",
        len(df),
        "Coverage:",
        df.date_time.min(),
        "to",
        df.date_time.max(),
    )
    return df


def split_data(df):
    """Add calendar features; split by time and label using training volume only."""
    df["hour"] = df.date_time.dt.hour
    df["day_of_week"] = df.date_time.dt.dayofweek
    df["month"] = df.date_time.dt.month
    df["is_weekend"] = df.day_of_week.ge(5).astype(int)
    n = len(df)
    train = df.iloc[: int(0.70 * n)].copy()
    valid = df.iloc[int(0.70 * n) : int(0.85 * n)].copy()
    test = df.iloc[int(0.85 * n) :].copy()
    volume_threshold = float(train.traffic_volume.quantile(0.70))
    for block in [train, valid, test, df]:
        block["high_traffic"] = block.traffic_volume.ge(volume_threshold).astype(int)
    assert train.date_time.max() < valid.date_time.min() < test.date_time.min()
    assert not set(train.date_time) & set(valid.date_time)
    assert not set(train.date_time) & set(test.date_time)
    assert volume_threshold == float(train.traffic_volume.quantile(0.70))
    split_summary = []
    for name, block in [("train", train), ("validation", valid), ("test", test)]:
        assert block.high_traffic.nunique() == 2
        split_summary.append(
            dict(
                partition=name,
                hours=len(block),
                start=str(block.date_time.min()),
                end=str(block.date_time.max()),
                high_fraction=float(block.high_traffic.mean()),
            )
        )
    display(pd.DataFrame(split_summary))
    print(
        "High Traffic cutoff:",
        volume_threshold,
        "vehicles/hour (training 70th percentile)",
    )
    display(
        Markdown(
            f"**Decision:** use ≥{volume_threshold:,.0f} vehicles/hour as High Traffic "
            "in every partition and in the app. It is a historical count label, not a universal road-capacity limit."
        )
    )
    return {"train": train, "validation": valid, "test": test}, volume_threshold


def plot_volume(train, volume_threshold):
    """Show volume distribution and the average hourly pattern."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(train.traffic_volume, bins=40, ax=axes[0])
    axes[0].axvline(
        volume_threshold, color="#f59e0b", ls="--", label="Training 70th percentile"
    )
    axes[0].set(
        xlabel="Vehicles per hour", title="Distribution of recorded hourly volume"
    )
    axes[0].legend()
    train.groupby("hour").traffic_volume.mean().plot(marker="o", ax=axes[1])
    axes[1].set(
        xlabel="Local hour",
        ylabel="Mean vehicles/hour",
        title="Traffic changes across the day",
        xticks=range(0, 24, 3),
    )
    plt.tight_layout()
    plt.show()
    peak_hour = int(train.groupby("hour").traffic_volume.mean().idxmax())
    display(
        Markdown(
            f"**Observed:** the largest training mean occurs at local hour {peak_hour:02d}:00. "
            "The full distribution separates low-volume and busy hours. **Decision:** encode hour "
            "categorically rather than force a straight-line increase from midnight to 23:00."
        )
    )


def plot_calendar(train):
    """Compare weekdays, weekends and weather groups."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    daily = train.groupby("day_of_week").traffic_volume.mean()
    daily.plot.bar(ax=axes[0], rot=30)
    axes[0].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[0].set(title="Average volume by weekday", ylabel="Vehicles/hour", xlabel="")
    train.groupby("weather_main").traffic_volume.mean().sort_values().plot.barh(
        ax=axes[1]
    )
    axes[1].set(title="Average volume by main weather", xlabel="Vehicles/hour")
    train.groupby("is_weekend").traffic_volume.mean().rename(
        index={0: "Weekday", 1: "Weekend"}
    ).plot.bar(ax=axes[2], rot=0)
    axes[2].set(title="Weekday versus weekend", ylabel="Vehicles/hour", xlabel="")
    plt.tight_layout()
    plt.show()
    display(
        Markdown(
            "**Interpretation → decision:** day and weather groups have different observed "
            "means, but weather groups also occur at different times and have unequal sample sizes. "
            "Retain weekday and a compact weather group for the controlled feature comparison; "
            "do not treat these unadjusted differences as causal weather effects."
        )
    )


def plot_weather(train):
    """Inspect rainfall, class balance and numeric correlations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    rain_plot = train[train.rain_1h.notna()].sample(
        min(3000, train.rain_1h.notna().sum()), random_state=SEED
    )
    axes[0].scatter(rain_plot.rain_1h, rain_plot.traffic_volume, s=7, alpha=0.2)
    axes[0].set(
        xscale="symlog",
        xlabel="Rainfall mm (symlog display)",
        ylabel="Vehicles/hour",
        title="Rainfall and volume; extremes retained",
    )
    train.high_traffic.value_counts().sort_index().plot.bar(ax=axes[1], rot=0)
    axes[1].set_xticklabels(["Normal", "High"])
    axes[1].set(title="Training class balance", ylabel="Unique hours")
    sns.heatmap(
        train[["temp_c", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]].corr(),
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        ax=axes[2],
    )
    axes[2].set_title("Training numerical correlations")
    plt.tight_layout()
    plt.show()
    display(
        Markdown(
            "**Observed → decision:** precipitation is strongly skewed and simple weather–volume "
            "correlations do not isolate calendar effects. All nonnegative extreme measurements remain "
            "in this simple model; their plausibility and leverage are limitations. Imputation and scaling "
            "must fit training only, and actual validation determines regularization."
        )
    )


time_numeric = ["is_holiday"]
time_categorical = ["hour", "day_of_week", "month"]
weather_numeric = ["temp_c", "rain_1h", "snow_1h", "clouds_all"]
all_numeric = time_numeric + weather_numeric
all_categorical = time_categorical + ["weather_main"]
feature_groups = {
    "time_only": (time_numeric, time_categorical),
    "time_weather": (all_numeric, all_categorical),
}


def make_pipeline(numeric, categorical):
    """Create readable train-fit preprocessing and Binary Logistic Regression."""
    preprocessing = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessing),
            (
                "classifier",
                LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED),
            ),
        ]
    )


def train_models(parts):
    """Tune two Logistic Regression feature groups with chronological validation."""
    train, valid = parts["train"], parts["validation"]
    pool = pd.concat([train, valid], ignore_index=True)
    fold = PredefinedSplit(
        np.r_[np.full(len(train), -1), np.zeros(len(valid), dtype=int)]
    )
    models, tuning_rows = {}, []
    for name, (numeric, categorical) in feature_groups.items():
        pipeline = make_pipeline(numeric, categorical)
        search = GridSearchCV(
            pipeline,
            {"classifier__C": [0.1, 1.0, 10.0]},
            cv=fold,
            scoring="f1",
            refit=False,
            return_train_score=True,
            error_score="raise",
            n_jobs=1,
        )
        search.fit(pool[numeric + categorical], pool.high_traffic)
        fitted = make_pipeline(numeric, categorical).set_params(**search.best_params_)
        fitted.fit(train[numeric + categorical], train.high_traffic)
        assert fitted.named_steps["classifier"].n_iter_.max() < 2000
        models[name] = fitted
        for params, tr, va in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_train_score"],
            search.cv_results_["mean_test_score"],
        ):
            tuning_rows.append(
                dict(
                    model=name,
                    C=params["classifier__C"],
                    training_f1=float(tr),
                    validation_f1=float(va),
                )
            )
    tuning = pd.DataFrame(tuning_rows)
    display(tuning)
    model = models["time_weather"]
    display(
        Markdown(
            "**Decision:** freeze the validation-selected C for each feature group and keep "
            "probability cutoff 0.5. The tuning table exposes train–validation gaps; a high training "
            "score alone would not establish future-period usefulness."
        )
    )
    return models, tuning


def evaluate(y, decision, probability):
    """Calculate classification metrics from actual predictions."""
    return dict(
        accuracy=float(accuracy_score(y, decision)),
        precision=float(precision_score(y, decision, zero_division=0)),
        recall=float(recall_score(y, decision, zero_division=0)),
        f1=float(f1_score(y, decision, zero_division=0)),
        roc_auc=float(roc_auc_score(y, probability)),
    )


def evaluate_models(models, parts):
    """Report real held-out metrics and the weather-versus-time experiment."""
    train, valid, test = parts["train"], parts["validation"], parts["test"]
    model = models["time_weather"]
    rows = []
    for partition, block in [("train", train), ("validation", valid), ("test", test)]:
        for name, fitted in models.items():
            numeric, categorical = feature_groups[name]
            X = block[numeric + categorical]
            rows.append(
                dict(
                    partition=partition,
                    model=name,
                    **evaluate(
                        block.high_traffic,
                        fitted.predict(X),
                        fitted.predict_proba(X)[:, 1],
                    ),
                )
            )
    evaluation = pd.DataFrame(rows)
    dummy = DummyClassifier(strategy="most_frequent").fit(
        train[["hour"]], train.high_traffic
    )
    baseline = evaluate(
        test.high_traffic,
        dummy.predict(test[["hour"]]),
        dummy.predict_proba(test[["hour"]])[:, 1],
    )
    X_test = test[all_numeric + all_categorical]
    predicted = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(test.high_traffic, predicted, labels=[0, 1])
    report = classification_report(
        test.high_traffic,
        predicted,
        target_names=["Normal Traffic", "High Traffic"],
        output_dict=True,
        zero_division=0,
    )
    fpr, tpr, _ = roc_curve(test.high_traffic, probability)
    display(
        evaluation,
        pd.Series(baseline, name="majority baseline"),
        pd.DataFrame(report).T,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="GnBu",
        cbar=False,
        xticklabels=["Normal", "High"],
        yticklabels=["Normal", "High"],
        ax=axes[0],
    )
    axes[0].set(
        xlabel="Predicted",
        ylabel="Actual",
        title="Primary model: test confusion matrix",
    )
    axes[1].plot(fpr, tpr, label="Time + weather")
    axes[1].plot([0, 1], [0, 1], "k--", label="Chance")
    axes[1].legend()
    axes[1].set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Final test ROC",
    )
    plt.tight_layout()
    plt.show()
    comparison = evaluation[evaluation.partition.isin(["validation", "test"])]
    display(comparison[["partition", "model", "f1", "roc_auc"]])
    weather_gain = (
        evaluation.query("partition == 'test'")
        .set_index("model")
        .loc["time_weather", ["f1", "roc_auc"]]
        - evaluation.query("partition == 'test'")
        .set_index("model")
        .loc["time_only", ["f1", "roc_auc"]]
    )
    test_scores = evaluate(test.high_traffic, predicted, probability)
    display(
        Markdown(
            f'**Actual findings:** primary test F1 {test_scores["f1"]:.3f}, AUC '
            f'{test_scores["roc_auc"]:.3f}. Weather-minus-time F1 {weather_gain.f1:+.3f}, '
            f"AUC {weather_gain.roc_auc:+.3f}. These differences do not prove that weather "
            "causes changes in volume, and the test is not reused for tuning."
        )
    )
    return {
        "evaluation": evaluation,
        "baseline": baseline,
        "predicted": predicted,
        "probability": probability,
        "cm": cm,
        "report": report,
        "fpr": fpr,
        "tpr": tpr,
        "comparison": comparison,
        "weather_gain": weather_gain,
        "test_scores": test_scores,
    }


def interpret_model(model):
    """Show fitted coefficient associations without claiming causation."""
    names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pd.DataFrame(
        {"feature": names, "coefficient": model.named_steps["classifier"].coef_[0]}
    )
    coefficients["magnitude"] = coefficients.coefficient.abs()
    coefficients = coefficients.sort_values("magnitude", ascending=False)
    display(coefficients.head(15))
    display(
        Markdown(
            "**Observed:** the largest standardized/encoded associations are "
            + ", ".join(coefficients.head(5).feature)
            + ". **Decision:** report their signs and "
            "conditional associations, without claiming causal feature importance."
        )
    )
    return coefficients


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
    X_test = test[all_numeric + all_categorical]
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
        features=all_numeric + all_categorical,
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


def show_conclusion(raw, df, volume_threshold, results):
    """Summarize the actual measured results."""
    test_scores = results["test_scores"]
    weather_gain = results["weather_gain"]
    display(
        Markdown(
            f"**Completed execution:** {len(raw):,} source observations became "
            f"{len(df):,} unique hours. High Traffic is ≥{volume_threshold:,.0f} vehicles/hour. "
            f'Test accuracy {test_scores["accuracy"]:.2%}, precision {test_scores["precision"]:.3f}, '
            f'recall {test_scores["recall"]:.3f}, F1 {test_scores["f1"]:.3f}, AUC {test_scores["roc_auc"]:.3f}. '
            f"Weather changes test F1 by {weather_gain.f1:+.3f} and AUC by {weather_gain.roc_auc:+.3f}. "
            "The same fitted pipeline and definition are exported for the website; results estimate "
            "historical volume categories, not jams or crashes."
        )
    )
    print("All cells completed and artifact round-trip verified.")


def run_analysis():
    """Run the same analysis from a terminal, without needing Jupyter."""
    raw = load_data()
    df = clean_data(raw)
    parts, threshold = split_data(df)
    plot_volume(parts["train"], threshold)
    plot_calendar(parts["train"])
    plot_weather(parts["train"])
    models, tuning = train_models(parts)
    results = evaluate_models(models, parts)
    model = models["time_weather"]
    coefficients = interpret_model(model)
    export_results(raw, df, parts, threshold, model, tuning, results, coefficients)
    show_conclusion(raw, df, threshold, results)


if __name__ == "__main__":
    run_analysis()
