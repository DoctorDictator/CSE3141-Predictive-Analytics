# Traffic Analysis and Prediction
## TrafficRush — Predicting High Traffic Conditions Using Machine Learning

CSE3141 Predictive Analytics mini project, Manipal University Jaipur.

[Website](https://doctordictator.github.io/CSE3141-Predictive-Analytics/) ·
[Executed notebook](notebooks/Traffic_Analysis_and_Prediction.ipynb) ·
[UCI dataset](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)

## Purpose and actual results

Can weather, time of day, weekday and holidays help identify high-volume highway hours?
**High Traffic means at least 4,722 vehicles/hour**, the training-period 70th percentile.
It does not necessarily mean a jam, crash risk, or live traffic condition.

The genuine UCI dataset contains **48,204 source records**, covering October 2012 to
September 2018 at westbound I-94 ATR station 301, Minneapolis–Saint Paul. After
investigating repeated timestamps and retaining one consistent source weather report
per hour, **40,575 unique hours** remain. Temperature is converted from Kelvin to
Celsius; precipitation is in millimetres/hour.

| Final chronological test metric | Time + weather |
|---|---:|
| Accuracy | 92.84% |
| Precision | 0.8849 |
| Recall | 0.8885 |
| F1 | 0.8867 |
| ROC-AUC | 0.9770 |

Majority-class test accuracy: **68.46%**. The original experiment uses the same
Logistic Regression algorithm, tuning opportunities and chronological partitions:

| Feature group | Test F1 | Test ROC-AUC |
|---|---:|---:|
| Time only | 0.8835 | 0.9761 |
| Time + weather | 0.8867 | 0.9770 |

Weather adds a small improvement: **+0.0032 F1**, **+0.0009 AUC**. This is not a
causal or statistical-significance claim. All numbers come from actual execution.

## Notebook and validation

The **24-cell notebook** follows the straightforward pandas/scikit-learn style of
the supplied airline reference (20 cells), inspected before implementation.
The actual CSE3141 handout was also reviewed. This project demonstrates its
collection, preprocessing, missingness/outlier investigation, EDA, visualization,
feature engineering/selection, Binary Logistic Regression and validation topics.
It does not add unrelated algorithms just to reproduce the entire syllabus.

The implementation is now split into three straightforward Python files:

- **`traffic_analysis.py`** ? readable functions for data loading, cleaning, plots,
  training, evaluation and model export. It can also run as a normal Python script.
- **`make_notebook.py`** ? creates the notebook with short function calls and academic
  explanations; `--execute` runs every cell and saves real outputs.
- **`run.py`** ? starts the backend and frontend together, reuses already-running
  services, and stops only the services it started when you press Ctrl+C.

The notebook keeps its explanations, charts and actual results, with only 23 lines
of Python calls across its code cells. The original splits, label cutoff, fitted
pipeline and exported metrics are unchanged. `backend/main.py` only serves frozen
artifacts and never trains a model.

- Download/cache the official UCI archive. If unavailable, save the official ZIP as
  `data/raw/metro_interstate_traffic_volume.zip` and rerun.
- Audit missingness, exact duplicates, repeated timestamps, conflicting hourly volume,
  physical invalidity and IQR flags. Zero Kelvin becomes missing for training-fit
  imputation. Unusual nonnegative weather values are investigated and retained.
- Consistent repeated hours keep the first source weather record, without target-based
  ranking. Conflicting-volume hours would be quarantined. No missing hours are invented.
- Holiday status is a known calendar-day property; source midnight markers are propagated
  within the date. Weekend is omitted from predictors because weekday determines it.
- Chronological split: **28,402 training / 6,086 validation / 6,087 test hours**.
  Duplicate hours cannot enter different partitions. The volume cutoff uses training only.
- Pipeline: training-fit median/most-frequent imputation, numeric scaling and categorical
  one-hot encoding with unknown-category handling. Hour, weekday and month are categorical,
  allowing nonmonotonic rush-hour patterns. Volume and weather_description are not inputs.
- C grid {0.1, 1, 10} uses one predefined chronological validation fold and F1.
  Primary selected C: **0.1**. This is a holdout validation fold, not repeated random CV.
  Models remain training-fit; the probability decision cutoff stays **0.5**.
- Time + weather is the predeclared primary model; time only is the research ablation.
  Final test outcomes never select features or hyperparameters.
- Metrics, confusion matrix, classification report, ROC, coefficients and historical
  aggregates are exported alongside the complete fitted Joblib pipeline.
  Export/reload probabilities are verified.

## Local setup

Tested with **Python 3.13.9** and **Node 24**. Use Python 3.13 with the pinned
scientific dependencies so the Joblib artifact loads consistently.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m jupyterlab notebooks/Traffic_Analysis_and_Prediction.ipynb
```

Select the environment's Python kernel and **Restart Kernel and Run All Cells**.
Launch at the repository root or inside `notebooks/`. The notebook exports model
artifacts to `backend/artifacts/` and real saved dashboard JSON to
`frontend/public/data/`.

**Start the whole app with one command from the repository root:**

```powershell
.\.venv\Scripts\python run.py
```

Open **http://127.0.0.1:5173/CSE3141-Predictive-Analytics/**. Keep the command running;
press Ctrl+C to stop its services. If frontend dependencies are not installed yet,
run `npm ci` inside `frontend/` once. When using the existing Anaconda Python on this
machine, `py -3.13 run.py` works too.

**Regenerate and execute the short notebook:**

```powershell
.\.venv\Scripts\python make_notebook.py --execute
```

Running `make_notebook.py` without `--execute` creates the notebook without outputs;
restart and run all cells in Jupyter afterward. To inspect or edit the analytical
code, open `traffic_analysis.py`. `python traffic_analysis.py` runs the same analysis
as a script (close each Matplotlib figure to proceed when using an interactive backend).

For separate development terminals, start the API from the repository root:

```powershell
.\.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open **http://127.0.0.1:5173/CSE3141-Predictive-Analytics/**.
The local frontend defaults to the local API. Override `VITE_API_URL` with an
untracked `.env.local` if needed; see `frontend/.env.example`.

## Architecture and interface

**Notebook → fitted Joblib pipeline + genuine JSON → FastAPI → React**.

React/Vite, Tailwind CSS, Recharts, Lucide and Fetch provide five pages:
Overview, Traffic Analysis, Traffic Prediction, Model Performance and About.
The form accepts local date/hour, holiday, Celsius temperature, rain/snow, clouds
and weather. FastAPI derives calendar fields only. Learned imputation, encoding
and scaling remain inside the exact exported pipeline. No prediction formula or
hardcoded probability exists in JavaScript.

| Endpoint | Responsibility |
|---|---|
| GET `/api/health` | API and model-load state |
| GET `/api/overview` | Actual dataset summary and volume cutoff |
| GET `/api/analytics` | Precomputed historical chart aggregates |
| GET `/api/model/metrics` | Test results, experiment, coefficients and metadata |
| POST `/api/predict` | Real category, both probabilities and input snapshot |

Pydantic validates weather ranges, categories and local whole-hour timestamps.
Timezone offsets are rejected rather than silently converting the source's
local wall-clock convention. The default CORS list permits the actual Pages origin
and local Vite origins. `CORS_ORIGINS` can override it. Missing artifacts return useful
unavailable errors. Load only trusted local project Joblib artifacts.

## Deployment

**Frontend:** GitHub Pages uses `.github/workflows/deploy-pages.yml`, the correct
repository base path and hash routes, including direct page links. Saved chart JSON
contains authentic notebook exports; the UI labels this state **Saved analysis**.

**Backend:** local prediction is fully verified, but **public inference is pending
a connected Python hosting account**. Render was discovered as available but is
not connected, and no hosting credentials are present. No public API URL or
working public prediction feature is claimed. The site shows an explanation/error
instead of invented predictions.

A prepared `render.yaml` deploys the existing FastAPI service and checked-in artifacts.
Render's [current free-service documentation](https://render.com/docs/free) supports
Python web services, with 15-minute idle spin-down, roughly one-minute cold starts
and workspace quotas. Account eligibility and remaining allowance must be checked
in the user's account; no paid resource or free hosting entitlement is assumed.
See [Render FastAPI deployment](https://render.com/docs/deploy-fastapi).

To finish public inference after connecting hosting:

1. Deploy the blueprint, or create a Python web service with root directory `backend`.
2. Build with `pip install -r requirements.txt`.
3. Start with `uvicorn main:app --host 0.0.0.0 --port $PORT`; use Python 3.13.9.
4. Set `CORS_ORIGINS=https://doctordictator.github.io`.
5. Verify the returned HTTPS `/api/health` reports `model_loaded: true` and test a POST.
6. Set repository Actions variable `VITE_API_URL` to that exact HTTPS API origin.
7. Rerun **Deploy TrafficRush to GitHub Pages** and verify the public form.

Until then, the public site provides historical analysis and model results, while
the full real prediction workflow runs locally. Cold-start services may need a retry.

## Verification

- Executed all 24 notebook cells and verified a fresh-kernel restart/run-all.
- After moving the logic into Python functions, regenerated and executed the notebook;
  all exported metrics exactly match the pre-refactor results.
- Exported pipeline reload reproduces notebook test probabilities.
- Four GET endpoints passed; three representative held-out inputs matched
  notebook/API class and probability to 1e-12.
- Seven invalid-input cases rejected; permitted and unlisted CORS origins checked.
- Browser smoke tests pass navigation across all five pages, actual API inference,
  validation, failed-API errors, mobile menu behavior and horizontal overflow.
- Desktop/mobile screenshots reviewed. Production build succeeds with prefixed assets.
- Public Pages verification passed all five routes, mobile layout and the unavailable-backend
  message. Four repository-prefixed assets returned HTTP 200; all three public JSON exports
  matched notebook artifacts. Public inference remains unavailable pending backend hosting.

```powershell
cd frontend
npm run build
npm run test:smoke
```

Browser tests need the local backend and Vite server. Windows tests use installed
Chrome; elsewhere run `npx playwright install chromium`. `TEST_URL` can target the
production preview. `QA_SCREENSHOT_DIR` saves optional screenshots outside the repo.
Public deployment checks are distinct from these local integration results.

## Limitations and improvements

This is one station's historical count dataset, not a live road network.
High counts do not establish low speeds or congestion. Selecting one source
weather record loses detail. Unusual precipitation, zero-temperature sentinels
and missing hours remain limitations. Supplied same-hour weather is assumed
available, not independently forecast. Adjacent hours are dependent, validation is
reused for model selection, and probability calibration is not independently tested.
Coefficients are observational associations rather than causes. Improve with new
stations/years, rolling-origin evaluation and an independent calibration block.

## Project structure

```text
traffic_analysis.py                           # readable analysis functions
make_notebook.py                              # create/execute the short notebook
run.py                                        # start the complete local app
notebooks/Traffic_Analysis_and_Prediction.ipynb  # explanations, short calls and outputs
backend/main.py                               # frozen-model serving only
backend/artifacts/                            # model and actual exported results
frontend/src/                                 # React dashboard and API client
frontend/public/data/                         # genuine saved notebook summaries
frontend/tests/smoke.mjs                       # browser integration checks
.github/workflows/deploy-pages.yml             # frontend deployment
render.yaml                                   # prepared backend hosting blueprint
requirements.txt
README.md
```

Raw data, environments, node_modules, temporary screenshots and credentials are
excluded from commits. The previous edited NearMiss project was preserved locally
outside this repository before replacement.

## Attribution

Hogue, J. (2019). *Metro Interstate Traffic Volume* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5X60B . **CC BY 4.0**. Aggregates and the model
derive from this attributed dataset; raw data is downloaded rather than committed.
UCI supplies a regression count target; this project creates its own training-defined
binary label. The airline ZIP informed code simplicity and architecture, not the
traffic data, trained model or reported scores. The supplied CSE3141 handout was reviewed.
