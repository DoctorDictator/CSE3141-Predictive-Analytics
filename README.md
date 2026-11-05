# NearMiss — Lane-Change Behavior Prediction

CSE3141 · Predictive Analytics · Manipal University Jaipur

**Question:** Given a vehicle's current conditions and past lateral motion, will it cross into an adjacent mainline lane within the next two seconds, or stay in its lane?

This project learns from actual NGSIM lane IDs. It does not predict verified accidents, certify maneuver safety or choose the best lane. Earlier close-following and conflict-proxy models have been removed. The interactive simulator remains a separate experiment using programmed movement, clearance and geometric contact rules.

## Website

- **[Open NearMiss](https://doctordictator.github.io/CSE3141-Predictive-Analytics/)**
- **[Prediction versus recorded outcome](https://doctordictator.github.io/CSE3141-Predictive-Analytics/#/replay)** — the primary demonstration
- **[Interactive simulator](https://doctordictator.github.io/CSE3141-Predictive-Analytics/#/simulator)**

GitHub Pages publishes pushes to `main`. The workflow tests the actual compiled site under its repository subpath before publication. Render updates the connected FastAPI service independently. Matching saved exports and exact fitted-model browser inference keep the frontend usable during backend updates/cold starts; incompatible older API responses are rejected.

## Run locally

Python 3.13/3.14 and Node.js 22.12+ are required. Tested with Python 3.14 and Node 24.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm ci
cd ..
.venv\Scripts\python.exe run.py
```

Open http://127.0.0.1:5173/. The launcher starts FastAPI on port 8000 and Vite on 5173; Ctrl+C stops services it started. The fitted model and exports are committed, so running the app does not require training or downloading the raw data.

For separate terminals:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```powershell
cd frontend
npm run dev
```

## The main demonstration

1. Open **Recorded replay** and choose a held-out vehicle example.
2. Read the frozen forecast: lane change or stay in lane, its uncalibrated score and the validation-selected cutoff.
3. Inspect the six inputs measured at assessment time. No future measurement enters that prediction.
4. Play the next two seconds of actual recorded positions. These observations, not the model's forecast, control the replay.
5. At 2.0 s, reveal the actual outcome and whether the forecast was correct.
6. Try examples with missed crossings and false alarms as well as correct forecasts. The 16 examples are selected to illustrate all four confusion-matrix outcomes, not to estimate overall accuracy.
7. Open **Model comparison** for the entire untouched test block, class counts and always-stay baseline. Open **Distributions** for actual measured histograms, descriptive statistics and correlations.

## Genuine dataset and labels

Source: [DOT NGSIM trajectories](https://data.transportation.gov/d/8ect-6jqj). [FHWA's US-101 description](https://www.fhwa.dot.gov/publications/research/operations/07030/index.cfm) explains the 0.1-second measurements, five mainline lanes and auxiliary lane. This experiment retains one entire recording origin, `1118846978900`: **1,180,598 rows, 2,169 vehicles, 952.8 seconds**. It is not the complete three-period release. Vehicle class is joined from an official compact lookup.

Original positions, dimensions and spacing are in feet; speed in ft/s; acceleration in ft/s². Convert using exactly 0.3048. Timestamps are milliseconds. DOT space headway is front-to-front, so leader length is subtracted for bumper gap. Mainline lanes 1–5 are retained; ramp transitions are outside this task. Invalid/missing physical measurements, duplicate keys and unavailable preceding references are investigated/excluded. A reported preceding ID of zero remains an explicitly leaderless observation.

For every current frame, inspect **all 20 subsequent observations**, each exactly 100 ms later:

- **0 — Stay in lane:** no lane crossing occurs in that complete future window.
- **1 — Lane change:** an adjacent mainline crossing occurs anywhere in the complete window. A change followed by a return still counts.
- Missing, truncated or nonadjacent-jump windows are **censored**, never assigned an invented negative label.

The horizon is our stated research choice; it is not supplied by DOT. The labels come from recorded future lane IDs, without crash, following-gap or safety thresholds. The recording contains **812 observed adjacent crossings** and **1,103,092 eligible 10 Hz windows** after current-feature and future-completeness checks. Positive windows can overlap around one crossing, so window counts are not unique maneuver counts.

## Simple notebook

[notebooks/NearMiss.ipynb](notebooks/NearMiss.ipynb) contains the entire analysis in **31 executed cells**: loading, cleaning, units, EDA, feature engineering, future labels, isolated splits, all three syllabus classifiers, tuning, evaluation, interpretation, replay and artifact exports. Its final section is **Conclusion**. No additional ML helper modules or custom model classes are used.

```powershell
.venv\Scripts\python.exe scripts/nearmiss_notebook.py --execute
```

This command recreates and executes the notebook, overwriting notebook edits. To preserve edits, run the notebook itself:

```powershell
.venv\Scripts\python.exe -m jupyterlab notebooks/NearMiss.ipynb
```

Raw files are ignored by Git. With no cache, the notebook downloads ordered official API pages and the vehicle-class lookup. It verifies source location, recording origin and live official row count, and exports SHA-256 provenance. A first download can take several minutes; there is no fabricated-data fallback.

### Six current/past features

| Feature | Meaning |
|---|---|
| `speed` | Current ego speed, m/s |
| `acceleration` | Current ego acceleration, m/s² |
| `lateral_speed` | Absolute average lateral motion over the preceding 0.5 s |
| `front_gap` | Current bumper gap, capped at 150 m |
| `closing_speed` | Ego speed minus the same-time leader's speed |
| `has_leader` | Whether the recorded current leader is available: 0/1 |

When there is no reported leader, encode `front_gap=150`, `closing_speed=0`, `has_leader=0`. This cap is a documented representation, not a measured distance to an unseen vehicle. Current leaders must match at the same timestamp/lane. Missing 0.5-second history makes the observation ineligible. Lateral motion may already be underway before lane-ID crossing; this is not a guarantee of recognizing hidden intention before any movement.

### Validation and actual results

Chronological 60/20/20 partitions exclude both ego and current leader trajectories crossing boundaries. A full two-second outcome embargo prevents labels crossing partitions. Thin to 1 Hz only after constructing labels from all frames. Vehicle IDs, timestamps, future lanes and target values are not model features.

All three algorithms use the same features and purged expanding training CV. Scaling fits within folds. Grid search tunes only syllabus classifiers. After training-only parameter search, each candidate's decision cutoff is selected on validation from a declared fixed grid. Validation F1 selects the model, with average precision/ROC-AUC breaking ties. Test is not used to select parameters or cutoffs. The exported training-fitted pipeline is retained without refitting on validation.

| Partition | Stay windows | Change windows | Total |
|---|---:|---:|---:|
| Training | 57,936 | 821 | 58,757 |
| Validation | 14,692 | 205 | 14,897 |
| Test | 17,488 | 137 | 17,625 |

The selected model is **Logistic Regression**, cutoff **0.10**. Untouched test results: precision **0.2270**, recall **0.2336**, F1 **0.2302**, ROC-AUC **0.8873**, average precision **0.1645**, accuracy **0.9879**. These are limited forecasting results, not a reliable safety system. The always-stay baseline achieves **0.9922 accuracy** but **zero lane-change recall/F1**; its average precision is **0.0078**. The model detects 32 of 137 positive windows and misses 105, with 109 false alarms. Do not advertise the near-99% accuracy as strong lane-change detection.

Class imbalance, limited current observables, noisy trajectories, already-underway lateral motion and one recording constrain performance. More independent recordings and better causal/current traffic context should be evaluated before stronger generalization claims. Probabilities are uncalibrated; class weighting/priors and selected cutoffs must be interpreted accordingly.

## Separate interactive simulator

The simulator retains 1–8 lanes, 2–30 vehicles, seeded generation, editable speed/acceleration, braking, smooth lane changes, adding/removing vehicles, camera controls, pause/reset/rewind/replay, event logs and actual geometric contacts.

Default generation adapts **11 coherent training-period snapshots**, preserving measured speed/acceleration/size combinations for selected vehicles. Mapping lanes to a different count and separating overlaps necessarily changes the scene. It does not guarantee an exact empirical distribution or reproduce a historical accident. Explicit synthetic random ranges and constructed presets remain available and labeled.

Movement integrates at 1/120 s; lane changes take 1.6 s. Automatic cruising, braking, front/rear checks, projected-path checks and bypass decisions are **programmed rules**, independent of the behavior prediction. A high predicted lane-change score does not command a maneuver, and a low score does not forbid one. Contacts occur only when actual simulated bodies overlap; contacted cars stop and remain obstacles. Snapshot rewind keeps the last 15 simulated seconds.

The **Six-lane bypass demo** and **Blocked Lane - No Safe Gap** show rule-based passing/waiting. **Unsafe Lane Change**, **Rear-End Collision Demo** and **Multiple-Vehicle Incident** deliberately demonstrate geometric contacts. These events are hypothetical, not labeled NGSIM accidents.

Behavior scores on generated traffic are illustrative applications outside the recorded evaluation. They require at least 0.5 s of motion history and supported feature values. They are not evaluated against the controller's programmed decisions as if those were ground truth.

## Backend and browser inference

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Readiness plus exact task identifier |
| `GET /api/model/info` | Features, ranges, units, cutoff, task and portable model |
| `GET /api/model/metrics` | Actual three-model comparison and test/baseline results |
| `GET /api/analytics` | Measured distributions and training snapshot seeds |
| `GET /api/replay` | Held-out examples with frozen forecasts and recorded futures |
| `POST /api/predict` | One valid observation |
| `POST /api/predict-batch` | 1–30 uniquely identified observations |

Single-observation example is saved at `backend/artifacts/example.json`. The six fields must match the table above. Batch body: `{"observations":[{"id":"vehicle-1","inputs":{...six fields...}}]}`. Old `pairs` bodies and conflict endpoints are no longer supported. Impossible/nonfinite/extra fields are rejected. Out-of-training-range items receive no probability. No retraining occurs during startup or requests.

`behavior.joblib` is the fitted pipeline. `info.json` also exports the same scaler and linear coefficients, or QDA parameters when applicable. Browser inference reproduces fitted scores; held-out checks match scikit-learn within `1e-12`. It applies the saved validation-selected cutoff and labels its source. No percentage is hardcoded or derived from a hand-designed risk formula.

## Deployment and verification

`render.yaml` retains the backend blueprint. Build: `pip install -r requirements.txt`; start: `uvicorn main:app --host 0.0.0.0 --port $PORT`, root `backend`. Commit `behavior.joblib` and the four JSON exports. `CORS_ORIGINS` must include the exact frontend origin. Backend environment variables are read from the process; a `.env` file is not loaded automatically.

`VITE_API_URL` sets the API origin at build time. `VITE_BASE_PATH` defaults to `/`; GitHub Pages CI sets the repository subpath. Hash routes preserve refresh behavior. Matching saved research exports start immediately during cold starts; incompatible prior-task responses are rejected rather than causing a blank page.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest backend/test_api.py -q
cd frontend
npm test
npm run build
# With local frontend/backend running:
npm run test:smoke
# With a build made using VITE_BASE_PATH=/CSE3141-Predictive-Analytics/:
npm run test:production
```

The production-browser suite runs on the built files before GitHub Pages publication. It checks all six routes, every replay example and frozen forecast, empirical initialization, independent simulator behavior, blocked lanes, deliberate contacts, reset/rewind/replay, mobile layout and legacy-task rejection. Windows uses installed Chrome; CI installs Playwright Chromium. Screenshots are saved under ignored `frontend/test-results/`.
