# NearMiss lane-change behavior — verification

Updated: 5 November 2026

## Dataset and notebook

- Source: Automatum Full Highway drone dataset, 114 recordings; the 4.23 GB ZIP passed SHA-256 and ZIP CRC checks. The notebook streams the archive and does not unpack the roughly 35 GiB of JSON trajectories.
- Processing produced 27,359,541 frame observations and 30.6 recording hours. After eligibility checks, the model table contains 679,705 one-second assessment windows from 114 recordings; 28,646 (4.21%) have an observed adjacent lane-ID crossing in the next complete two seconds.
- The new feature build uses `vy_vec` as vehicle-frame lateral velocity and projects neighbor positions onto the ego heading. The prior draft treated global map movement as lateral speed and measured longitudinal gaps on global `x`; both definitions are corrected. The generated schema is `behavior-3`.
- Train/validation/test splits are by whole recording: 68/22/24 recordings and 406,853/143,985/128,867 windows. Three-fold grouped CV and validation-only threshold selection are used.
- `notebooks/NearMiss.ipynb`: 24 cells, 13 code cells executed, no notebook error outputs, no “viva” text. The generated local replay remains in ignored `data/processed/`.

## Held-out model result

- Selected classifier: QDA; decision threshold: 0.40, selected on validation.
- Test precision: 0.1302; recall: 0.4978; F1: 0.2064; ROC-AUC: 0.8003; average precision: 0.1206; balanced accuracy: 0.6904.
- Confusion matrix: `[[109923, 14565], [2199, 2180]]`.
- Always-stay baseline: accuracy 0.9660, with zero lane-change recall and F1.
- Scores are modest. The target is observed lane-change behavior; the dataset has no verified crash labels. The model does not certify a safe maneuver or control the simulator.

## Integration checks

- Python compilation: passed.
- Backend/API/notebook tests: 23 passed. Readiness, fitted-model equivalence, browser/Python probability agreement, batch support, feature limits, missing-neighbor encoding, mismatched artifacts, task/version schema and CORS were covered. Tiny generated recordings validate rotated-road gaps, vehicle-frame lateral speed, actual lane crossings and exclusion of road changes, divider states, nonadjacent jumps and missing frames.
- Frontend unit tests: 37 passed. New cases cover deployment mismatches, API retry delays and recovery, changing observation inputs, instantaneous lateral velocity, dataset highway speeds, and merging leaders remaining visible.
- Production build: passed with GitHub Pages base path `/CSE3141-Predictive-Analytics/`.
- Production browser suite: passed on five routes with repeated reloads, simulator initialization and movement, lane-change bypass, blocked-lane behavior, collision/replay cases, mobile layout, and no JavaScript errors, duplicate navigation keys or missing assets. API outage/backoff and retrying a failed static dataset profile also passed.
- Live end-to-end smoke: FastAPI and Vite ran together; the browser suite passed all five routes and received 10 successful batch responses from the real API.
- Static schema checks: backend and browser aggregate exports match byte-for-byte; fitted model feature names match the 19-field API schema; `git diff --check` passed.

## Audit fixes

- Model fingerprints are attached to notebook exports, static exports, API health and batch predictions. The backend verifies the fitted artifact and its metadata; the browser accepts matching models and complete response IDs only.
- Delayed API scores are discarded when any current feature changes. Simulated observations use current lateral velocity immediately, matching the notebook's current-state input contract.
- Removed duplicate Simulator navigation and corrected the Automatum truck indicator legend and outdated one-recording description.
- Leaders remain visible while merging, dataset speeds do not undergo the old 45 m/s clamp, and preset changes preserve the loaded behavior model and traffic profile.
- GitHub Actions now checks the API/notebook and frontend tests before deployment, on Ubuntu 24.04 to match the browser package's supported runner.

The existing Render URL is retained as requested: `https://trafficrush-api.onrender.com`. During the audit it returned the old TrafficRush OpenAPI schema and 404 for `/api/model/info`; local backend integration passed. A push triggers the user's configured Render deployment. The frontend continues browser inference if that service is not yet compatible; remote API readiness must be checked after deployment before claiming it serves this model. `README.md` contains the hosted website and simulator links.
