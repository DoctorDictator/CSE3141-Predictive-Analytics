# Verification — 4 November 2026

## Frontend production regression — 5 November 2026

Reproduced the live GitHub Pages blank screen: the previous traffic-volume API returned HTTP 200 for research requests, but its payload lacked NearMiss fields. Dashboard rendering threw `Cannot read properties of undefined (reading 'toFixed')` and React cleared the page. This was a runtime data-contract failure despite a successful build/deployment.

The frontend now validates research and prediction responses, uses validated saved NearMiss research exports for incompatible or unavailable APIs, and provides a page-level recovery screen for unexpected rendering errors. There is no automatic page-reload loop. README includes the public website and simulator links.

All **25 frontend unit/engine tests passed**. The production build was tested under `/CSE3141-Predictive-Analytics/`: **12 repeated research-page loads** against HTTP-200 legacy responses, successful saved-data fallback, correct static paths, malformed prediction handling and continuing simulation, with **zero missing assets or rendering errors**. The GitHub Pages workflow runs this production-browser regression before publication. Notebook and backend code were unchanged.

## Initial implementation evidence

Executed locally in `C:\Users\USER\predictive-analytics`.

| Check | Actual result |
|---|---|
| Notebook generator/execution | 26 cells; all code cells executed without errors |
| Genuine source | Cached complete DOT US-101 recording; live official row-count check; official class lookup downloaded |
| Data size | 1,180,598 observations; 2,169 vehicles; 952.8 seconds |
| Eligible 10 Hz windows | 993,381; positive fraction 0.2700 |
| Isolated 1 Hz model splits | 53,545 / 13,671 / 15,009 train / validation / test |
| Models actually fitted | Logistic Regression, LDA, QDA; three purged chronological CV folds each |
| Selected model | Logistic Regression; validation F1 0.9143 |
| Untouched test | Accuracy 0.9613; precision 0.9021; recall 0.8411; F1 0.8705; ROC-AUC 0.9815 |
| Persistence baseline | Test F1 0.8276 |
| Engine tests | 17 passed using Node's test runner |
| API/notebook tests | 17 passed using pytest |
| Production frontend build | `npm run build` passed |
| Real browser workflow | Passed using headless installed Chrome, desktop 1600 px and mobile 390 px |
| Actual browser inference | 9 successful real batch responses during the final workflow run |
| Local launcher | Started API, detected running Vite and reported readiness |

Engine coverage includes seeded generation across all 1–8 lane / 2–30 vehicle settings, no initial overlaps, road compression, integrated movement, progressive target-speed changes, braking, continuous lateral motion, rear-end and lane-change contacts, sequential incidents, reset, replay, valid pair construction and automatic avoidance.

API coverage compares the returned probability directly against the fitted artifact, checks endpoints/CORS, batch IDs and bounds, unsupported inputs, body overlaps, nonfinite values, unavailable artifacts, executed notebook outputs, current-only feature schema, and the notebook's actual future-window code using controlled complete, missing-frame, changed-leader and stopped-vehicle examples.

The browser exercised the exact 4-lane / 12-vehicle generation journey, visible movement, vehicle selection and controls, reset, real batch requests, all three incident presets, pre-impact rewind/replay, 30 cars, mobile layout, API outage recovery and rejection of delayed responses from discarded scenes. Desktop, mobile and collision screenshots under ignored `frontend/test-results/` were visually inspected. Narrow-screen clipped controls and SVG vehicle hit regions were corrected during this verification.

Known limits: one recording, uncalibrated probabilities, target includes persistence, and future eligibility is conditional on a stable leader/lane. Simulator contacts are geometric and hypothetical. No public frontend or live Render deployment was performed. Physical crash dynamics, optional automatic spawning and cross-recording external validation are not implemented. A dependency deprecation warning from Starlette's httpx TestClient was observed; tests passed.
