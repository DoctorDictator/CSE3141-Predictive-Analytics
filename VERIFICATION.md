# Verification — 4 November 2026

## Blocked-lane avoidance and observed-maneuver ML — 5 November 2026

Fixed the speed-dependent automatic-lane trigger that could leave an uncollided stopped follower stuck behind an obstacle. The normal frontend preset enables automatic bypass; explicit controls show whether it is enabled. Added early congestion detection, zero-speed reassessment, front/rear clearance, rear closing-speed checks, projected lateral path checks, simultaneous-merge reservations and progress-advantage checks. Completion events are recorded after contact detection, so a collision in the completion step cannot be mislabeled as a successful maneuver.

The 33-cell notebook executed in full. It found 812 actual adjacent mainline lane transitions, retaining 481 complete observed maneuvers. Separate lane-change Logistic Regression/LDA/QDA training uses five pre-crossing inputs, isolated chronological partitions (296/78/49), purged training CV and actual future close-conflict proxy labels. Selected Logistic Regression test F1 is 0.8333 versus always-conflict baseline 0.7901; ROC-AUC is 0.9099. These results are based on just 49 test maneuvers and are not verified crash prediction or counterfactual safety evidence.

**40 frontend tests and 20 API/notebook tests passed.** Coverage includes the zero-speed empty-lane case, safe six-lane bypass, all-lanes-blocked waiting, fast rear traffic, an unsafe lateral path despite an empty target lane, simultaneous merging, already-collided obstacles, model veto, stale neighbor identity and acceleration overrides. Exact browser fitted-model outputs match held-out scikit-learn predictions to `1e-12`; no probability is invented for unsupported inputs. API outputs match the same joblib artifact. Controlled future-window tests verify actual outcomes and exclusion of missing observations.

The production browser test passed under the GitHub Pages subpath with legacy API responses: safe bypass with no contact, waiting when blocked, a visible genuine local fitted score, deliberate unsafe contact, actual recorded-trajectory playback, mobile layout and no render/missing-asset errors. The original full browser smoke workflow also passed with 12 successful actual following-pair inference batches. Desktop and mobile screenshots were inspected. The API supports the new lane-change model independently; simulation and portable lane-change inference remain usable during a backend cold start.

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
