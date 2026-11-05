# Lane-change behavior migration — 5 November 2026

The old close-following and lane-change conflict models have been replaced by one directly observed behavior target: any adjacent mainline lane-ID crossing during the next complete two seconds. Binary labels are derived from actual future lane IDs; missing/truncated windows are censored. Classification does not use accident or headway thresholds.

- **31 notebook cells**, all executed without errors.
- Genuine cached US-101 source: 1,180,598 rows / 2,169 vehicles / 952.8 seconds; official live row-count and origin checks executed.
- **812 observed adjacent crossings**, **1,103,092 eligible 10 Hz windows**; 1 Hz isolated model partitions: **58,757 / 14,897 / 17,625**.
- Logistic Regression, LDA and QDA fitted with identical data, fold-fitted scaling, purged chronological CV and validation-selected decision cutoffs.
- Selected **Logistic Regression**, cutoff **0.10**. Test F1 **0.2302**, precision **0.2270**, recall **0.2336**, ROC-AUC **0.8873**, average precision **0.1645**. Confusion matrix `[[17379,109],[105,32]]`.
- Always-stay baseline accuracy **0.9922**, but lane-change F1/recall **0**. Selected accuracy **0.9879**. Rare positive frequency and limited current observations mean forecasting remains weak; this is explicitly shown in the notebook, website and README.
- **16 held-out recorded replay examples** cover all four confusion-matrix outcomes. Each frozen score is computed from assessment inputs; the actual future trajectory drives playback independently.
- **11 training-period traffic snapshots** provide seeded, explicitly adapted initialization for generated scenes. Preset and synthetic scenes remain distinct from recorded replay.
- Old conflict artifacts, endpoints and model-based safety vetoes removed. Behavior predictions do not control lane selection or geometric contact detection.

Executed checks:

- 31 frontend/engine tests: actual portable-model/sklearn equivalence, QDA portable math, correct cutoff, legacy-task rejection, past-only simulation features, snapshot reproducibility and retained motion/contact/bypass mechanics.
- 17 API/notebook tests: genuine fitted output, batch limits/IDs, physical/finite inputs, unsupported feature handling, unavailable artifacts, recorded labels, executed cells and direct controlled-window cases including return crossings, missing frames, nonadjacent jumps and truncated futures.
- Production build passed.
- Production browser suite passed under the GitHub Pages subpath: six routes and refreshes, all 16 observed outcomes, frozen predictions, actual replay playback, 4-lane/12-car adapted initialization, independent bypass, blocked alternatives, three contact/replay scenarios, 30 cars, mobile layout and incompatible old API responses. No render errors or missing built assets.

Limitations: one recording, rare lane crossings, uncalibrated scores, missing hidden driver intent and possible cues from lateral motion already underway. No verified crash prediction, safe-maneuver certification, causal recommendation or exact-distribution claim for adapted synthetic scenes is made. Direct replay is the evidence-backed demonstration; generated scenarios are separate simulation experiments.
