# NearMiss — Predicting Risky Vehicle Encounters and Future Vehicle Trajectories

A completed **CSE3141 Predictive Analytics mini project**, Manipal University Jaipur.
All executable project Python is inside one
[executed Jupyter Notebook](NearMiss_Complete_Predictive_Analytics.ipynb).

## Research questions

1. Can a currently noncritical same-lane following interaction enter a defined low-TTC condition within three seconds, and does recent history help predict it?
2. Can Gaussian identity-link GLMs forecast vehicle X/Y coordinates one second ahead, and do acceleration/variability features improve on basic past displacement?

Stage 1 was implemented, executed and verified before Stage 2 was added. In every
fresh sequential run, an explicit Stage 1 gate must pass before regression begins.
These are separate targets and models, sharing cleaned data.

## Real data and provenance

[USDOT NGSIM official catalog](https://data.transportation.gov/d/8ect-6jqj).
The notebook inspects the official schema and recording origins, then downloads the
**complete earliest US-101 recording**, origin 1118846978900, rather than an arbitrary page.
It contains **1,180,598 observations, 2,169 vehicles**, at 100-ms intervals.
There are 1,180,598 cleaned observations and 1,119,307 geometry-validated
same-lane pair observations. Original measurements use feet, feet/s and feet/s²;
the notebook converts to SI with 0.3048.

[Official US-101 dictionary](https://data.transportation.gov/api/views/8ect-6jqj/files/ddb2c29d-2ef4-4b67-94ea-b55169229bd9?download=true&filename=1-+US+101+Metadata+Documentation.pdf).
Local_X/Y are front-center coordinates. Clear gap is leader Y minus leader length
minus follower Y. Recorded leader/follower references, lane, ordering and timestamp
must agree. Source vehicle IDs repeat across recordings, so recording identity is
explicit. [FHWA program attribution and data access](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm).
Raw data is not redistributed in GitHub.

## Setup and run

Tested with **Python 3.13.9**. Use Python 3.13 for the pinned
scientific stack in requirements.txt.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m jupyterlab NearMiss_Complete_Predictive_Analytics.ipynb
```

Launch from this repository root, choose the environment's Python kernel, then
**Restart Kernel and Run All Cells**. All statistical logic, helpers, integrity
assertions, plots, fitted models and evaluations are in the notebook. No .py
modules, dashboard, traffic simulator or monitoring application is required.

The first full download is about 100 MB on disk and may take several minutes.
API calls use bounded pagination, retry/backoff for rate limits/server failures
and interrupted response bodies, and per-page caching in data/raw/. A SHA-256
manifest verifies subsequent local reuse (local integrity, not a publisher checksum).
The source-row count and recording-origin checks prevent silently accepting a
truncated interval. Subsequent runs reuse the cached recording.

**Local-file fallback:** export the selected complete recording from
https://data.transportation.gov/resource/8ect-6jqj.csv with
location='us-101' AND global_time-frame_id*100=1118846978900, requesting the columns
shown in the notebook. Save it as data/raw/us101_complete_recording.csv.
With no manifest, the notebook validates schema, exact expected row count and
recording origin before accepting it. Cached official catalog/recording metadata
or network access to those endpoints is needed to verify provenance. Failed
downloads produce a specific error; synthetic substitutes are never used.

## Shared preparation and evaluation controls

Missingness, malformed values, exact/conflicting duplicates, physical validity,
IQR/Z-score flags, continuity and leader-reference checks are reported. Statistical
extremes are investigated, not blindly removed. No interpolation bridges missing
observations. EDA includes distributions, speed/position traces, geometry,
descriptive statistics and correlations with purpose/interpretation.

One complete source recording supplies chronological 60%/20%/20% blocks with
a five-second embargo on either side of boundaries. Entire following encounters
(Stage 1) or vehicle trajectories (Stage 2) crossing blocks are excluded.
Every history-to-target support interval is checked against its block. There is
no random row split. This is one **purged predefined validation fold**, not
independent-day testing or repeated random cross-validation. Validation is reused
for selection and its estimates are optimistic. All preprocessing fits training only.

## Stage 1 — future surrogate risk classification

Primary label: any observed TTC ≤2 seconds in (t,t+3s], for a currently noncritical
pair. The cutoff is a project-defined experimental threshold; training/validation
sensitivity at 1.5/2/2.5 s is included. A complete unchanged, same-lane pair horizon
is required; missing future observations and leader/lane changes are censored,
never automatically safe. TTC uses positive closing speed; nonclosing cases have
infinite TTC and zero inverse TTC.

A two-second trailing window supplies historical summaries. Current-state features
are gap, closing speed and inverse TTC. The enriched model adds past gap trend,
relative-speed change, variability, mean acceleration and braking fraction.
Anchors are one second apart; overlapping windows stay within their partition.

Binary Logistic Regression uses a training-fit imputer/scaler Pipeline, L2
regularization, fixed balanced class weights and C∈{0.1,1,10} grid selection by
validation average precision. Probability thresholds maximize validation F1.
The current-TTC rule chooses its warning cutoff on validation, not test.

Training/validation/test anchors: **47,764 / 12,521 / 16,330**.
Test onset prevalence: **2.71%**.

| Model | precision | recall | f1 | roc_auc | average_precision |
|---|---:|---:|---:|---:|---:|
| TTC_rule | 0.268 | 0.305 | 0.285 | 0.747 | 0.198 |
| current | 0.257 | 0.293 | 0.274 | 0.862 | 0.201 |
| history | 0.277 | 0.269 | 0.273 | 0.869 | 0.202 |

**Research finding:** history-minus-current F1 -0.001, ROC-AUC
+0.006. History does not improve every metric; the rule's F1
is 0.285. History-model false positives:
310; false negatives: 324. Never warning would yield
97.29% accuracy with zero onset recall, illustrating why
accuracy alone is misleading.

Class-weighted probabilities are poorly calibrated: history Brier score
0.262, versus
0.027 for a constant training-prevalence probability.
ROC/PR curves, confusion counts and a calibration plot are included. Ranking
performance is not evidence of reliable real-world accident probabilities.

## Stage 2 — coordinate forecasting

Two **statsmodels Gaussian GLMs with identity links** predict one-second ΔX/ΔY.
Add predicted displacement to current coordinates to reconstruct X/Y. The basic
feature set uses past one-second displacement; the enriched set adds speed,
acceleration, changes in velocity and displacement variability. A constant-velocity
baseline extrapolates the same past one-second average velocity. All methods use
the same samples; no future coordinate is an input.

Training/validation/test endpoints: **60,399 / 14,850 / 18,020**,
with disjoint vehicle IDs. Validation selected **enriched** GLMs by mean
Euclidean endpoint error. All errors below are metres; R² is on displacement.

| Model | MAE_X | MAE_Y | RMSE_X | RMSE_Y | R2_delta_X | R2_delta_Y | mean_position_error | median_position_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| constant_velocity | 0.135 | 0.638 | 0.247 | 0.932 | -0.506 | 0.961 | 0.684 | 0.541 |
| basic | 0.111 | 0.657 | 0.195 | 0.934 | 0.057 | 0.961 | 0.690 | 0.504 |
| enriched | 0.104 | 0.393 | 0.187 | 0.636 | 0.141 | 0.982 | 0.433 | 0.286 |

Enriched-minus-basic mean endpoint error: **-0.257 m** (negative
means improvement). Charts show actual/predicted X/Y, error distributions, residual
diagnostics, feature-group comparisons and a real held-out endpoint forecast.
Coordinate forecasting is not demonstrated to improve Stage 1 risk predictions.

## Limitations

This predicts a recorded **surrogate condition**, not confirmed accidents.
One recording, reconstruction noise, temporal drift, within-encounter dependence
and complete-horizon selection limit generalization. Excluded lane changes and
short/disappearing encounters are outside the stable-following cohort.
Stage 1 needs independent probability calibration. Stage 2 Gaussian working
assumptions are diagnosed, not presumed; ordinary independent-error inferential
p-values are not used as evidence. No causal, autonomous-driving or deployment
claim is made. Future studies need independent intervals/sites, grouped temporal
validation, cluster uncertainty and a separate calibration block.

## Submission structure

```text
NearMiss_Complete_Predictive_Analytics.ipynb   # all code, outputs and conclusions
README.md
requirements.txt
.gitignore
data/raw/.gitkeep                            # raw recordings remain local
```

All charts and results are embedded in the notebook; no external Python or model
files are needed. README and requirements are populated by its final executed cell.

## References

- USDOT/FHWA NGSIM catalog and US-101 data dictionary linked above.
- [FHWA Surrogate Safety Assessment Model](https://www.fhwa.dot.gov/publications/research/safety/08051/): TTC context; the two-second onset cutoff is this project's design choice.
- [statsmodels GLM documentation](https://www.statsmodels.org/stable/glm.html).
- [scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
