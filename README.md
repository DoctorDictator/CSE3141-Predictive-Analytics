# NearMiss ? Highway Lane-Change Behavior Prediction

CSE3141 ? Predictive Analytics ? Manipal University Jaipur

**Research question:** Using a vehicle's present state, its vehicle-frame lateral velocity, and traffic in the current and adjacent lanes, can we predict whether its recorded lane ID will move to an adjacent lane during the next two seconds?

The target is observed lane-change behavior. It is not an accident label, a safety score, a lane recommendation, or proof that a maneuver is safe. The interactive simulator uses explicit movement, gap-assessment, braking, and collision-geometry rules; the statistical prediction does not control those rules.

## Website

- **[Open NearMiss](https://doctordictator.github.io/CSE3141-Predictive-Analytics/)**
- **[Open the lane-change simulator](https://doctordictator.github.io/CSE3141-Predictive-Analytics/#/simulator)**
- **[Open the model comparison](https://doctordictator.github.io/CSE3141-Predictive-Analytics/#/performance)**

The website uses aggregate notebook exports for its charts and browser-side inference from the saved fitted model. Render runs the FastAPI service independently. A matching static export keeps the interface available while the API starts or is offline.

## Dataset and labels

The project uses the public [Automatum Full Highway drone dataset](https://huggingface.co/datasets/AutomatumData/automatum-data-full-highway), which contains 114 German Autobahn recordings. It includes frame-rate vehicle positions and velocities, acceleration, lane and road IDs, dimensions, and current front/rear neighbors in the same and adjacent lanes. The source records no verified crashes.

The notebook samples assessment points once per second. It labels a point **lane change** if the same vehicle makes an adjacent non-divider lane-ID transition on the same road during the following complete two seconds; otherwise it labels **stay in lane**. Incomplete windows, lane-zero states, road changes, large frame gaps, and non-adjacent lane jumps are excluded. The feature vector includes longitudinal speed, acceleration, vehicle-frame lateral velocity, vehicle type, current leader gap and closing speed, plus front/rear gap, relative speed, and presence in both neighboring lanes. Neighbor gaps are projected onto the vehicle's heading, so rotated UTM roads are handled correctly. Missing neighbors use gap `150 m`, relative speed `0`, and presence `0`.

Recordings, rather than individual time points, are split into training, validation, and held-out test groups. The notebook compares Logistic Regression, LDA, and QDA using grouped cross-validation, validation-selected thresholds, F1, precision, recall, ROC-AUC, average precision, balanced accuracy, confusion matrices, and an always-stay baseline. It includes descriptive statistics, missing-value checks, IQR outlier counts and box plots. Unusual but physically valid braking and lateral movement are inspected instead of automatically dropped.

### Current held-out result

The current recording-level test selected QDA at a validation threshold of `0.40`: precision `0.130`, recall `0.498`, F1 `0.206`, ROC-AUC `0.800`, and average precision `0.121`. Lane-change windows are about `4.2%` of the eligible examples. Always predicting “stay” reaches `96.6%` accuracy but has zero lane-change recall and F1. These results show a useful behavior-pattern signal, but they are not strong enough for a safety alert or autonomous lane-change decision.

### Dataset license

Automatum publishes this dataset under [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/). The downloader verifies the publisher's SHA-256 and the archive's ZIP integrity. Raw files and processed vehicle-level samples remain in ignored local data folders. Local notebook replay uses measured trajectories; the public website exports aggregates and fitted model parameters, and generates new illustrative traffic scenes from aggregate distributions rather than publishing modified vehicle tracks.

Citation: Spannaus, Paul, Peter Zechel, and Kilian Lenz. ?AUTOMATUM DATA: Drone-based highway dataset for development and validation of automated driving software.? IEEE Intelligent Vehicles Symposium, 2021.

## Reproduce the analysis

Python 3.13+ and Node.js 22+ are recommended. From the repository root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\download_automatum.py
.venv\Scripts\python.exe scripts\nearmiss_notebook.py --execute
```

The download is about 4.2 GB. The analysis streams each recording directly from the archive; unpacking the approximately 39 GB expanded dataset is unnecessary. The processed sample cache and local replay are written under ignored `data/processed/`. The command regenerates `notebooks/NearMiss.ipynb`, trains the three syllabus models, and refreshes the backend and static aggregate artifacts.

To run the web app locally after installing dependencies:

```powershell
cd frontend
npm ci
cd ..
.venv\Scripts\python.exe run.py
```

Open http://127.0.0.1:5173/. The launcher starts FastAPI on port 8000 and Vite on 5173. Press Ctrl+C to stop services it started.

## What the simulator shows

?Automatum data distributions? generates a new traffic scene by sampling aggregate speed, acceleration, truck share, and vehicle dimensions. It does not copy a historical scene. The six-lane bypass and other presets use the simulator's programmed lane-change, clearance, braking, and collision rules. A model score estimates recorded lane-change behavior from the current simulated scene; it does not trigger a lane change or predict a crash.

## API

- `GET /api/health`
- `GET /api/model/info`
- `GET /api/model/metrics`
- `GET /api/analytics`
- `POST /api/predict`
- `POST /api/predict-batch`

Prediction inputs must follow the feature names and training ranges in `backend/artifacts/info.json`. Requests with missing-neighbor flags must use the documented capped-gap encoding. The model returns an estimated lane-change score and the validation-selected decision.

## Repository outputs

- `notebooks/NearMiss.ipynb` ? reproducible data processing, EDA, classification, evaluation, and conclusion.
- `scripts/nearmiss_notebook.py` ? notebook generator and execution entry point.
- `scripts/download_automatum.py` ? public dataset downloader with integrity checks.
- `backend/artifacts/` ? fitted model, schema, aggregate metrics, and analytics used by FastAPI.
- `frontend/public/data/` ? matching aggregate exports for the hosted static website.
