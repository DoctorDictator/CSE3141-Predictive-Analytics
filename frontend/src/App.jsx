import { Component, useEffect, useState } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  BarChart3,
  Car,
  Database,
  GitBranch,
  Info,
  Layers,
  Pause,
  Play,
  RotateCcw,
  TrafficCone,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import Simulator from "./Simulator.jsx";
import "./replay.css";
import {
  Button,
  Field,
  Stat,
  useResearch,
  pct,
  number,
} from "./components.jsx";

class PageBoundary extends Component {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <section role="alert" className="panel methodology">
        <h1>This page could not load.</h1>
        <p>Try again or choose another page.</p>
        <Button onClick={() => this.setState({ failed: false })}>
          Try again
        </Button>
      </section>
    ) : (
      this.props.children
    );
  }
}
function Heading({ eyebrow, title, children, source }) {
  return (
    <div className="page-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{children}</p>
      </div>
      {source && <span className="status">{source}</span>}
    </div>
  );
}
function Plot({ data, label }) {
  return (
    <ResponsiveContainer width="100%" height={230}>
      <BarChart data={data}>
        <CartesianGrid vertical={false} stroke="#e3eadf" />
        <XAxis
          dataKey="value"
          tick={{ fontSize: 10 }}
          tickFormatter={(v) =>
            typeof v === "number" ? Number(v).toFixed(1) : v
          }
          minTickGap={20}
        />
        <YAxis tick={{ fontSize: 10 }} width={60} />
        <Tooltip />
        <Bar
          dataKey="count"
          name={label || "Observations"}
          fill="#208773"
          radius={[3, 3, 0, 0]}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
function Dashboard() {
  const { data: a, source } = useResearch("analytics"),
    { data: m } = useResearch("metrics");
  return (
    <>
      <Heading
        eyebrow="NGSIM / OBSERVED DRIVER BEHAVIOR"
        title="Predict the next lane change."
        source={source}
      >
        Compare what the model expects with what recorded vehicles actually did.
      </Heading>
      {a ? (
        <>
          <div className="overview-stats">
            <Stat
              label="Recorded observations"
              value={number(a.rows)}
              note={`${number(a.vehicles)} vehicles · one recording`}
            />
            <Stat
              label="Observed lane crossings"
              value={number(a.recorded_crossings)}
              note="Actual adjacent mainline transitions"
            />
            <Stat
              label="Eligible future windows"
              value={number(a.eligible_rows)}
              note="Complete next 2 seconds at 10 Hz"
            />
            <Stat
              label="Selected classifier"
              value={m?.selected_model || "Loading…"}
              note={m ? `Test lane-change F1 ${pct(m.test.f1)}` : ""}
            />
          </div>
          <div className="dashboard-grid">
            <div className="panel feature-card">
              <Car size={34} />
              <div className="eyebrow">PREDICT / REPLAY / COMPARE</div>
              <h2>A forecast you can check.</h2>
              <p>
                Freeze a prediction using current and past measurements. Then
                play the next two seconds of real NGSIM traffic and reveal the
                observed outcome.
              </p>
              <NavLink className="link-button primary" to="/replay">
                Open recorded replay <ArrowRight size={16} />
              </NavLink>
              <div className="mini-road">
                {[0, 1, 2].map((i) => (
                  <div key={i}>
                    <Car
                      size={35}
                      style={{
                        marginLeft: `${30 + i * 40}px`,
                        color: "#62d5b6",
                      }}
                    />
                    <Car
                      size={35}
                      style={{ marginLeft: 70, color: "#97b7ff" }}
                    />
                  </div>
                ))}
              </div>
            </div>
            <div className="panel chart-card">
              <h2>Recorded vehicle speed distribution</h2>
              <Plot data={a.histograms.v_vel} />
              <p className="micro">
                Speed (m/s) versus observation count. These are measured data,
                not generated crash percentages.
              </p>
            </div>
          </div>
          <div className="panel methodology">
            <h2>One observed outcome. A separate simulation.</h2>
            <div>
              <p>
                <b>Prediction</b>
                <br />
                Will this vehicle cross into an adjacent lane during the next
                two seconds?
              </p>
              <p>
                <b>Recorded replay</b>
                <br />
                Actual positions unfold independently of the prediction,
                including incorrect forecasts.
              </p>
              <p>
                <b>Interactive simulator</b>
                <br />
                Programmed movement and clearance rules let you experiment.
                Model scores do not choose lanes or certify safety.
              </p>
            </div>
          </div>
          {m && (
            <div className="panel methodology">
              <h2>Rare events require honest metrics</h2>
              <p>
                Only {pct(m.class_counts.test["1"] / m.split_sizes.test)} of
                test windows contain a lane crossing. Always predicting “stay”
                achieves {pct(m.baseline.accuracy)} accuracy but zero
                lane-change recall and F1. The selected model has precision{" "}
                {pct(m.test.precision)}, recall {pct(m.test.recall)} and F1{" "}
                {pct(m.test.f1)}. This is a behavior-prediction experiment with
                limitations.
              </p>
            </div>
          )}
        </>
      ) : (
        <p>{source}</p>
      )}
    </>
  );
}

function RecordedReplay() {
  const { data, source } = useResearch("replay"),
    { data: info } = useResearch("info");
  const [selected, setSelected] = useState(0),
    [step, setStep] = useState(0),
    [playing, setPlaying] = useState(false),
    [rate, setRate] = useState(0.5);
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(
      () =>
        setStep((s) => {
          if (s >= 20) {
            setPlaying(false);
            return 20;
          }
          return s + 1;
        }),
      100 / rate,
    );
    return () => clearInterval(timer);
  }, [playing, rate]);
  if (!data)
    return (
      <>
        <Heading
          eyebrow="HELD-OUT NGSIM DATA"
          title="Prediction versus recorded outcome"
          source={source}
        >
          The forecast is frozen before the next two seconds are shown.
        </Heading>
        <p>{source}</p>
      </>
    );
  const e = data.examples[selected],
    frame = e.snapshots[step],
    ego = frame.cars.find((c) => c.id === e.vehicle_id),
    revealed = step === 20;
  const all = e.snapshots.flatMap((s) => s.cars),
    minX = Math.min(...all.map((c) => c.x - c.length)) - 10,
    maxX = Math.max(...all.map((c) => c.x)) + 15,
    width = Math.max(90, maxX - minX);
  const label = (v) => (v ? "Lane change" : "Stay in lane");
  const reset = () => {
    setPlaying(false);
    setStep(0);
  };
  return (
    <>
      <Heading
        eyebrow="HELD-OUT NGSIM DATA / 0.1 s OBSERVATIONS"
        title="Prediction versus recorded outcome"
        source={source}
      >
        The forecast uses assessment-time inputs only. Actual future positions
        control the replay.
      </Heading>
      <div className="replay-layout">
        <section>
          <div className="panel replay-road">
            <div className="road-toolbar">
              <span className="live-label">ACTUAL RECORDED TRAFFIC</span>
              <span className="spacer" />
              <span className="mono" data-testid="replay-time">
                {frame.seconds.toFixed(1)} / 2.0 s
              </span>
            </div>
            <svg
              className="recorded-highway"
              role="img"
              aria-label="Actual NGSIM traffic replay"
              viewBox={`${minX} 0 ${width} 21.5`}
            >
              <rect x={minX} width={width} height="21.5" fill="#2d4437" />
              <rect
                x={minX}
                y=".5"
                width={width}
                height="18.3"
                fill="#252f2b"
              />
              {[1, 2, 3, 4].map((i) => (
                <line
                  key={i}
                  x1={minX}
                  x2={minX + width}
                  y1={i * 3.6576}
                  y2={i * 3.6576}
                  stroke="#b4c0af"
                  strokeWidth=".12"
                  strokeDasharray="3 4"
                />
              ))}
              {[0.4, 18.5].map((y) => (
                <line
                  key={y}
                  x1={minX}
                  x2={minX + width}
                  y1={y}
                  y2={y}
                  stroke="#dbc97f"
                  strokeWidth=".12"
                />
              ))}
              {frame.cars.map((c) => (
                <g key={c.id}>
                  <rect
                    x={c.x - c.length}
                    y={c.y - c.width / 2}
                    width={c.length}
                    height={c.width}
                    rx=".25"
                    fill={c.id === e.vehicle_id ? "#4ed8bb" : "#99acc1"}
                    stroke={c.id === e.vehicle_id ? "#f5f7df" : "#12241c"}
                    strokeWidth=".15"
                  />
                  <rect
                    x={c.x - 1.2}
                    y={c.y - c.width * 0.35}
                    width=".5"
                    height={c.width * 0.7}
                    fill="#294447"
                  />
                  {c.id === e.vehicle_id && (
                    <text
                      x={c.x - c.length / 2}
                      y={c.y - c.width / 2 - 1.1}
                      textAnchor="middle"
                      fontSize="1.3"
                      fill="white"
                    >
                      Vehicle {c.id}
                    </text>
                  )}
                </g>
              ))}
              <text x={minX + 4} y="20.5" fill="#c0d3bd" fontSize="1.1">
                Recorded front-center positions · travel →
              </text>
            </svg>
            <div className="playback">
              <Button
                className="primary"
                icon={playing ? Pause : Play}
                onClick={() => {
                  if (step === 20) setStep(0);
                  setPlaying(!playing);
                }}
              >
                {playing ? "Pause replay" : "Play recorded future"}
              </Button>
              <Button icon={RotateCcw} onClick={reset}>
                Reset replay
              </Button>
              <select
                aria-label="Replay speed"
                value={rate}
                onChange={(ev) => setRate(+ev.target.value)}
              >
                {[0.5, 1, 2].map((v) => (
                  <option key={v} value={v}>
                    {v}× speed
                  </option>
                ))}
              </select>
            </div>
            <div className="replay-scrubber">
              <input
                aria-label="Recorded replay time"
                type="range"
                min="0"
                max="20"
                value={step}
                onChange={(ev) => {
                  setPlaying(false);
                  setStep(+ev.target.value);
                }}
              />
              <span>Assessment at 0.0 s</span>
              <span>Observed window complete at 2.0 s</span>
            </div>
          </div>
          <div className="prediction-grid">
            <div className="panel result-card">
              <span className="eyebrow">FROZEN MODEL FORECAST</span>
              <h2 data-testid="replay-prediction">{label(e.predicted)}</h2>
              <strong>{pct(e.probability)}</strong>
              <p>
                Estimated lane-change score at 0.0 s ·{" "}
                {info?.name || "Selected classifier"}
              </p>
              <small>
                Threshold {info ? pct(info.decision_threshold) : "…"};
                uncalibrated score.{" "}
                {e.supported
                  ? "Within marginal training ranges."
                  : "Offline test evaluation outside marginal training ranges; live inference would be withheld."}
              </small>
            </div>
            <div
              className={`panel result-card ${revealed ? (e.predicted === e.actual ? "correct-result" : "incorrect-result") : ""}`}
            >
              <span className="eyebrow">RECORDED OUTCOME</span>
              <h2 data-testid="replay-outcome">
                {revealed ? label(e.actual) : "Play to reveal"}
              </h2>
              <strong>
                {revealed
                  ? e.predicted === e.actual
                    ? "Correct"
                    : "Incorrect"
                  : "—"}
              </strong>
              <p>
                {revealed
                  ? "Checked against actual future lane IDs, not a rule-based animation."
                  : `Current recorded lane: ${ego?.lane ?? "outside view"}. Outcome is revealed after the complete window.`}
              </p>
            </div>
          </div>
          <p className="footnote">
            <Info size={14} />
            Examples cover correct and incorrect forecasts where available; they
            are not a random estimate of performance. Full test metrics appear
            in Model Comparison. Road markings are illustrative; vehicle
            positions/dimensions are recorded.
          </p>
        </section>
        <aside className="panel replay-inspector">
          <h2>Recorded observation</h2>
          <Field label="Held-out example">
            <select
              aria-label="Held-out example"
              value={selected}
              onChange={(ev) => {
                setSelected(+ev.target.value);
                reset();
              }}
            >
              {data.examples.map((v, i) => (
                <option key={v.id} value={i}>
                  Example {i + 1} · Vehicle {v.vehicle_id} · Lane{" "}
                  {v.initial_lane}
                </option>
              ))}
            </select>
          </Field>
          <p className="micro">
            Source: held-out recording block. Nearby vehicles are recorded scene
            context. No future measurements enter the forecast.
          </p>
          <div className="pair-info">
            {info?.features.map((key) => (
              <div key={key}>
                <span>{key.replaceAll("_", " ")}</span>
                <b>
                  {Number(e.inputs[key]).toFixed(key === "has_leader" ? 0 : 2)}
                </b>
              </div>
            ))}
          </div>
          <p className="micro">
            Lateral speed uses the previous 0.5 s. A maneuver may already be
            underway before its lane ID changes. This predicts crossing
            behavior, not a driver's hidden intention or maneuver safety.
          </p>
          <NavLink to="/performance" className="link-button primary">
            View full evaluation
          </NavLink>
        </aside>
      </div>
    </>
  );
}

function Analysis() {
  const { data: a, source } = useResearch("analytics");
  return (
    <>
      <Heading
        eyebrow="REAL TRAFFIC / DISTRIBUTIONS"
        title="How the recorded traffic behaves"
        source={source}
      >
        Observed measurements and lane-crossing frequencies, before any
        simulation rules.
      </Heading>
      {a ? (
        <>
          <div className="chart-grid">
            {[
              ["v_vel", "Speed · m/s"],
              ["v_acc", "Acceleration · m/s²"],
              ["front_gap", "Capped front bumper gap · m"],
              ["headway", "Front time headway · s (below 10)"],
              ["lane_id", "Recorded mainline lane occupancy"],
              ["v_class", "Vehicle type · 1 motorcycle / 2 car / 3 truck"],
            ].map(([key, title]) => (
              <div className="panel chart-card" key={key}>
                <h2>{title}</h2>
                <Plot data={a.histograms[key]} />
              </div>
            ))}
            <div className="panel chart-card">
              <h2>Complete future-window classes</h2>
              <Plot
                data={[
                  { value: "Stay in lane", count: a.class_counts["0"] || 0 },
                  { value: "Lane change", count: a.class_counts["1"] || 0 },
                ]}
              />
              <p className="micro">
                Overlapping positive windows can belong to the same crossing.
                These are observation windows, not unique maneuvers or crash
                labels.
              </p>
            </div>
            <div className="panel methodology">
              <h2>What is actually recorded?</h2>
              <p>
                {number(a.recorded_crossings)} adjacent crossings,{" "}
                {number(a.vehicles)} vehicles and {number(a.rows)} observations
                across {a.duration_seconds.toFixed(1)} seconds in one recording.
              </p>
              <p>
                Generated scenes can initialize from {a.traffic_seeds.length}{" "}
                coherent training-period snapshots. Adapting them to a different
                lane count or preventing initial overlaps changes the original
                scene. The separate Recorded replay preserves observed
                positions.
              </p>
            </div>
          </div>
          <div className="panel table-panel">
            <h2>Current-feature correlations</h2>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    {Object.keys(a.correlation).map((k) => (
                      <th key={k}>{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(a.correlation).map(([key, row]) => (
                    <tr key={key}>
                      <th>{key}</th>
                      {Object.keys(a.correlation).map((k) => (
                        <td key={k}>{row[k].toFixed(2)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="panel table-panel">
            <h2>Descriptive statistics</h2>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Measurement</th>
                    {["mean", "median", "var", "std", "min", "max"].map((k) => (
                      <th key={k}>{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(a.statistics).map(([key, row]) => (
                    <tr key={key}>
                      <th>{key}</th>
                      {["mean", "median", "var", "std", "min", "max"].map(
                        (k) => (
                          <td key={k}>{row[k].toFixed(2)}</td>
                        ),
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : (
        <p>{source}</p>
      )}
    </>
  );
}

function Comparison() {
  const { data: m, source } = useResearch("metrics"),
    { data: info } = useResearch("info");
  return (
    <>
      <Heading
        eyebrow="ONE TARGET / THREE SYLLABUS CLASSIFIERS"
        title="Compare actual behavior predictions"
        source={source}
      >
        Stay in lane versus a recorded adjacent crossing within two seconds.
      </Heading>
      {m ? (
        <>
          <div className="overview-stats">
            <Stat label="Selected model" value={m.selected_model} />
            <Stat label="Test lane-change F1" value={pct(m.test.f1)} />
            <Stat label="Test lane-change recall" value={pct(m.test.recall)} />
            <Stat
              label="Test average precision"
              value={pct(m.test.average_precision)}
            />
          </div>
          <div className="panel table-panel">
            <h2>Validation comparison</h2>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Algorithm</th>
                    <th>Train F1</th>
                    <th>CV F1 at 0.5</th>
                    <th>Validation cutoff</th>
                    {[
                      "precision",
                      "recall",
                      "f1",
                      "roc_auc",
                      "average_precision",
                    ].map((k) => (
                      <th key={k}>{k.replaceAll("_", " ")}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {m.comparison.map((row) => (
                    <tr
                      key={row.name}
                      className={
                        row.name === m.selected_model ? "selected-row" : ""
                      }
                    >
                      <th>{row.name}</th>
                      <td>{pct(row.training.f1)}</td>
                      <td>{pct(row.cv_f1)}</td>
                      <td>{row.threshold.toFixed(2)}</td>
                      {[
                        "precision",
                        "recall",
                        "f1",
                        "roc_auc",
                        "average_precision",
                      ].map((k) => (
                        <td key={k}>{pct(row.validation[k])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="chart-grid">
            <div className="panel matrix">
              <h2>Selected model · untouched test</h2>
              <p className="micro">
                Rows observed stay/change; columns predicted stay/change.
              </p>
              <div>
                {m.test.confusion_matrix.flat().map((v, i) => (
                  <strong key={i}>{number(v)}</strong>
                ))}
              </div>
              <p className="micro">
                Precision {pct(m.test.precision)} · Recall {pct(m.test.recall)}{" "}
                · F1 {pct(m.test.f1)} · ROC-AUC {pct(m.test.roc_auc)} · Average
                precision {pct(m.test.average_precision)}
              </p>
            </div>
            <div className="panel methodology">
              <h2>Always-stay baseline</h2>
              <p>
                Accuracy: {pct(m.baseline.accuracy)}
                <br />
                Lane-change precision, recall and F1: 0%
                <br />
                Average precision: {pct(m.baseline.average_precision)}
              </p>
              <p>
                Selected model accuracy: {pct(m.test.accuracy)}. High accuracy
                alone is not success when lane changes are rare.
              </p>
              <p>
                Decision cutoff: {info?.decision_threshold}. Cutoffs are
                selected on validation using a fixed grid, not on test.
              </p>
            </div>
          </div>
          <div className="panel table-panel">
            <h2>Partition distributions</h2>
            <table>
              <thead>
                <tr>
                  <th>Partition</th>
                  <th>Stay windows</th>
                  <th>Change windows</th>
                  <th>Total</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(m.class_counts).map(([name, counts]) => (
                  <tr key={name}>
                    <th>{name}</th>
                    <td>{number(counts["0"] || 0)}</td>
                    <td>{number(counts["1"] || 0)}</td>
                    <td>{number(m.split_sizes[name])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="panel methodology">
            <h2>Evaluation and limits</h2>
            <p>
              {m.protocol}. No vehicle ID is a model feature. Scaling fits
              within training folds. The test block is used only after
              selection.
            </p>
            <p>
              One recording, class imbalance and lateral movement already
              underway limit conclusions. The classifier predicts observed
              crossing behavior; it does not infer accident likelihood,
              recommend a lane, or establish safe driving.
            </p>
          </div>
        </>
      ) : (
        <p>{source}</p>
      )}
    </>
  );
}

function About() {
  const { data: info } = useResearch("info");
  return (
    <>
      <Heading
        eyebrow="CSE3141 / MANIPAL UNIVERSITY JAIPUR"
        title="A question the recording can answer."
      >
        NearMiss — Lane-Change Behavior Prediction
      </Heading>
      <div className="panel about">
        <h2>What does the model predict?</h2>
        <p>
          Whether a vehicle makes any recorded adjacent mainline lane-ID
          crossing within the next two seconds. Label 0 means the vehicle stays
          in its current lane throughout a complete window. Label 1 means a
          crossing was observed. Missing or truncated windows are excluded.
        </p>
        <h2>What does NGSIM contain?</h2>
        <p>
          <a
            href="https://www.fhwa.dot.gov/publications/research/operations/07030/index.cfm"
            target="_blank"
            rel="noreferrer"
          >
            FHWA's US-101 trajectory recording ↗
          </a>{" "}
          provides vehicle positions, lanes, speeds and dimensions every 0.1 s.
          This project uses one complete origin, not the entire three-period
          release. Vehicle classes are joined from the official source. There
          are no verified crash labels.
        </p>
        <h2>What goes into the prediction?</h2>
        <p>
          Current speed, acceleration, leader gap, closing speed and leader
          presence, plus absolute lateral motion over the preceding 0.5 s.
          Current reported leaders are matched at the same timestamp and lane. A
          maneuver can already be underway before lane-ID crossing, so this is
          not a guaranteed prediction of hidden intention before any movement.
        </p>
        <h2>What controls the cars?</h2>
        <p>
          Recorded replay uses actual future positions, independent of the
          forecast. Interactive simulation uses deterministic cruising, braking,
          clearance and lane-change rules. The behavior score neither selects a
          lane nor acts as a safety veto. Vehicle-body overlap detects
          hypothetical simulated contact. Already-collided vehicles remain
          stopped obstacles.
        </p>
        <h2>Are generated cars sampled from the dataset?</h2>
        <p>
          The default generator adapts coherent training-period snapshots. It
          preserves the selected vehicles' measured initial speeds,
          accelerations and sizes, while remapping lane indices and separating
          overlaps when necessary. Explicit synthetic random ranges and
          deliberately constructed presets are also available. Adapted scenes do
          not guarantee exact empirical distributions or reproduce historical
          accidents.
        </p>
        <h2>What changed from the earlier project?</h2>
        <p>
          Close-following and lane-change conflict proxy targets, models,
          metrics and safety-score controls have been replaced. All three
          syllabus classifiers now learn the directly observed stay/change
          outcome. Programmed simulation mechanics are retained separately.
        </p>
        {info && (
          <>
            <h2>Exported fitted model</h2>
            <p>
              {info.name} · {info.version} · Two-second horizon · Uncalibrated
              estimates
            </p>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Units</th>
                    <th>Training support</th>
                  </tr>
                </thead>
                <tbody>
                  {info.features.map((key) => (
                    <tr key={key}>
                      <td>{key}</td>
                      <td>{info.units[key]}</td>
                      <td>
                        {info.ranges[key].map((v) => v.toFixed(3)).join(" – ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </>
  );
}
const nav = [
  ["/", "Dashboard", Layers],
  ["/replay", "Recorded replay", Play],
  ["/analysis", "Distributions", BarChart3],
  ["/simulator", "Simulator", Car],
  ["/performance", "Model comparison", GitBranch],
  ["/about", "About", Info],
];
export default function App() {
  const location = useLocation();
  return (
    <div className="app">
      <header>
        <NavLink to="/" className="brand">
          <span>
            <TrafficCone size={23} />
          </span>
          <div>
            NearMiss<small>LANE-CHANGE BEHAVIOR PREDICTION</small>
          </div>
        </NavLink>
        <nav>
          {nav.map(([to, label, Icon]) => (
            <NavLink end={to === "/"} to={to} key={to}>
              <Icon size={15} />
              {label}
            </NavLink>
          ))}
        </nav>
        <span className="course">
          CSE3141<span>RESEARCH LAB</span>
        </span>
      </header>
      <main>
        <PageBoundary key={location.pathname}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/replay" element={<RecordedReplay />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/simulator" element={<Simulator />} />
            <Route path="/prediction" element={<RecordedReplay />} />
            <Route path="/performance" element={<Comparison />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<Dashboard />} />
          </Routes>
        </PageBoundary>
      </main>
      <footer>
        <span>NearMiss / Recorded lane-change behavior</span>
        <span>
          Observed outcomes. Honest forecasts. Separate simulation rules.
        </span>
      </footer>
    </div>
  );
}
