import { Component, useEffect, useState } from "react";
import { Navigate, NavLink, Route, Routes, useLocation } from "react-router-dom";
import {
  ArrowRight,
  BarChart3,
  Car,
  GitBranch,
  Info,
  Layers,
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
        eyebrow="AUTOMATUM / OBSERVED DRIVER BEHAVIOR"
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
              note="1-second assessment points; complete 2-second future"
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
                explore dataset-wide lane-change patterns, then run a model-scored
                traffic simulation.
              </p>
              <NavLink className="link-button primary" to="/simulator">
                Open the lane-change simulator <ArrowRight size={16} />
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
                <b>Data-informed simulation</b>
                <br />
                Generated scenes follow observed distributions; their traffic is not a recorded replay.
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
  return <Navigate to="/simulator" replace />;
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
                across {a.duration_seconds.toFixed(1)} seconds in {a.recording_count} recording blocks.
              </p>
              <p>
                Generated scenes sample the aggregate speed, acceleration, vehicle-type
                and size distributions. They are new simulations, not copies of
                historical vehicle positions. The notebook keeps source-trajectory
                replay local to avoid publishing modified tracks.
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
        <h2>What does Automatum contain?</h2>
        <p>
          <a
            href="https://huggingface.co/datasets/AutomatumData/automatum-data-full-highway"
            target="_blank"
            rel="noreferrer"
          >
            Automatum Full Highway dataset ?
          </a>{" "}
          contains drone-tracked German Autobahn traffic, including positions,
          lane and road IDs, speeds, acceleration, vehicle dimensions and
          current-lane/neighbor relationships at about 30 frames per second. It
          records no verified crash labels. The published data uses CC BY-ND
          4.0, so this public site shows aggregate results rather than modified
          source tracks.
        </p>
        <h2>What goes into the prediction?</h2>
        <p>
          Longitudinal speed, acceleration, vehicle-frame lateral velocity, vehicle type,
          front-leader gap and closing speed, plus front/rear gaps and relative
          speeds in each adjacent lane. Missing neighbors use an explicit
          presence flag and a capped-gap value. The target is an observed lane-ID
          crossing in the next two seconds; a maneuver can already be underway.
        </p>
        <h2>What controls the cars?</h2>
        <p>
          The website generates new traffic scenes from aggregate dataset
          distributions. Its lane-change and braking rules are programmed and
          separate from the learned behavior score. The score does not choose a
          lane or act as a safety veto. Vehicle-body overlap detects hypothetical
          simulated contact; the dataset provides no crash outcome labels.
        </p>
        <h2>Are generated cars sampled from the dataset?</h2>
        <p>
          The default generator samples aggregate speed, acceleration, truck
          share and vehicle dimensions from the training data. Positions and lane
          assignments are newly generated, and the scene is not a historical
          replay. Fully synthetic ranges and deliberately constructed presets
          remain available.
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
  ["/simulator", "Lane-change lab", Car],
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
