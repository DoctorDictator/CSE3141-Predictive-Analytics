import { Component, useEffect, useRef, useState } from "react";
import { NavLink, Routes, Route, useLocation } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  BarChart3,
  Car,
  ChevronLeft,
  ChevronRight,
  Database,
  Gauge,
  GitBranch,
  Info,
  Layers,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Scan,
  Settings2,
  ShieldAlert,
  TrafficCone,
  Trash2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ScatterChart,
  Scatter,
} from "recharts";
import {
  createWorld,
  generate,
  saveInitial,
  reset,
  tick,
  STEP,
  LANE_WIDTH,
  SCENARIOS,
  leader,
  brake,
  changeLane,
  setLanes,
  loadScenario,
  addCar,
  rewind,
  observations,
  event,
  laneAssessment,
  requestLaneChange,
  laneObservations,
} from "./simulation";
import { request, loadResearch } from "./services/api";
import { validatePredictions } from "./services/research-schema";

class PageBoundary extends Component {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (this.state.failed)
      return (
        <section className="panel methodology" role="alert">
          <h1>This page couldn't load.</h1>
          <p>Try again or choose another page from the navigation.</p>
          <Button onClick={() => this.setState({ failed: false })}>
            Try again
          </Button>
        </section>
      );
    return this.props.children;
  }
}

const pct = (x) => `${(x * 100).toFixed(1)}%`;
const num = (x) => Number(x).toLocaleString("en-US");
function Button({ children, icon: Icon, ...props }) {
  return (
    <button {...props}>
      {Icon && <Icon size={15} />} {children}
    </button>
  );
}
function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}
function Stat({ label, value, note }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
      {note && <small>{note}</small>}
    </div>
  );
}
function useResearch(name) {
  const [state, set] = useState({
    data: null,
    source: "Loading notebook exports…",
  });
  useEffect(() => {
    let active = true;
    loadResearch(name)
      .then((value) => active && set(value))
      .catch((e) => active && set({ data: null, source: e.message }));
    return () => {
      active = false;
    };
  }, [name]);
  return state;
}

function Simulator() {
  const world = useRef(null);
  if (!world.current) {
    world.current = createWorld();
    loadScenario(world.current, SCENARIOS[0]);
  }
  const w = world.current,
    [running, setRunning] = useState(false),
    runRef = useRef(false),
    [version, render] = useState(0);
  const { data: laneInfo, source: laneSource } = useResearch("lane-info");
  useEffect(() => {
    w.laneModel = laneInfo;
  }, [laneInfo]);
  const [count, setCount] = useState(6),
    [seed, setSeed] = useState("42"),
    [scenario, setScenario] = useState(SCENARIOS[0]);
  const [selected, select] = useState(1),
    [rate, setRate] = useState(1),
    rateRef = useRef(1);
  const [camera, setCamera] = useState("fit"),
    [zoom, setZoom] = useState(1),
    [pan, setPan] = useState(0);
  const [scores, setScores] = useState({}),
    [apiStatus, setApiStatus] = useState("Connecting to model…");
  const generation = useRef(0),
    scoreRef = useRef({}),
    lastElevated = useRef(new Set());
  const update = () => render((v) => v + 1);
  const clearScores = () => {
    generation.current++;
    scoreRef.current = {};
    lastElevated.current = new Set();
    w.laneScores = {};
    setScores({});
  };
  const mutate = (fn) => {
    fn();
    w.epoch++;
    clearScores();
    update();
  };
  useEffect(() => {
    runRef.current = running;
  }, [running]);
  useEffect(() => {
    rateRef.current = rate;
  }, [rate]);
  useEffect(() => {
    let frame,
      previous = performance.now(),
      accumulator = 0,
      lastDraw = 0;
    const animate = (now) => {
      const elapsed = Math.min(0.1, (now - previous) / 1000);
      previous = now;
      if (runRef.current) {
        accumulator += elapsed * rateRef.current;
        while (accumulator >= STEP) {
          tick(world.current, STEP);
          accumulator -= STEP;
        }
      } else accumulator = 0;
      if (now - lastDraw > 40) {
        render((v) => v + 1);
        lastDraw = now;
      }
      frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, []);
  useEffect(() => {
    let disposed = false,
      timer,
      controller;
    const poll = async () => {
      const current = world.current,
        candidates = laneObservations(current),
        stamp = generation.current,
        epoch = current.epoch;
      if (candidates.length) {
        controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 8000);
        try {
          const response = validatePredictions(
            await request("/api/lane-change/predict-batch", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              signal: controller.signal,
              body: JSON.stringify({
                pairs: candidates.map(({ id, inputs }) => ({ id, inputs })),
              }),
            }),
          );
          if (
            !disposed &&
            stamp === generation.current &&
            epoch === current.epoch
          ) {
            const now = laneObservations(current),
              next = {};
            for (const p of response.predictions) {
              const before = candidates.find((c) => c.id === p.id),
                after = now.find((c) => c.id === p.id);
              if (before && after && before.signature === after.signature)
                next[p.id] = {
                  ...p,
                  signature: after.signature,
                  at: current.time,
                  wall: Date.now(),
                };
            }
            current.laneScores = next;
          }
        } catch {
          if (!disposed && stamp === generation.current)
            current.laneScores = {};
        } finally {
          clearTimeout(timeout);
        }
      }
      if (!disposed) timer = setTimeout(poll, 1000);
    };
    poll();
    return () => {
      disposed = true;
      clearTimeout(timer);
      controller?.abort();
    };
  }, []);
  useEffect(() => {
    let disposed = false,
      timer,
      controller;
    const poll = async () => {
      const current = world.current,
        pairs = observations(current),
        stamp = generation.current,
        epoch = current.epoch;
      if (!pairs.length) {
        setScores({});
        scoreRef.current = {};
        setApiStatus("No applicable following-pair prediction");
      } else {
        controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 12000);
        try {
          const response = await request("/api/predict-batch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: controller.signal,
            body: JSON.stringify({
              pairs: pairs.map(({ id, inputs }) => ({ id, inputs })),
            }),
          });
          if (
            disposed ||
            stamp !== generation.current ||
            epoch !== current.epoch
          ) {
            if (!disposed) timer = setTimeout(poll, 1000);
            return;
          }
          const currentPairs = observations(current),
            mapped = {};
          for (const prediction of validatePredictions(response).predictions) {
            const before = pairs.find((p) => p.id === prediction.id),
              now = currentPairs.find((p) => p.id === prediction.id);
            if (!before || !now || before.leaderId !== now.leaderId) continue;
            mapped[prediction.id] = {
              ...prediction,
              at: current.time,
              wall: Date.now(),
              leaderId: now.leaderId,
            };
            if (
              prediction.eligible &&
              prediction.probability >= 0.7 &&
              !lastElevated.current.has(prediction.id)
            )
              event(
                current,
                `Vehicle ${prediction.id}: elevated model close-following score (${pct(prediction.probability)})`,
                "risk",
              );
          }
          lastElevated.current = new Set(
            Object.values(mapped)
              .filter((p) => p.eligible && p.probability >= 0.7)
              .map((p) => p.id),
          );
          scoreRef.current = mapped;
          setScores(mapped);
          setApiStatus("Model connected · batch updates");
        } catch (e) {
          if (!disposed) {
            setApiStatus("Predictions unavailable · retrying");
            setScores({});
            scoreRef.current = {};
          }
        } finally {
          clearTimeout(timeout);
        }
      }
      if (!disposed) timer = setTimeout(poll, 1000);
    };
    poll();
    return () => {
      disposed = true;
      clearTimeout(timer);
      controller?.abort();
    };
  }, []);
  const freshScore = (c) => {
    const s = scores[c.id],
      l = leader(w, c);
    return s &&
      l &&
      l.id === s.leaderId &&
      Date.now() - s.wall < 5000 &&
      !c.crashed &&
      !c.changing
      ? s
      : null;
  };
  const car = w.cars.find((c) => c.id === selected),
    ahead = car && leader(w, car);
  const laneOptions = car
    ? [car.lane - 1, car.lane + 1]
        .filter((l) => l >= 0 && l < w.lanes)
        .map((l) => laneAssessment(w, car, l))
    : [];
  const valid = w.cars.map(freshScore).filter((s) => s?.eligible),
    average =
      w.cars.reduce((a, c) => a + c.speed, 0) / Math.max(1, w.cars.length);
  let minX = Math.min(...w.cars.map((c) => c.x - c.length - 20), 0),
    maxX = Math.max(130, ...w.cars.map((c) => c.x + 20));
  // Fit tracks the moving group; it must not keep the origin pinned to the entrance forever.
  if (w.time > 0 && w.cars.length)
    minX = Math.min(...w.cars.map((c) => c.x - c.length - 20));
  let width = Math.max(120, maxX - minX) / zoom;
  let origin = (camera === "follow" && car ? car.x - width * 0.35 : minX) + pan;
  const roadHeight = w.lanes * LANE_WIDTH;
  const height = Math.max(roadHeight + 10, width * 0.34),
    top = (height - roadHeight) / 2;
  const roadY = (y) => top + y;
  const resetScene = () => {
    setRunning(false);
    mutate(() => reset(w));
    setPan(0);
  };
  const randomize = () => {
    setRunning(false);
    mutate(() => {
      w.cars = generate(
        w.lanes,
        count,
        seed === "" ? Date.now() : Number(seed),
      );
      w.time = 0;
      w.events = [];
      w.collisions = [];
      w.history = [];
      w.laneChanges = 0;
      w.completedLaneChanges = 0;
      w.historyClock = 0;
      w.nextId = count + 1;
      saveInitial(w);
    });
    select(1);
  };
  const score = car && freshScore(car);
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">INTERACTIVE LAB / 01</div>
          <h1>Highway simulator</h1>
          <p>
            Build a traffic scene. Change its behavior. Observe what happens.
          </p>
        </div>
        <span
          className={`status ${apiStatus.startsWith("Model connected") ? "online" : ""}`}
        >
          <i />
          {apiStatus}
        </span>
      </div>
      <div className="lane-banner panel">
        <div>
          <strong>Can the car get around a blocked lane?</strong>
          <p>
            Assess the target lane, watch the maneuver, then check whether
            contact occurred.
          </p>
        </div>
        <Button
          className="primary"
          onClick={() => {
            setRunning(false);
            mutate(() => loadScenario(w, "Six-Lane Blockage - Safe Bypass"));
            setScenario("Six-Lane Blockage - Safe Bypass");
            setCount(w.cars.length);
            select(2);
            setPan(0);
          }}
        >
          Six-lane bypass demo
        </Button>
      </div>
      <div className="sim-layout">
        <aside className="panel config">
          <div className="panel-title">
            <Settings2 size={17} />
            <h2>Traffic setup</h2>
          </div>
          <Field label="Number of Lanes">
            <select
              aria-label="Number of Lanes"
              value={w.lanes}
              onChange={(e) => {
                setRunning(false);
                mutate(() => {
                  setLanes(w, Number(e.target.value));
                  saveInitial(w);
                });
              }}
            >
              {Array.from({ length: 8 }, (_, i) => (
                <option key={i} value={i + 1}>
                  {i + 1} {i === 0 ? "lane" : "lanes"}
                </option>
              ))}
            </select>
          </Field>
          <Field label={`Number of vehicles · ${count}`}>
            <input
              aria-label="Number of vehicles"
              type="range"
              min="2"
              max="30"
              value={count}
              onChange={(e) => setCount(+e.target.value)}
            />
            <div className="range-ends">
              <span>2 vehicles</span>
              <span>30 vehicles</span>
            </div>
          </Field>
          <Field label="Random seed (optional)">
            <input
              aria-label="Random seed"
              type="number"
              value={seed}
              placeholder="Random"
              onChange={(e) => setSeed(e.target.value)}
            />
          </Field>
          <Button className="primary full" icon={RotateCcw} onClick={randomize}>
            Generate Random Traffic
          </Button>
          <div className="divider" />
          <div className="eyebrow">SCENARIO LIBRARY</div>
          <Field label="Reproducible scenario">
            <select
              aria-label="Scenario"
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
            >
              {SCENARIOS.map((s) => (
                <option key={s}>{s}</option>
              ))}
            </select>
          </Field>
          <Button
            className="full"
            icon={Layers}
            onClick={() => {
              setRunning(false);
              mutate(() => loadScenario(w, scenario));
              setCount(w.cars.length);
              select(w.cars.find((c) => c.speed > 0)?.id || w.cars[0].id);
              setPan(0);
            }}
          >
            Load scenario
          </Button>
          <div className="divider" />
          <Field label="Traffic mode">
            <select
              aria-label="Traffic mode"
              value={w.mode}
              onChange={(e) => mutate(() => (w.mode = e.target.value))}
            >
              <option>Manual Traffic</option>
              <option>Automatic Traffic</option>
            </select>
          </Field>
          <label className="check">
            <input
              type="checkbox"
              checked={w.avoidance}
              onChange={(e) => mutate(() => (w.avoidance = e.target.checked))}
            />
            Automatic collision avoidance
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={w.autoLane}
              onChange={(e) => mutate(() => (w.autoLane = e.target.checked))}
            />
            Automatic lane decisions
          </label>
          <p className="micro">
            Automatic mode checks adjacent lanes before stopping, including an
            empty lane. Geometry checks the full path; a supported fitted-model
            score can veto an elevated close-conflict estimate. Already-collided
            cars remain obstacles.
          </p>
        </aside>
        <section className="simulation-main">
          <div className="panel road-panel">
            <div className="road-toolbar">
              <div className="live-label">
                <span className={running ? "pulse" : ""} />
                {running ? "SIMULATION RUNNING" : "SIMULATION PAUSED"}
              </div>
              <span className="mono" data-testid="sim-time">
                {w.time.toFixed(1)} s
              </span>
              <span className="spacer" />
              <Button
                icon={Plus}
                onClick={() =>
                  mutate(() => {
                    const c = addCar(w);
                    if (c) select(c.id);
                  })
                }
              >
                Add vehicle
              </Button>
            </div>
            <div className="road-stage">
              <div className="road-caption">
                <span>HIGHWAY / {w.lanes} LANES</span>
                <span>
                  TRAVEL DIRECTION <ArrowRight size={15} />
                </span>
              </div>
              <svg
                className="highway"
                role="img"
                aria-label="Interactive moving highway; select a vehicle to edit"
                viewBox={`${origin} 0 ${width} ${height}`}
              >
                <defs>
                  <pattern
                    id="terrain"
                    width="7"
                    height="7"
                    patternUnits="userSpaceOnUse"
                  >
                    <circle cx="2" cy="2" r=".18" fill="#547464" />
                  </pattern>
                  <pattern
                    id="asphalt"
                    width="1.8"
                    height="1.8"
                    patternUnits="userSpaceOnUse"
                  >
                    <circle cx=".5" cy=".5" r=".035" fill="#819287" />
                  </pattern>
                </defs>
                <rect x={origin} width={width} height={height} fill="#314c3d" />
                <rect
                  x={origin}
                  width={width}
                  height={height}
                  fill="url(#terrain)"
                />
                <rect
                  x={origin}
                  y={top - 1.4}
                  width={width}
                  height={roadHeight + 2.8}
                  fill="#879187"
                />
                <rect
                  x={origin}
                  y={top - 0.8}
                  width={width}
                  height={roadHeight + 1.6}
                  fill="#242e2b"
                />
                <rect
                  x={origin}
                  y={top}
                  width={width}
                  height={roadHeight}
                  fill="url(#asphalt)"
                />
                {[top - 0.35, top + roadHeight + 0.35].map((y) => (
                  <line
                    key={y}
                    x1={origin}
                    x2={origin + width}
                    y1={y}
                    y2={y}
                    stroke="#e3cc78"
                    strokeWidth=".12"
                  />
                ))}
                {Array.from({ length: w.lanes - 1 }, (_, i) => (
                  <line
                    key={i}
                    x1={origin}
                    x2={origin + width}
                    y1={roadY((i + 1) * LANE_WIDTH)}
                    y2={roadY((i + 1) * LANE_WIDTH)}
                    stroke="#c0c8bb"
                    strokeWidth=".12"
                    strokeDasharray="3 5"
                  />
                ))}
                {Array.from({ length: w.lanes }, (_, i) => (
                  <text
                    key={i}
                    x={origin + 3}
                    y={roadY((i + 0.5) * LANE_WIDTH) + 0.4}
                    fill="#a0afa3"
                    fontSize="1.1"
                  >
                    L{i + 1} →
                  </text>
                ))}
                {w.cars.map((c) => {
                  const l = leader(w, c),
                    s = freshScore(c);
                  return s?.eligible && l ? (
                    <line
                      key={`gap-${c.id}`}
                      x1={c.x}
                      x2={l.x - l.length}
                      y1={roadY(c.y)}
                      y2={roadY(c.y)}
                      stroke={s.probability >= 0.7 ? "#ffb65d" : "#57d9b5"}
                      strokeDasharray=".6 .6"
                      strokeWidth=".13"
                      pointerEvents="none"
                    />
                  ) : null;
                })}
                {w.cars.map((c) => {
                  const l = leader(w, c),
                    s = freshScore(c),
                    risk = s?.eligible ? s.probability : null;
                  const color = c.crashed
                    ? "#ff6c66"
                    : risk != null && risk >= 0.7
                      ? "#ffb65d"
                      : c.color;
                  return (
                    <g
                      key={c.id}
                      onClick={() => select(c.id)}
                      tabIndex="0"
                      role="button"
                      aria-label={`Vehicle ${c.id}`}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") select(c.id);
                      }}
                      style={{ cursor: "pointer" }}
                    >
                      {selected === c.id && (
                        <rect
                          x={c.x - c.length - 0.5}
                          y={roadY(c.y) - c.width / 2 - 0.5}
                          width={c.length + 1}
                          height={c.width + 1}
                          rx=".6"
                          fill="none"
                          stroke="white"
                          strokeWidth=".12"
                        />
                      )}
                      <rect
                        x={c.x - c.length}
                        y={roadY(c.y) - c.width / 2}
                        width={c.length}
                        height={c.width}
                        rx=".35"
                        fill={color}
                        stroke="#15251f"
                        strokeWidth=".12"
                      />
                      <rect
                        x={c.x - 1.5}
                        y={roadY(c.y) - c.width * 0.38}
                        width=".75"
                        height={c.width * 0.76}
                        rx=".1"
                        fill="#284a47"
                      />
                      <rect
                        x={c.x - c.length + 0.55}
                        y={roadY(c.y) - c.width * 0.38}
                        width=".55"
                        height={c.width * 0.76}
                        rx=".1"
                        fill="#284a47"
                      />
                      <line
                        x1={c.x - 0.15}
                        x2={c.x - 0.15}
                        y1={roadY(c.y) - c.width * 0.35}
                        y2={roadY(c.y) + c.width * 0.35}
                        stroke="#fff5bd"
                        strokeWidth=".16"
                      />
                      {(c.actualAccel ?? c.accel) < -0.5 && (
                        <line
                          x1={c.x - c.length + 0.1}
                          x2={c.x - c.length + 0.1}
                          y1={roadY(c.y) - c.width * 0.4}
                          y2={roadY(c.y) + c.width * 0.4}
                          stroke="#ff3a36"
                          strokeWidth=".28"
                        />
                      )}
                      <text
                        x={c.x - c.length / 2}
                        y={roadY(c.y) - c.width / 2 - 1}
                        textAnchor="middle"
                        fontSize="1.35"
                        fill="#f1f6ef"
                        fontWeight="600"
                      >
                        {c.id}
                        {risk != null ? ` · ${pct(risk)}` : ""}
                      </text>
                      {c.crashed && (
                        <text
                          x={c.x - c.length / 2}
                          y={roadY(c.y) + 0.5}
                          textAnchor="middle"
                          fill="#631e1b"
                          fontSize="1.8"
                        >
                          ×
                        </text>
                      )}
                      <rect
                        x={c.x - c.length - 1}
                        y={roadY(c.y) - c.width / 2 - 3}
                        width={c.length + 2}
                        height={c.width + 4}
                        fill="transparent"
                      />
                    </g>
                  );
                })}
                {w.collisions.map((c, i) => (
                  <g key={i}>
                    <circle
                      cx={c.x}
                      cy={roadY(c.y)}
                      r="2.6"
                      fill="none"
                      stroke="#ff7971"
                      strokeWidth=".2"
                      strokeDasharray=".5 .4"
                    />
                  </g>
                ))}
                <line
                  x1={origin + width - 25}
                  x2={origin + width - 5}
                  y1={height - 4}
                  y2={height - 4}
                  stroke="#d1ddcf"
                  strokeWidth=".15"
                />
                <text
                  x={origin + width - 15}
                  y={height - 2}
                  textAnchor="middle"
                  fill="#d1ddcf"
                  fontSize="1.3"
                >
                  20 m
                </text>
              </svg>
              <div className="road-legend">
                <span>
                  <i style={{ background: "#4ed8bb" }} />
                  Vehicle
                </span>
                <span>
                  <i style={{ background: "#ffb65d" }} />
                  Elevated model score
                </span>
                <span>
                  <i style={{ background: "#ff6c66" }} />
                  Simulated contact
                </span>
              </div>
            </div>
            <div className="playback">
              <Button
                className="primary"
                icon={running ? Pause : Play}
                onClick={() => setRunning(!running)}
              >
                {running
                  ? "Pause Simulation"
                  : w.time
                    ? "Resume Simulation"
                    : "Start Simulation"}
              </Button>
              <Button icon={RotateCcw} onClick={resetScene}>
                Reset
              </Button>
              <Button
                onClick={() => {
                  setRunning(false);
                  mutate(() => rewind(w));
                }}
              >
                Rewind 2 s
              </Button>
              <Button
                onClick={() => {
                  mutate(() => rewind(w));
                  setRunning(true);
                }}
              >
                Replay
              </Button>
              <span className="spacer" />
              <select
                aria-label="Simulation speed"
                value={rate}
                onChange={(e) => setRate(+e.target.value)}
              >
                {[0.5, 1, 2, 4].map((v) => (
                  <option key={v} value={v}>
                    {v}× speed
                  </option>
                ))}
              </select>
            </div>
            <div className="camera">
              <span>CAMERA</span>
              <Button
                icon={Scan}
                onClick={() => {
                  setCamera("fit");
                  setPan(0);
                  setZoom(1);
                }}
              >
                Fit all
              </Button>
              <Button
                onClick={() => {
                  setCamera("follow");
                  setPan(0);
                }}
              >
                Follow selected
              </Button>
              <Button
                aria-label="Zoom in"
                icon={ZoomIn}
                onClick={() => setZoom(Math.min(5, zoom * 1.3))}
              />
              <Button
                aria-label="Zoom out"
                icon={ZoomOut}
                onClick={() => setZoom(Math.max(0.4, zoom / 1.3))}
              />
              <Button
                aria-label="Pan left"
                icon={ChevronLeft}
                onClick={() => setPan(pan - 30)}
              />
              <Button
                aria-label="Pan right"
                icon={ChevronRight}
                onClick={() => setPan(pan + 30)}
              />
            </div>
          </div>
          {w.collisions.length > 0 && (
            <div className="collision-alert" role="status">
              <ShieldAlert size={20} />
              <div>
                <strong>Hypothetical Simulated Collision</strong>
                <p>
                  {w.collisions.length} geometric contact
                  {w.collisions.length === 1 ? "" : "s"} detected. Rewind to
                  inspect or change controls and compare.
                </p>
              </div>
              <Button onClick={() => setRunning(false)}>Pause scene</Button>
            </div>
          )}
          <div className="live-stats">
            <Stat label="Number of lanes" value={w.lanes} />
            <Stat label="Active vehicles" value={w.cars.length} />
            <Stat
              label="Average speed"
              value={`${(average * 3.6).toFixed(0)} km/h`}
            />
            <Stat
              label="Braking / lane changes"
              value={`${w.cars.filter((c) => (c.actualAccel ?? c.accel) < -0.5).length} / ${w.laneChanges}`}
            />
            <Stat label="Simulated contacts" value={w.collisions.length} />
            <Stat label="Valid model scores" value={valid.length} />
            <Stat
              label="Lane changes completed"
              value={w.completedLaneChanges ?? 0}
            />
            <Stat
              label="Highest score"
              value={
                valid.length
                  ? pct(Math.max(...valid.map((s) => s.probability)))
                  : "—"
              }
            />
          </div>
          <div className="panel timeline">
            <div className="panel-title">
              <Activity size={17} />
              <h2>Event timeline</h2>
              <span className="spacer" />
              <span className="micro">Actual simulation events</span>
            </div>
            {w.events.length ? (
              w.events.slice(0, 10).map((e, i) => (
                <div className={`event ${e.type}`} key={`${i}-${e.time}`}>
                  <time>{e.time.toFixed(1)} s</time>
                  <i />
                  <span>{e.text}</span>
                </div>
              ))
            ) : (
              <div className="empty">
                Start a scenario or adjust a vehicle. Its events appear here.
              </div>
            )}
          </div>
        </section>
        <aside className="panel inspector">
          <div className="panel-title">
            <Car size={18} />
            <h2>Vehicle inspector</h2>
          </div>
          <Field label="Selected vehicle">
            <select
              aria-label="Selected vehicle"
              value={selected}
              onChange={(e) => select(+e.target.value)}
            >
              {!car && <option value={selected}>Choose a vehicle</option>}
              {w.cars.map((c) => (
                <option key={c.id} value={c.id}>
                  Vehicle {c.id} · Lane {c.lane + 1}
                </option>
              ))}
            </select>
          </Field>
          {car ? (
            <>
              <div
                className="vehicle-preview"
                style={{ "--car-color": car.color }}
              >
                <Car size={65} />
                <strong>V{String(car.id).padStart(2, "0")}</strong>
                <span>
                  {car.type} · {car.length} m
                </span>
              </div>
              <div className="pair-info">
                <div>
                  <span>Position</span>
                  <b>{car.x.toFixed(1)} m</b>
                </div>
                <div>
                  <span>Current speed</span>
                  <b>{(car.speed * 3.6).toFixed(1)} km/h</b>
                </div>
                <div>
                  <span>Leading vehicle</span>
                  <b>{ahead ? `V${ahead.id}` : "—"}</b>
                </div>
                <div>
                  <span>Bumper gap</span>
                  <b>
                    {ahead
                      ? `${Math.max(0, ahead.x - ahead.length - car.x).toFixed(1)} m`
                      : "—"}
                  </b>
                </div>
                <div>
                  <span>State</span>
                  <b>
                    {car.crashed
                      ? "Simulated contact"
                      : car.changing
                        ? "Changing lane"
                        : (car.actualAccel ?? car.accel) < -0.5
                          ? "Braking"
                          : "Cruising"}
                  </b>
                </div>
              </div>
              <div className="lane-assessment" data-testid="lane-assessment">
                <h2>Lane-change assessment</h2>
                <p className="micro">
                  {w.mode === "Automatic Traffic" && w.autoLane && w.avoidance
                    ? "Automatic bypass enabled"
                    : "Automatic bypass is off"}
                </p>
                {!(
                  w.mode === "Automatic Traffic" &&
                  w.autoLane &&
                  w.avoidance
                ) && (
                  <Button
                    className="full"
                    onClick={() =>
                      mutate(() => {
                        w.mode = "Automatic Traffic";
                        w.autoLane = true;
                        w.avoidance = true;
                      })
                    }
                  >
                    Enable automatic bypass
                  </Button>
                )}
                {laneOptions.map((option) => (
                  <div
                    className={`lane-option ${option.clear ? "lane-clear" : "lane-wait"}`}
                    key={option.lane}
                  >
                    <strong>
                      Lane {option.lane + 1}:{" "}
                      {car.changing
                        ? "Changing lane"
                        : option.clear
                          ? "Clear to attempt"
                          : "Wait"}
                    </strong>
                    <p>{option.reason}</p>
                    <small>
                      Front gap:{" "}
                      {option.frontGap != null
                        ? `${option.frontGap.toFixed(1)} m`
                        : "no vehicle"}{" "}
                      · Rear gap:{" "}
                      {option.rearGap != null
                        ? `${option.rearGap.toFixed(1)} m`
                        : "no vehicle"}
                    </small>
                    {option.probability != null ? (
                      <>
                        <b>
                          Learned close-conflict score:{" "}
                          {pct(option.probability)}
                        </b>
                        <small>{option.modelSource}</small>
                      </>
                    ) : (
                      <small>
                        {option.modelReason ||
                          "No fitted score for this configuration; geometric assessment shown."}
                      </small>
                    )}
                  </div>
                ))}
                {car.laneDecision && (
                  <p className="micro">
                    Last decision: {car.laneDecision.reason}
                  </p>
                )}
                <p className="micro">
                  Learned scores use actual recorded lane changes. They are not
                  verified crash probabilities.{" "}
                  {laneInfo ? laneSource : "Loading fitted lane model…"}
                </p>
              </div>
              <div className="risk-box">
                <span>FUTURE CLOSE-FOLLOWING SCORE</span>
                <strong>
                  {score?.eligible ? pct(score.probability) : "—"}
                </strong>
                <small>
                  {score?.eligible
                    ? `Model output at ${score.at.toFixed(1)} s · next 2 s`
                    : score?.reason ||
                      "No applicable following-pair prediction"}
                </small>
              </div>
              <Field label="Target speed (km/h)">
                <input
                  aria-label="Vehicle speed"
                  disabled={car.crashed}
                  type="number"
                  min="0"
                  max="162"
                  value={Math.round(car.desired * 3.6)}
                  onChange={(e) =>
                    mutate(() => {
                      car.desired = Math.max(
                        0,
                        Math.min(45, +e.target.value / 3.6),
                      );
                      car.targetActive = true;
                      event(
                        w,
                        `Vehicle ${car.id} speed set to ${(car.desired * 3.6).toFixed(0)} km/h`,
                      );
                    })
                  }
                />
              </Field>
              <Field label="Acceleration (m/s²)">
                <input
                  aria-label="Vehicle acceleration"
                  disabled={car.crashed}
                  type="number"
                  min="-8"
                  max="4"
                  step=".25"
                  value={car.accel}
                  onChange={(e) =>
                    mutate(() => {
                      car.accel = Math.max(-8, Math.min(4, +e.target.value));
                      car.actualAccel = car.accel;
                      car.manualAccelUntil =
                        w.mode === "Automatic Traffic" ? w.time + 3 : undefined;
                      car.targetActive = false;
                    })
                  }
                />
              </Field>
              <div className="button-row">
                <Button
                  onClick={() => mutate(() => brake(w, car))}
                  disabled={car.crashed}
                >
                  Apply brakes
                </Button>
                <Button
                  onClick={() =>
                    mutate(() => {
                      car.targetActive = false;
                      car.speed = 0;
                      car.desired = 0;
                      car.accel = 0;
                      car.actualAccel = 0;
                      event(w, `Vehicle ${car.id} stopped`);
                    })
                  }
                >
                  Stop
                </Button>
              </div>
              <div className="button-row">
                <Button
                  icon={ChevronLeft}
                  disabled={car.lane === 0 || car.changing || car.crashed}
                  onClick={() =>
                    mutate(() => requestLaneChange(w, car, car.lane - 1))
                  }
                >
                  Left lane
                </Button>
                <Button
                  icon={ChevronRight}
                  disabled={
                    car.lane === w.lanes - 1 || car.changing || car.crashed
                  }
                  onClick={() =>
                    mutate(() => requestLaneChange(w, car, car.lane + 1))
                  }
                >
                  Right lane
                </Button>
              </div>
              <Button
                className="full"
                onClick={() =>
                  mutate(() => {
                    const c = generate(
                      w.lanes,
                      1,
                      Number(seed) + car.id + Math.round(w.time * 100),
                    )[0];
                    car.speed = c.speed;
                    car.desired = c.desired;
                    car.accel = c.accel;
                    car.color = c.color;
                    event(w, `Vehicle ${car.id} behavior regenerated`);
                  })
                }
                disabled={car.crashed}
              >
                Regenerate selected
              </Button>
              <Button
                className="full danger"
                icon={Trash2}
                onClick={() =>
                  mutate(() => {
                    w.cars = w.cars.filter((c) => c.id !== car.id);
                    select(w.cars[0]?.id || 0);
                  })
                }
              >
                Remove vehicle
              </Button>
            </>
          ) : (
            <p className="empty">Add or select a vehicle to inspect it.</p>
          )}
        </aside>
      </div>
      <p className="footnote">
        <Info size={14} /> Model scores estimate an observed short-headway
        event, not a verified accident probability. Collisions use vehicle
        geometry. Simulation behavior can differ from NGSIM.
      </p>
    </>
  );
}

function Plot({ data, x = "value", y = "count", scatter = false }) {
  return (
    <ResponsiveContainer width="100%" height={230}>
      {scatter ? (
        <ScatterChart>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" dataKey={x} name="Speed" unit=" m/s" />
          <YAxis type="number" dataKey={y} name="Gap" unit=" m" />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
          <Scatter data={data} fill="#208777" />
        </ScatterChart>
      ) : (
        <BarChart data={data}>
          <CartesianGrid vertical={false} stroke="#e7ece7" />
          <XAxis
            dataKey={x}
            tickFormatter={(v) => Number(v).toFixed(1)}
            tick={{ fontSize: 11 }}
            minTickGap={24}
          />
          <YAxis tick={{ fontSize: 11 }} width={55} />
          <Tooltip />
          <Bar dataKey={y} fill="#218b76" radius={[3, 3, 0, 0]} />
        </BarChart>
      )}
    </ResponsiveContainer>
  );
}
function Dashboard() {
  const { data: a, source } = useResearch("analytics"),
    { data: m } = useResearch("metrics");
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">NGSIM / US-101</div>
          <h1>Traffic, understood through motion.</h1>
          <p>
            Real trajectory analysis meets an interactive highway laboratory.
          </p>
        </div>
        <NavLink className="link-button primary" to="/simulator">
          Open simulator <ArrowRight size={16} />
        </NavLink>
      </div>
      {!a ? (
        <p>{source}</p>
      ) : (
        <>
          <div className="overview-stats">
            <Stat
              label="Recorded observations"
              value={num(a.rows)}
              note="Genuine DOT trajectory records"
            />
            <Stat
              label="Unique vehicles"
              value={num(a.vehicles)}
              note={`${a.recording_count} complete recording · ${a.duration_seconds.toFixed(0)} seconds`}
            />
            <Stat
              label="Eligible future windows"
              value={num(a.eligible_rows)}
              note="Complete 2-second same-leader windows"
            />
            <Stat
              label="Selected model"
              value={m?.selected_model || "Loading…"}
              note={m ? `Test F1 ${pct(m.test.f1)}` : ""}
            />
          </div>
          <div className="dashboard-grid">
            <div className="panel feature-card">
              <TrafficCone size={34} />
              <div className="eyebrow">YOUR TRAFFIC LABORATORY</div>
              <h2>A road you can experiment with.</h2>
              <p>
                Generate up to 30 vehicles across 1–8 lanes. Brake a leader,
                change a lane and replay the resulting contact.
              </p>
              <NavLink to="/simulator" className="link-button primary">
                Build a scenario <ArrowRight size={16} />
              </NavLink>
              <div className="mini-road">
                {[0, 1, 2].map((l) => (
                  <div key={l}>
                    {[0, 1].map((c) => (
                      <Car
                        key={c}
                        size={36}
                        style={{
                          marginLeft: `${30 + l * 12 + c * 20}px`,
                          color: ["#4ed8bb", "#f2c46f", "#8daeff"][l],
                        }}
                      />
                    ))}
                  </div>
                ))}
              </div>
            </div>
            <div className="panel chart-card">
              <div className="panel-title">
                <Gauge size={18} />
                <h2>Recorded vehicle speeds</h2>
              </div>
              <Plot data={a.histograms.v_vel} />
              <p className="micro">
                Speed in m/s · observation counts from the cleaned recording.
              </p>
            </div>
          </div>
          <div className="panel methodology">
            <h2>One project. Two distinct systems.</h2>
            <div>
              <p>
                <b>Predictive analytics</b>
                <br />
                Logistic Regression, LDA and QDA learn future short-following
                events from historical trajectories.
              </p>
              <p>
                <b>Traffic simulation</b>
                <br />
                Movement rules, smooth lateral motion and body overlaps produce
                hypothetical incidents.
              </p>
              <p>
                <b>Honest integration</b>
                <br />
                Eligible current pairs receive actual fitted-model scores.
                Unavailable or unsupported scores stay blank.
              </p>
            </div>
          </div>
          <p className="footnote">
            {source} · Dataset SHA-256: {a.sha256.slice(0, 16)}…
          </p>
        </>
      )}
    </>
  );
}
function Analysis() {
  const { data: a, source } = useResearch("analytics");
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">RESEARCH / DATA EXPLORATION</div>
          <h1>Inside the trajectories</h1>
          <p>Actual observations from one complete US-101 recording.</p>
        </div>
        <span className="status">{source}</span>
      </div>
      {a ? (
        <>
          <div className="chart-grid">
            {[
              ["v_vel", "Vehicle speed · m/s"],
              ["v_acc", "Acceleration · m/s²"],
              ["gap", "Bumper following gap · m"],
              ["headway", "Front-to-front time headway · s (capped at 10)"],
              ["lane_id", "Lane distribution"],
              ["v_class", "Vehicle type (1 motorcycle, 2 car, 3 truck)"],
            ].map(([key, title]) => (
              <div className="panel chart-card" key={key}>
                <h2>{title}</h2>
                <Plot data={a.histograms[key]} />
              </div>
            ))}
            <div className="panel chart-card">
              <h2>Speed versus bumper gap</h2>
              <Plot data={a.speed_gap} x="speed" y="gap" scatter />
            </div>
            <div className="panel chart-card">
              <h2>Target threshold sensitivity</h2>
              <Plot
                data={Object.entries(a.threshold_sensitivity).map(
                  ([value, count]) => ({ value, count: count * 100 }),
                )}
              />
              <p className="micro">
                Threshold (seconds) versus positive windows (%). Threshold
                choice changes the research question.
              </p>
            </div>
          </div>
          <div className="panel table-panel">
            <h2>Feature correlations</h2>
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
                        <td
                          key={k}
                          style={{
                            background: `rgba(32,135,119,${Math.abs(row[k]) * 0.16})`,
                          }}
                        >
                          {row[k].toFixed(2)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <p className="footnote">
            Vehicle types come from the official per-vehicle class lookup.
            Future windows require a constant valid leader/lane. Positives
            include persistence of an already short headway.
          </p>
        </>
      ) : (
        <p>{source}</p>
      )}
    </>
  );
}
function RecordedManeuver({ data }) {
  const [selected, setSelected] = useState(0),
    [index, setIndex] = useState(0),
    [playing, setPlaying] = useState(false);
  const example = data.examples[selected],
    trace = example?.trace || [];
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(
      () => setIndex((i) => (i + 1) % trace.length),
      200,
    );
    return () => clearInterval(timer);
  }, [playing, trace.length]);
  if (!trace.length) return null;
  const point = trace[Math.min(index, trace.length - 1)],
    minY = Math.min(...trace.map((p) => p.lateral)) - 3.5,
    maxY = Math.max(...trace.map((p) => p.lateral)) + 3.5,
    width = Math.max(35, ...trace.map((p) => p.x + 10));
  return (
    <div className="panel recorded-maneuver">
      <h2>Replay an actual NGSIM lane transition</h2>
      <p>
        Recorded positions, 0.1 s apart. This is trajectory evidence, not a
        recorded accident.
      </p>
      <Field label="Recorded maneuver">
        <select
          aria-label="Recorded maneuver"
          value={selected}
          onChange={(e) => {
            setSelected(+e.target.value);
            setIndex(0);
          }}
        >
          {data.examples.map((e, i) => (
            <option key={i} value={i}>
              Vehicle {e.vehicle_id}: Lane {e.from_lane} to {e.target_lane}
            </option>
          ))}
        </select>
      </Field>
      <svg
        role="img"
        aria-label="Actual recorded lane-change positions"
        viewBox={`-5 ${minY} ${width} ${maxY - minY}`}
      >
        <rect
          x="-5"
          y={minY}
          width={width}
          height={maxY - minY}
          fill="#263e33"
        />
        <line
          x1="-5"
          x2={width}
          y1={(minY + maxY) / 2}
          y2={(minY + maxY) / 2}
          stroke="#bdccbd"
          strokeWidth=".1"
          strokeDasharray="2 2"
        />
        <polyline
          points={trace.map((p) => `${p.x},${p.lateral}`).join(" ")}
          fill="none"
          stroke="#72c1a4"
          strokeWidth=".12"
          strokeDasharray=".4 .4"
        />
        <rect
          x={point.x - 4.6}
          y={point.lateral - 0.9}
          width="4.6"
          height="1.8"
          rx=".3"
          fill="#58dab3"
        />
      </svg>
      <div className="button-row">
        <Button onClick={() => setPlaying(!playing)}>
          {playing ? "Pause recorded replay" : "Play recorded replay"}
        </Button>
        <span className="micro">
          {point.seconds.toFixed(1)} s · recorded Lane {point.lane}
        </span>
      </div>
      <input
        aria-label="Recorded replay time"
        type="range"
        min="0"
        max={trace.length - 1}
        step="1"
        value={index}
        onChange={(e) => setIndex(+e.target.value)}
      />
      <p className="micro">
        Observed proxy conflict after crossing:{" "}
        {example.conflict ? "Yes" : "No"}. Lane boundary and vehicle size are
        illustrative; the plotted position samples are real. No verified
        collision label is available.
      </p>
    </div>
  );
}
function LaneChangeResearch() {
  const { data, source } = useResearch("lane-metrics");
  if (!data) return <p className="micro">Lane-change study: {source}</p>;
  return (
    <section className="panel methodology lane-research">
      <div className="eyebrow">LEARNED LANE-CHANGE ASSESSMENT</div>
      <h2>What happened after recorded lane changes?</h2>
      <p>
        {num(data.recorded_transitions)} observed transitions;{" "}
        {num(data.eligible_maneuvers)} usable maneuvers with complete front/rear
        observations. Select and replay a real recorded example below.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Algorithm</th>
              <th>Validation F1</th>
              <th>Validation ROC-AUC</th>
            </tr>
          </thead>
          <tbody>
            {data.comparison.map((row) => (
              <tr key={row.name}>
                <th>{row.name}</th>
                <td>{pct(row.validation.f1)}</td>
                <td>{pct(row.validation.roc_auc)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        <b>Selected: {data.selected_model}</b> · test F1 {pct(data.test.f1)} ·
        test ROC-AUC {pct(data.test.roc_auc)} · always-conflict baseline F1{" "}
        {pct(data.baseline_f1)}. Test sample: {data.split_sizes.test} maneuvers.
        High F1 partly reflects class imbalance; it is not proof of safe lane
        selection.
      </p>
      <p>
        Assessment is one second before the recorded lane-ID crossing. The
        outcome is an observed short front/rear headway or very small bumper gap
        during the next second after crossing. These are uncalibrated
        conflict-proxy scores, not accident predictions or counterfactual
        guarantees.
      </p>
      <RecordedManeuver data={data} />
      <p className="micro">{source}</p>
    </section>
  );
}
function Comparison() {
  const { data: m, source } = useResearch("metrics");
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">RESEARCH / CLASSIFICATION</div>
          <h1>Let the validation decide.</h1>
          <p>
            Three syllabus algorithms. Identical features and data partitions.
          </p>
        </div>
        <span className="status">{source}</span>
      </div>
      {m ? (
        <>
          <div className="overview-stats">
            <Stat label="Selected pipeline" value={m.selected_model} />
            <Stat label="Untouched test F1" value={pct(m.test.f1)} />
            <Stat label="Untouched test ROC-AUC" value={pct(m.test.roc_auc)} />
            <Stat
              label="Persistence baseline F1"
              value={pct(m.persistence_baseline_f1)}
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
                    <th>CV F1</th>
                    {["accuracy", "precision", "recall", "f1", "roc_auc"].map(
                      (k) => (
                        <th key={k}>{k.replace("_", " ").toUpperCase()}</th>
                      ),
                    )}
                  </tr>
                </thead>
                <tbody>
                  {m.comparison.map((r) => (
                    <tr
                      key={r.name}
                      className={
                        r.name === m.selected_model ? "selected-row" : ""
                      }
                    >
                      <th>
                        {r.name}
                        {r.name === m.selected_model && (
                          <span className="tag">SELECTED</span>
                        )}
                      </th>
                      <td>{pct(r.training.f1)}</td>
                      <td>{pct(r.cv_f1)}</td>
                      {["accuracy", "precision", "recall", "f1", "roc_auc"].map(
                        (k) => (
                          <td key={k}>{pct(r.validation[k])}</td>
                        ),
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="chart-grid">
            {m.comparison.map((r) => (
              <div className="panel matrix" key={r.name}>
                <h2>{r.name} · validation</h2>
                <p className="micro">
                  Rows = observed 0 / 1 · Columns = predicted 0 / 1
                </p>
                <div>
                  {r.validation.confusion_matrix.flat().map((n, i) => (
                    <strong key={i}>{num(n)}</strong>
                  ))}
                </div>
                <small>Parameters: {JSON.stringify(r.parameters)}</small>
              </div>
            ))}
          </div>
          <div className="panel methodology">
            <h2>Validation protocol & limitations</h2>
            <p>
              {m.protocol}. Split sizes:{" "}
              {Object.entries(m.split_sizes)
                .map(([k, v]) => `${k}: ${num(v)}`)
                .join(" · ")}
              .
            </p>
            <p>
              Model selection uses validation F1; the selected training-fitted
              pipeline is evaluated once on test. Probabilities are
              uncalibrated. Compare the persistence baseline before interpreting
              scores as advance warning. One recording does not establish
              generalization across roads or dates.
            </p>
          </div>
          <div className="panel table-panel">
            <h2>Selected model · untouched test</h2>
            <p>
              {Object.entries(m.test)
                .filter(([k]) => k !== "confusion_matrix")
                .map(([k, v]) => `${k}: ${pct(v)}`)
                .join(" · ")}
            </p>
          </div>
          <LaneChangeResearch />
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
      <div className="page-heading">
        <div>
          <div className="eyebrow">CSE3141 / MANIPAL UNIVERSITY JAIPUR</div>
          <h1>A clear question. An honest experiment.</h1>
          <p>NearMiss — Traffic Analysis and Prediction</p>
        </div>
      </div>
      <div className="panel about">
        <h2>What does NearMiss predict?</h2>
        <p>
          Whether a front-to-front time headway below 1.5 seconds appears in the
          next two seconds of a stable moving following pair. The target
          includes persistence of an already short gap. It does not predict
          verified accidents.
        </p>
        <h2>Where does the data come from?</h2>
        <p>
          <a
            href="https://data.transportation.gov/d/8ect-6jqj"
            target="_blank"
            rel="noreferrer"
          >
            U.S. Department of Transportation NGSIM US-101 ↗
          </a>
          . One coherent recording retains all frames for labeling. Feet are
          converted to metres. Leaders are matched at the same timestamp.
          Missing references and incomplete windows are excluded.
        </p>
        <h2>How does the simulator work?</h2>
        <p>
          Positions update at 120 fixed steps/second. Longitudinal motion
          integrates speed and acceleration; lane changes use a 1.6-second
          smooth lateral interpolation. Axis-aligned vehicle bodies determine
          contacts, including while changing lanes. Contacted vehicles stop and
          remain obstacles, permitting sequential incidents. This is an
          illustrative contact response, not a validated crash-dynamics model.
        </p>
        <p>
          Manual mode applies user inputs. Automatic mode assesses both adjacent
          lanes before stopping behind an obstacle. It checks front and rear
          gaps, approaching vehicles and the projected lateral path. A
          separately fitted lane-change conflict model can veto a supported
          high-conflict estimate. Empty lanes use explicit geometry. Movement
          rules are programmed; the conflict score is learned from observed
          NGSIM lane changes. No automatic vehicle spawning is enabled. The
          camera follows traffic as it moves.
        </p>
        <h2>What do the model scores mean?</h2>
        <p>
          Actual output of the exported fitted pipeline, for supported same-lane
          pairs. No leader, lane transition, collision, unsupported values or
          API failure means no displayed prediction. Observations are batched
          once per second; older responses cannot overwrite a reset scene.
          Training support checks do not establish joint-distribution similarity
          to real traffic.
        </p>
        <h2>Research limitations</h2>
        <p>
          The 1.5-second threshold is experimental. Official vehicle types are
          shown in EDA; the model uses current interaction features and leader
          length. Future labels require no leader/lane change; arbitrary
          simulator scenarios may violate this conditioning. Scores are
          uncalibrated. The provided syllabus topic list, rather than an
          unavailable handout, determines the model choices.
        </p>
        {info && (
          <>
            <h2>Exported model schema</h2>
            <p>
              {info.name} · {info.version}
            </p>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Unit</th>
                    <th>Training support</th>
                  </tr>
                </thead>
                <tbody>
                  {info.features.map((f) => (
                    <tr key={f}>
                      <td>{f}</td>
                      <td>{info.units[f]}</td>
                      <td>
                        {info.ranges[f].map((n) => n.toFixed(2)).join(" – ")}
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
  ["/analysis", "Data analysis", BarChart3],
  ["/simulator", "Highway simulator", Car],
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
            NearMiss<small>TRAFFIC ANALYSIS & PREDICTION</small>
          </div>
        </NavLink>
        <nav>
          {nav.map(([to, label, Icon]) => (
            <NavLink end={to === "/"} to={to} key={to}>
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>
        <span className="course">
          CSE3141 <span>RESEARCH LAB</span>
        </span>
      </header>
      <main>
        <PageBoundary key={location.pathname}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/simulator" element={<Simulator />} />
            <Route path="/prediction" element={<Simulator />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/performance" element={<Comparison />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<Dashboard />} />
          </Routes>
        </PageBoundary>
      </main>
      <footer>
        <span>
          NearMiss <b> / </b> A laboratory for traffic interactions
        </span>
        <span>
          Real trajectories. Hypothetical collisions. Clear boundaries.
        </span>
      </footer>
    </div>
  );
}
