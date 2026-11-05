import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  Car,
  ChevronLeft,
  ChevronRight,
  Info,
  Layers,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Scan,
  Settings2,
  ShieldAlert,
  Trash2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import {
  createWorld,
  generate,
  generateFromProfile,
  saveInitial,
  reset,
  tick,
  STEP,
  LANE_WIDTH,
  MAX_SPEED,
  SCENARIOS,
  leader,
  brake,
  setLanes,
  loadScenario,
  addCar,
  rewind,
  observations,
  event,
  laneAssessment,
  requestLaneChange,
  behaviorObservation,
} from "./simulation.js";
import { request } from "./services/api.js";
import { createBehaviorClient } from "./services/behavior-client.js";
import { inferBehavior } from "./services/behavior-inference.js";
import { Button, Field, Stat, useResearch } from "./components.jsx";
const pct = (x) => `${(x * 100).toFixed(1)}%`;
export default function Simulator() {
  const world = useRef(null);
  if (!world.current) {
    world.current = createWorld();
    loadScenario(world.current, SCENARIOS[0]);
  }
  const w = world.current,
    [running, setRunning] = useState(false),
    runRef = useRef(false),
    [version, render] = useState(0);
  const { data: behaviorInfo } = useResearch("info");
  const { data: analytics, source: analyticsStatus, error: analyticsError, retry: retryAnalytics } = useResearch("analytics");
  const [initialization, setInitialization] = useState("profile");
  useEffect(() => {
    w.behaviorModel = behaviorInfo;
  }, [behaviorInfo]);
  useEffect(() => {
    w.trafficProfile = analytics?.simulator_profile;
  }, [analytics]);
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
    clientRef = useRef(null);
  const update = () => render((v) => v + 1);
  const clearScores = () => {
    generation.current++;
    setScores({});
  };
  const mutate = (fn) => {
    fn();
    w.epoch++;
    clearScores();
    update();
  };
  useEffect(() => {
    clientRef.current = behaviorInfo ? createBehaviorClient(behaviorInfo, { request }) : null;
    return () => { clientRef.current = null; };
  }, [behaviorInfo]);
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
        rows = observations(current),
        stamp = generation.current,
        epoch = current.epoch,
        at = current.time;
      if (rows.length && clientRef.current) {
        controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 8000);
        try {
          const response = await clientRef.current.predict(rows, controller.signal);
          if (
            !disposed &&
            stamp === generation.current &&
            epoch === current.epoch
          ) {
            const now = observations(current),
              mapped = {};
            for (const p of response.predictions) {
              const prior = rows.find((r) => r.id === p.id),
                latest = now.find((r) => r.id === p.id);
              if (prior && latest && prior.signature === latest.signature)
                mapped[p.id] = {
                  ...p,
                  signature: latest.signature,
                  at,
                  wall: Date.now(),
                  source: response.source,
                };
            }
            setScores(mapped);
            setApiStatus(response.source);
          }
        } catch {
          if (!disposed) {
            setScores({});
            setApiStatus(
              current.behaviorModel
                ? "Saved fitted behavior model - local inference"
                : "Behavior estimates unavailable",
            );
          }
        } finally {
          clearTimeout(timeout);
        }
      } else if (!disposed)
        setApiStatus(
          current.behaviorModel
            ? "Saved behavior model ready"
            : "Loading behavior model",
        );
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
    const observation = behaviorObservation(w, c);
    if (!observation)
      return {
        eligible: false,
        reason: c.crashed
          ? "No behavior estimate for a collided vehicle"
          : "Vehicle state unavailable",
      };
    const cached = scores[c.id];
    if (
      cached &&
      cached.signature === observation.signature &&
      w.time >= cached.at &&
      w.time - cached.at < 1.5 &&
      Date.now() - cached.wall < 4000
    )
      return cached;
    return behaviorInfo
      ? inferBehavior(behaviorInfo, observation.inputs)
      : { eligible: false, reason: "Behavior model loading" };
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
      const value = seed === "" ? Date.now() : Number(seed);
      w.cars =
        initialization === "profile"
          ? generateFromProfile(w.lanes, count, value, w.trafficProfile)
          : generate(w.lanes, count, value);
      w.initializationSource =
        initialization === "profile"
          ? "New scene sampled from Automatum distributions"
          : "Configured synthetic random ranges";
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
          <Field label="Traffic initialization">
            <select
              aria-label="Traffic initialization"
              value={initialization}
              onChange={(e) => setInitialization(e.target.value)}
            >
              <option value="profile">Automatum data distributions</option>
              <option value="synthetic">Synthetic random ranges</option>
            </select>
          </Field>
          <p className="micro">
            {w.initializationSource ||
              "Generated scenes follow aggregate Automatum measurements; they are not historical replays."}
          </p>
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
          <Button
            className="primary full"
            icon={RotateCcw}
            onClick={randomize}
            disabled={
              initialization === "profile" && !analytics?.simulator_profile
            }
          >
            Generate Random Traffic
          </Button>
          {initialization === "profile" && analyticsError && (
            <div role="alert">
              <p className="micro">{analyticsStatus}</p>
              <Button onClick={retryAnalytics}>Retry dataset profile</Button>
            </div>
          )}
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
            empty lane. Geometry checks the full path. Already-collided cars
            remain obstacles.
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
                      stroke={
                        s.probability >=
                        (behaviorInfo?.decision_threshold ?? 0.5)
                          ? "#ffb65d"
                          : "#57d9b5"
                      }
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
                    : risk != null &&
                        risk >= (behaviorInfo?.decision_threshold ?? 0.5)
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
                  Predicted lane change
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
            <Stat label="Behavior estimates" value={valid.length} />
            <Stat
              label="Lane changes completed"
              value={w.completedLaneChanges ?? 0}
            />
            <Stat
              label="Highest change score"
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
                  {car.type} · {car.length.toFixed(1)} m
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
                <h2>Simulation clearance rules</h2>
                <p className="micro">
                  The behavior prediction does not choose a lane or judge
                  safety.
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
                      {option.clear ? "Path clear" : "Wait"}
                    </strong>
                    <p>{option.reason}</p>
                    <small>
                      Front:{" "}
                      {option.frontGap != null
                        ? option.frontGap.toFixed(1) + " m"
                        : "no vehicle"}{" "}
                      / Rear:{" "}
                      {option.rearGap != null
                        ? option.rearGap.toFixed(1) + " m"
                        : "no vehicle"}
                    </small>
                  </div>
                ))}
                {car.laneDecision && (
                  <p className="micro">
                    Controller decision: {car.laneDecision.reason}
                  </p>
                )}
              </div>
              <div className="risk-box">
                <span>LANE CHANGE WITHIN NEXT 2 SECONDS</span>
                <strong>
                  {score?.eligible ? pct(score.probability) : "—"}
                </strong>
                <small>
                  {score?.eligible
                    ? `Behavior estimate at ${w.time.toFixed(1)} s · next 2 s`
                    : score?.reason || "No supported behavior estimate"}
                </small>
              </div>
              <Field label="Target speed (km/h)">
                <input
                  aria-label="Vehicle speed"
                  disabled={car.crashed}
                  type="number"
                  min="0"
                  max={MAX_SPEED * 3.6}
                  value={Math.round(car.desired * 3.6)}
                  onChange={(e) =>
                    mutate(() => {
                      car.desired = Math.max(
                        0,
                        Math.min(MAX_SPEED, +e.target.value / 3.6),
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
                  value={Number(car.accel.toFixed(3))}
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
                    car.actualAccel = c.accel;
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
        <Info size={14} />
        Behavior scores estimate a recorded lane crossing, not accident risk or
        the best lane. Generated scenes use a separate rule-based controller and
        geometric contact detection. The model score is an estimate, not a
        command to change lanes.
      </p>
    </>
  );
}
