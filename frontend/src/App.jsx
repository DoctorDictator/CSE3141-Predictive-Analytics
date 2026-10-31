import { useEffect, useState } from "react";
import { NavLink, Routes, Route, Link, useLocation } from "react-router-dom";
import {
  Activity,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  CalendarDays,
  Check,
  ChevronRight,
  CircleHelp,
  CloudSun,
  Database,
  ExternalLink,
  Gauge,
  GitBranch,
  Layers3,
  LoaderCircle,
  Menu,
  Route as RoadIcon,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  X,
} from "lucide-react";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { API_URL, loadResearch, predictTraffic } from "./services/api";

const repo = "https://github.com/DoctorDictator/CSE3141-Predictive-Analytics";
const navigation = [
  ["/", "Overview", Layers3],
  ["/analysis", "Traffic analysis", BarChart3],
  ["/prediction", "Predict traffic", SlidersHorizontal],
  ["/performance", "Model performance", Activity],
  ["/about", "About the project", CircleHelp],
];
const number = (value) =>
  Number(value).toLocaleString("en-US", { maximumFractionDigits: 0 });
const percent = (value) => `${(Number(value) * 100).toFixed(1)}%`;
const tooltipStyle = {
  borderRadius: 12,
  border: "1px solid #e2e7df",
  fontSize: 12,
  boxShadow: "0 8px 30px #163a3010",
};

function Chart({
  data,
  x,
  bar = false,
  color = "#208777",
  reference,
  value = "volume",
}) {
  const Container = bar ? BarChart : AreaChart;
  return (
    <ResponsiveContainer width="100%" height="100%">
      <Container
        data={data}
        margin={{ top: 15, right: 12, left: -16, bottom: 5 }}
      >
        <defs>
          <linearGradient id="volumeFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.2} />
            <stop offset="100%" stopColor={color} stopOpacity={0.01} />
          </linearGradient>
        </defs>
        <CartesianGrid
          vertical={false}
          stroke="#e8ece6"
          strokeDasharray="3 4"
        />
        <XAxis
          dataKey={x}
          tickLine={false}
          axisLine={false}
          tick={{ fontSize: 11, fill: "#778278" }}
          minTickGap={20}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          tick={{ fontSize: 11, fill: "#778278" }}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          formatter={(v) => [number(v), "Mean vehicles / hour"]}
        />
        {reference && (
          <ReferenceLine y={reference} stroke="#d6a340" strokeDasharray="5 5" />
        )}
        {bar ? (
          <Bar
            isAnimationActive={false}
            dataKey={value}
            fill={color}
            radius={[5, 5, 0, 0]}
            maxBarSize={42}
          />
        ) : (
          <Area
            isAnimationActive={false}
            type="monotone"
            dataKey={value}
            stroke={color}
            strokeWidth={2.5}
            fill="url(#volumeFill)"
          />
        )}
      </Container>
    </ResponsiveContainer>
  );
}

function Panel({ title, subtitle, children, aside, className = "" }) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {aside}
      </div>
      {children}
    </section>
  );
}
function Metric({ label, value, note, icon: Icon = Gauge, accent }) {
  return (
    <article className={`metric ${accent ? "accent" : ""}`}>
      <div className="metric-top">
        <span>{label}</span>
        <Icon size={18} />
      </div>
      <strong>{value}</strong>
      <p>{note}</p>
    </article>
  );
}
function Heading({ eyebrow, title, description, children }) {
  return (
    <header className="page-heading">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {children}
    </header>
  );
}
function Highway({ high = false, muted = false }) {
  return (
    <div className={`highway ${muted ? "muted" : ""}`} aria-hidden="true">
      <span className="road-mark road-mark-a" />
      <span className="road-mark road-mark-b" />
      {Array.from({ length: high ? 9 : 4 }, (_, i) => (
        <span
          key={i}
          className={`car car-${i % 3}`}
          style={{
            left: `${12 + ((i * 19) % 75)}%`,
            top: `${21 + (i % 3) * 25}%`,
          }}
        />
      ))}
      <span className="illustration-label">Illustration only</span>
    </div>
  );
}

function Overview({ overview: o, analytics: a, metrics: m }) {
  return (
    <>
      <Heading
        eyebrow="THE BIG PICTURE"
        title="Traffic, in perspective."
        description="Explore historical I-94 patterns. Put busy hours into context."
      >
        <Link className="button primary" to="/prediction">
          <Sparkles size={16} /> Predict traffic <ArrowRight size={16} />
        </Link>
      </Heading>
      <div className="metrics-grid">
        <Metric
          icon={Database}
          label="Recorded observations"
          value={number(o.raw_observations)}
          note={`${number(o.unique_hours)} unique hours after cleaning`}
        />
        <Metric
          icon={CalendarDays}
          label="Historical coverage"
          value="2012–2018"
          note="One westbound I-94 traffic station"
        />
        <Metric
          icon={Gauge}
          label="High-volume hours"
          value={`${o.high_percent.toFixed(1)}%`}
          note={`At least ${number(o.volume_threshold)} vehicles / hour`}
          accent
        />
        <Metric
          icon={ShieldCheck}
          label="Test accuracy"
          value={percent(m.test.accuracy)}
          note="On the final chronological holdout"
        />
      </div>
      <div className="main-grid">
        <Panel
          title="The rhythm of a day"
          subtitle="Mean historical traffic volume by local hour"
          aside={<span className="tag">All unique hours</span>}
        >
          <div className="chart large">
            <Chart data={a.hourly} x="hour" reference={o.volume_threshold} />
          </div>
          <div className="chart-caption">
            <span>
              <i className="dot teal" /> Mean hourly volume
            </span>
            <span>
              <i className="dash" /> High Traffic cutoff ·{" "}
              {number(o.volume_threshold)}
            </span>
          </div>
        </Panel>
        <section className="scenario-card">
          <span className="small-label">
            <Sparkles size={14} /> CONDITIONS → CONTEXT
          </span>
          <h2>
            Every hour has
            <br />a different story.
          </h2>
          <p>
            Explore how a time, a day and the weather relate to historical
            traffic volume.
          </p>
          <Highway />
          <Link to="/prediction" className="button light">
            Explore a scenario <ArrowRight size={17} />
          </Link>
          <span className="scenario-note">
            An estimate from a trained model.
            <br />A live traffic report is not available.
          </span>
        </section>
      </div>
      <div className="secondary-grid">
        <Panel title="The weekly pattern" subtitle="Average vehicles per hour">
          <div className="chart">
            <Chart data={a.daily} x="day" bar />
          </div>
        </Panel>
        <Panel
          title="Weather in the picture"
          subtitle="Historical means; not causal effects"
        >
          <div className="chart">
            <Chart data={a.weather} x="weather" bar color="#b5aa72" />
          </div>
        </Panel>
        <Panel
          title="What does “High” mean?"
          subtitle="A transparent historical definition"
          className="definition-card"
        >
          <div className="threshold-value">
            {number(o.volume_threshold)}
            <span>vehicles / hour</span>
          </div>
          <p>
            The 70th percentile of training-period volume. This same cutoff
            defines every label and prediction.
          </p>
          <div className="notice compact">
            <CircleHelp size={17} />
            <span>High volume does not necessarily mean congestion.</span>
          </div>
          <Link to="/about" className="text-link">
            Understand the model <ArrowRight size={14} />
          </Link>
        </Panel>
      </div>
      <div className="source-line">
        <Database size={14} />
        <span>
          UCI Metro Interstate Traffic Volume · ATR station 301 ·
          Minneapolis–Saint Paul
        </span>
        <a href={o.source_url} target="_blank" rel="noreferrer">
          View source <ExternalLink size={12} />
        </a>
      </div>
    </>
  );
}

function Analysis({ analytics: a }) {
  return (
    <>
      <Heading
        eyebrow="EXPLORE THE RECORD"
        title="Patterns behind the numbers."
        description="Descriptive summaries of unique historical hours, exported by the notebook."
      />
      <div className="analysis-grid">
        <Panel title="Hour of day" subtitle="Local hour · mean vehicles / hour">
          <div className="chart large">
            <Chart data={a.hourly} x="hour" />
          </div>
        </Panel>
        <Panel
          title="Day of week"
          subtitle="Daily mean · unequal numbers of observed hours"
        >
          <div className="chart large">
            <Chart data={a.daily} x="day" bar />
          </div>
        </Panel>
        <Panel
          title="Weather conditions"
          subtitle="Unadjusted group means, not weather effects"
        >
          <div className="chart large">
            <Chart data={a.weather} x="weather" bar color="#b5aa72" />
          </div>
        </Panel>
        <Panel
          title="Rainfall and volume"
          subtitle="Hourly precipitation in millimetres"
        >
          <div className="chart large">
            <Chart data={a.rain} x="rain" bar />
          </div>
        </Panel>
        <Panel title="Weekday or weekend?" subtitle="A descriptive comparison">
          <div className="chart">
            <Chart data={a.weekend} x="period" bar color="#b5aa72" />
          </div>
        </Panel>
        <Panel
          title="Across the seasons"
          subtitle="Month 1–12 · mean vehicles / hour"
        >
          <div className="chart">
            <Chart data={a.monthly} x="month" />
          </div>
        </Panel>
      </div>
      <div className="notice">
        <CircleHelp size={19} />
        <p>
          Weather and calendar groups overlap. These summaries describe the
          records; the controlled time-only versus time-and-weather experiment
          is on the Model Performance page.
        </p>
      </div>
    </>
  );
}

function Prediction({ metrics: m, overview: o }) {
  const [form, setForm] = useState({
    date: "2018-09-28",
    hour: "17",
    is_holiday: false,
    temp_c: "18",
    rain_1h: "0",
    snow_1h: "0",
    clouds_all: "30",
    weather_main: "Clear",
  });
  const [result, setResult] = useState(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const update = (name, value) => {
    setForm((previous) => ({ ...previous, [name]: value }));
    setError("");
    setResult(null);
  };
  const date = new Date(`${form.date}T12:00:00`);
  async function submit(event) {
    event.preventDefault();
    setError("");
    setResult(null);
    setBusy(true);
    try {
      setResult(
        await predictTraffic({
          date_time: `${form.date}T${form.hour.padStart(2, "0")}:00:00`,
          is_holiday: form.is_holiday,
          temp_c: Number(form.temp_c),
          rain_1h: Number(form.rain_1h),
          snow_1h: Number(form.snow_1h),
          clouds_all: Number(form.clouds_all),
          weather_main: form.weather_main,
        }),
      );
    } catch (problem) {
      setError(
        problem.message ||
          "The prediction service could not be reached. Start the backend and try again.",
      );
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <Heading
        eyebrow="YOUR CONDITIONS, THE MODEL’S ESTIMATE"
        title="What does this hour look like?"
        description="Choose a scenario. Get an estimate from the actual trained Logistic Regression model."
      />
      {!API_URL && (
        <div className="notice warning">
          <CircleHelp size={20} />
          <p>
            Historical analysis is available. Live model inference requires the
            separately hosted FastAPI service, which has not been configured for
            this site.{" "}
            <a href={`${repo}#local-setup`} target="_blank" rel="noreferrer">
              Run locally
            </a>{" "}
            to use predictions.
          </p>
        </div>
      )}
      <div className="prediction-grid">
        <Panel
          title="Set the scene"
          subtitle="Use the local date and hour for the Minnesota dataset"
        >
          <form onSubmit={submit} className="prediction-form">
            <div className="form-section-label">
              <CalendarDays size={16} /> TIME & CALENDAR
            </div>
            <div className="field-row">
              <label>
                Date
                <input
                  name="date"
                  type="date"
                  value={form.date}
                  min="2012-01-01"
                  max="2100-12-31"
                  required
                  onChange={(e) => update("date", e.target.value)}
                />
              </label>
              <label>
                Local hour
                <select
                  name="hour"
                  value={form.hour}
                  onChange={(e) => update("hour", e.target.value)}
                >
                  {Array.from({ length: 24 }, (_, i) => (
                    <option key={i} value={i}>
                      {String(i).padStart(2, "0")}:00
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="derived-fields">
              {Number.isNaN(date.getTime())
                ? "Choose a date"
                : date.toLocaleDateString("en-US", {
                    weekday: "long",
                    month: "long",
                    day: "numeric",
                  })}
              <span>Weekday and month are derived automatically</span>
            </div>
            <label className="check-field">
              <input
                name="is_holiday"
                type="checkbox"
                checked={form.is_holiday}
                onChange={(e) => update("is_holiday", e.target.checked)}
              />
              <span>This date is a holiday</span>
            </label>
            <div className="form-section-label weather-label">
              <CloudSun size={16} /> WEATHER CONDITIONS
            </div>
            <div className="field-row">
              <label>
                Temperature <span>°C</span>
                <input
                  name="temp_c"
                  type="number"
                  min="-60"
                  max="60"
                  step="0.1"
                  required
                  value={form.temp_c}
                  onChange={(e) => update("temp_c", e.target.value)}
                />
              </label>
              <label>
                Main condition
                <select
                  name="weather_main"
                  value={form.weather_main}
                  onChange={(e) => update("weather_main", e.target.value)}
                >
                  {m.weather_categories.map((weather) => (
                    <option key={weather}>{weather}</option>
                  ))}
                </select>
              </label>
            </div>
            <div className="field-row">
              <label>
                Rainfall <span>mm / hour</span>
                <input
                  name="rain_1h"
                  type="number"
                  min="0"
                  max="1000"
                  step="0.1"
                  required
                  value={form.rain_1h}
                  onChange={(e) => update("rain_1h", e.target.value)}
                />
              </label>
              <label>
                Snowfall <span>mm / hour</span>
                <input
                  name="snow_1h"
                  type="number"
                  min="0"
                  max="1000"
                  step="0.1"
                  required
                  value={form.snow_1h}
                  onChange={(e) => update("snow_1h", e.target.value)}
                />
              </label>
            </div>
            <label className="cloud-label">
              Cloud coverage <output>{form.clouds_all}%</output>
              <input
                name="clouds_all"
                type="range"
                min="0"
                max="100"
                value={form.clouds_all}
                onChange={(e) => update("clouds_all", e.target.value)}
              />
            </label>
            <div className="range-labels">
              <span>Clear sky</span>
              <span>Overcast</span>
            </div>
            {error && (
              <div className="error-message" role="alert">
                {error}
              </div>
            )}
            <button
              className="button primary full-width"
              type="submit"
              disabled={busy}
            >
              {busy ? (
                <LoaderCircle className="spin" size={17} />
              ) : (
                <Sparkles size={17} />
              )}{" "}
              {busy ? "Calculating estimate…" : "Predict Traffic"}{" "}
              <ArrowRight size={16} />
            </button>
            <p className="form-footnote">
              No live location or traffic feed is used.
            </p>
          </form>
        </Panel>
        <div className="prediction-side">
          <section
            className={`result-card ${result ? "has-result" : ""}`}
            aria-live="polite"
          >
            <span className="small-label">
              {result ? "MODEL RESULT" : "THE ROAD AHEAD"}
            </span>
            {result ? (
              <>
                <span
                  className={`result-tag ${result.prediction ? "high" : ""}`}
                >
                  <span className="dot" />
                  {result.category}
                </span>
                <div className="probability-value">
                  {percent(result.high_probability)}
                  <span>estimated High Traffic probability</span>
                </div>
                <div className="probability-track">
                  <span
                    style={{ width: `${result.high_probability * 100}%` }}
                  />
                </div>
                <div className="probability-labels">
                  <span>Normal {percent(result.normal_probability)}</span>
                  <span>High {percent(result.high_probability)}</span>
                </div>
                <Highway high={result.prediction === 1} />
                <div className="result-explanation">
                  <h3>What this means</h3>
                  <p>
                    The estimated probability that hourly volume meets or
                    exceeds {number(result.volume_threshold)} vehicles, given
                    these conditions.
                  </p>
                  <div className="result-inputs">
                    <span>{result.inputs.date_time.replace("T", " · ")}</span>
                    <span>
                      {result.inputs.temp_c}°C · {result.inputs.weather_main}
                    </span>
                  </div>
                  {result.extrapolation && (
                    <p className="extrapolation">
                      This date is outside the 2012–2018 training source.
                      Historical patterns may have changed.
                    </p>
                  )}
                </div>
              </>
            ) : (
              <>
                <h2>
                  A little context.
                  <br />A clearer estimate.
                </h2>
                <p>
                  Set the conditions and we’ll let the trained model do the
                  calculating.
                </p>
                <Highway muted />
                <div className="empty-result">
                  <Gauge size={25} />
                  <span>Your result will appear here</span>
                  <small>Real model output. No simulated probabilities.</small>
                </div>
              </>
            )}
          </section>
          <div className="notice">
            <ShieldCheck size={20} />
            <p>
              “High Traffic” is a volume category. It does not indicate a
              traffic jam or crash risk. The model uses historical data, not
              live road conditions.
            </p>
          </div>
          <div className="mini-method">
            <span>THE SAME MODEL, END TO END</span>
            <p>Notebook-trained pipeline → FastAPI → your estimate</p>
            <span className="tag">
              Cutoff · {number(o.volume_threshold)} vehicles / hour
            </span>
          </div>
        </div>
      </div>
    </>
  );
}

function Performance({ metrics: m }) {
  const coefficients = m.coefficients
    .slice(0, 12)
    .map((c) => ({ ...c, feature: c.feature.replace(/^(cat|num)__/, "") }));
  const comparisons = m.comparison
    .filter((row) => row.partition === "test")
    .map((row) => ({
      ...row,
      name: row.model === "time_only" ? "Time only" : "Time + weather",
    }));
  return (
    <>
      <Heading
        eyebrow="EVIDENCE OVER ASSUMPTION"
        title="How well does it hold up?"
        description="Genuine notebook results from the final chronological test period."
      />
      <div className="metrics-grid five">
        {[
          ["Accuracy", m.test.accuracy],
          ["Precision", m.test.precision],
          ["Recall", m.test.recall],
          ["F1 score", m.test.f1],
          ["ROC-AUC", m.test.roc_auc],
        ].map(([label, value]) => (
          <Metric
            key={label}
            label={label}
            value={label === "ROC-AUC" ? value.toFixed(3) : percent(value)}
            note="Time + weather · held-out test"
          />
        ))}
      </div>
      <div className="analysis-grid">
        <Panel
          title="Where predictions land"
          subtitle="Actual rows × predicted columns"
        >
          <div className="confusion">
            <div />
            <span>Predicted Normal</span>
            <span>Predicted High</span>
            <span>Actual Normal</span>
            <strong className="cm-good">
              {number(m.confusion_matrix[0][0])}
              <small>True negative</small>
            </strong>
            <strong>
              {number(m.confusion_matrix[0][1])}
              <small>False positive</small>
            </strong>
            <span>Actual High</span>
            <strong>
              {number(m.confusion_matrix[1][0])}
              <small>False negative</small>
            </strong>
            <strong className="cm-good">
              {number(m.confusion_matrix[1][1])}
              <small>True positive</small>
            </strong>
          </div>
        </Panel>
        <Panel
          title="Discrimination across thresholds"
          subtitle="ROC curve · chance reference shown dashed"
        >
          <div className="chart large">
            <ResponsiveContainer>
              <LineChart
                data={m.roc}
                margin={{ left: -12, right: 15, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 4" />
                <XAxis
                  dataKey="fpr"
                  type="number"
                  domain={[0, 1]}
                  tick={{ fontSize: 11 }}
                />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line
                  dataKey="tpr"
                  stroke="#208777"
                  strokeWidth={2}
                  dot={false}
                />
                <ReferenceLine
                  segment={[
                    { x: 0, y: 0 },
                    { x: 1, y: 1 },
                  ]}
                  stroke="#b9c2b8"
                  strokeDasharray="4 4"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="panel-footnote">
            X: false positive rate · Y: true positive rate
          </p>
        </Panel>
        <Panel
          title="Does weather add useful information?"
          subtitle="Identical test hours and algorithm; different feature groups"
        >
          <div className="chart large">
            <ResponsiveContainer>
              <BarChart data={comparisons} margin={{ left: -25 }}>
                <CartesianGrid vertical={false} strokeDasharray="3 4" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 1]} />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend />
                <Bar
                  dataKey="f1"
                  name="F1 score"
                  fill="#208777"
                  radius={[4, 4, 0, 0]}
                />
                <Bar
                  dataKey="roc_auc"
                  name="ROC-AUC"
                  fill="#c5b97f"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="panel-footnote">
            The primary model remains time + weather; differences are reported
            without test retuning.
          </p>
        </Panel>
        <Panel
          title="What the model associates with High Traffic"
          subtitle="Largest encoded/standardized coefficients; not causal importance"
        >
          <div className="chart coefficient-chart">
            <ResponsiveContainer>
              <BarChart
                data={coefficients}
                layout="vertical"
                margin={{ left: 65, right: 20 }}
              >
                <CartesianGrid horizontal={false} strokeDasharray="3 4" />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis
                  dataKey="feature"
                  type="category"
                  width={90}
                  tick={{ fontSize: 10 }}
                />
                <Tooltip contentStyle={tooltipStyle} />
                <ReferenceLine x={0} stroke="#8a9588" />
                <Bar dataKey="coefficient">
                  {coefficients.map((c, i) => (
                    <Cell
                      key={i}
                      fill={c.coefficient >= 0 ? "#208777" : "#c5b97f"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>
      <div className="validation-strip">
        <GitBranch size={21} />
        <div>
          <h3>Chronology matters.</h3>
          <p>
            Training → validation → testing. Unique hours stay in one partition.
            The volume cutoff and preprocessing are fitted with training data
            only. Validation selects regularization; the final test does not
            tune it.
          </p>
        </div>
        <span className="tag">
          Majority baseline accuracy · {percent(m.baseline.accuracy)}
        </span>
      </div>
    </>
  );
}

function About({ overview: o, metrics: m }) {
  return (
    <>
      <Heading
        eyebrow="CSE3141 · PREDICTIVE ANALYTICS"
        title="A simple model. A transparent story."
        description="Traffic Analysis and Prediction — an academic project at Manipal University Jaipur."
      />
      <div className="about-grid">
        <Panel title="The question" subtitle="TrafficRush">
          <p className="about-lead">
            Can weather, time of day, weekday and holidays help identify
            high-volume highway hours?
          </p>
          <p>
            We use the real UCI Metro Interstate Traffic Volume dataset:
            historical hourly westbound I-94 counts at station 301 near
            Minneapolis–Saint Paul. High Traffic is defined as at least{" "}
            <strong>{number(o.volume_threshold)} vehicles per hour</strong>, the
            training-period 70th percentile.
          </p>
          <div className="about-links">
            <a href={o.source_url} target="_blank" rel="noreferrer">
              UCI dataset <ExternalLink size={14} />
            </a>
            <a
              href={`${repo}/blob/main/notebooks/Traffic_Analysis_and_Prediction.ipynb`}
              target="_blank"
              rel="noreferrer"
            >
              Executed notebook <ExternalLink size={14} />
            </a>
            <a href={repo} target="_blank" rel="noreferrer">
              GitHub repository <ExternalLink size={14} />
            </a>
          </div>
        </Panel>
        <Panel title="From records to a result">
          <ol className="workflow">
            {[
              [
                "Collect & understand",
                "Official UCI download, missingness, repeated hours and unusual weather.",
              ],
              [
                "Explore & prepare",
                "Descriptive plots, calendar features and chronological partitions.",
              ],
              [
                "Train & validate",
                "Binary Logistic Regression with training-fit imputation, scaling and encoding.",
              ],
              [
                "Evaluate & serve",
                "Held-out metrics, a weather ablation and the same fitted pipeline in FastAPI.",
              ],
            ].map(([title, body], i) => (
              <li key={title}>
                <span>{i + 1}</span>
                <div>
                  <h3>{title}</h3>
                  <p>{body}</p>
                </div>
              </li>
            ))}
          </ol>
        </Panel>
        <Panel title="What goes into the model">
          <div className="feature-tags">
            {m.features.map((f) => (
              <span key={f}>{f.replaceAll("_", " ")}</span>
            ))}
          </div>
          <p className="panel-footnote">
            Weekend is omitted as a predictor because weekday determines it.
            Weather description and observed traffic volume are not input
            features.
          </p>
        </Panel>
        <Panel title="Know the boundaries">
          <ul className="limitations">
            <li>
              One westbound station and 2012–2018 conditions limit transfer to
              other roads and years.
            </li>
            <li>
              High vehicle count is not necessarily congestion, a crash risk, or
              a live traffic condition.
            </li>
            <li>
              Unusual weather readings and missing hours remain dataset
              limitations.
            </li>
            <li>
              Observational associations are not causal effects. Probabilities
              have not been independently calibrated.
            </li>
            <li>
              The actual CSE3141 handout was reviewed; this project focuses on
              preparation, EDA, feature selection, Logistic Regression and
              validation.
            </li>
          </ul>
        </Panel>
      </div>
      <div className="source-line">
        <Database size={14} />
        <span>
          Hogue, J. (2019). Metro Interstate Traffic Volume. UCI · CC BY 4.0.
        </span>
        <a
          href="https://doi.org/10.24432/C5X60B"
          target="_blank"
          rel="noreferrer"
        >
          Dataset citation <ExternalLink size={12} />
        </a>
      </div>
    </>
  );
}

export default function App() {
  const [research, setResearch] = useState(null),
    [error, setError] = useState(""),
    [open, setOpen] = useState(false);
  const location = useLocation();
  useEffect(() => {
    let active = true;
    loadResearch()
      .then((value) => {
        if (active) setResearch(value);
      })
      .catch((problem) => {
        if (active) setError(problem.message);
      });
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    setOpen(false);
    window.scrollTo(0, 0);
  }, [location.pathname]);
  const title =
    navigation.find(([path]) => path === location.pathname)?.[1] || "Overview";
  return (
    <div className="app-shell">
      {open && (
        <button
          className="mobile-scrim"
          aria-label="Close navigation"
          onClick={() => setOpen(false)}
        />
      )}
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <Link className="brand" to="/">
          <span className="brand-mark">
            <RoadIcon size={23} />
          </span>
          <span>
            TrafficRush<small>TRAFFIC ANALYSIS & PREDICTION</small>
          </span>
        </Link>
        <div className="sidebar-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {navigation.map(([path, label, Icon]) => (
            <NavLink key={path} to={path} end={path === "/"}>
              <Icon size={18} />
              <span>{label}</span>
              <ChevronRight size={13} />
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="dataset-label">
            <span className="dot" /> REAL DATA. CLEARER CONTEXT.
          </div>
          <p>
            Built on historical I-94 traffic.
            <br />
            Designed to make patterns understandable.
          </p>
          <a href={repo} target="_blank" rel="noreferrer">
            <GitBranch size={15} /> View project on GitHub{" "}
            <ExternalLink size={12} />
          </a>
          <div className="course-label">
            CSE3141 · Manipal University Jaipur
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="mobile-menu"
              aria-label="Open navigation"
              onClick={() => setOpen(true)}
            >
              <Menu size={21} />
            </button>
            <span>Workspace</span>
            <ChevronRight size={13} />
            <strong>{title}</strong>
          </div>
          <div className="topbar-right">
            <span
              className={`connection ${research?.source === "api" ? "connected" : ""}`}
            >
              <span className="dot" />
              {research?.source === "api"
                ? "Model connected"
                : "Saved analysis"}
            </span>
            <a
              className="source-button"
              href="https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume"
              target="_blank"
              rel="noreferrer"
            >
              UCI source <ExternalLink size={13} />
            </a>
          </div>
        </header>
        <main>
          {error ? (
            <div className="error-message" role="alert">
              Analysis could not load: {error}. Refresh or start the backend.
            </div>
          ) : !research ? (
            <div className="loading">
              <LoaderCircle className="spin" /> Loading actual notebook results…
            </div>
          ) : (
            <Routes>
              <Route path="/" element={<Overview {...research} />} />
              <Route path="/analysis" element={<Analysis {...research} />} />
              <Route
                path="/prediction"
                element={<Prediction {...research} />}
              />
              <Route
                path="/performance"
                element={<Performance {...research} />}
              />
              <Route path="/about" element={<About {...research} />} />
              <Route
                path="*"
                element={
                  <div className="notice">
                    Page not found. <Link to="/">Return to overview</Link>
                  </div>
                }
              />
            </Routes>
          )}
        </main>
        <footer>
          <span>
            TrafficRush <i>·</i> Historical patterns, transparent predictions.
          </span>
          <span>
            <span className="dot" /> Academic project · Not a live traffic
            service
          </span>
        </footer>
      </div>
    </div>
  );
}
