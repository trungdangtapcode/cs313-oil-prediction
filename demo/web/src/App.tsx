import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  Brain,
  CheckCircle2,
  Database,
  Gauge,
  GitBranch,
  LineChart as LineChartIcon,
  Microscope,
  PlayCircle,
  Rocket,
  ShieldCheck,
  Smartphone,
  Trophy,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { DataTable } from "./components/DataTable";
import { KpiCard } from "./components/KpiCard";
import { ProbabilityStrip } from "./components/ProbabilityStrip";
import { Section } from "./components/Section";
import { formatNumber, formatPercent, metricTone, compactHash } from "./data/format";
import { loadApiHealth, loadDemoData, loadLiveExamples, runLivePrediction } from "./data/client";
import type { DemoData, LeaderboardRow, LiveExample, LiveExamplesPayload, LivePredictionResult, PredictionDay } from "./data/types";
import "./styles/app.css";

const navItems = [
  { id: "mission", label: "Mission", icon: Activity },
  { id: "data", label: "Data Mine", icon: Database },
  { id: "models", label: "Model Arena", icon: Trophy },
  { id: "decision", label: "Microscope", icon: Microscope },
  { id: "live", label: "Live Demo", icon: PlayCircle },
  { id: "confidence", label: "Confidence", icon: Gauge },
  { id: "audit", label: "Audit", icon: ShieldCheck },
  { id: "ops", label: "DevOps", icon: Rocket },
] as const;

type PageId = (typeof navItems)[number]["id"];

function metricLabel(row: LeaderboardRow): string {
  return row.ModelConfig || row.Model;
}

function modelColor(experiment: string): string {
  const colors: Record<string, string> = {
    ensemble: "#31d0aa",
    weight_decay: "#f6b14a",
    baseline: "#7dd3fc",
    deep_learning: "#c084fc",
    feature_selection: "#f87171",
  };
  return colors[experiment] ?? "#a7b0be";
}

function Pill({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "good" | "bad" | "neutral" | "info" }) {
  return <span className={`pill pill-${tone}`}>{children}</span>;
}

function App() {
  const [data, setData] = useState<DemoData | null>(null);
  const [apiHealth, setApiHealth] = useState<unknown | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activePage, setActivePage] = useState<PageId>("mission");

  useEffect(() => {
    let mounted = true;
    loadDemoData()
      .then((payload) => {
        if (mounted) {
          setData(payload);
        }
      })
      .catch((loadError: Error) => {
        if (mounted) {
          setError(loadError.message);
        }
      });

    loadApiHealth()
      .then((payload) => {
        if (mounted) {
          setApiHealth(payload);
        }
      })
      .catch(() => {
        if (mounted) {
          setApiHealth(null);
        }
      });

    return () => {
      mounted = false;
    };
  }, []);

  if (error) {
    return (
      <main className="loading-shell">
        <ShieldCheck size={36} />
        <h1>Data bundle failed validation in browser</h1>
        <p>{error}</p>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="loading-shell">
        <Brain size={36} />
        <h1>Mining latest oil-direction artifacts</h1>
        <p>Loading model leaderboard, decision log, feature audit, and MLOps manifest.</p>
      </main>
    );
  }

  const page = {
    mission: <MissionControl data={data} apiHealth={apiHealth} />,
    data: <DataMine data={data} />,
    models: <ModelArena data={data} />,
    decision: <DecisionMicroscope data={data} />,
    live: <LivePredictionDemo />,
    confidence: <ConfidenceLab data={data} />,
    audit: <LeakageAudit data={data} />,
    ops: <DevOpsMlOps data={data} apiHealth={apiHealth} />,
  }[activePage];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand__mark">
            <Brain size={22} />
          </span>
          <div>
            <strong>Oil Direction</strong>
            <small>Signal Mine</small>
          </div>
        </div>
        <nav className="nav-list" aria-label="Primary">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                className={activePage === item.id ? "nav-item nav-item--active" : "nav-item"}
                key={item.id}
                onClick={() => setActivePage(item.id)}
                type="button"
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
        <div className="sidebar-status">
          <Pill tone="good">ML report locked</Pill>
          <small>{data.manifest.report_timestamp}</small>
        </div>
      </aside>

      <main className="content-shell">
        <header className="topbar">
          <div>
            <span className="eyebrow">Data-mining and machine learning demo</span>
            <h1>{navItems.find((item) => item.id === activePage)?.label}</h1>
          </div>
          <div className="topbar__meta">
            <Pill tone={apiHealth ? "good" : "info"}>{apiHealth ? "API online" : "static bundle"}</Pill>
            <Pill tone="neutral">GCP {data.manifest.local_control_status.gcp_project}</Pill>
          </div>
        </header>
        {page}
      </main>

      <nav className="mobile-tabs" aria-label="Mobile primary">
        {navItems.slice(0, 5).map((item) => {
          const Icon = item.icon;
          return (
            <button
              aria-label={item.label}
              className={activePage === item.id ? "mobile-tab mobile-tab--active" : "mobile-tab"}
              key={item.id}
              onClick={() => setActivePage(item.id)}
              type="button"
            >
              <Icon size={19} />
            </button>
          );
        })}
      </nav>
    </div>
  );
}

function MissionControl({ data, apiHealth }: { data: DemoData; apiHealth: unknown | null }) {
  const topRows = data.leaderboard.rows.slice(0, 5);
  const chartRows = topRows.map((row) => ({
    model: metricLabel(row),
    f1: row.F1_macro,
    accuracy: row.Accuracy,
  }));

  return (
    <div className="page-grid">
      <section className="hero-panel">
        <div>
          <span className="eyebrow">Forecasting-safe historical demo</span>
          <h2>Mine weak daily oil signals without pretending they are easy.</h2>
          <p>
            The product turns cleaned market, macro, supply, conflict, and news signals into an auditable next-day
            UP/DOWN probability. The point is a rigorous signal factory, not inflated accuracy.
          </p>
        </div>
        <div className="hero-panel__aside">
          <strong>{data.manifest.best_model.model_config}</strong>
          <span>current primary classifier</span>
        </div>
      </section>

      <div className="kpi-grid">
        <KpiCard
          label="Accuracy"
          value={formatPercent(data.manifest.best_model.accuracy, 1)}
          detail="full test coverage"
          tone={metricTone(data.manifest.best_model.accuracy)}
          icon={<CheckCircle2 size={18} />}
        />
        <KpiCard
          label="F1 macro"
          value={formatPercent(data.manifest.best_model.f1_macro, 1)}
          detail="primary sort metric"
          tone={metricTone(data.manifest.best_model.f1_macro)}
          icon={<BarChart3 size={18} />}
        />
        <KpiCard
          label="AUC"
          value={formatPercent(data.manifest.best_model.auc, 1)}
          detail="rank quality"
          tone={metricTone(data.manifest.best_model.auc)}
          icon={<LineChartIcon size={18} />}
        />
        <KpiCard
          label="Test days"
          value={String(data.manifest.best_model.n)}
          detail="target date >= 2023"
          tone="info"
          icon={<Database size={18} />}
        />
      </div>

      <Section title="Pipeline Control" eyebrow="Data mining path">
        <div className="pipeline">
          {["Raw sources", "Cleaning", "Leakage cleanup", "Feature refinery", "Model arena", "Decision audit"].map(
            (stage, index) => (
              <div className="pipeline__stage" key={stage}>
                <span>{index + 1}</span>
                <strong>{stage}</strong>
              </div>
            ),
          )}
        </div>
      </Section>

      <Section title="Top Classifiers" eyebrow="Latest primary leaderboard">
        <div className="chart-block">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={chartRows}>
              <CartesianGrid strokeDasharray="3 3" stroke="#25313c" />
              <XAxis dataKey="model" tick={{ fill: "#a7b0be", fontSize: 12 }} />
              <YAxis domain={[0.45, 0.57]} tickFormatter={(value) => formatPercent(Number(value), 0)} />
              <Tooltip formatter={(value: number) => formatPercent(value, 2)} />
              <Bar dataKey="f1" fill="#31d0aa" name="F1 macro" radius={[4, 4, 0, 0]} />
              <Bar dataKey="accuracy" fill="#f6b14a" name="Accuracy" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <Section title="Runtime Contract" eyebrow="Web, API, mobile">
        <div className="status-grid">
          <div className="status-tile">
            <GitBranch size={20} />
            <strong>{data.manifest.local_control_status.github_repo}</strong>
            <span>GitHub Actions source repo</span>
          </div>
          <div className="status-tile">
            <Rocket size={20} />
            <strong>{data.manifest.local_control_status.gcp_project}</strong>
            <span>GCP deploy target</span>
          </div>
          <div className="status-tile">
            <Smartphone size={20} />
            <strong>Capacitor Android</strong>
            <span>same Vite build, offline data bundle</span>
          </div>
          <div className="status-tile">
            <Activity size={20} />
            <strong>{apiHealth ? "API health connected" : "Static-first mode"}</strong>
            <span>{data.manifest.local_control_status.ci_note}</span>
          </div>
        </div>
      </Section>
    </div>
  );
}

function DataMine({ data }: { data: DemoData }) {
  const qualityImage = data.assets.rows.find((asset) => asset.file.includes("data_quality"));
  const targetImage = data.assets.rows.find((asset) => asset.file.includes("target_over_time"));

  return (
    <div className="page-grid">
      <Section title="Dataset Evolution" eyebrow="Preprocess to Step5C">
        <DataTable
          compact
          rows={data.datasets.rows}
          columns={[
            { key: "file", header: "Stage", render: (row) => row.file.replace("data/processed/", "") },
            { key: "rows", header: "Rows", align: "right", render: (row) => row.rows.toLocaleString() },
            { key: "cols", header: "Columns", align: "right", render: (row) => row.columns },
            { key: "range", header: "Date range", render: (row) => `${row.date_min} to ${row.date_max}` },
          ]}
        />
      </Section>

      <Section title="Feature Groups" eyebrow="Mined signal families">
        <div className="group-grid">
          {data.featureGroups.rows.map((group) => (
            <article className="group-tile" key={group.group}>
              <div>
                <strong>{group.group}</strong>
                <small>
                  {group.selected}/{group.total} selected
                </small>
              </div>
              <ul>
                {group.features.map((feature) => (
                  <li key={feature}>{feature}</li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      </Section>

      <Section title="EDA Evidence" eyebrow="Forecasting-safe branch">
        <div className="image-grid">
          {[qualityImage, targetImage].filter(Boolean).map((asset) => (
            <figure className="asset-figure" key={asset!.path}>
              <img src={asset!.path} alt={asset!.name} />
              <figcaption>{asset!.name}</figcaption>
            </figure>
          ))}
        </div>
      </Section>
    </div>
  );
}

function ModelArena({ data }: { data: DemoData }) {
  const rows = data.leaderboard.rows;
  const scatter = rows.map((row) => ({
    x: row.AUC,
    y: row.F1_macro,
    z: Math.max(80, row.Accuracy * 420),
    name: metricLabel(row),
    experiment: row.Experiment,
    fill: modelColor(row.Experiment),
  }));

  return (
    <div className="page-grid">
      <Section title="Model Tournament" eyebrow="AUC versus F1 macro">
        <div className="chart-block chart-block--large">
          <ResponsiveContainer width="100%" height={330}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#25313c" />
              <XAxis
                dataKey="x"
                name="AUC"
                domain={[0.45, 0.58]}
                tickFormatter={(value) => formatPercent(Number(value), 0)}
              />
              <YAxis
                dataKey="y"
                name="F1 macro"
                domain={[0.43, 0.56]}
                tickFormatter={(value) => formatPercent(Number(value), 0)}
              />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                formatter={(value: number, name: string) => [formatPercent(value, 2), name]}
                labelFormatter={(_, payload) => payload?.[0]?.payload?.name ?? "model"}
              />
              <Scatter data={scatter} fill="#31d0aa" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <Section title="Leaderboard" eyebrow="Primary fixed 0.5 policy">
        <DataTable
          rows={rows.slice(0, 12)}
          columns={[
            { key: "rank", header: "#", align: "right", render: (row) => row.rank },
            { key: "model", header: "Model", render: (row) => <strong>{metricLabel(row)}</strong> },
            { key: "exp", header: "Experiment", render: (row) => <Pill tone="info">{row.Experiment}</Pill> },
            { key: "acc", header: "Accuracy", align: "right", render: (row) => formatPercent(row.Accuracy, 1) },
            { key: "f1", header: "F1 macro", align: "right", render: (row) => formatPercent(row.F1_macro, 1) },
            { key: "auc", header: "AUC", align: "right", render: (row) => formatPercent(row.AUC, 1) },
          ]}
        />
      </Section>
    </div>
  );
}

function DecisionMicroscope({ data }: { data: DemoData }) {
  const [index, setIndex] = useState(0);
  const rows = data.predictions.rows;
  const day: PredictionDay = rows[index] ?? rows[0];
  const decision = data.decisions.rows[index] ?? data.decisions.rows[0];
  const timeline = data.decisions.rows.filter((_, rowIndex) => rowIndex % 10 === 0).map((row) => ({
    date: row.date,
    proba: row.proba_up,
    correct: row.correct ? 1 : 0,
  }));

  return (
    <div className="page-grid">
      <Section
        title="Decision Replay"
        eyebrow="ENS_FINAL3 test period"
        action={<Pill tone={decision.correct ? "good" : "bad"}>{decision.correct ? "correct" : "miss"}</Pill>}
      >
        <div className="decision-layout">
          <div className="decision-main">
            <div className="decision-date">
              <strong>{day.date}</strong>
              <span>
                Actual {decision.actual_label}, predicted {decision.pred_label}
              </span>
            </div>
            <input
              aria-label="Prediction date"
              max={rows.length - 1}
              min={0}
              onChange={(event) => setIndex(Number(event.target.value))}
              type="range"
              value={index}
            />
            <div className="probability-meter">
              <span>DOWN</span>
              <div>
                <i style={{ left: `${decision.proba_up * 100}%` }} />
              </div>
              <span>UP</span>
            </div>
            <div className="decision-kpis">
              <KpiCard label="UP probability" value={formatPercent(decision.proba_up, 1)} tone="info" />
              <KpiCard label="Margin from 50%" value={formatPercent(decision.confidence_margin, 1)} tone="neutral" />
              <KpiCard label="Validation threshold" value={formatPercent(decision.val_threshold, 1)} tone="neutral" />
            </div>
          </div>
          <ProbabilityStrip day={day} />
        </div>
      </Section>

      <Section title="Probability Timeline" eyebrow="Sampled every 10th test day">
        <div className="chart-block">
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={timeline}>
              <CartesianGrid strokeDasharray="3 3" stroke="#25313c" />
              <XAxis dataKey="date" tick={{ fill: "#a7b0be", fontSize: 11 }} minTickGap={32} />
              <YAxis domain={[0, 1]} tickFormatter={(value) => formatPercent(Number(value), 0)} />
              <Tooltip formatter={(value: number) => formatPercent(value, 1)} />
              <Area dataKey="proba" stroke="#31d0aa" fill="#1f7f6c55" name="UP probability" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Section>
    </div>
  );
}

function LivePredictionDemo() {
  const [examples, setExamples] = useState<LiveExamplesPayload | null>(null);
  const [selectedId, setSelectedId] = useState<string>("");
  const [features, setFeatures] = useState<Record<string, number>>({});
  const [result, setResult] = useState<LivePredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let mounted = true;
    loadLiveExamples()
      .then((payload) => {
        if (!mounted) {
          return;
        }
        setExamples(payload);
        const first = payload.rows[0];
        if (first) {
          setSelectedId(first.id);
          setFeatures(first.features);
        }
      })
      .catch((loadError: Error) => {
        if (mounted) {
          setError(loadError.message);
        }
      });
    return () => {
      mounted = false;
    };
  }, []);

  const selected = examples?.rows.find((row) => row.id === selectedId) ?? examples?.rows[0];
  const visibleFeatures = examples?.feature_columns ?? [];

  function selectExample(example: LiveExample) {
    setSelectedId(example.id);
    setFeatures(example.features);
    setResult(null);
    setError(null);
  }

  async function submit(mode: "replay" | "score") {
    if (!selected) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const payload =
        mode === "replay"
          ? { exampleId: selected.id }
          : { sourceExampleId: selected.id, features };
      setResult(await runLivePrediction(payload));
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Live prediction failed");
    } finally {
      setBusy(false);
    }
  }

  if (!examples) {
    return (
      <div className="page-grid">
        <Section title="Live Demo Loading" eyebrow="Backend API">
          <p className="muted-copy">{error ?? "Loading selectable prediction examples from the backend."}</p>
        </Section>
      </div>
    );
  }

  return (
    <div className="page-grid">
      <Section title="As-Of Prediction Demo" eyebrow="Backend-connected">
        <div className="live-layout">
          <div className="live-picker">
            <label htmlFor="live-example">Example row</label>
            <select
              id="live-example"
              onChange={(event) => {
                const next = examples.rows.find((row) => row.id === event.target.value);
                if (next) {
                  selectExample(next);
                }
              }}
              value={selected?.id ?? ""}
            >
              {examples.rows.map((example) => (
                <option key={example.id} value={example.id}>
                  {example.label} - {example.target_date}
                </option>
              ))}
            </select>
            {selected ? (
              <div className="live-example-card">
                <strong>{selected.label}</strong>
                <span>{selected.description}</span>
                <small>
                  Source {selected.source_date}, target {selected.target_date}, historical prediction{" "}
                  {formatPercent(selected.prediction.proba_up, 1)} UP
                </small>
              </div>
            ) : null}
            <div className="live-actions">
              <button disabled={busy} onClick={() => submit("replay")} type="button">
                Replay exact example
              </button>
              <button disabled={busy} onClick={() => submit("score")} type="button">
                Score edited fields
              </button>
            </div>
            {error ? <p className="form-error">{error}</p> : null}
          </div>

          {result ? (
            <div className="live-result">
              <span className="eyebrow">{result.mode.replace(/_/g, " ")}</span>
              <strong>{formatPercent(result.probaUp, 1)} UP</strong>
              <Pill tone={result.predLabel === "UP" ? "good" : "bad"}>{result.predLabel}</Pill>
              <small>Margin {formatPercent(result.confidenceMargin, 1)}</small>
              {typeof result.featureCompleteness === "number" ? (
                <small>Completeness {formatPercent(result.featureCompleteness, 0)}</small>
              ) : null}
            </div>
          ) : (
            <div className="live-result live-result--empty">
              <PlayCircle size={24} />
              <span>Select an example, then replay it or edit fields and score.</span>
            </div>
          )}
        </div>
      </Section>

      <Section title="Editable Feature Row" eyebrow="Examples prefill all fields">
        <div className="feature-input-grid">
          {visibleFeatures.map((feature) => (
            <label key={feature}>
              <span>{feature}</span>
              <input
                inputMode="decimal"
                onChange={(event) => {
                  const value = Number(event.target.value);
                  setFeatures((current) => ({
                    ...current,
                    [feature]: Number.isFinite(value) ? value : 0,
                  }));
                }}
                step="0.001"
                type="number"
                value={Number(features[feature] ?? 0).toFixed(4)}
              />
            </label>
          ))}
        </div>
      </Section>

      <Section title="Prediction Audit" eyebrow="What the backend did">
        {result ? (
          <div className="audit-stack">
            {result.audit.map((item) => (
              <div className="ops-rule" key={item}>
                <CheckCircle2 size={18} />
                <span>{item}</span>
              </div>
            ))}
            <DataTable
              compact
              rows={result.neighbors.slice(0, 5)}
              columns={[
                { key: "date", header: "Neighbor", render: (row) => row.targetDate },
                { key: "distance", header: "Distance", align: "right", render: (row) => formatNumber(row.distance, 3) },
                { key: "proba", header: "UP prob", align: "right", render: (row) => formatPercent(row.probaUp, 1) },
                { key: "actual", header: "Actual", render: (row) => row.actualLabel },
              ]}
            />
          </div>
        ) : (
          <p className="muted-copy">
            Exact replay uses the stored ENS_FINAL3 prediction. Edited fields use nearest historical analogs until full
            runtime model inference is added.
          </p>
        )}
      </Section>
    </div>
  );
}

function ConfidenceLab({ data }: { data: DemoData }) {
  const confidenceRows = data.confidence.rows.map((row) => ({
    margin: row.min_margin ?? 0,
    accuracy: row.accuracy ?? 0,
    f1: row.f1_macro ?? 0,
    coverage: row.coverage_rate ?? 0,
  }));
  const thresholdRows = data.thresholds.rows.map((row) => ({
    threshold: row.threshold ?? 0,
    accuracy: row.accuracy ?? 0,
    f1: row.f1_macro ?? 0,
    posRate: row.pos_rate ?? 0,
  }));

  return (
    <div className="page-grid">
      <Section title="Coverage Tradeoff" eyebrow="Confidence margin">
        <div className="chart-block chart-block--large">
          <ResponsiveContainer width="100%" height={310}>
            <LineChart data={confidenceRows}>
              <CartesianGrid strokeDasharray="3 3" stroke="#25313c" />
              <XAxis dataKey="margin" tickFormatter={(value) => formatPercent(Number(value), 0)} />
              <YAxis domain={[0, 1]} tickFormatter={(value) => formatPercent(Number(value), 0)} />
              <Tooltip formatter={(value: number) => formatPercent(value, 1)} />
              <Line dataKey="accuracy" stroke="#31d0aa" strokeWidth={2} name="Accuracy" />
              <Line dataKey="f1" stroke="#f6b14a" strokeWidth={2} name="F1 macro" />
              <Line dataKey="coverage" stroke="#7dd3fc" strokeWidth={2} name="Coverage" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <Section title="Threshold Curve" eyebrow="ENS_FINAL3">
        <div className="chart-block">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={thresholdRows}>
              <CartesianGrid strokeDasharray="3 3" stroke="#25313c" />
              <XAxis dataKey="threshold" tickFormatter={(value) => formatPercent(Number(value), 0)} />
              <YAxis domain={[0, 0.8]} tickFormatter={(value) => formatPercent(Number(value), 0)} />
              <Tooltip formatter={(value: number) => formatPercent(value, 1)} />
              <Line dataKey="accuracy" stroke="#31d0aa" strokeWidth={2} name="Accuracy" />
              <Line dataKey="f1" stroke="#f6b14a" strokeWidth={2} name="F1 macro" />
              <Line dataKey="posRate" stroke="#f87171" strokeWidth={2} name="Predicted UP rate" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Section>
    </div>
  );
}

function LeakageAudit({ data }: { data: DemoData }) {
  const selectedFeatures = data.features.rows.filter((row) => row.selected).slice(0, 18);
  const riskRows = data.leakage.rows.slice(0, 24);

  return (
    <div className="page-grid">
      <Section title="Selected Feature Audit" eyebrow="Risk and signal">
        <DataTable
          rows={selectedFeatures}
          columns={[
            { key: "rank", header: "#", align: "right", render: (row) => row.rank },
            { key: "feature", header: "Feature", render: (row) => <strong>{row.feature}</strong> },
            { key: "group", header: "Group", render: (row) => row.group },
            { key: "risk", header: "Risk", render: (row) => <Pill tone={row.risk === "low" ? "good" : "bad"}>{row.risk}</Pill> },
            { key: "score", header: "Mix score", align: "right", render: (row) => formatNumber(row.mix_score, 3) },
          ]}
        />
      </Section>

      <Section title="Leakage Controls" eyebrow="Forecasting-safe contract">
        <DataTable
          compact
          rows={riskRows}
          columns={[
            { key: "feature", header: "Feature", render: (row) => row.feature },
            { key: "group", header: "Group", render: (row) => row.group },
            { key: "risk", header: "Risk", render: (row) => row.risk },
            { key: "why", header: "Rationale", render: (row) => row.why },
          ]}
        />
      </Section>
    </div>
  );
}

function DevOpsMlOps({ data, apiHealth }: { data: DemoData; apiHealth: unknown | null }) {
  const hashes = Object.entries(data.mlops.source_hashes).slice(0, 8);

  return (
    <div className="page-grid">
      <Section title="CI/CD Control Plane" eyebrow="Current implementation target">
        <div className="status-grid">
          <div className="status-tile">
            <GitBranch size={20} />
            <strong>GitHub Actions</strong>
            <span>CI, web deploy, backend deploy, APK build, MLOps validation</span>
          </div>
          <div className="status-tile">
            <Rocket size={20} />
            <strong>Cloud Run backend</strong>
            <span>Containerized API using generated ML data</span>
          </div>
          <div className="status-tile">
            <Smartphone size={20} />
            <strong>Android APK</strong>
            <span>Capacitor wraps the same production Vite build</span>
          </div>
          <div className="status-tile">
            <Activity size={20} />
            <strong>{apiHealth ? "Backend reachable" : "API URL not configured"}</strong>
            <span>Set VITE_API_BASE_URL for deployed API health checks</span>
          </div>
        </div>
      </Section>

      <Section title="MLOps Gate" eyebrow="Artifact validation">
        <div className="ops-grid">
          {data.mlops.validation_rules.map((rule) => (
            <div className="ops-rule" key={rule}>
              <CheckCircle2 size={18} />
              <span>{rule}</span>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Source Hashes" eyebrow={data.mlops.report_timestamp}>
        <DataTable
          compact
          rows={hashes}
          columns={[
            { key: "path", header: "Source", render: ([path]) => path },
            { key: "hash", header: "SHA-256", render: ([, hash]) => compactHash(hash) },
          ]}
        />
      </Section>
    </div>
  );
}

export default App;
