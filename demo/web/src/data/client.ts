import type { DemoData, LiveExamplesPayload, LivePredictionResult } from "./types";

const dataFiles = {
  manifest: "demo_manifest.json",
  leaderboard: "leaderboard.json",
  predictions: "prediction_wide_test.json",
  decisions: "ens_final3_decision_log.json",
  features: "feature_ranking.json",
  featureGroups: "feature_groups.json",
  confidence: "confidence_curve.json",
  thresholds: "threshold_curve.json",
  datasets: "dataset_evolution.json",
  leakage: "leakage_audit_table.json",
  assets: "asset_index.json",
  mlops: "mlops_status.json",
  trading: "trading_strategy_summary.json",
} as const;

async function fetchJson<T>(file: string): Promise<T> {
  const response = await fetch(`/data/${file}`);
  if (!response.ok) {
    throw new Error(`Failed to load ${file}: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function apiUrl(path: string): string {
  const apiBase = import.meta.env.VITE_API_BASE_URL;
  if (!apiBase) {
    return path;
  }
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

export async function loadDemoData(): Promise<DemoData> {
  const entries = await Promise.all(
    Object.entries(dataFiles).map(async ([key, file]) => [key, await fetchJson(file)] as const),
  );
  return Object.fromEntries(entries) as unknown as DemoData;
}

export async function loadApiHealth(): Promise<unknown | null> {
  const response = await fetch(apiUrl("/health"));
  if (!response.ok) {
    throw new Error(`API health failed: ${response.status}`);
  }
  return response.json();
}

export async function loadLiveExamples(): Promise<LiveExamplesPayload> {
  const response = await fetch(apiUrl("/api/live/examples"));
  if (!response.ok) {
    throw new Error(`Live examples failed: ${response.status}`);
  }
  return response.json() as Promise<LiveExamplesPayload>;
}

export async function runLivePrediction(payload: {
  exampleId?: string;
  sourceExampleId?: string;
  features?: Record<string, number>;
}): Promise<LivePredictionResult> {
  const response = await fetch(apiUrl("/api/live/predict"), {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Live prediction failed: ${response.status}`);
  }
  return response.json() as Promise<LivePredictionResult>;
}
