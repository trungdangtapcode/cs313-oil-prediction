import compression from "compression";
import cors from "cors";
import express from "express";
import helmet from "helmet";

import { listDataFiles, readJson, sliceRows } from "./data.js";

type Manifest = {
  app_name: string;
  report_timestamp: string;
  best_model: {
    model: string;
    model_config: string;
    accuracy: number;
    f1_macro: number;
    auc: number;
    n: number;
  };
};

type RowsPayload<T = unknown> = {
  rows: T[];
  [key: string]: unknown;
};

type LivePredictionRow = {
  id: string;
  source_date: string;
  target_date: string;
  features: Record<string, number>;
  actual: {
    target: number;
    label: "UP" | "DOWN";
  };
  prediction: {
    model: string;
    proba_up: number;
    pred: number;
    label: "UP" | "DOWN";
    correct: boolean;
    confidence_margin: number;
  };
  label?: string;
  description?: string;
};

type LiveReferencePayload = {
  mode: string;
  model: string;
  feature_columns: string[];
  feature_stats: Record<string, { mean: number; std: number; min: number; max: number }>;
  rows: LivePredictionRow[];
  note: string;
};

type LivePredictRequest = {
  exampleId?: string;
  sourceExampleId?: string;
  features?: Record<string, unknown>;
  k?: unknown;
};

type FeatureAttribution = {
  feature: string;
  value: number;
  absValue: number;
  direction: "UP" | "DOWN";
  featureValue: number;
};

type ScoredNeighbor = {
  row: LivePredictionRow;
  distance: number;
  weight: number;
};

type FeatureVectorBuild = {
  featureVector: Record<string, number>;
  providedFeatures: string[];
  imputedFeatures: string[];
};

type VectorScore = {
  probaUp: number;
  pred: number;
  predLabel: "UP" | "DOWN";
  confidenceMargin: number;
  neighbors: ReturnType<typeof neighborSummary>[];
  scored: ScoredNeighbor[];
};

function asyncRoute(
  handler: express.RequestHandler,
): express.RequestHandler {
  return (req, res, next) => {
    Promise.resolve(handler(req, res, next)).catch(next);
  };
}

export function createApp(): express.Express {
  const app = express();

  app.disable("x-powered-by");
  app.use(express.json({ limit: "1mb" }));
  app.use(helmet({ crossOriginResourcePolicy: { policy: "cross-origin" } }));
  app.use(compression());
  app.use(
    cors({
      origin: process.env.CORS_ORIGIN?.split(",").map((origin) => origin.trim()) ?? true,
    }),
  );

  app.get(
    "/health",
    asyncRoute(async (_req, res) => {
      const manifest = await readJson<Manifest>("demo_manifest.json");
      res.json({
        status: "ok",
        service: "oil-signal-mine-backend",
        reportTimestamp: manifest.report_timestamp,
        bestModel: manifest.best_model,
      });
    }),
  );

  app.get(
    "/api",
    asyncRoute(async (_req, res) => {
      const [manifest, files] = await Promise.all([
        readJson<Manifest>("demo_manifest.json"),
        listDataFiles(),
      ]);
      res.json({
        name: manifest.app_name,
        reportTimestamp: manifest.report_timestamp,
        bestModel: manifest.best_model,
        files,
        endpoints: [
          "/api/manifest",
          "/api/summary",
          "/api/leaderboard",
          "/api/features",
          "/api/feature-groups",
          "/api/predictions?limit=100",
          "/api/decision-log?limit=100",
          "/api/confidence",
          "/api/thresholds",
          "/api/leakage",
          "/api/assets",
          "/api/mlops/status",
          "/api/live/examples",
          "POST /api/live/predict",
          "/api/trading/summary",
        ],
      });
    }),
  );

  app.get("/api/manifest", asyncRoute(async (_req, res) => res.json(await readJson("demo_manifest.json"))));

  app.get(
    "/api/summary",
    asyncRoute(async (_req, res) => {
      const [manifest, leaderboard, features, confidence, mlops] = await Promise.all([
        readJson<Manifest>("demo_manifest.json"),
        readJson<RowsPayload>("leaderboard.json"),
        readJson<RowsPayload>("feature_ranking.json"),
        readJson<RowsPayload>("confidence_curve.json"),
        readJson("mlops_status.json"),
      ]);

      res.json({
        manifest,
        leaderboardTop5: leaderboard.rows.slice(0, 5),
        topFeatures: features.rows.slice(0, 10),
        confidenceCurve: confidence.rows,
        mlops,
      });
    }),
  );

  app.get("/api/leaderboard", asyncRoute(async (_req, res) => res.json(await readJson("leaderboard.json"))));
  app.get("/api/features", asyncRoute(async (_req, res) => res.json(await readJson("feature_ranking.json"))));
  app.get("/api/feature-groups", asyncRoute(async (_req, res) => res.json(await readJson("feature_groups.json"))));
  app.get("/api/confidence", asyncRoute(async (_req, res) => res.json(await readJson("confidence_curve.json"))));
  app.get("/api/thresholds", asyncRoute(async (_req, res) => res.json(await readJson("threshold_curve.json"))));
  app.get("/api/leakage", asyncRoute(async (_req, res) => res.json(await readJson("leakage_audit_table.json"))));
  app.get("/api/assets", asyncRoute(async (_req, res) => res.json(await readJson("asset_index.json"))));
  app.get("/api/mlops/status", asyncRoute(async (_req, res) => res.json(await readJson("mlops_status.json"))));
  app.get("/api/live/examples", asyncRoute(async (_req, res) => res.json(await readJson("live_prediction_examples.json"))));
  app.get("/api/trading/summary", asyncRoute(async (_req, res) => res.json(await readJson("trading_strategy_summary.json"))));

  app.post(
    "/api/live/predict",
    asyncRoute(async (req, res) => {
      const body = req.body as LivePredictRequest;
      const [examples, reference] = await Promise.all([
        readJson<LiveReferencePayload>("live_prediction_examples.json"),
        readJson<LiveReferencePayload>("live_prediction_reference.json"),
      ]);

      if (body.exampleId && !body.features) {
        const example = examples.rows.find((row) => row.id === body.exampleId)
          ?? reference.rows.find((row) => row.id === body.exampleId);
        if (!example) {
          res.status(404).json({ error: "example_not_found" });
          return;
        }

        res.json({
          mode: "historical_replay",
          model: reference.model,
          sourceExampleId: example.id,
          targetDate: example.target_date,
          sourceDate: example.source_date,
          probaUp: example.prediction.proba_up,
          pred: example.prediction.pred,
          predLabel: example.prediction.label,
          confidenceMargin: example.prediction.confidence_margin,
          actual: example.actual,
          correct: example.prediction.correct,
          audit: [
            "This is an exact replay of a validated historical test-period prediction.",
            "No model retraining or live market fetch was performed.",
            "LIME/SHAP panels are model-agnostic analog explanations from the demo scorer.",
          ],
          neighbors: [neighborSummary(example, 0)],
          explanations: buildLocalExplanations(reference, example.features, 7),
        });
        return;
      }

      const features = body.features ?? {};
      const k = clampInteger(body.k, 7, 1, 15);
      const result = scoreNearestNeighbors(reference, features, k);
      res.json({
        ...result,
        mode: "nearest_neighbor_demo_scorer",
        model: reference.model,
        sourceExampleId: body.sourceExampleId ?? body.exampleId ?? null,
        audit: [
          "Edited/custom fields are scored by nearest historical analogs in the scaled feature space.",
          "This is a live demo scorer, not production model inference from the serialized LightGBM ensemble.",
          "Missing fields are imputed from the test-period feature means and listed in the response.",
          "LIME/SHAP panels explain the live demo scorer; production TreeSHAP requires serving the serialized model stack.",
        ],
      });
    }),
  );

  app.get(
    "/api/predictions",
    asyncRoute(async (req, res) => {
      const payload = await readJson<RowsPayload>("prediction_wide_test.json");
      res.json({
        ...payload,
        rows: sliceRows(payload.rows, req.query.limit, req.query.offset),
        totalRows: payload.rows.length,
      });
    }),
  );

  app.get(
    "/api/decision-log",
    asyncRoute(async (req, res) => {
      const payload = await readJson<RowsPayload>("ens_final3_decision_log.json");
      res.json({
        ...payload,
        rows: sliceRows(payload.rows, req.query.limit, req.query.offset),
        totalRows: payload.rows.length,
      });
    }),
  );

  app.use((_req, res) => {
    res.status(404).json({ error: "not_found" });
  });

  app.use(
    (
      error: Error,
      _req: express.Request,
      res: express.Response,
      _next: express.NextFunction,
    ) => {
      console.error(error);
      res.status(500).json({
        error: "internal_error",
        message: process.env.NODE_ENV === "production" ? "Internal server error" : error.message,
      });
    },
  );

  return app;
}

function clampInteger(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = Number(value ?? fallback);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, Math.trunc(parsed)));
}

function neighborSummary(row: LivePredictionRow, distance: number) {
  return {
    id: row.id,
    targetDate: row.target_date,
    sourceDate: row.source_date,
    distance,
    probaUp: row.prediction.proba_up,
    predLabel: row.prediction.label,
    actualLabel: row.actual.label,
    correct: row.prediction.correct,
  };
}

function buildFeatureVector(
  reference: LiveReferencePayload,
  rawFeatures: Record<string, unknown>,
): FeatureVectorBuild {
  const imputedFeatures: string[] = [];
  const providedFeatures: string[] = [];
  const featureVector: Record<string, number> = {};

  for (const feature of reference.feature_columns) {
    const rawValue = rawFeatures[feature];
    const value = typeof rawValue === "number" ? rawValue : Number(rawValue);
    if (Number.isFinite(value)) {
      featureVector[feature] = value;
      providedFeatures.push(feature);
    } else {
      featureVector[feature] = reference.feature_stats[feature]?.mean ?? 0;
      imputedFeatures.push(feature);
    }
  }

  return { featureVector, providedFeatures, imputedFeatures };
}

function scoreFeatureVector(
  reference: LiveReferencePayload,
  featureVector: Record<string, number>,
  k: number,
): VectorScore {
  const scored = reference.rows
    .map((row) => {
      let sumSquared = 0;
      for (const feature of reference.feature_columns) {
        const stats = reference.feature_stats[feature];
        const scale = stats?.std && stats.std > 0 ? stats.std : 1;
        const delta = (featureVector[feature] - row.features[feature]) / scale;
        sumSquared += delta * delta;
      }
      const distance = Math.sqrt(sumSquared / reference.feature_columns.length);
      return { row, distance, weight: 1 / Math.max(distance, 0.05) };
    })
    .sort((a, b) => a.distance - b.distance)
    .slice(0, k);

  const weightTotal = scored.reduce((sum, item) => sum + item.weight, 0);
  const probaUp = scored.reduce(
    (sum, item) => sum + item.row.prediction.proba_up * item.weight,
    0,
  ) / (weightTotal || 1);
  const pred = probaUp >= 0.5 ? 1 : 0;

  return {
    probaUp,
    pred,
    predLabel: pred === 1 ? "UP" : "DOWN",
    confidenceMargin: Math.abs(probaUp - 0.5),
    neighbors: scored.map((item) => neighborSummary(item.row, item.distance)),
    scored,
  };
}

function scoreNearestNeighbors(
  reference: LiveReferencePayload,
  rawFeatures: Record<string, unknown>,
  k: number,
) {
  const { featureVector, providedFeatures, imputedFeatures } = buildFeatureVector(reference, rawFeatures);
  const score = scoreFeatureVector(reference, featureVector, k);

  return {
    targetDate: "custom-as-of-demo",
    probaUp: score.probaUp,
    pred: score.pred,
    predLabel: score.predLabel,
    confidenceMargin: score.confidenceMargin,
    featureCompleteness: providedFeatures.length / reference.feature_columns.length,
    providedFeatures,
    imputedFeatures,
    neighbors: score.neighbors,
    explanations: buildLocalExplanations(reference, featureVector, k),
  };
}

function standardizeFeature(
  reference: LiveReferencePayload,
  feature: string,
  value: number,
): number {
  const stats = reference.feature_stats[feature];
  const scale = stats?.std && stats.std > 0 ? stats.std : 1;
  return (value - (stats?.mean ?? 0)) / scale;
}

function attribution(
  feature: string,
  value: number,
  featureVector: Record<string, number>,
): FeatureAttribution {
  return {
    feature,
    value,
    absValue: Math.abs(value),
    direction: value >= 0 ? "UP" : "DOWN",
    featureValue: featureVector[feature],
  };
}

function topAttributions(rows: FeatureAttribution[], limit = 10): FeatureAttribution[] {
  return [...rows]
    .sort((a, b) => b.absValue - a.absValue)
    .slice(0, limit);
}

function buildLocalExplanations(
  reference: LiveReferencePayload,
  featureVector: Record<string, number>,
  k: number,
) {
  const shap = buildShapStyleExplanation(reference, featureVector, k);
  const lime = buildLimeStyleExplanation(reference, featureVector);

  return {
    note: "These are dependency-free, model-agnostic explanations for the live demo scorer. They are not exact TreeSHAP values from the serialized ensemble.",
    shap,
    lime,
  };
}

function buildShapStyleExplanation(
  reference: LiveReferencePayload,
  featureVector: Record<string, number>,
  k: number,
) {
  const fullScore = scoreFeatureVector(reference, featureVector, k);
  const baselineVector = Object.fromEntries(
    reference.feature_columns.map((feature) => [feature, reference.feature_stats[feature]?.mean ?? 0]),
  ) as Record<string, number>;
  const baselineScore = scoreFeatureVector(reference, baselineVector, k);

  const rows = reference.feature_columns.map((feature) => {
    const masked = { ...featureVector, [feature]: reference.feature_stats[feature]?.mean ?? 0 };
    const maskedScore = scoreFeatureVector(reference, masked, k);
    return attribution(feature, fullScore.probaUp - maskedScore.probaUp, featureVector);
  });

  return {
    method: "SHAP-style mean-baseline leave-one-feature-out",
    baselineProba: baselineScore.probaUp,
    predictionProba: fullScore.probaUp,
    features: topAttributions(rows),
  };
}

function buildLimeStyleExplanation(
  reference: LiveReferencePayload,
  featureVector: Record<string, number>,
) {
  const localRows = reference.rows
    .map((row) => {
      let sumSquared = 0;
      for (const feature of reference.feature_columns) {
        const inputValue = standardizeFeature(reference, feature, featureVector[feature]);
        const rowValue = standardizeFeature(reference, feature, row.features[feature]);
        sumSquared += (inputValue - rowValue) ** 2;
      }
      const distance = Math.sqrt(sumSquared / reference.feature_columns.length);
      const kernelWidth = 0.75;
      return {
        row,
        distance,
        weight: Math.exp(-(distance ** 2) / (kernelWidth ** 2)),
      };
    })
    .sort((a, b) => a.distance - b.distance)
    .slice(0, Math.max(120, reference.feature_columns.length * 5));

  const p = reference.feature_columns.length + 1;
  const xtwx = Array.from({ length: p }, () => Array.from({ length: p }, () => 0));
  const xtwy = Array.from({ length: p }, () => 0);

  for (const item of localRows) {
    const x = [
      1,
      ...reference.feature_columns.map((feature) => (
        standardizeFeature(reference, feature, item.row.features[feature])
      )),
    ];
    const y = item.row.prediction.proba_up;
    for (let i = 0; i < p; i += 1) {
      xtwy[i] += item.weight * x[i] * y;
      for (let j = 0; j < p; j += 1) {
        xtwx[i][j] += item.weight * x[i] * x[j];
      }
    }
  }

  for (let i = 1; i < p; i += 1) {
    xtwx[i][i] += 0.05;
  }

  const beta = solveLinearSystem(xtwx, xtwy) ?? Array.from({ length: p }, () => 0);
  const inputZ = reference.feature_columns.map((feature) => (
    standardizeFeature(reference, feature, featureVector[feature])
  ));
  const approximationProba = clampProbability(
    beta[0] + inputZ.reduce((sum, value, index) => sum + beta[index + 1] * value, 0),
  );
  const rows = reference.feature_columns.map((feature, index) => (
    attribution(feature, beta[index + 1] * inputZ[index], featureVector)
  ));

  return {
    method: "LIME-style weighted local linear surrogate",
    localRows: localRows.length,
    fidelityR2: weightedR2(reference, localRows, beta),
    intercept: beta[0],
    approximationProba,
    features: topAttributions(rows),
  };
}

function clampProbability(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function weightedR2(
  reference: LiveReferencePayload,
  rows: Array<{ row: LivePredictionRow; weight: number }>,
  beta: number[],
): number {
  const weightTotal = rows.reduce((sum, item) => sum + item.weight, 0);
  if (!weightTotal) {
    return 0;
  }
  const meanY = rows.reduce((sum, item) => sum + item.weight * item.row.prediction.proba_up, 0) / weightTotal;
  let sse = 0;
  let sst = 0;
  for (const item of rows) {
    const x = [
      1,
      ...reference.feature_columns.map((feature) => (
        standardizeFeature(reference, feature, item.row.features[feature])
      )),
    ];
    const yHat = beta.reduce((sum, value, index) => sum + value * x[index], 0);
    const y = item.row.prediction.proba_up;
    sse += item.weight * (y - yHat) ** 2;
    sst += item.weight * (y - meanY) ** 2;
  }
  return sst > 0 ? 1 - sse / sst : 0;
}

function solveLinearSystem(matrix: number[][], vector: number[]): number[] | null {
  const n = vector.length;
  const augmented = matrix.map((row, index) => [...row, vector[index]]);

  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(augmented[row][col]) > Math.abs(augmented[pivot][col])) {
        pivot = row;
      }
    }
    if (Math.abs(augmented[pivot][col]) < 1e-10) {
      return null;
    }
    [augmented[col], augmented[pivot]] = [augmented[pivot], augmented[col]];

    const pivotValue = augmented[col][col];
    for (let j = col; j <= n; j += 1) {
      augmented[col][j] /= pivotValue;
    }
    for (let row = 0; row < n; row += 1) {
      if (row === col) {
        continue;
      }
      const factor = augmented[row][col];
      for (let j = col; j <= n; j += 1) {
        augmented[row][j] -= factor * augmented[col][j];
      }
    }
  }

  return augmented.map((row) => row[n]);
}
