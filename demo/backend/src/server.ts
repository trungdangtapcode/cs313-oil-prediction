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
          ],
          neighbors: [neighborSummary(example, 0)],
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

function scoreNearestNeighbors(
  reference: LiveReferencePayload,
  rawFeatures: Record<string, unknown>,
  k: number,
) {
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
  ) / weightTotal;
  const pred = probaUp >= 0.5 ? 1 : 0;

  return {
    targetDate: "custom-as-of-demo",
    probaUp,
    pred,
    predLabel: pred === 1 ? "UP" : "DOWN",
    confidenceMargin: Math.abs(probaUp - 0.5),
    featureCompleteness: providedFeatures.length / reference.feature_columns.length,
    providedFeatures,
    imputedFeatures,
    neighbors: scored.map((item) => neighborSummary(item.row, item.distance)),
  };
}
