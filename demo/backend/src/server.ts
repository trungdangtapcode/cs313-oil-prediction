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
