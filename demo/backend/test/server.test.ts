import request from "supertest";
import { describe, expect, it } from "vitest";

import { createApp } from "../src/server.js";

describe("oil signal mine backend", () => {
  const app = createApp();

  it("returns health with the current ML report timestamp", async () => {
    const response = await request(app).get("/health").expect(200);

    expect(response.body.status).toBe("ok");
    expect(response.body.reportTimestamp).toBe("2026-05-14 12:08:11 UTC");
    expect(response.body.bestModel.model_config).toBe("ENS_FINAL3");
  });

  it("serves paginated prediction rows", async () => {
    const response = await request(app).get("/api/predictions?limit=3").expect(200);

    expect(response.body.totalRows).toBe(840);
    expect(response.body.rows).toHaveLength(3);
    expect(response.body.models).toContain("ENS_FINAL3");
  });

  it("serves the MLOps status bundle", async () => {
    const response = await request(app).get("/api/mlops/status").expect(200);

    expect(response.body.report_timestamp).toBe("2026-05-14 12:08:11 UTC");
    expect(response.body.validation_rules.length).toBeGreaterThan(0);
  });

  it("serves live demo examples and replays one prediction", async () => {
    const examples = await request(app).get("/api/live/examples").expect(200);

    expect(examples.body.rows.length).toBeGreaterThanOrEqual(5);

    const response = await request(app)
      .post("/api/live/predict")
      .send({ exampleId: examples.body.rows[0].id })
      .expect(200);

    expect(response.body.mode).toBe("historical_replay");
    expect(response.body.model).toBe("ENS_FINAL3");
    expect(response.body.probaUp).toBeGreaterThanOrEqual(0);
    expect(response.body.probaUp).toBeLessThanOrEqual(1);
  });

  it("scores edited live demo fields with nearest historical analogs", async () => {
    const examples = await request(app).get("/api/live/examples").expect(200);
    const example = examples.body.rows[0];

    const response = await request(app)
      .post("/api/live/predict")
      .send({
        sourceExampleId: example.id,
        features: {
          ...example.features,
          oil_return: Number(example.features.oil_return ?? 0) + 0.25,
        },
      })
      .expect(200);

    expect(response.body.mode).toBe("nearest_neighbor_demo_scorer");
    expect(response.body.neighbors.length).toBeGreaterThan(0);
    expect(response.body.featureCompleteness).toBe(1);
  });
});
