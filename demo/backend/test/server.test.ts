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
});
