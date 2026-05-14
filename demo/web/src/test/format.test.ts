import { describe, expect, it } from "vitest";

import { compactHash, directionLabel, formatPercent, metricTone } from "../data/format";

describe("format helpers", () => {
  it("formats metrics for dashboard display", () => {
    expect(formatPercent(0.547619, 1)).toBe("54.8%");
    expect(directionLabel(1)).toBe("UP");
    expect(directionLabel(0)).toBe("DOWN");
  });

  it("compresses source hashes without losing endpoints", () => {
    expect(compactHash("1234567890abcdef")).toBe("1234567...0abcdef");
  });

  it("assigns metric tone around a baseline", () => {
    expect(metricTone(0.55)).toBe("good");
    expect(metricTone(0.5)).toBe("neutral");
    expect(metricTone(0.45)).toBe("bad");
  });
});
