export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return value.toFixed(digits);
}

export function directionLabel(value: number): "UP" | "DOWN" {
  return value === 1 ? "UP" : "DOWN";
}

export function compactHash(hash: string): string {
  return `${hash.slice(0, 7)}...${hash.slice(-7)}`;
}

export function metricTone(value: number, baseline = 0.5): "good" | "neutral" | "bad" {
  if (value >= baseline + 0.03) {
    return "good";
  }
  if (value <= baseline - 0.03) {
    return "bad";
  }
  return "neutral";
}
