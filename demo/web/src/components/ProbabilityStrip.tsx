import type { PredictionDay } from "../data/types";
import { formatPercent } from "../data/format";

type ProbabilityStripProps = {
  day: PredictionDay;
};

export function ProbabilityStrip({ day }: ProbabilityStripProps) {
  const entries = Object.entries(day.models).sort(([, a], [, b]) => b.proba_up - a.proba_up);

  return (
    <div className="prob-strip">
      {entries.map(([model, value]) => (
        <div className="prob-strip__row" key={model}>
          <span>{model}</span>
          <div className="prob-strip__bar" aria-label={`${model} UP probability`}>
            <i style={{ width: `${Math.max(2, Math.min(98, value.proba_up * 100))}%` }} />
          </div>
          <strong>{formatPercent(value.proba_up, 1)}</strong>
        </div>
      ))}
    </div>
  );
}
