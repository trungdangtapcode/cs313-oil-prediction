import type { ReactNode } from "react";

type KpiCardProps = {
  label: string;
  value: string;
  detail?: string;
  tone?: "good" | "neutral" | "bad" | "info";
  icon?: ReactNode;
};

export function KpiCard({ label, value, detail, tone = "neutral", icon }: KpiCardProps) {
  return (
    <article className={`kpi-card tone-${tone}`}>
      <div className="kpi-card__top">
        <span>{label}</span>
        {icon ? <span className="kpi-card__icon">{icon}</span> : null}
      </div>
      <strong>{value}</strong>
      {detail ? <small>{detail}</small> : null}
    </article>
  );
}
