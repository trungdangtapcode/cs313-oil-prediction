import type { ReactNode } from "react";

type SectionProps = {
  title: string;
  eyebrow?: string;
  children: ReactNode;
  action?: ReactNode;
};

export function Section({ title, eyebrow, children, action }: SectionProps) {
  return (
    <section className="section">
      <header className="section__header">
        <div>
          {eyebrow ? <span className="eyebrow">{eyebrow}</span> : null}
          <h2>{title}</h2>
        </div>
        {action ? <div className="section__action">{action}</div> : null}
      </header>
      {children}
    </section>
  );
}
