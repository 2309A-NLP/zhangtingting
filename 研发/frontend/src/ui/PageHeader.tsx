export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow: string;
  title: string;
  description: string;
  actions?: React.ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-col justify-between gap-4 md:flex-row md:items-end">
      <div>
        <div className="mb-2 text-[12px] uppercase tracking-[0.28em] text-[color:var(--muted)]">
          {eyebrow}
        </div>
        <h2 className="brand-title text-4xl font-semibold">{title}</h2>
        <p className="mt-3 max-w-2xl text-[15px] leading-7 text-[color:var(--muted)]">
          {description}
        </p>
      </div>
      {actions}
    </div>
  );
}
