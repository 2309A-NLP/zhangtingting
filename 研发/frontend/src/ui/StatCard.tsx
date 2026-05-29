export function StatCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div className="gradient-stroke glass-panel rounded-[24px] p-5">
      <div className="text-[11px] uppercase tracking-[0.24em] text-[color:var(--muted)]">
        {label}
      </div>
      <div className="mt-3 brand-title text-3xl font-semibold">{value}</div>
      <div className="mt-2 text-sm leading-7 text-[color:var(--muted)]">{hint}</div>
    </div>
  );
}
