import { ArrowRight, BriefcaseMedical, Gavel, Landmark, LineChart } from "lucide-react";

import { formatRoleCategory, formatRoleType } from "../lib/labels";
import type { RoleRecord } from "../types";

const iconMap = {
  lawyer: Gavel,
  doctor: BriefcaseMedical,
  stock: LineChart,
  history: Landmark,
  general: ArrowRight,
};

export function RoleCard({
  role,
  active,
  onSelect,
}: {
  role: RoleRecord;
  active?: boolean;
  onSelect: (role: RoleRecord) => void;
}) {
  const Icon = iconMap[role.category as keyof typeof iconMap] ?? ArrowRight;

  return (
    <button
      type="button"
      onClick={() => onSelect(role)}
      className={[
        "gradient-stroke glass-panel w-full rounded-[24px] p-5 text-left transition duration-200",
        active
          ? "bg-[rgba(245,227,205,0.98)] shadow-[0_22px_60px_rgba(121,71,33,0.18)]"
          : "hover:-translate-y-0.5 hover:bg-[rgba(255,249,239,0.98)]",
      ].join(" ")}
    >
      <div className="mb-5 flex items-start justify-between">
        <div className="rounded-2xl bg-[color:var(--accent-soft)] p-3 text-[color:var(--accent-dark)]">
          <Icon size={18} />
        </div>
        <span className="rounded-full bg-[rgba(255,255,255,0.72)] px-3 py-1 text-xs uppercase tracking-[0.22em] text-[color:var(--muted)]">
          {formatRoleType(role.role_type)}
        </span>
      </div>
      <div className="text-xl font-semibold">{role.name}</div>
      <div className="mt-2 text-sm uppercase tracking-[0.18em] text-[color:var(--muted)]">
        {formatRoleCategory(role.category)}
      </div>
      <p className="mt-4 text-sm leading-7 text-[color:var(--muted)]">
        {role.system_prompt}
      </p>
    </button>
  );
}
