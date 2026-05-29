import { DatabaseZap, LogOut, MessagesSquare, ShieldUser, Sparkles } from "lucide-react";
import { NavLink, useNavigate } from "react-router-dom";

import { useAuthStore } from "../stores/auth";

const navigation = [
  { to: "/app/chat", label: "对话中心", icon: MessagesSquare },
  { to: "/app/roles", label: "角色工作台", icon: ShieldUser },
  { to: "/app/knowledge", label: "知识库", icon: DatabaseZap },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const user = useAuthStore((state) => state.user);
  const clearSession = useAuthStore((state) => state.clearSession);

  return (
    <div className="app-shell-grid">
      <aside className="border-r border-[color:var(--line)] bg-[rgba(255,248,239,0.75)] px-6 py-7 backdrop-blur-xl">
        <div className="gradient-stroke glass-panel animated-in rounded-[28px] px-5 py-5">
          <div className="mb-8 flex items-center justify-between">
            <div>
              <div className="mb-2 flex items-center gap-2 text-[13px] uppercase tracking-[0.26em] text-[color:var(--muted)]">
                <Sparkles size={14} />
                多角色 RAG 控制台
              </div>
              <h1 className="brand-title text-3xl font-semibold">Aster Desk</h1>
            </div>
            <div className="rounded-full bg-[color:var(--accent-soft)] p-3 text-[color:var(--accent-dark)]">
              <Sparkles size={18} />
            </div>
          </div>

          <div className="mb-8 rounded-3xl bg-[rgba(252,243,230,0.92)] p-4">
            <div className="text-xs uppercase tracking-[0.24em] text-[color:var(--muted)]">当前账号</div>
            <div className="mt-2 text-lg font-semibold">{user?.username ?? "访客"}</div>
          </div>

          <nav className="space-y-2">
            {navigation.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  [
                    "flex items-center gap-3 rounded-2xl px-4 py-3 transition-all",
                    isActive
                      ? "bg-[color:var(--accent)] text-white shadow-[0_16px_34px_rgba(188,86,55,0.25)]"
                      : "text-[color:var(--ink)] hover:bg-[rgba(252,240,228,0.95)]",
                  ].join(" ")
                }
              >
                <Icon size={18} />
                <span>{label}</span>
              </NavLink>
            ))}
          </nav>

          <button
            type="button"
            onClick={() => {
              clearSession();
              navigate("/auth");
            }}
            className="mt-8 flex w-full items-center justify-center gap-2 rounded-2xl border border-[color:var(--line)] px-4 py-3 text-[color:var(--muted)] transition hover:border-[rgba(188,86,55,0.4)] hover:text-[color:var(--accent-dark)]"
          >
            <LogOut size={16} />
            退出登录
          </button>
        </div>
      </aside>

      <main className="px-5 py-5 md:px-8 md:py-7">
        <div className="animated-in min-h-[calc(100vh-40px)] rounded-[32px] border border-[color:var(--line)] bg-[rgba(255,252,246,0.62)] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.65)] md:p-6">
          {children}
        </div>
      </main>
    </div>
  );
}
