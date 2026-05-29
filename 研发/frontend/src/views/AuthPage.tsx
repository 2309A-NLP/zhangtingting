import { Sparkles } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { api } from "../lib/api";
import { useAuthStore } from "../stores/auth";

export function AuthPage() {
  const navigate = useNavigate();
  const setSession = useAuthStore((state) => state.setSession);

  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    const trimmedUsername = username.trim();
    const trimmedEmail = email.trim();

    if (trimmedUsername.length < 3) {
      setError("用户名至少需要 3 个字符。");
      return;
    }
    if (password.length < 8) {
      setError("密码至少需要 8 个字符。");
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      if (mode === "register") {
        await api.register({
          username: trimmedUsername,
          password,
          email: trimmedEmail || undefined,
        });
      }

      const tokenPayload = await api.login({
        username: trimmedUsername,
        password,
      });
      setSession(tokenPayload.access_token, {
        user_id: tokenPayload.user_id,
        username: tokenPayload.username,
      });
      navigate("/app/roles");
    } catch (err) {
      setError(err instanceof Error ? err.message : "认证失败，请稍后重试。");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-5 py-8">
      <div className="grid w-full max-w-6xl gap-6 md:grid-cols-[1.1fr_0.9fr]">
        <section className="gradient-stroke glass-panel animated-in rounded-[36px] p-8 md:p-10">
          <div className="inline-flex items-center gap-2 rounded-full bg-[rgba(255,255,255,0.55)] px-4 py-2 text-xs uppercase tracking-[0.22em] text-[color:var(--muted)]">
            <Sparkles size={14} />
            多角色 RAG
          </div>
          <h1 className="brand-title mt-6 max-w-xl text-5xl font-semibold leading-[1.02]">
            多角色知识对话工作台
          </h1>
          <p className="mt-6 max-w-2xl text-[15px] leading-8 text-[color:var(--muted)]">
            从注册登录开始，把角色切换、知识导入和检索问答这条主链路顺畅跑通，再继续细化体验。
          </p>

          <div className="mt-8 grid gap-4 md:grid-cols-3">
            {[
              ["角色", "预设角色和自定义角色放在同一工作区内统一管理。"],
              ["流式", "对话过程中可以看到实时返回和来源片段。"],
              ["知识", "每个角色有独立的知识边界和导入记录。"],
            ].map(([title, desc]) => (
              <div key={title} className="rounded-[28px] bg-[rgba(255,249,239,0.82)] p-5">
                <div className="text-sm uppercase tracking-[0.24em] text-[color:var(--muted)]">{title}</div>
                <div className="mt-3 text-sm leading-7 text-[color:var(--ink)]">{desc}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="gradient-stroke glass-panel animated-in rounded-[36px] p-8 md:p-10">
          <div className="mb-3 flex rounded-full bg-[rgba(248,239,227,0.95)] p-1">
            {[
              ["login", "登录"],
              ["register", "注册"],
            ].map(([key, label]) => (
              <button
                key={key}
                type="button"
                onClick={() => setMode(key as "login" | "register")}
                className={[
                  "flex-1 rounded-full px-4 py-3 text-sm transition",
                  mode === key
                    ? "bg-[color:var(--accent)] text-white"
                    : "text-[color:var(--muted)]",
                ].join(" ")}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="mb-6 text-sm leading-7 text-[color:var(--muted)]">
            {mode === "login" ? "使用已有账号进入工作区。" : "注册完成后会自动登录并进入对话中心。"}
          </div>

          <div className="space-y-4">
            <label className="block">
              <span className="mb-2 block text-sm text-[color:var(--muted)]">用户名</span>
              <input
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                className="w-full rounded-2xl border border-[color:var(--line)] bg-[rgba(255,255,255,0.8)] px-4 py-3 outline-none focus:border-[rgba(188,86,55,0.5)]"
                placeholder="请输入用户名"
              />
            </label>

            {mode === "register" ? (
              <label className="block">
                <span className="mb-2 block text-sm text-[color:var(--muted)]">邮箱</span>
                <input
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  className="w-full rounded-2xl border border-[color:var(--line)] bg-[rgba(255,255,255,0.8)] px-4 py-3 outline-none focus:border-[rgba(188,86,55,0.5)]"
                  placeholder="选填"
                />
              </label>
            ) : null}

            <label className="block">
              <span className="mb-2 block text-sm text-[color:var(--muted)]">密码</span>
              <input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                className="w-full rounded-2xl border border-[color:var(--line)] bg-[rgba(255,255,255,0.8)] px-4 py-3 outline-none focus:border-[rgba(188,86,55,0.5)]"
                placeholder="至少 8 位"
              />
            </label>
          </div>

          {error ? (
            <div className="mt-5 rounded-2xl border border-[rgba(188,86,55,0.22)] bg-[rgba(250,231,224,0.9)] px-4 py-3 text-sm text-[color:var(--accent-dark)]">
              {error}
            </div>
          ) : null}

          <button
            type="button"
            onClick={submit}
            disabled={submitting}
            className="mt-6 w-full rounded-full bg-[color:var(--accent)] px-5 py-4 text-sm font-semibold text-white transition hover:bg-[color:var(--accent-dark)] disabled:opacity-60"
          >
            {submitting ? "提交中..." : mode === "login" ? "登录" : "注册"}
          </button>
        </section>
      </div>
    </div>
  );
}
