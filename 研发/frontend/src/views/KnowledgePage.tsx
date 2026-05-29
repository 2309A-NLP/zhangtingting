import { FileUp, FolderUp } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { api } from "../lib/api";
import { formatKnowledgeMode, formatKnowledgeStatus } from "../lib/labels";
import { useAppStore } from "../stores/app";
import { useAuthStore } from "../stores/auth";
import type { KnowledgeTaskStatus } from "../types";
import { PageHeader } from "../ui/PageHeader";

const ACTIVE_STATUSES = new Set(["queued", "processing"]);

const t = {
  eyebrow: "\u77e5\u8bc6",
  title: "\u77e5\u8bc6\u5bfc\u5165",
  description:
    "\u628a PDF\u3001TXT\u3001JSON \u6216 HTML \u6587\u4ef6\u6302\u5230\u5f53\u524d\u6fc0\u6d3b\u89d2\u8272\u4e0b\u3002\u4e0a\u4f20\u4f1a\u7acb\u523b\u5165\u961f\uff0c\u5148\u5199\u5165 MinIO\uff0c\u518d\u7531\u540e\u7aef\u5f02\u6b65\u5b8c\u6210\u5bfc\u5165\u3002",
  defaultStatus:
    "\u4e0a\u4f20\u6587\u4ef6\u540e\u4f1a\u8fdb\u5165\u5bfc\u5165\u961f\u5217\uff0c\u9875\u9762\u4f1a\u6301\u7eed\u8f6e\u8be2\u4efb\u52a1\u72b6\u6001\u76f4\u5230\u540e\u7aef\u5904\u7406\u5b8c\u6210\u3002",
  targetRole: "\u76ee\u6807\u89d2\u8272",
  targetRoleHint:
    "\u4e0a\u4f20\u540e\u7684\u77e5\u8bc6\u4f1a\u4e25\u683c\u9694\u79bb\u5728\u5f53\u524d\u7528\u6237\u548c\u5f53\u524d\u89d2\u8272\u8303\u56f4\u5185\uff0c\u68c0\u7d22\u3001\u7f13\u5b58\u548c\u8bb0\u5fc6\u90fd\u4e0d\u4f1a\u8d8a\u51fa\u8fd9\u4e2a\u79df\u6237\u8fb9\u754c\u3002",
  noRole:
    "\u8fd8\u6ca1\u6709\u9009\u4e2d\u89d2\u8272\u3002\u8bf7\u5148\u5230\u89d2\u8272\u5de5\u4f5c\u53f0\u9009\u62e9\u89d2\u8272\uff0c\u518d\u56de\u6765\u4e0a\u4f20\u77e5\u8bc6\u6587\u4ef6\u3002",
  uploadDoc: "\u4e0a\u4f20\u6587\u6863",
  importMode: "\u5bfc\u5165\u6a21\u5f0f",
  overwrite: "\u8986\u76d6\u5df2\u6709\u540c\u5185\u5bb9\u6587\u4ef6",
  overwriteHint:
    "\u9ed8\u8ba4\u4f1a\u963b\u6b62\u91cd\u590d\u4e0a\u4f20\u3002\u5f00\u542f\u540e\uff0c\u540c\u4e00 role \u4e0b\u68c0\u6d4b\u5230\u76f8\u540c\u5185\u5bb9\u65f6\uff0c\u4f1a\u7528\u65b0\u6587\u4ef6\u66ff\u6362\u65e7\u5411\u91cf\u3002",
  pickFile: "\u628a\u6587\u4ef6\u62d6\u5230\u8fd9\u91cc\uff0c\u6216\u70b9\u51fb\u9009\u62e9\u6587\u4ef6",
  fileHint:
    "\u652f\u6301 PDF / TXT / JSON / HTML\u3002\u6bcf\u4e2a\u6587\u4ef6\u90fd\u4f1a\u5148\u843d\u5230 MinIO\uff0c\u518d\u8fdb\u5165\u540e\u53f0\u5bfc\u5165\u6d41\u7a0b\u3002",
  uploading: "\u6b63\u5728\u4e0a\u4f20\u6587\u4ef6\u5e76\u52a0\u5165\u540e\u7aef\u5bfc\u5165\u961f\u5217...",
  uploadFailed: "\u4e0a\u4f20\u5931\u8d25\u3002",
  uploadButton: "\u63d0\u4ea4\u77e5\u8bc6\u6587\u4ef6",
  uploadButtonBusy: "\u4e0a\u4f20\u4e2d...",
  pollingFailed: "\u4efb\u52a1\u8f6e\u8be2\u5931\u8d25\u3002",
  status: "\u72b6\u6001",
  task: "\u4efb\u52a1",
  chunkCount: "\u5206\u5757\u6570",
  sourceFile: "\u6e90\u6587\u4ef6",
  parsedArtifact: "\u89e3\u6790\u4ea7\u7269",
  polling: "\u6b63\u5728\u8f6e\u8be2\u4efb\u52a1\u72b6\u6001...",
  queue: "\u961f\u5217",
  queueCopy: "\u4e0a\u4f20\u4f1a\u7acb\u523b\u8fd4\u56de task id\uff0c\u540e\u7aef\u5de5\u4f5c\u8fdb\u7a0b\u518d\u5f02\u6b65\u5b8c\u6210\u89e3\u6790\u548c\u5411\u91cf\u5316\u3002",
  storage: "\u5b58\u50a8",
  storageCopy: "\u539f\u59cb\u6587\u4ef6\u4f1a\u5148\u843d\u5230 MinIO\uff0c\u5bfc\u5165\u6210\u529f\u540e\u518d\u5355\u72ec\u5199\u5165\u89e3\u6790\u4ea7\u7269\u3002",
  isolation: "\u9694\u79bb",
  isolationCopy: "\u6bcf\u6b21\u4e0a\u4f20\u90fd\u7ed1\u5b9a\u5728\u5f53\u524d user-role \u79df\u6237\u4e0b\uff0c\u4e0e Redis \u548c Milvus \u7684\u5206\u533a\u8fb9\u754c\u4fdd\u6301\u4e00\u81f4\u3002",
  queued: "\u4efb\u52a1 ",
  queuedSuffix: " \u5df2\u8fdb\u5165\u961f\u5217\uff0c\u540e\u7aef\u5de5\u4f5c\u8fdb\u7a0b\u5c1a\u672a\u5f00\u59cb\u5904\u7406\u8be5\u6587\u4ef6\u3002",
  processingSuffix: " \u6b63\u5728\u5904\u7406\u4e2d\uff0c\u6587\u4ef6\u4f1a\u88ab\u89e3\u6790\u5e76\u5199\u5165\u5f53\u524d\u89d2\u8272\u7684\u77e5\u8bc6\u7a7a\u95f4\u3002",
  successPrefix: "\u4efb\u52a1 ",
  successMiddle: " \u5df2\u6210\u529f\u5b8c\u6210\u3002",
  successChunkPrefix: " \u5171\u751f\u6210 ",
  successChunkSuffix: " \u4e2a\u5206\u5757\u3002",
  failedPrefix: "\u4efb\u52a1 ",
  failedMiddle: " \u5931\u8d25\u3002",
};

function formatTaskCopy(task: KnowledgeTaskStatus | null) {
  if (!task) {
    return t.defaultStatus;
  }

  if (task.status === "queued") {
    return `${t.queued}${task.task_id}${t.queuedSuffix}`;
  }
  if (task.status === "processing") {
    return `${t.queued}${task.task_id}${t.processingSuffix}`;
  }
  if (task.status === "success") {
    const chunkInfo = task.chunk_count
      ? `${t.successChunkPrefix}${task.chunk_count}${t.successChunkSuffix}`
      : "";
    return `${t.successPrefix}${task.task_id}${t.successMiddle}${chunkInfo}`;
  }
  return `${t.failedPrefix}${task.task_id}${t.failedMiddle}${task.error_message ? task.error_message : ""}`;
}

export function KnowledgePage() {
  const token = useAuthStore((state) => state.token)!;
  const user = useAuthStore((state) => state.user)!;
  const activeRole = useAppStore((state) => state.activeRole);

  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<"incremental" | "full">("incremental");
  const [overwrite, setOverwrite] = useState(false);
  const [status, setStatus] = useState<string>(t.defaultStatus);
  const [uploading, setUploading] = useState(false);
  const [currentTask, setCurrentTask] = useState<KnowledgeTaskStatus | null>(null);
  const [isPolling, setIsPolling] = useState(false);

  useEffect(() => {
    if (!currentTask || !activeRole || !ACTIVE_STATUSES.has(currentTask.status)) {
      setIsPolling(false);
      return undefined;
    }

    setIsPolling(true);
    let cancelled = false;
    const poll = async () => {
      try {
        const nextTask = await api.getKnowledgeTaskStatus(token, {
          task_id: currentTask.task_id,
          user_id: user.user_id,
          role_id: activeRole.role_id,
        });
        if (cancelled) {
          return;
        }
        setCurrentTask(nextTask);
        setStatus(formatTaskCopy(nextTask));
        if (!ACTIVE_STATUSES.has(nextTask.status)) {
          setIsPolling(false);
        }
      } catch (error) {
        if (cancelled) {
          return;
        }
        setIsPolling(false);
        setStatus(error instanceof Error ? error.message : t.pollingFailed);
      }
    };

    const timer = window.setInterval(poll, 2500);
    void poll();
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [activeRole, currentTask, token, user.user_id]);

  const taskMeta = useMemo(
    () =>
      [
        [t.status, formatKnowledgeStatus(currentTask?.status ?? "idle")],
        [t.task, currentTask?.task_id ?? "-"],
        [t.chunkCount, currentTask?.chunk_count?.toString() ?? "-"],
      ] as const,
    [currentTask],
  );

  return (
    <div>
      <PageHeader eyebrow={t.eyebrow} title={t.title} description={t.description} />

      <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
        <section className="gradient-stroke glass-panel rounded-[30px] p-6">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold">
            <FolderUp size={16} />
            {t.targetRole}
          </div>
          {activeRole ? (
            <div className="rounded-[26px] bg-[rgba(255,248,239,0.86)] p-5">
              <div className="text-2xl font-semibold">{activeRole.name}</div>
              <div className="mt-2 text-sm uppercase tracking-[0.18em] text-[color:var(--muted)]">
                {activeRole.role_id}
              </div>
              <p className="mt-4 text-[15px] leading-7 text-[color:var(--muted)]">{t.targetRoleHint}</p>
            </div>
          ) : (
            <div className="rounded-[26px] border border-dashed border-[rgba(69,49,30,0.18)] px-5 py-8 text-[15px] leading-7 text-[color:var(--muted)]">
              {t.noRole}
            </div>
          )}
        </section>

        <section className="gradient-stroke glass-panel rounded-[30px] p-6">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold">
            <FileUp size={16} />
            {t.uploadDoc}
          </div>
          <label className="block">
            <span className="mb-2 block text-sm text-[color:var(--muted)]">{t.importMode}</span>
            <select
              value={mode}
              onChange={(event) => setMode(event.target.value as "incremental" | "full")}
              className="w-full rounded-2xl border border-[color:var(--line)] bg-[rgba(255,255,255,0.82)] px-4 py-3 outline-none"
            >
              <option value="incremental">{formatKnowledgeMode("incremental")}</option>
              <option value="full">{formatKnowledgeMode("full")}</option>
            </select>
          </label>

          <label className="mt-4 flex min-h-[220px] cursor-pointer flex-col items-center justify-center rounded-[28px] border border-dashed border-[rgba(188,86,55,0.28)] bg-[rgba(255,248,241,0.75)] px-6 py-8 text-center transition hover:bg-[rgba(255,245,233,0.9)]">
            <div className="rounded-full bg-[color:var(--accent-soft)] p-4 text-[color:var(--accent-dark)]">
              <FileUp size={22} />
            </div>
            <div className="mt-4 text-lg font-semibold">{file ? file.name : t.pickFile}</div>
            <div className="mt-2 text-sm leading-7 text-[color:var(--muted)]">{t.fileHint}</div>
            <input
              type="file"
              className="hidden"
              accept=".pdf,.txt,.json,.html"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
          </label>

          <label className="mt-4 flex items-start gap-3 rounded-[24px] bg-[rgba(255,249,241,0.86)] px-4 py-4 text-sm leading-7 text-[color:var(--muted)]">
            <input
              type="checkbox"
              checked={overwrite}
              onChange={(event) => setOverwrite(event.target.checked)}
              className="mt-1 h-4 w-4"
            />
            <span>
              <span className="block font-semibold text-[color:var(--ink)]">{t.overwrite}</span>
              <span>{t.overwriteHint}</span>
            </span>
          </label>

          <button
            type="button"
            disabled={!file || !activeRole || uploading}
            onClick={async () => {
              if (!file || !activeRole) return;
              setUploading(true);
              setStatus(t.uploading);
              setCurrentTask(null);
              try {
                const queued = await api.uploadKnowledge(token, {
                  user_id: user.user_id,
                  role_id: activeRole.role_id,
                  mode,
                  file,
                  overwrite,
                });
                const queuedTask: KnowledgeTaskStatus = {
                  task_id: queued.task_id,
                  user_id: queued.user_id,
                  role_id: queued.role_id,
                  mode: queued.mode,
                  status: queued.status,
                };
                setCurrentTask(queuedTask);
                setStatus(formatTaskCopy(queuedTask));
              } catch (err) {
                setStatus(err instanceof Error ? err.message : t.uploadFailed);
              } finally {
                setUploading(false);
              }
            }}
            className="mt-5 rounded-full bg-[color:var(--accent)] px-5 py-3 text-sm text-white disabled:opacity-60"
          >
            {uploading ? t.uploadButtonBusy : t.uploadButton}
          </button>

          <div className="mt-5 rounded-[24px] bg-[rgba(255,249,241,0.86)] px-4 py-4 text-sm leading-7 text-[color:var(--muted)]">
            {status}
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            {taskMeta.map(([label, value]) => (
              <div key={label} className="rounded-[22px] bg-[rgba(255,255,255,0.76)] px-4 py-4">
                <div className="text-[11px] uppercase tracking-[0.18em] text-[color:var(--muted)]">{label}</div>
                <div className="mt-2 text-sm font-semibold">{value}</div>
              </div>
            ))}
          </div>

          {currentTask?.source_uri ? (
            <div className="mt-4 rounded-[20px] border border-[rgba(69,49,30,0.12)] bg-[rgba(255,255,255,0.72)] px-4 py-4 text-xs leading-6 text-[color:var(--muted)]">
              <div className="font-semibold text-[color:var(--ink)]">{t.sourceFile}</div>
              <div className="mt-1 break-all">{currentTask.source_uri}</div>
              {currentTask.parsed_artifact_uri ? (
                <>
                  <div className="mt-3 font-semibold text-[color:var(--ink)]">{t.parsedArtifact}</div>
                  <div className="mt-1 break-all">{currentTask.parsed_artifact_uri}</div>
                </>
              ) : null}
            </div>
          ) : null}

          {isPolling ? (
            <div className="mt-4 text-xs uppercase tracking-[0.18em] text-[color:var(--accent-dark)]">
              {t.polling}
            </div>
          ) : null}
        </section>
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-3">
        {[
          [t.queue, t.queueCopy],
          [t.storage, t.storageCopy],
          [t.isolation, t.isolationCopy],
        ].map(([title, copy]) => (
          <div key={title} className="gradient-stroke glass-panel rounded-[24px] p-5">
            <div className="text-xs uppercase tracking-[0.22em] text-[color:var(--muted)]">{title}</div>
            <div className="mt-3 text-sm leading-7 text-[color:var(--muted)]">{copy}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
