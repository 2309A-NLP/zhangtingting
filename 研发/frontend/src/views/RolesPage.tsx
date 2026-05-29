import { FileUp, Plus, Radar, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { api } from "../lib/api";
import { formatRoleCategory, formatRoleType } from "../lib/labels";
import { useAppStore } from "../stores/app";
import { useAuthStore } from "../stores/auth";
import type { KnowledgeTaskStatus, RoleRecord } from "../types";
import { PageHeader } from "../ui/PageHeader";
import { RoleCard } from "../ui/RoleCard";
import { StatCard } from "../ui/StatCard";

const ACTIVE_TASK_STATUSES = new Set<KnowledgeTaskStatus["status"]>(["queued", "processing"]);

const t = {
  role: "\u89d2\u8272",
  title: "\u89d2\u8272\u5de5\u4f5c\u53f0",
  description:
    "\u5207\u6362\u9884\u8bbe\u89d2\u8272\u3001\u521b\u5efa\u81ea\u5b9a\u4e49\u89d2\u8272\uff0c\u6216\u8005\u6839\u636e\u95ee\u9898\u81ea\u52a8\u5339\u914d\u5408\u9002\u89d2\u8272\u3002\u5f53\u524d\u6fc0\u6d3b\u89d2\u8272\u4f1a\u76f4\u63a5\u51b3\u5b9a\u5bf9\u8bdd\u8bed\u6c14\u3001\u8bb0\u5fc6\u8303\u56f4\u548c\u68c0\u7d22\u8fb9\u754c\u3002",
  totalRoles: "\u89d2\u8272\u603b\u6570",
  presetRoles: "\u9884\u8bbe\u89d2\u8272",
  customRoles: "\u81ea\u5b9a\u4e49\u89d2\u8272",
  totalHint: "\u9884\u8bbe\u89d2\u8272\u548c\u81ea\u5b9a\u4e49\u89d2\u8272\u90fd\u5728\u5f53\u524d\u7528\u6237\u5de5\u4f5c\u533a\u5185\u7edf\u4e00\u7ba1\u7406\u3002",
  presetHint: "\u9002\u5408\u76f4\u63a5\u6d4b\u8bd5\u548c\u7a33\u5b9a\u4f7f\u7528\u7684\u9ed8\u8ba4\u89d2\u8272\u3002",
  customHint: "\u9002\u5408\u4e34\u65f6\u5b9e\u9a8c\u3001\u5782\u76f4\u573a\u666f\u548c\u4e2a\u6027\u5316\u63d0\u793a\u8bcd\u3002",
  filterRoles: "\u7b5b\u9009\u89d2\u8272",
  searchPlaceholder: "\u6309\u89d2\u8272\u540d\u3001\u5206\u7c7b\u6216 role id \u641c\u7d22",
  currentSelection: "\u5f53\u524d\u9009\u62e9",
  noActiveRole: "\u8fd8\u6ca1\u6709\u9009\u4e2d\u89d2\u8272\u3002\u8bf7\u5148\u4ece\u5de6\u4fa7\u5217\u8868\u4e2d\u9009\u62e9\u4e00\u4e2a\u89d2\u8272\u3002",
  emptyRoles: "\u5f53\u524d\u8fd8\u6ca1\u6709\u53ef\u5c55\u793a\u7684\u89d2\u8272\u3002\u8bf7\u786e\u8ba4\u767b\u5f55\u72b6\u6001\u6b63\u5e38\uff0c\u6216\u7a0d\u540e\u91cd\u65b0\u52a0\u8f7d\u3002",
  loadFailed: "\u89d2\u8272\u5217\u8868\u52a0\u8f7d\u5931\u8d25\uff0c\u8bf7\u7a0d\u540e\u91cd\u8bd5\u3002",
  autoDetect: "\u81ea\u52a8\u8bc6\u522b\u89d2\u8272",
  detectPlaceholder: "\u4f8b\u5982\uff1a\u6211\u7684\u52b3\u52a8\u5408\u540c\u5230\u671f\u4e86\uff0c\u516c\u53f8\u4e0d\u7eed\u7b7e\uff0c\u6211\u5e94\u8be5\u600e\u4e48\u5904\u7406\uff1f",
  detectButton: "\u8bc6\u522b\u6700\u4f73\u89d2\u8272",
  createCustomRole: "\u521b\u5efa\u81ea\u5b9a\u4e49\u89d2\u8272",
  roleNamePlaceholder: "\u89d2\u8272\u540d\u79f0\uff0c\u4f8b\u5982\uff1a\u4f01\u4e1a\u6cd5\u52a1\u987e\u95ee",
  promptPlaceholder: "\u5b9a\u4e49\u8bed\u6c14\u3001\u8fb9\u754c\u3001\u56de\u7b54\u98ce\u683c\u548c\u77e5\u8bc6\u8303\u56f4\u3002",
  createButton: "\u521b\u5efa\u89d2\u8272",
  updateButton: "保存角色设置",
  createModeButton: "切换为新建",
  editModeTitle: "编辑当前自定义角色",
  createModeTitle: "创建自定义角色",
  roleUpdated: "已更新角色",
  localLegal: "\u672c\u5730\u5173\u952e\u8bcd\u5339\u914d\u5230\u6cd5\u5f8b\u95ee\u9898",
  localDoctor: "\u672c\u5730\u5173\u952e\u8bcd\u5339\u914d\u5230\u533b\u7597\u95ee\u9898",
  localStock: "\u672c\u5730\u5173\u952e\u8bcd\u5339\u914d\u5230\u6295\u8d44\u95ee\u9898",
  localHistory: "\u672c\u5730\u5173\u952e\u8bcd\u5339\u914d\u5230\u5386\u53f2\u95ee\u9898",
  localMiss: "\u672c\u5730\u672a\u547d\u4e2d\uff0c\u8bf7\u624b\u52a8\u9009\u62e9\u89d2\u8272\u3002",
  matchedRolePrefix: "\u5df2\u672c\u5730\u5339\u914d\u89d2\u8272\uff1a",
  matchedReasonPrefix: "\u539f\u56e0\uff1a",
  createdRole: "\u5df2\u521b\u5efa\u89d2\u8272",
  createFailed: "\u89d2\u8272\u521b\u5efa\u5931\u8d25",
  categoryGeneral: "\u901a\u7528",
  categoryLawyer: "\u6cd5\u5f8b",
  categoryDoctor: "\u533b\u7597",
  categoryStock: "\u6295\u8d44",
  categoryHistory: "\u5386\u53f2",
  optionalUpload: "创建时可选上传知识文件",
  uploadHint: "支持 PDF / TXT / JSON / HTML。创建时可以不上传；编辑自定义角色时也能继续上传。",
  importModeIncremental: "导入模式：增量",
  importModeFull: "导入模式：全量",
  createFilePlaceholder: "点击选择一个可选文件",
  noFileSelected: "当前未选择文件",
  uploadForRole: "给当前自定义角色上传文件",
  overwriteUpload: "如果是同内容文件，允许新上传覆盖旧知识",
  presetRoleUploadBlocked: "当前选中的是预设角色，请先选择或创建自定义角色后再在这里上传文件。",
  uploadButton: "保存并上传",
  uploadBusy: "上传中...",
  deleteRole: "删除当前角色",
  deleteConfirm: "确定删除当前自定义角色吗？该角色的本地对话和文件记录也会一起移除。",
  deleteFailed: "角色删除失败",
  deleteSuccess: "已删除角色：",
  uploading: "正在上传文件并加入导入队列...",
  uploadFailed: "文件上传失败",
  queuePrefix: "上传任务 ",
  queueSuffix: " 已入队，后端稍后会继续处理。",
  processingSuffix: " 正在处理中，文件会被解析并写入当前角色知识空间。",
  successMiddle: " 已完成。",
  successChunkPrefix: " 共生成 ",
  successChunkSuffix: " 个分块。",
  failedMiddle: " 失败。",
};

const LOCAL_ROLE_KEYWORDS: Array<{
  roleId: string;
  reason: string;
  keywords: string[];
}> = [
  {
    roleId: "lawyer_01",
    reason: t.localLegal,
    keywords: [
      "\u5408\u540c",
      "\u52b3\u52a8\u5408\u540c",
      "\u5f8b\u5e08",
      "\u516c\u53f8",
      "\u7ea0\u7eb7",
      "\u8d77\u8bc9",
      "\u6cd5\u5f8b",
      "\u52b3\u52a8",
      "\u8fdd\u7ea6",
      "\u8d54\u507f",
      "\u4ef2\u88c1",
      "legal",
      "contract",
      "lawsuit",
    ],
  },
  {
    roleId: "doctor_01",
    reason: t.localDoctor,
    keywords: [
      "\u75c7\u72b6",
      "\u6cbb\u7597",
      "\u53d1\u70ed",
      "\u75be\u75c5",
      "\u533b\u751f",
      "\u533b\u7597",
      "\u7528\u836f",
      "\u533b\u9662",
      "\u75bc",
      "symptom",
      "treatment",
      "fever",
    ],
  },
  {
    roleId: "stock_01",
    reason: t.localStock,
    keywords: [
      "\u80a1\u7968",
      "\u57fa\u91d1",
      "\u6295\u8d44",
      "\u7406\u8d22",
      "\u8bc1\u5238",
      "\u884c\u60c5",
      "a\u80a1",
      "\u6e2f\u80a1",
      "\u4f30\u503c",
      "stock",
      "fund",
      "market",
      "invest",
    ],
  },
  {
    roleId: "history_01",
    reason: t.localHistory,
    keywords: [
      "\u5386\u53f2",
      "\u738b\u671d",
      "\u4eba\u7269",
      "\u674e\u4e16\u6c11",
      "\u79e6\u59cb\u7687",
      "\u4f20\u8bb0",
      "\u671d\u4ee3",
      "\u7687\u5e1d",
      "history",
      "dynasty",
      "biography",
      "historical",
    ],
  },
];

function detectRoleLocally(query: string, roles: RoleRecord[]) {
  const lowered = query.trim().toLowerCase();
  let bestMatch: { role: RoleRecord; reason: string; score: number } | null = null;

  for (const candidate of LOCAL_ROLE_KEYWORDS) {
    const score = candidate.keywords.reduce((total, keyword) => {
      return lowered.includes(keyword.toLowerCase()) ? total + 1 : total;
    }, 0);
    if (score === 0) {
      continue;
    }

    const role = roles.find((item) => item.role_id === candidate.roleId);
    if (role && (!bestMatch || score > bestMatch.score)) {
      bestMatch = { role, reason: candidate.reason, score };
    }
  }

  return bestMatch;
}

function formatTaskMessage(task: KnowledgeTaskStatus | null) {
  if (!task) {
    return null;
  }

  if (task.status === "queued") {
    return `${t.queuePrefix}${task.task_id}${t.queueSuffix}`;
  }
  if (task.status === "processing") {
    return `${t.queuePrefix}${task.task_id}${t.processingSuffix}`;
  }
  if (task.status === "success") {
    const chunkInfo = task.chunk_count
      ? `${t.successChunkPrefix}${task.chunk_count}${t.successChunkSuffix}`
      : "";
    return `${t.queuePrefix}${task.task_id}${t.successMiddle}${chunkInfo}`;
  }
  return `${t.queuePrefix}${task.task_id}${t.failedMiddle}${task.error_message ?? ""}`;
}

export function RolesPage() {
  const token = useAuthStore((state) => state.token)!;
  const user = useAuthStore((state) => state.user)!;
  const activeRole = useAppStore((state) => state.activeRole);
  const setActiveRole = useAppStore((state) => state.setActiveRole);
  const removeRoleState = useAppStore((state) => state.removeRoleState);

  const [roles, setRoles] = useState<RoleRecord[]>([]);
  const [detectQuery, setDetectQuery] = useState("");
  const [createName, setCreateName] = useState("");
  const [createPrompt, setCreatePrompt] = useState("");
  const [createCategory, setCreateCategory] = useState("general");
  const [createMode, setCreateMode] = useState<"incremental" | "full">("incremental");
  const [createFile, setCreateFile] = useState<File | null>(null);
  const [createOverwrite, setCreateOverwrite] = useState(false);
  const [roleEditorMode, setRoleEditorMode] = useState<"create" | "edit">("create");
  const [keyword, setKeyword] = useState("");
  const [busy, setBusy] = useState(false);
  const [detectMessage, setDetectMessage] = useState<string | null>(null);
  const [selectionMessage, setSelectionMessage] = useState<string | null>(null);
  const [editorMessage, setEditorMessage] = useState<string | null>(null);
  const [knowledgeTask, setKnowledgeTask] = useState<KnowledgeTaskStatus | null>(null);

  const mutableActiveRole = activeRole && activeRole.role_type !== "preset" ? activeRole : null;

  async function loadRoles() {
    const list = await api.getRoles(token, user.user_id);
    setRoles(list);
    return list;
  }

  async function queueKnowledgeUpload(
    roleId: string,
    file: File,
    mode: "incremental" | "full",
    overwrite: boolean,
  ) {
    const queued = await api.uploadKnowledge(token, {
      user_id: user.user_id,
      role_id: roleId,
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
    setKnowledgeTask(queuedTask);
    setEditorMessage(formatTaskMessage(queuedTask));
    return queuedTask;
  }

  async function handleDetectRole() {
    setBusy(true);
    setDetectMessage(null);
    try {
      const list = roles.length > 0 ? roles : await loadRoles();
      const localMatch = detectRoleLocally(detectQuery, list);
      if (localMatch) {
        setActiveRole(localMatch.role);
        setDetectMessage(`${t.matchedRolePrefix}${localMatch.role.name}\u3002${t.matchedReasonPrefix}${localMatch.reason}`);
        return;
      }
      setDetectMessage(t.localMiss);
    } catch (err) {
      setDetectMessage(err instanceof Error ? err.message : t.localMiss);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void loadRoles().catch((err) => {
      setSelectionMessage(err instanceof Error ? err.message : t.loadFailed);
    });
  }, []);

  useEffect(() => {
    if (mutableActiveRole) {
      setRoleEditorMode("edit");
      setCreateName(mutableActiveRole.name);
      setCreatePrompt(mutableActiveRole.system_prompt);
      setCreateCategory(mutableActiveRole.category);
      setCreateFile(null);
      setCreateMode("incremental");
      setCreateOverwrite(false);
      return;
    }

    setRoleEditorMode("create");
    setCreateName("");
    setCreatePrompt("");
    setCreateCategory("general");
    setCreateFile(null);
    setCreateMode("incremental");
    setCreateOverwrite(false);
  }, [mutableActiveRole?.role_id]);

  useEffect(() => {
    if (!knowledgeTask || !ACTIVE_TASK_STATUSES.has(knowledgeTask.status)) {
      return undefined;
    }

    let cancelled = false;
    const poll = async () => {
      try {
        const nextTask = await api.getKnowledgeTaskStatus(token, {
          task_id: knowledgeTask.task_id,
          user_id: user.user_id,
          role_id: knowledgeTask.role_id,
        });
        if (cancelled) {
          return;
        }
        setKnowledgeTask(nextTask);
        setEditorMessage(formatTaskMessage(nextTask));
      } catch (err) {
        if (cancelled) {
          return;
        }
        setEditorMessage(err instanceof Error ? err.message : t.uploadFailed);
      }
    };

    const timer = window.setInterval(() => {
      void poll();
    }, 2500);
    void poll();

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [knowledgeTask, token, user.user_id]);

  useEffect(() => {
    if (!detectMessage) {
      return undefined;
    }
    const timer = window.setTimeout(() => setDetectMessage(null), 10000);
    return () => window.clearTimeout(timer);
  }, [detectMessage]);

  useEffect(() => {
    if (!selectionMessage) {
      return undefined;
    }
    const timer = window.setTimeout(() => setSelectionMessage(null), 10000);
    return () => window.clearTimeout(timer);
  }, [selectionMessage]);

  useEffect(() => {
    if (!editorMessage) {
      return undefined;
    }
    const timer = window.setTimeout(() => setEditorMessage(null), 10000);
    return () => window.clearTimeout(timer);
  }, [editorMessage]);

  const grouped = useMemo(() => {
    const term = keyword.trim().toLowerCase();
    const filtered = roles.filter((role) => {
      if (!term) return true;
      return (
        role.name.toLowerCase().includes(term) ||
        role.category.toLowerCase().includes(term) ||
        role.role_id.toLowerCase().includes(term)
      );
    });

    return filtered.reduce<Record<string, RoleRecord[]>>((acc, role) => {
      acc[role.category] ??= [];
      acc[role.category].push(role);
      return acc;
    }, {});
  }, [keyword, roles]);

  const presetCount = roles.filter((role) => role.role_type === "preset").length;
  const customCount = roles.filter((role) => role.role_type !== "preset").length;

  return (
    <div>
      <PageHeader eyebrow={t.role} title={t.title} description={t.description} />

      <div className="mb-6 grid gap-4 md:grid-cols-3">
        <StatCard label={t.totalRoles} value={String(roles.length).padStart(2, "0")} hint={t.totalHint} />
        <StatCard label={t.presetRoles} value={String(presetCount).padStart(2, "0")} hint={t.presetHint} />
        <StatCard label={t.customRoles} value={String(customCount).padStart(2, "0")} hint={t.customHint} />
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <section className="space-y-6">
          <div className="gradient-stroke glass-panel rounded-[28px] p-5">
            <div className="mb-3 text-xs uppercase tracking-[0.22em] text-[color:var(--muted)]">
              {t.filterRoles}
            </div>
            <input
              value={keyword}
              onChange={(event) => setKeyword(event.target.value)}
              placeholder={t.searchPlaceholder}
              className="w-full rounded-2xl border border-[color:var(--line)] bg-[rgba(255,255,255,0.8)] px-4 py-3 outline-none"
            />
          </div>

          {Object.entries(grouped).map(([category, items]) => (
            <div key={category}>
              <div className="mb-4 text-xs uppercase tracking-[0.24em] text-[color:var(--muted)]">
                {formatRoleCategory(category)}
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                {items.map((role) => (
                  <RoleCard
                    key={role.role_id}
                    role={role}
                    active={activeRole?.role_id === role.role_id}
                    onSelect={setActiveRole}
                  />
                ))}
              </div>
            </div>
          ))}
        </section>

        <section className="space-y-6">
          <div className="gradient-stroke glass-panel rounded-[28px] p-5">
            <div className="mb-4 text-sm font-semibold">{t.currentSelection}</div>
            {activeRole ? (
              <>
                <div className="brand-title text-3xl font-semibold">{activeRole.name}</div>
                <div className="mt-2 text-xs uppercase tracking-[0.22em] text-[color:var(--muted)]">
                  {activeRole.role_id} {"\u00b7"} {formatRoleCategory(activeRole.category)} {"\u00b7"} {formatRoleType(activeRole.role_type)}
                </div>
                <p className="mt-4 text-sm leading-7 text-[color:var(--muted)]">{activeRole.system_prompt}</p>
                {mutableActiveRole ? (
                  <button
                    type="button"
                    disabled={busy}
                    onClick={async () => {
                      if (!window.confirm(t.deleteConfirm)) {
                        return;
                      }
                      setBusy(true);
                      setSelectionMessage(null);
                      try {
                        const deletingRole = mutableActiveRole;
                        await api.deleteRole(token, {
                          user_id: user.user_id,
                          role_id: deletingRole.role_id,
                        });
                        removeRoleState(deletingRole.role_id);
                        setKnowledgeTask(null);
                        setCreateFile(null);
                        setRoleEditorMode("create");
                        const list = await loadRoles();
                        setActiveRole(list[0] ?? null);
                        setSelectionMessage(`${t.deleteSuccess}${deletingRole.name}`);
                      } catch (err) {
                        setSelectionMessage(err instanceof Error ? err.message : t.deleteFailed);
                      } finally {
                        setBusy(false);
                      }
                    }}
                    className="mt-5 inline-flex items-center gap-2 rounded-full border border-[rgba(188,86,55,0.28)] px-4 py-3 text-sm text-[color:var(--accent-dark)] transition hover:bg-[rgba(255,245,233,0.9)] disabled:opacity-60"
                  >
                    <Trash2 size={16} />
                    {t.deleteRole}
                  </button>
                ) : null}
                {selectionMessage ? (
                  <div className="mt-4 rounded-2xl border border-[rgba(188,86,55,0.22)] bg-[rgba(255,250,243,0.9)] px-4 py-3 text-sm leading-7 text-[color:var(--muted)]">
                    {selectionMessage}
                  </div>
                ) : null}
              </>
            ) : (
              <div className="text-sm leading-7 text-[color:var(--muted)]">{t.noActiveRole}</div>
            )}
          </div>

          {roles.length === 0 && !selectionMessage ? (
            <div className="rounded-[28px] border border-dashed border-[rgba(69,49,30,0.18)] px-5 py-6 text-sm leading-7 text-[color:var(--muted)]">
              {t.emptyRoles}
            </div>
          ) : null}

          <div className="gradient-stroke glass-panel rounded-[28px] p-5">
            <div className="mb-4 flex items-center gap-2 text-sm font-semibold">
              <Radar size={16} />
              {t.autoDetect}
            </div>
            <textarea
              value={detectQuery}
              onChange={(event) => setDetectQuery(event.target.value)}
              rows={5}
              placeholder={t.detectPlaceholder}
              className="w-full rounded-2xl border border-[color:var(--line)] bg-[rgba(255,255,255,0.8)] px-4 py-3 outline-none"
            />
            <button
              type="button"
              disabled={busy || !detectQuery.trim()}
              onClick={handleDetectRole}
              className="mt-4 rounded-full bg-[color:var(--accent)] px-5 py-3 text-sm text-white disabled:opacity-60"
            >
              {t.detectButton}
            </button>
            {detectMessage ? (
              <div className="mt-4 rounded-2xl border border-[rgba(188,86,55,0.22)] bg-[rgba(255,250,243,0.9)] px-4 py-3 text-sm leading-7 text-[color:var(--muted)]">
                {detectMessage}
              </div>
            ) : null}
          </div>

          <div className="gradient-stroke glass-panel rounded-[28px] p-5">
            <div className="mb-4 flex items-center gap-2 text-sm font-semibold">
              <Plus size={16} />
              {roleEditorMode === "edit" ? t.editModeTitle : t.createModeTitle}
            </div>
            <div className="mb-4 rounded-2xl bg-[rgba(255,249,241,0.86)] px-4 py-3 text-sm leading-7 text-[color:var(--muted)]">
              <div>{t.optionalUpload}</div>
              <div>{t.uploadHint}</div>
            </div>
            {roleEditorMode === "edit" ? (
              <button
                type="button"
                disabled={busy}
                onClick={() => {
                  setRoleEditorMode("create");
                  setCreateName("");
                  setCreatePrompt("");
                  setCreateCategory("general");
                  setCreateFile(null);
                  setCreateMode("incremental");
                }}
                className="mb-4 rounded-full border border-[color:var(--line)] px-4 py-2 text-sm text-[color:var(--muted)] transition hover:border-[rgba(188,86,55,0.3)] disabled:opacity-60"
              >
                {t.createModeButton}
              </button>
            ) : null}
            <div className="space-y-4">
              <input
                value={createName}
                onChange={(event) => setCreateName(event.target.value)}
                placeholder={t.roleNamePlaceholder}
                className="w-full rounded-2xl border border-[color:var(--line)] bg-[rgba(255,255,255,0.8)] px-4 py-3 outline-none"
              />
              <select
                value={createCategory}
                onChange={(event) => setCreateCategory(event.target.value)}
                className="w-full rounded-2xl border border-[color:var(--line)] bg-[rgba(255,255,255,0.8)] px-4 py-3 outline-none"
              >
                {[
                  ["general", t.categoryGeneral],
                  ["lawyer", t.categoryLawyer],
                  ["doctor", t.categoryDoctor],
                  ["stock", t.categoryStock],
                  ["history", t.categoryHistory],
                ].map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
              <textarea
                value={createPrompt}
                onChange={(event) => setCreatePrompt(event.target.value)}
                rows={6}
                placeholder={t.promptPlaceholder}
                className="w-full rounded-2xl border border-[color:var(--line)] bg-[rgba(255,255,255,0.8)] px-4 py-3 outline-none"
              />
              <select
                value={createMode}
                onChange={(event) => setCreateMode(event.target.value as "incremental" | "full")}
                className="w-full rounded-2xl border border-[color:var(--line)] bg-[rgba(255,255,255,0.8)] px-4 py-3 outline-none"
              >
                <option value="incremental">{t.importModeIncremental}</option>
                <option value="full">{t.importModeFull}</option>
              </select>
              <label className="block rounded-2xl border border-dashed border-[rgba(188,86,55,0.28)] bg-[rgba(255,248,241,0.75)] px-4 py-4 text-sm text-[color:var(--muted)]">
                <div className="flex items-center gap-2 font-semibold text-[color:var(--ink)]">
                  <FileUp size={16} />
                  {t.createFilePlaceholder}
                </div>
                <div className="mt-2">{createFile ? createFile.name : t.noFileSelected}</div>
                <input
                  type="file"
                  className="mt-3 block w-full text-sm"
                  accept=".pdf,.txt,.json,.html"
                  onChange={(event) => setCreateFile(event.target.files?.[0] ?? null)}
                />
              </label>
              <label className="flex items-start gap-3 rounded-2xl bg-[rgba(255,249,241,0.86)] px-4 py-4 text-sm leading-7 text-[color:var(--muted)]">
                <input
                  type="checkbox"
                  checked={createOverwrite}
                  onChange={(event) => setCreateOverwrite(event.target.checked)}
                  className="mt-1 h-4 w-4"
                />
                <span>{t.overwriteUpload}</span>
              </label>
            </div>
            <button
              type="button"
              disabled={busy || !createName.trim() || !createPrompt.trim()}
              onClick={async () => {
                setBusy(true);
                setEditorMessage(null);
                try {
                  const saved =
                    roleEditorMode === "edit" && mutableActiveRole
                      ? await api.updateRole(token, {
                          user_id: user.user_id,
                          role_id: mutableActiveRole.role_id,
                          name: createName,
                          category: createCategory,
                          system_prompt: createPrompt,
                        })
                      : await api.createRole(token, {
                          user_id: user.user_id,
                          name: createName,
                          category: createCategory,
                          system_prompt: createPrompt,
                        });
                  if (createFile) {
                    setEditorMessage(t.uploading);
                    await queueKnowledgeUpload(saved.role_id, createFile, createMode, createOverwrite);
                  } else {
                    setKnowledgeTask(null);
                    setEditorMessage(
                      roleEditorMode === "edit"
                        ? `${t.roleUpdated}：${saved.name}`
                        : `${t.createdRole}\uff1a${saved.name}`,
                    );
                  }
                  setCreateFile(null);
                  setCreateMode("incremental");
                  const list = await loadRoles();
                  const matched = list.find((item) => item.role_id === saved.role_id);
                  if (matched) setActiveRole(matched);
                  if (roleEditorMode === "create" && !matched) {
                    setCreateName("");
                    setCreatePrompt("");
                    setCreateCategory("general");
                  }
                } catch (err) {
                  setEditorMessage(err instanceof Error ? err.message : t.createFailed);
                } finally {
                  setBusy(false);
                }
              }}
              className="mt-4 rounded-full bg-[color:var(--accent-dark)] px-5 py-3 text-sm text-white disabled:opacity-60"
            >
              {busy ? t.uploadBusy : createFile ? t.uploadButton : roleEditorMode === "edit" ? t.updateButton : t.createButton}
            </button>
            {editorMessage ? (
              <div className="mt-4 rounded-2xl border border-[rgba(188,86,55,0.22)] bg-[rgba(255,250,243,0.9)] px-4 py-3 text-sm leading-7 text-[color:var(--muted)]">
                {editorMessage}
              </div>
            ) : null}
          </div>
        </section>
      </div>
    </div>
  );
}
