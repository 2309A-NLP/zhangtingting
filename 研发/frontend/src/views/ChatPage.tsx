import { Eraser, MessageSquarePlus, MessagesSquare, Orbit, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { api } from "../lib/api";
import { formatRoleCategory, formatRoleType } from "../lib/labels";
import { useAppStore } from "../stores/app";
import { useAuthStore } from "../stores/auth";
import type { ContextSource } from "../types";
import { ChatBubble } from "../ui/ChatBubble";
import { Composer } from "../ui/Composer";
import { PageHeader } from "../ui/PageHeader";

function createMessageId() {
  return (
    globalThis.crypto?.randomUUID?.() ??
    `msg_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`
  );
}

function formatConversationTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function filterVisibleSources(sources: ContextSource[]) {
  return sources.filter((source) => source.score >= 0.5).slice(0, 5);
}

export function ChatPage() {
  const token = useAuthStore((state) => state.token)!;
  const user = useAuthStore((state) => state.user)!;
  const activeRole = useAppStore((state) => state.activeRole);
  const conversationsByRole = useAppStore((state) => state.conversationsByRole);
  const activeConversationIdByRole = useAppStore((state) => state.activeConversationIdByRole);
  const createConversation = useAppStore((state) => state.createConversation);
  const ensureConversation = useAppStore((state) => state.ensureConversation);
  const setActiveConversation = useAppStore((state) => state.setActiveConversation);
  const appendConversationMessage = useAppStore((state) => state.appendConversationMessage);
  const patchLastAssistantMessage = useAppStore((state) => state.patchLastAssistantMessage);
  const clearConversationMessages = useAppStore((state) => state.clearConversationMessages);
  const deleteConversation = useAppStore((state) => state.deleteConversation);
  const setConversationMessages = useAppStore((state) => state.setConversationMessages);

  const roleId = activeRole?.role_id ?? "";
  const conversations = roleId ? conversationsByRole[roleId] ?? [] : [];
  const activeConversationId = roleId ? activeConversationIdByRole[roleId] : "";
  const activeConversation = useMemo(
    () => conversations.find((item) => item.id === activeConversationId) ?? conversations[0] ?? null,
    [activeConversationId, conversations],
  );
  const messages = activeConversation?.messages ?? [];

  const [streaming, setStreaming] = useState(false);
  const [status, setStatus] = useState("请选择一个角色后开始对话。");
  const [latestSources, setLatestSources] = useState<ContextSource[]>([]);

  useEffect(() => {
    if (!activeRole) {
      setStatus("请选择一个角色后开始对话。");
      setLatestSources([]);
      return;
    }
    ensureConversation(activeRole.role_id);
    setStatus(`当前角色已切换为 ${activeRole.name}。`);
    setLatestSources([]);
  }, [activeRole?.role_id]);

  async function handleSend(query: string) {
    if (!activeRole) return;
    const conversationId = activeConversation?.id ?? ensureConversation(activeRole.role_id);

    appendConversationMessage(activeRole.role_id, conversationId, {
      id: createMessageId(),
      role: "user",
      content: query,
    });
    const assistantId = createMessageId();
    appendConversationMessage(activeRole.role_id, conversationId, {
      id: assistantId,
      role: "assistant",
      content: "正在生成，请稍候...",
      pending: true,
      sources: [],
    });

    setStreaming(true);
    setStatus(`正在以 ${activeRole.name} 的身份生成回答...`);
    setLatestSources([]);

    try {
      const response = await api.chat(token, {
        user_id: user.user_id,
        role_id: activeRole.role_id,
        session_id: conversationId,
        query,
        stream: false,
      });
      patchLastAssistantMessage(activeRole.role_id, conversationId, response.response);
      setLatestSources(filterVisibleSources(response.context_sources));

      const next = [
        ...(useAppStore
          .getState()
          .conversationsByRole[activeRole.role_id]?.find((item) => item.id === conversationId)
          ?.messages ?? []),
      ];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].id === assistantId) {
          next[index] = {
            ...next[index],
            content: response.response,
            sources: filterVisibleSources(response.context_sources) as ContextSource[],
            pending: false,
          };
          break;
        }
      }
      setConversationMessages(activeRole.role_id, conversationId, next);
      setStatus("回答完成。");
    } catch (err) {
      patchLastAssistantMessage(
        activeRole.role_id,
        conversationId,
        err instanceof Error ? err.message : "请求失败",
      );
      setStatus("请求失败，请检查后端服务是否可用。");
    } finally {
      setStreaming(false);
    }
  }

  function handleNewConversation() {
    if (!activeRole) return;
    createConversation(activeRole.role_id);
    setLatestSources([]);
    setStatus(`已为 ${activeRole.name} 新建对话。`);
  }

  async function handleClearCurrentConversation() {
    if (!activeRole || !activeConversation) return;
    await api.clearChat(token, {
      user_id: user.user_id,
      role_id: activeRole.role_id,
      session_id: activeConversation.id,
    });
    clearConversationMessages(activeRole.role_id, activeConversation.id);
    setLatestSources([]);
    setStatus("已清空当前对话。");
  }

  async function handleDeleteConversation(conversationId: string) {
    if (!activeRole) return;
    await api.clearChat(token, {
      user_id: user.user_id,
      role_id: activeRole.role_id,
      session_id: conversationId,
    });
    deleteConversation(activeRole.role_id, conversationId);
    setLatestSources([]);
    setStatus("已删除对话。");
  }

  return (
    <div className="flex h-[calc(100vh-8rem)] min-h-0 flex-col">
      <div className="grid min-h-0 flex-1 gap-6 xl:grid-cols-[1.12fr_0.88fr]">
        <section className="flex min-h-0 flex-col">
          <PageHeader
            eyebrow="对话"
            title="对话中心"
            description="每个角色可以拥有多个独立对话。切换角色或切换对话时，只显示对应会话的历史记录。"
            actions={
              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  disabled={!activeRole}
                  onClick={handleNewConversation}
                  className="inline-flex items-center gap-2 rounded-full bg-[color:var(--accent)] px-4 py-3 text-sm text-white transition hover:bg-[color:var(--accent-dark)] disabled:opacity-60"
                >
                  <MessageSquarePlus size={16} />
                  新建对话
                </button>
                <button
                  type="button"
                  disabled={!activeRole || !activeConversation}
                  onClick={handleClearCurrentConversation}
                  className="inline-flex items-center gap-2 rounded-full border border-[color:var(--line)] px-4 py-3 text-sm text-[color:var(--muted)] transition hover:border-[rgba(188,86,55,0.3)] disabled:opacity-60"
                >
                  <Eraser size={16} />
                  清空当前对话
                </button>
              </div>
            }
          />

          <div className="gradient-stroke glass-panel flex min-h-0 flex-1 flex-col overflow-hidden rounded-[30px] p-4">
            <div className="mb-4 flex items-center justify-between rounded-[24px] bg-[rgba(253,247,239,0.92)] px-4 py-3 text-sm text-[color:var(--muted)]">
              <div className="flex items-center gap-2">
                <Orbit size={16} />
                {activeRole ? `当前角色：${activeRole.name}` : "尚未选择角色"}
              </div>
              <div>{status}</div>
            </div>

            <div className="scroll-area min-h-0 flex-1 space-y-4 overflow-y-auto px-1 pb-3">
              {messages.length === 0 ? (
                <div className="rounded-[28px] border border-dashed border-[rgba(69,49,30,0.14)] bg-[rgba(255,250,243,0.72)] px-5 py-8 text-[15px] leading-8 text-[color:var(--muted)]">
                  当前对话还没有消息。选择角色后可以直接提问，也可以先新建一个空白对话。
                </div>
              ) : (
                messages.map((message) => <ChatBubble key={message.id} message={message} />)
              )}
            </div>
          </div>
        </section>

        <section className="flex min-h-0 flex-col gap-6">
          <div className="gradient-stroke glass-panel rounded-[30px] p-5">
            <div className="mb-3 text-xs uppercase tracking-[0.22em] text-[color:var(--muted)]">
              当前角色
            </div>
            {activeRole ? (
              <>
                <div className="text-2xl font-semibold">{activeRole.name}</div>
                <div className="mt-2 text-sm uppercase tracking-[0.18em] text-[color:var(--muted)]">
                  {formatRoleCategory(activeRole.category)} · {formatRoleType(activeRole.role_type)}
                </div>
                <p className="mt-4 text-[15px] leading-7 text-[color:var(--muted)]">
                  {activeRole.system_prompt}
                </p>
              </>
            ) : (
              <div className="text-[15px] leading-7 text-[color:var(--muted)]">
                当前还没有激活角色。请先去角色工作台选择角色，再回来发起对话。
              </div>
            )}
          </div>

          <div className="gradient-stroke glass-panel flex min-h-0 flex-1 flex-col rounded-[30px] p-5">
            <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.22em] text-[color:var(--muted)]">
              <MessagesSquare size={15} />
              对话列表
            </div>
            {activeRole ? (
              <div className="scroll-area min-h-0 flex-1 space-y-2 overflow-y-auto">
                {conversations.map((conversation, index) => (
                  <div
                    key={conversation.id}
                    className={[
                      "flex items-stretch gap-2 rounded-[18px] border p-2 transition",
                      conversation.id === activeConversation?.id
                        ? "border-[rgba(188,86,55,0.35)] bg-[rgba(245,227,205,0.9)]"
                        : "border-[color:var(--line)] bg-[rgba(255,248,239,0.72)] hover:border-[rgba(188,86,55,0.24)]",
                    ].join(" ")}
                  >
                    <button
                      type="button"
                      onClick={() => {
                        setActiveConversation(activeRole.role_id, conversation.id);
                        setLatestSources([]);
                      }}
                      className="min-w-0 flex-1 px-2 py-1 text-left"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div className="min-w-0 truncate text-sm font-semibold">
                          {conversation.title || `对话 ${index + 1}`}
                        </div>
                        <div className="shrink-0 text-xs text-[color:var(--muted)]">
                          {conversation.messages.length}
                        </div>
                      </div>
                      <div className="mt-1 text-xs text-[color:var(--muted)]">
                        {formatConversationTime(conversation.updated_at)}
                      </div>
                    </button>
                    <button
                      type="button"
                      aria-label="删除对话"
                      title="删除对话"
                      disabled={streaming}
                      onClick={() => void handleDeleteConversation(conversation.id)}
                      className="flex w-10 shrink-0 items-center justify-center rounded-xl text-[color:var(--muted)] transition hover:bg-[rgba(188,86,55,0.12)] hover:text-[color:var(--accent-dark)] disabled:opacity-50"
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm leading-7 text-[color:var(--muted)]">
                请选择角色后查看对话列表。
              </div>
            )}
          </div>

          <div className="gradient-stroke glass-panel flex max-h-[280px] min-h-0 flex-col rounded-[30px] p-5">
            <div className="mb-3 text-xs uppercase tracking-[0.22em] text-[color:var(--muted)]">
              最新来源
            </div>
            {latestSources.length > 0 ? (
              <div className="scroll-area min-h-0 flex-1 space-y-2 overflow-y-auto">
                {latestSources.map((source, index) => (
                  <div
                    key={`${source.doc_id}-${source.chunk_id}-${index}`}
                    className="rounded-2xl border border-[rgba(69,49,30,0.08)] bg-white/70 px-4 py-3 text-sm text-[color:var(--muted)]"
                  >
                    <div className="font-medium text-[color:var(--ink)]">{source.doc_id}</div>
                    <div className="mt-1">匹配度：{source.score.toFixed(3)}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm leading-7 text-[color:var(--muted)]">
                当前对话的来源片段会显示在这里。
              </div>
            )}
          </div>

          <Composer loading={streaming} disabled={!activeRole || !activeConversation} onSend={handleSend} />
        </section>
      </div>
    </div>
  );
}
