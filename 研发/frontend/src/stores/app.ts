import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { ChatConversation, ChatMessage, RoleRecord } from "../types";

function createId(prefix: string) {
  return (
    globalThis.crypto?.randomUUID?.() ??
    `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`
  );
}

function createConversation(title = "新对话"): ChatConversation {
  const now = new Date().toISOString();
  return {
    id: createId("conv"),
    title,
    created_at: now,
    updated_at: now,
    messages: [],
  };
}

function titleFromMessage(content: string) {
  const normalized = content.trim().replace(/\s+/g, " ");
  if (!normalized) return "新对话";
  return normalized.length > 18 ? `${normalized.slice(0, 18)}...` : normalized;
}

function getConversations(state: AppState, roleId: string) {
  return state.conversationsByRole[roleId] ?? [];
}

interface AppState {
  activeRole: RoleRecord | null;
  conversationsByRole: Record<string, ChatConversation[]>;
  activeConversationIdByRole: Record<string, string>;
  setActiveRole: (role: RoleRecord | null) => void;
  removeRoleState: (roleId: string) => void;
  createConversation: (roleId: string) => string;
  ensureConversation: (roleId: string) => string;
  setActiveConversation: (roleId: string, conversationId: string) => void;
  setConversationMessages: (roleId: string, conversationId: string, messages: ChatMessage[]) => void;
  appendConversationMessage: (roleId: string, conversationId: string, message: ChatMessage) => void;
  patchLastAssistantMessage: (roleId: string, conversationId: string, content: string) => void;
  clearConversationMessages: (roleId: string, conversationId: string) => void;
  deleteConversation: (roleId: string, conversationId: string) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      activeRole: null,
      conversationsByRole: {},
      activeConversationIdByRole: {},
      setActiveRole: (role) =>
        set((state) => {
          if (!role) return { activeRole: null };
          const conversations = getConversations(state, role.role_id);
          if (conversations.length > 0) {
            return { activeRole: role };
          }
          const conversation = createConversation();
          return {
            activeRole: role,
            conversationsByRole: {
              ...state.conversationsByRole,
              [role.role_id]: [conversation],
            },
            activeConversationIdByRole: {
              ...state.activeConversationIdByRole,
              [role.role_id]: conversation.id,
            },
          };
        }),
      removeRoleState: (roleId) =>
        set((state) => {
          const conversationsByRole = { ...state.conversationsByRole };
          const activeConversationIdByRole = { ...state.activeConversationIdByRole };
          delete conversationsByRole[roleId];
          delete activeConversationIdByRole[roleId];
          return {
            activeRole: state.activeRole?.role_id === roleId ? null : state.activeRole,
            conversationsByRole,
            activeConversationIdByRole,
          };
        }),
      createConversation: (roleId) => {
        const conversation = createConversation();
        set((state) => ({
          conversationsByRole: {
            ...state.conversationsByRole,
            [roleId]: [conversation, ...getConversations(state, roleId)],
          },
          activeConversationIdByRole: {
            ...state.activeConversationIdByRole,
            [roleId]: conversation.id,
          },
        }));
        return conversation.id;
      },
      ensureConversation: (roleId) => {
        const state = get();
        const conversations = getConversations(state, roleId);
        const activeId = state.activeConversationIdByRole[roleId];
        if (activeId && conversations.some((item) => item.id === activeId)) {
          return activeId;
        }
        if (conversations[0]) {
          set((current) => ({
            activeConversationIdByRole: {
              ...current.activeConversationIdByRole,
              [roleId]: conversations[0].id,
            },
          }));
          return conversations[0].id;
        }
        return get().createConversation(roleId);
      },
      setActiveConversation: (roleId, conversationId) =>
        set((state) => ({
          activeConversationIdByRole: {
            ...state.activeConversationIdByRole,
            [roleId]: conversationId,
          },
        })),
      setConversationMessages: (roleId, conversationId, messages) =>
        set((state) => ({
          conversationsByRole: {
            ...state.conversationsByRole,
            [roleId]: getConversations(state, roleId).map((conversation) =>
              conversation.id === conversationId
                ? {
                    ...conversation,
                    messages,
                    title:
                      conversation.title === "新对话" && messages[0]?.role === "user"
                        ? titleFromMessage(messages[0].content)
                        : conversation.title,
                    updated_at: new Date().toISOString(),
                  }
                : conversation,
            ),
          },
        })),
      appendConversationMessage: (roleId, conversationId, message) =>
        set((state) => ({
          conversationsByRole: {
            ...state.conversationsByRole,
            [roleId]: getConversations(state, roleId).map((conversation) => {
              if (conversation.id !== conversationId) return conversation;
              const messages = [...conversation.messages, message];
              return {
                ...conversation,
                messages,
                title:
                  conversation.title === "新对话" && message.role === "user"
                    ? titleFromMessage(message.content)
                    : conversation.title,
                updated_at: new Date().toISOString(),
              };
            }),
          },
        })),
      patchLastAssistantMessage: (roleId, conversationId, content) =>
        set((state) => ({
          conversationsByRole: {
            ...state.conversationsByRole,
            [roleId]: getConversations(state, roleId).map((conversation) => {
              if (conversation.id !== conversationId) return conversation;
              const messages = [...conversation.messages];
              for (let index = messages.length - 1; index >= 0; index -= 1) {
                if (messages[index].role === "assistant") {
                  messages[index] = {
                    ...messages[index],
                    content,
                    pending: false,
                  };
                  break;
                }
              }
              return {
                ...conversation,
                messages,
                updated_at: new Date().toISOString(),
              };
            }),
          },
        })),
      clearConversationMessages: (roleId, conversationId) =>
        set((state) => ({
          conversationsByRole: {
            ...state.conversationsByRole,
            [roleId]: getConversations(state, roleId).map((conversation) =>
              conversation.id === conversationId
                ? {
                    ...conversation,
                    title: "新对话",
                    messages: [],
                    updated_at: new Date().toISOString(),
              }
                : conversation,
            ),
          },
        })),
      deleteConversation: (roleId, conversationId) =>
        set((state) => {
          const remaining = getConversations(state, roleId).filter(
            (conversation) => conversation.id !== conversationId,
          );
          const nextConversations = remaining.length > 0 ? remaining : [createConversation()];
          const currentActiveId = state.activeConversationIdByRole[roleId];
          const nextActiveId =
            currentActiveId === conversationId || !nextConversations.some((item) => item.id === currentActiveId)
              ? nextConversations[0].id
              : currentActiveId;
          return {
            conversationsByRole: {
              ...state.conversationsByRole,
              [roleId]: nextConversations,
            },
            activeConversationIdByRole: {
              ...state.activeConversationIdByRole,
              [roleId]: nextActiveId,
            },
          };
        }),
    }),
    {
      name: "rag-console-app",
      partialize: (state) => ({
        activeRole: state.activeRole,
        conversationsByRole: state.conversationsByRole,
        activeConversationIdByRole: state.activeConversationIdByRole,
      }),
      merge: (persisted, current) => {
        const saved = persisted as Partial<AppState> & {
          messagesByRole?: Record<string, ChatMessage[]>;
        };
        const conversationsByRole = saved.conversationsByRole ?? {};
        if (saved.messagesByRole) {
          for (const [roleId, messages] of Object.entries(saved.messagesByRole)) {
            if (!conversationsByRole[roleId]) {
              conversationsByRole[roleId] = [
                {
                  ...createConversation(messages[0]?.role === "user" ? titleFromMessage(messages[0].content) : "历史对话"),
                  messages,
                },
              ];
            }
          }
        }
        return {
          ...current,
          ...saved,
          conversationsByRole,
          activeConversationIdByRole: saved.activeConversationIdByRole ?? {},
        } as AppState;
      },
    },
  ),
);
