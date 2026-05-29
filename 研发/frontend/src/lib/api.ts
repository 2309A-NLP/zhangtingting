import type {
  ChatResponse,
  DeleteCustomRoleResponse,
  KnowledgeTaskStatus,
  KnowledgeUploadResponse,
  RoleRecord,
  TokenPayload,
  UserProfile,
} from "../types";

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "";
let resolvedApiBaseUrl: string | null = null;

function containsCjk(value: string) {
  return /[\u4e00-\u9fff]/.test(value);
}

function repairMojibake(value: string | null | undefined) {
  if (!value) return value ?? "";
  if (containsCjk(value)) return value;

  try {
    const bytes = Uint8Array.from(value, (char) => char.charCodeAt(0));
    const repaired = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    if (containsCjk(repaired)) {
      return repaired;
    }
  } catch {
    return value;
  }

  return value;
}

function normalizeRoleRecord(role: RoleRecord): RoleRecord {
  return {
    ...role,
    name: repairMojibake(role.name),
    system_prompt: repairMojibake(role.system_prompt),
    category: repairMojibake(role.category),
  };
}

function collectApiBaseCandidates() {
  const candidates = new Set<string>();

  if (configuredApiBaseUrl) {
    candidates.add(configuredApiBaseUrl);
  }

  if (typeof window !== "undefined") {
    const { protocol, hostname } = window.location;
    if (hostname) {
      candidates.add(`${protocol}//${hostname}:8000`);
    }
  }

  candidates.add("http://localhost:8000");
  return Array.from(candidates);
}

function buildHeaders(token?: string, extra?: HeadersInit): HeadersInit {
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...extra,
  };
}

function unwrapEnvelope<T>(payload: unknown): T {
  if (
    payload &&
    typeof payload === "object" &&
    "success" in payload &&
    "data" in payload
  ) {
    return (payload as { data: T }).data;
  }
  return payload as T;
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const message = await safeErrorMessage(response);
    throw new Error(message);
  }
  const payload = await response.json();
  return unwrapEnvelope<T>(payload);
}

async function safeErrorMessage(response: Response) {
  try {
    const body = await response.json();
    return (
      body?.error?.message ||
      body?.detail ||
      body?.message ||
      `Request failed: ${response.status}`
    );
  } catch {
    return `Request failed: ${response.status}`;
  }
}

function buildUrl(baseUrl: string, path: string) {
  return `${baseUrl}${path}`;
}

function isNetworkError(error: unknown) {
  return error instanceof TypeError;
}

function formatNetworkErrorMessage(attemptedBaseUrls: string[]) {
  return `无法连接后端服务，请确认 API 已启动。已尝试：${attemptedBaseUrls.join("、")}`;
}

async function fetchWithFallback(
  path: string,
  init?: RequestInit,
  options?: { allowRetryOnHttpError?: boolean },
) {
  const candidates = resolvedApiBaseUrl
    ? [resolvedApiBaseUrl, ...collectApiBaseCandidates().filter((item) => item !== resolvedApiBaseUrl)]
    : collectApiBaseCandidates();
  const attemptedBaseUrls: string[] = [];
  let lastError: unknown = null;

  for (const baseUrl of candidates) {
    attemptedBaseUrls.push(baseUrl);
    try {
      const response = await fetch(buildUrl(baseUrl, path), init);
      if (!response.ok && options?.allowRetryOnHttpError && response.status >= 500) {
        lastError = new Error(await safeErrorMessage(response));
        continue;
      }
      resolvedApiBaseUrl = baseUrl;
      return response;
    } catch (error) {
      lastError = error;
      if (!isNetworkError(error)) {
        throw error;
      }
    }
  }

  if (isNetworkError(lastError)) {
    throw new Error(formatNetworkErrorMessage(attemptedBaseUrls));
  }
  throw lastError instanceof Error ? lastError : new Error("请求失败，请稍后重试。");
}

export const api = {
  async register(payload: {
    username: string;
    password: string;
    email?: string;
  }): Promise<UserProfile> {
    const response = await fetchWithFallback("/api/v1/auth/register", {
      method: "POST",
      headers: buildHeaders(),
      body: JSON.stringify(payload),
    });
    return parseJson<UserProfile>(response);
  },

  async login(payload: {
    username: string;
    password: string;
  }): Promise<TokenPayload> {
    const response = await fetchWithFallback("/api/v1/auth/login", {
      method: "POST",
      headers: buildHeaders(),
      body: JSON.stringify(payload),
    });
    return parseJson<TokenPayload>(response);
  },

  async me(token: string): Promise<UserProfile> {
    const response = await fetchWithFallback("/api/v1/auth/me", {
      headers: buildHeaders(token),
    });
    return parseJson<UserProfile>(response);
  },

  async getRoles(token: string, userId?: string): Promise<RoleRecord[]> {
    const query = userId ? `?user_id=${encodeURIComponent(userId)}` : "";
    const response = await fetchWithFallback(`/api/v1/roles${query}`, {
      headers: buildHeaders(token),
    });
    const payload = await parseJson<{ total: number; items: RoleRecord[] }>(response);
    return payload.items.map(normalizeRoleRecord);
  },

  async createRole(
    token: string,
    payload: {
      user_id: string;
      name: string;
      category: string;
      system_prompt: string;
    },
  ): Promise<RoleRecord> {
    const response = await fetchWithFallback("/api/v1/roles/custom", {
      method: "POST",
      headers: buildHeaders(token),
      body: JSON.stringify(payload),
    });
    return normalizeRoleRecord(await parseJson<RoleRecord>(response));
  },

  async deleteRole(
    token: string,
    payload: {
      user_id: string;
      role_id: string;
    },
  ): Promise<DeleteCustomRoleResponse> {
    const query = new URLSearchParams({
      user_id: payload.user_id,
    });
    const response = await fetchWithFallback(
      `/api/v1/roles/custom/${encodeURIComponent(payload.role_id)}?${query.toString()}`,
      {
        method: "DELETE",
        headers: buildHeaders(token),
      },
    );
    return parseJson<DeleteCustomRoleResponse>(response);
  },

  async updateRole(
    token: string,
    payload: {
      user_id: string;
      role_id: string;
      name: string;
      category: string;
      system_prompt: string;
    },
  ): Promise<RoleRecord> {
    const response = await fetchWithFallback(`/api/v1/roles/custom/${encodeURIComponent(payload.role_id)}`, {
      method: "PUT",
      headers: buildHeaders(token),
      body: JSON.stringify({
        user_id: payload.user_id,
        name: payload.name,
        category: payload.category,
        system_prompt: payload.system_prompt,
      }),
    });
    return normalizeRoleRecord(await parseJson<RoleRecord>(response));
  },

  async chat(
    token: string,
    payload: {
      user_id: string;
      role_id: string;
      query: string;
      session_id?: string;
      stream?: false;
    },
  ): Promise<ChatResponse> {
    const response = await fetchWithFallback("/api/v1/chat", {
      method: "POST",
      headers: buildHeaders(token),
      body: JSON.stringify({ ...payload, stream: false }),
    });
    return parseJson<ChatResponse>(response);
  },

  async clearChat(token: string, payload: { user_id: string; role_id: string; session_id?: string }) {
    const response = await fetchWithFallback("/api/v1/chat/clear", {
      method: "POST",
      headers: buildHeaders(token),
      body: JSON.stringify(payload),
    });
    return parseJson<{ success: boolean; cleared_keys: string[] }>(response);
  },

  async uploadKnowledge(
    token: string,
    payload: {
      user_id: string;
      role_id: string;
      mode: "full" | "incremental";
      file: File;
      overwrite?: boolean;
    },
  ): Promise<KnowledgeUploadResponse> {
    const form = new FormData();
    form.append("user_id", payload.user_id);
    form.append("role_id", payload.role_id);
    form.append("mode", payload.mode);
    form.append("overwrite", payload.overwrite ? "true" : "false");
    form.append("file", payload.file);

    const response = await fetchWithFallback("/api/v1/knowledge/upload", {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      body: form,
    });
    return parseJson<KnowledgeUploadResponse>(response);
  },

  async getKnowledgeTaskStatus(
    token: string,
    payload: { task_id: string; user_id: string; role_id: string },
  ): Promise<KnowledgeTaskStatus> {
    const query = new URLSearchParams({
      user_id: payload.user_id,
      role_id: payload.role_id,
    });
    const response = await fetchWithFallback(
      `/api/v1/knowledge/tasks/${encodeURIComponent(payload.task_id)}?${query.toString()}`,
      {
        headers: buildHeaders(token),
      },
    );
    return parseJson<KnowledgeTaskStatus>(response);
  },
};

export async function streamChat(
  token: string,
  payload: {
    user_id: string;
    role_id: string;
    query: string;
    session_id?: string;
    stream: true;
  },
  handlers: {
    onStart?: (data: { request_id: string; session_id: string }) => void;
    onSource?: (data: { doc_id: string; chunk_id: string; source: string; score: number }) => void;
    onDelta?: (content: string) => void;
    onEnd?: () => void;
  },
) {
  const response = await fetchWithFallback("/api/v1/chat", {
    method: "POST",
    headers: buildHeaders(token),
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    throw new Error(await safeErrorMessage(response));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "message";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const lines = chunk.split("\n");
      let data = "";
      currentEvent = "message";
      for (const line of lines) {
        if (line.startsWith("event:")) {
          currentEvent = line.slice(6).trim();
        }
        if (line.startsWith("data:")) {
          data += line.slice(5).trim();
        }
      }
      if (!data) continue;
      const parsed = JSON.parse(data);
      if (currentEvent === "start") handlers.onStart?.(parsed);
      if (currentEvent === "source") handlers.onSource?.(parsed);
      if (currentEvent === "delta") handlers.onDelta?.(parsed.content ?? "");
      if (currentEvent === "end") handlers.onEnd?.();
    }
  }
}
