export type RoleType = "preset" | "custom" | "auto";

export interface UserProfile {
  user_id: string;
  username: string;
  email?: string | null;
}

export interface TokenPayload {
  access_token: string;
  token_type: "bearer";
  expires_in_seconds: number;
  user_id: string;
  username: string;
}

export interface RoleRecord {
  role_id: string;
  name: string;
  category: string;
  role_type: RoleType;
  system_prompt: string;
  knowledge_base_id?: string | null;
  created_at?: string | null;
}

export interface ContextSource {
  doc_id: string;
  chunk_id: string;
  source: string;
  score: number;
}

export interface ChatResponse {
  request_id: string;
  role_id: string;
  role_name: string;
  session_id: string;
  response: string;
  context_sources: ContextSource[];
  tokens_used: number;
  latency_ms: number;
  model: string;
  degraded_to_online_api: boolean;
  rewritten_query: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  pending?: boolean;
  sources?: ContextSource[];
}

export interface ChatConversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
}

export interface DeleteCustomRoleResponse {
  success: boolean;
  role_id: string;
}

export interface KnowledgeUploadResponse {
  task_id: string;
  user_id: string;
  role_id: string;
  mode: "incremental" | "full";
  status: "queued" | "processing" | "success" | "failed";
  overwrite: boolean;
  duplicate_of_file_id?: string | null;
  uploaded_at: string;
}

export interface KnowledgeTaskStatus {
  task_id: string;
  user_id: string;
  role_id: string;
  mode: "incremental" | "full";
  status: "queued" | "processing" | "success" | "failed";
  doc_id?: string | null;
  source_uri?: string | null;
  parsed_artifact_uri?: string | null;
  chunk_count?: number | null;
  error_message?: string | null;
  started_at?: number | null;
  finished_at?: number | null;
}
