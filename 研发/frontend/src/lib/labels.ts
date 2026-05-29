import type { KnowledgeTaskStatus, RoleType } from "../types";

export function formatRoleCategory(category: string) {
  const labels: Record<string, string> = {
    general: "通用",
    lawyer: "法律",
    doctor: "医疗",
    stock: "投资",
    history: "历史",
  };

  return labels[category] ?? category;
}

export function formatRoleType(roleType: RoleType) {
  const labels: Record<RoleType, string> = {
    preset: "预设",
    custom: "自定义",
    auto: "自动",
  };

  return labels[roleType] ?? roleType;
}

export function formatKnowledgeMode(mode: "incremental" | "full") {
  return mode === "incremental" ? "增量导入" : "全量重建";
}

export function formatKnowledgeStatus(status: KnowledgeTaskStatus["status"] | "idle") {
  const labels: Record<KnowledgeTaskStatus["status"] | "idle", string> = {
    idle: "待开始",
    queued: "排队中",
    processing: "处理中",
    success: "成功",
    failed: "失败",
  };

  return labels[status] ?? status;
}
