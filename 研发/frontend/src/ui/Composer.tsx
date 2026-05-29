import { LoaderCircle, SendHorizonal } from "lucide-react";
import { useState } from "react";

export function Composer({
  onSend,
  loading,
  disabled,
}: {
  onSend: (value: string) => Promise<void>;
  loading: boolean;
  disabled?: boolean;
}) {
  const [value, setValue] = useState("");

  return (
    <form
      className="gradient-stroke glass-panel rounded-[28px] p-4"
      onSubmit={async (event) => {
        event.preventDefault();
        const text = value.trim();
        if (!text || loading || disabled) return;
        setValue("");
        await onSend(text);
      }}
    >
      <textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        placeholder={disabled ? "请先在左侧选择角色" : "输入问题，支持流式回复"}
        rows={4}
        disabled={disabled || loading}
        className="w-full resize-none border-none bg-transparent px-2 py-2 text-[15px] leading-7 text-[color:var(--ink)] outline-none placeholder:text-[color:var(--muted)]"
      />
      <div className="mt-3 flex items-center justify-between">
        <div className="text-xs uppercase tracking-[0.22em] text-[color:var(--muted)]">
          流式对话工作区
        </div>
        <button
          type="submit"
          disabled={loading || disabled}
          className="inline-flex items-center gap-2 rounded-full bg-[color:var(--accent)] px-5 py-3 text-sm font-medium text-white transition hover:bg-[color:var(--accent-dark)] disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? <LoaderCircle className="animate-spin" size={16} /> : <SendHorizonal size={16} />}
          发送
        </button>
      </div>
    </form>
  );
}
