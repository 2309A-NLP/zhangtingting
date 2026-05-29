import type { ChatMessage } from "../types";

export function ChatBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={[
          "max-w-[82%] rounded-[28px] px-5 py-4 shadow-[0_14px_30px_rgba(52,36,24,0.08)]",
          isUser
            ? "bg-[color:var(--accent)] text-white"
            : "border border-[color:var(--line)] bg-[rgba(255,250,243,0.92)] text-[color:var(--ink)]",
        ].join(" ")}
      >
        <div className="whitespace-pre-wrap text-[15px] leading-7">
          {message.content || (message.pending ? "..." : "")}
        </div>

      </div>
    </div>
  );
}
