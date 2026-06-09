from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
import json
import os
import queue
import socket
import threading
import time
import uuid
from typing import Any, Generator

import gradio as gr
import requests


API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000/api")
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
PLACEHOLDER_DELAY_SECONDS = float(os.getenv("PLACEHOLDER_DELAY_SECONDS", "3"))


def _pick_port(default_port: int = 7860, max_tries: int = 20) -> int:
    env_port = os.getenv("GRADIO_SERVER_PORT")
    if env_port:
        return int(env_port)
    for port in range(default_port, default_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise OSError(f"Cannot find empty port in range: {default_port}-{default_port + max_tries - 1}")


def _new_session_id() -> str:
    return uuid.uuid4().hex


def _format_citations(items: list[dict[str, Any]]) -> str:
    formatted = []
    for item in items:
        metadata = item.get("metadata", {}) or {}
        chunk_id = item.get("chunk_id", "")
        preview_text = (item.get("text") or "").strip()
        page_type = metadata.get("page_type", "")
        object_id = metadata.get("table_id") or metadata.get("visual_id") or chunk_id
        if not preview_text:
            preview_parts = []
            if metadata.get("doc_name"):
                preview_parts.append(f"doc={metadata['doc_name']}")
            if metadata.get("profile"):
                preview_parts.append(f"profile={metadata['profile']}")
            if page_type:
                preview_parts.append(f"type={page_type}")
            if object_id:
                preview_parts.append(f"id={object_id}")
            preview_text = " | ".join(preview_parts)
        formatted.append(
            {
                "page_number": item.get("page_number", 0),
                "logical_page": item.get("logical_page"),
                "score": round(float(item.get("score", 0.0)), 4),
                "chunk_id": chunk_id,
                "page_type": page_type,
                "object_id": object_id,
                "doc_name": metadata.get("doc_name", ""),
                "profile": metadata.get("profile", ""),
                "corpus": metadata.get("corpus", "default"),
                "preview": preview_text[:180],
            }
        )
    return json.dumps(formatted, ensure_ascii=False, indent=2)


def _format_request_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            detail = exc.response.json()
        except ValueError:
            detail = exc.response.text.strip()
        if detail:
            return f"请求失败：{detail}"
    return f"请求失败：{exc}"


def _run_with_placeholder(task, placeholder_output):
    result_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()

    def worker() -> None:
        try:
            result_queue.put(("ok", task()))
        except Exception as exc:  # pragma: no cover
            result_queue.put(("error", _format_request_error(exc)))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    started_at = time.time()
    placeholder_sent = False
    while True:
        try:
            status, payload = result_queue.get(timeout=0.2)
            if status == "error":
                yield payload
                return
            yield payload
            return
        except queue.Empty:
            if not placeholder_sent and time.time() - started_at >= PLACEHOLDER_DELAY_SECONDS:
                placeholder_sent = True
                yield placeholder_output


def upload_pdf(pdf_path: str | None) -> tuple[str, str]:
    if not pdf_path:
        return "请先选择一个 PDF 文件。", "未上传"
    with open(pdf_path, "rb") as handle:
        files = {"pdf": (os.path.basename(pdf_path), handle, "application/pdf")}
        response = requests.post(f"{API_BASE}/upload-pdf", files=files, timeout=900)
    response.raise_for_status()
    data = response.json()
    return (
        f"上传并入库完成：{data['filename']}，共 {data['chunks']} 个分块，集合 {data['collection_name']}",
        data["filename"],
    )


def _build_answer_markdown(data: dict[str, Any], show_rewrite: bool) -> str:
    lines = [str(data.get("answer") or "")]
    rewritten_query = str(data.get("rewritten_query") or "").strip()
    if show_rewrite and rewritten_query:
        lines.append("")
        lines.append(f"> 改写问题：{rewritten_query}")
    resolved_company = str(data.get("resolved_company") or "").strip()
    if resolved_company:
        lines.append(f"> 解析公司：{resolved_company}")
    if data.get("used_history"):
        lines.append("> 使用了会话历史承接")
    return "\n".join(lines).strip()


def _fetch_conversation_state(session_id: str) -> dict[str, Any]:
    response = requests.get(f"{API_BASE}/conversation/{session_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def _clear_conversation_state(session_id: str) -> None:
    response = requests.delete(f"{API_BASE}/conversation/{session_id}", timeout=30)
    response.raise_for_status()


def _query_backend(query: str, language: str, corpus: str, session_id: str, enable_conversation: bool) -> dict[str, Any]:
    payload = {
        "query": query,
        "language": language,
        "use_llm": True,
        "corpus": corpus,
        "session_id": session_id,
        "enable_conversation": enable_conversation,
    }
    response = requests.post(f"{API_BASE}/query", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def _transcribe_audio(audio_path: str | None) -> dict[str, Any]:
    if not audio_path:
        raise ValueError("请先上传或录制语音。")
    with open(audio_path, "rb") as handle:
        files = {"audio": (os.path.basename(audio_path), handle, "audio/wav")}
        response = requests.post(f"{API_BASE}/transcribe", files=files, timeout=150)
    response.raise_for_status()
    return response.json()


def _render_history(state: dict[str, Any], show_rewrite: bool) -> list[dict[str, str]]:
    history = []
    for turn in state.get("history_turns", []) or []:
        user_text = str(turn.get("query") or "")
        lines = [str(turn.get("answer_summary") or "") or "(无摘要)"]
        rewritten_query = str(turn.get("rewritten_query") or "").strip()
        if show_rewrite and rewritten_query:
            lines.append(f"改写问题：{rewritten_query}")
        resolved_company = str(turn.get("resolved_company") or "").strip()
        if resolved_company:
            lines.append(f"解析公司：{resolved_company}")
        if turn.get("used_history"):
            lines.append("使用了会话历史承接")
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": "\n".join(lines).strip()})
    return history


def ask_question_stream(
    query: str,
    language: str,
    corpus: str,
    session_id: str,
    enable_conversation: bool,
    show_rewrite: bool,
    chat_history: list[dict[str, str]] | None,
) -> Generator[tuple[list[dict[str, str]], str, str, str], None, None]:
    history = list(chat_history or [])
    if not query.strip():
        yield history, "请输入问题。", "", session_id
        return
    active_session_id = session_id or _new_session_id()
    pending_history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": "正在检索，请稍后..."}]
    yield pending_history, "", "", active_session_id

    def task() -> tuple[list[dict[str, str]], str, str, str]:
        data = _query_backend(query, language, corpus, active_session_id, enable_conversation)
        answer_text = _build_answer_markdown(data, show_rewrite)
        final_history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": answer_text}]
        return final_history, _format_citations(data.get("citations", [])), "", active_session_id

    def placeholder() -> tuple[list[dict[str, str]], str, str, str]:
        return pending_history, "", "", active_session_id

    for payload in _run_with_placeholder(task, placeholder()):
        if isinstance(payload, str):
            error_history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": payload}]
            yield error_history, "", payload, active_session_id
            return
        yield payload


def ask_audio_question_stream(
    audio_path: str | None,
    language: str,
    corpus: str,
    session_id: str,
    enable_conversation: bool,
    show_rewrite: bool,
    chat_history: list[dict[str, str]] | None,
) -> Generator[tuple[list[dict[str, str]], str, str, str, str], None, None]:
    history = list(chat_history or [])
    active_session_id = session_id or _new_session_id()
    placeholder_history = history + [{"role": "assistant", "content": "正在识别语音并检索，请稍后..."}]
    yield placeholder_history, "", "", "", active_session_id

    def task() -> tuple[list[dict[str, str]], str, str, str, str]:
        transcript_data = _transcribe_audio(audio_path)
        transcript_text = str(transcript_data.get("text") or "").strip()
        if not transcript_text:
            return history, "", "", "未识别到有效语音内容。", active_session_id
        query_data = _query_backend(transcript_text, language, corpus, active_session_id, enable_conversation)
        answer_text = _build_answer_markdown(query_data, show_rewrite)
        final_history = history + [
            {"role": "user", "content": f"🎤 {transcript_text}"},
            {"role": "assistant", "content": answer_text},
        ]
        return final_history, transcript_text, _format_citations(query_data.get("citations", [])), "", active_session_id

    def placeholder() -> tuple[list[dict[str, str]], str, str, str, str]:
        return placeholder_history, "", "", "", active_session_id

    for payload in _run_with_placeholder(task, placeholder()):
        if isinstance(payload, str):
            error_history = history + [{"role": "assistant", "content": payload}]
            yield error_history, "", "", payload, active_session_id
            return
        yield payload


def start_new_conversation() -> tuple[list[dict[str, str]], str, str, str, str]:
    return [], _new_session_id(), "", "", "已创建新会话。"


def clear_current_conversation(session_id: str) -> tuple[list[dict[str, str]], str, str, str, str]:
    if session_id:
        _clear_conversation_state(session_id)
    new_session_id = _new_session_id()
    return [], new_session_id, "", "", "当前会话已清空。"


def reload_conversation(session_id: str, show_rewrite: bool) -> tuple[list[dict[str, str]], str]:
    if not session_id:
        return [], ""
    state = _fetch_conversation_state(session_id)
    return _render_history(state, show_rewrite), state.get("current_company", "")


def submit_feedback(chat_history: list[dict[str, str]] | None, helpful: bool) -> str:
    label = "有帮助" if helpful else "需改进"
    if not chat_history:
        return f"反馈已记录：{label}"
    last_answer = ""
    for item in reversed(chat_history):
        if item.get("role") == "assistant":
            last_answer = str(item.get("content") or "")
            break
    return f"反馈已记录：{label}\n\n答案摘要：\n{last_answer[:120]}"


with gr.Blocks(title="PDF RAG QA") as demo:
    session_state = gr.State(_new_session_id())
    chat_state = gr.State([])

    gr.Markdown(
        """
        # PDF 文档 RAG 问答系统
        支持默认招股书问答、上传 PDF 后问答、语音提问、多轮会话线程、证据引用与会话清空。
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="聊天线程", height=560)
        with gr.Column(scale=2):
            session_id_box = gr.Textbox(label="会话 ID", interactive=False)
            current_company_box = gr.Textbox(label="当前承接公司", interactive=False)
            show_rewrite = gr.Checkbox(label="显示改写后的问题", value=True)
            language = gr.Dropdown(choices=["auto", "zh", "en"], value="auto", label="语言")
            corpus_choice = gr.Radio(
                choices=[("默认招股书", "default"), ("当前上传 PDF", "uploaded")],
                value="default",
                label="查询范围",
            )
            enable_conversation = gr.Checkbox(label="启用多轮会话", value=True)
            citations = gr.Code(label="当前轮引用证据", language="json")
            status_box = gr.Textbox(label="状态", lines=4)

    with gr.Row():
        query = gr.Textbox(label="输入问题", lines=3, placeholder="例如：法定代表人是谁？然后追问：那它的注册资本呢？")

    with gr.Row():
        ask_button = gr.Button("发送")
        new_chat_button = gr.Button("新会话")
        clear_chat_button = gr.Button("清空当前会话")
        refresh_button = gr.Button("刷新线程")

    with gr.Accordion("PDF 上传", open=False):
        upload_input = gr.File(label="上传当前要问答的 PDF", file_types=[".pdf"], type="filepath")
        upload_button = gr.Button("上传并建索引")
        upload_status = gr.Textbox(label="上传状态", lines=3)
        active_pdf = gr.Textbox(label="当前上传文档", value="未上传", interactive=False)

    with gr.Accordion("语音问答", open=False):
        audio_input = gr.Audio(label="上传语音 / 录制语音", sources=["upload", "microphone"], type="filepath")
        audio_button = gr.Button("发送语音")
        transcript = gr.Textbox(label="识别文本", lines=4)

    with gr.Row():
        feedback_yes = gr.Button("有帮助")
        feedback_no = gr.Button("需改进")
    feedback_result = gr.Textbox(label="反馈结果", lines=3)

    demo.load(lambda sid: sid, inputs=[session_state], outputs=[session_id_box])
    upload_button.click(upload_pdf, inputs=[upload_input], outputs=[upload_status, active_pdf])
    ask_button.click(
        ask_question_stream,
        inputs=[query, language, corpus_choice, session_state, enable_conversation, show_rewrite, chat_state],
        outputs=[chatbot, citations, status_box, session_state],
    ).then(lambda history: history, inputs=[chatbot], outputs=[chat_state]).then(
        reload_conversation,
        inputs=[session_state, show_rewrite],
        outputs=[chatbot, current_company_box],
    ).then(lambda sid: sid, inputs=[session_state], outputs=[session_id_box]).then(
        lambda: "",
        outputs=[query],
    )
    audio_button.click(
        ask_audio_question_stream,
        inputs=[audio_input, language, corpus_choice, session_state, enable_conversation, show_rewrite, chat_state],
        outputs=[chatbot, transcript, citations, status_box, session_state],
    ).then(lambda history: history, inputs=[chatbot], outputs=[chat_state]).then(
        reload_conversation,
        inputs=[session_state, show_rewrite],
        outputs=[chatbot, current_company_box],
    ).then(lambda sid: sid, inputs=[session_state], outputs=[session_id_box])
    new_chat_button.click(
        start_new_conversation,
        outputs=[chatbot, session_state, citations, current_company_box, status_box],
    ).then(lambda history: history, inputs=[chatbot], outputs=[chat_state]).then(
        lambda sid: sid,
        inputs=[session_state],
        outputs=[session_id_box],
    )
    clear_chat_button.click(
        clear_current_conversation,
        inputs=[session_state],
        outputs=[chatbot, session_state, citations, current_company_box, status_box],
    ).then(lambda history: history, inputs=[chatbot], outputs=[chat_state]).then(
        lambda sid: sid,
        inputs=[session_state],
        outputs=[session_id_box],
    )
    refresh_button.click(
        reload_conversation,
        inputs=[session_state, show_rewrite],
        outputs=[chatbot, current_company_box],
    ).then(lambda history: history, inputs=[chatbot], outputs=[chat_state])
    feedback_yes.click(lambda history: submit_feedback(history, True), inputs=[chat_state], outputs=[feedback_result])
    feedback_no.click(lambda history: submit_feedback(history, False), inputs=[chat_state], outputs=[feedback_result])


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name=GRADIO_SERVER_NAME, server_port=_pick_port())
