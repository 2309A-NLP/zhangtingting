from __future__ import annotations

# 工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统
import json
import os
import queue
import socket
import threading
import time
from typing import Generator, Tuple

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


def _format_citations(items) -> str:
    return json.dumps(
        [
            {
                "page_number": item["page_number"],
                "logical_page": item.get("logical_page"),
                "score": round(item["score"], 4),
                "preview": item["text"][:180],
                "corpus": item.get("metadata", {}).get("corpus", "default"),
            }
            for item in items
        ],
        ensure_ascii=False,
        indent=2,
    )


def upload_pdf(pdf_path: str | None) -> Tuple[str, str]:
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


def ask_question(query: str, language: str, corpus: str) -> Tuple[str, str]:
    if not query.strip():
        return "请输入问题。", ""
    payload = {"query": query, "language": language, "use_llm": True, "corpus": corpus}
    response = requests.post(f"{API_BASE}/query", json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["answer"], _format_citations(data["citations"])


def _format_request_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            detail = exc.response.json()
        except ValueError:
            detail = exc.response.text.strip()
        if detail:
            return f"请求失败：{detail}"
    return f"请求失败：{exc}"


def _run_with_placeholder(
    task,
    placeholder_output: Tuple[str, ...],
) -> Generator[Tuple[str, ...], None, None]:
    result_queue: "queue.Queue[tuple[str, Tuple[str, ...] | str]]" = queue.Queue()

    def worker() -> None:
        try:
            result = task()
            result_queue.put(("ok", result))
        except Exception as exc:  # pragma: no cover - UI fallback path
            result_queue.put(("error", _format_request_error(exc)))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    started_at = time.time()
    placeholder_sent = False

    while True:
        try:
            status, payload = result_queue.get(timeout=0.2)
            if status == "error":
                message = f"请求失败：{payload}"
                if len(placeholder_output) == 2:
                    yield message, ""
                elif len(placeholder_output) == 3:
                    yield "", message, ""
                else:
                    yield (message,)
                return
            yield payload
            return
        except queue.Empty:
            if not placeholder_sent and time.time() - started_at >= PLACEHOLDER_DELAY_SECONDS:
                placeholder_sent = True
                yield placeholder_output


def ask_question_stream(query: str, language: str, corpus: str) -> Generator[Tuple[str, str], None, None]:
    if not query.strip():
        yield "请输入问题。", ""
        return
    yield from _run_with_placeholder(
        lambda: ask_question(query, language, corpus),
        ("正在检索，请稍后...", ""),
    )


def transcribe_audio(audio_path: str | None) -> Tuple[str, str]:
    if not audio_path:
        return "", "请先上传或录制语音。"
    with open(audio_path, "rb") as handle:
        files = {"audio": (os.path.basename(audio_path), handle, "audio/wav")}
        response = requests.post(f"{API_BASE}/transcribe", files=files, timeout=150)
    response.raise_for_status()
    data = response.json()
    return data["text"], f"识别完成，耗时约 {data['duration']:.1f} 秒，后端：{data['backend']}"


def answer_from_transcript(transcript_text: str, language: str, corpus: str) -> Tuple[str, str]:
    if not transcript_text.strip():
        return "请先完成语音识别。", ""
    return ask_question(transcript_text.strip(), language, corpus)


def ask_audio_question(audio_path: str | None, language: str, corpus: str) -> Tuple[str, str, str]:
    transcript, status = transcribe_audio(audio_path)
    if not transcript.strip():
        return status, "", ""
    answer, citations = answer_from_transcript(transcript, language, corpus)
    return transcript, answer, citations


def ask_audio_question_stream(
    audio_path: str | None,
    language: str,
    corpus: str,
) -> Generator[Tuple[str, str, str], None, None]:
    if not audio_path:
        yield "请先上传或录制语音。", "", ""
        return
    yield from _run_with_placeholder(
        lambda: ask_audio_question(audio_path, language, corpus),
        ("", "正在检索，请稍后...", ""),
    )


def submit_feedback(answer: str, helpful: bool) -> str:
    label = "有帮助" if helpful else "需改进"
    return f"反馈已记录：{label}\n\n答案摘要：\n{answer[:120]}"


with gr.Blocks(title="PDF RAG QA") as demo:
    gr.Markdown(
        """
        # PDF 文档 RAG 问答系统
        支持默认招股书问答、上传 PDF 后问答、语音提问、中英文问答、证据引用与反馈记录。
        """
    )

    with gr.Tab("PDF 上传"):
        upload_input = gr.File(label="上传当前要问答的 PDF", file_types=[".pdf"], type="filepath")
        upload_button = gr.Button("上传并建索引")
        upload_status = gr.Textbox(label="上传状态", lines=3)
        active_pdf = gr.Textbox(label="当前上传文档", value="未上传", interactive=False)

    with gr.Tab("文本问答"):
        with gr.Row():
            query = gr.Textbox(label="问题 / Question", lines=3, placeholder="例如：法定代表人是谁？")
            language = gr.Dropdown(choices=["auto", "zh", "en"], value="auto", label="语言")
        corpus_choice = gr.Radio(
            choices=[("默认招股书", "default"), ("当前上传 PDF", "uploaded")],
            value="default",
            label="查询范围",
        )
        ask_button = gr.Button("文本提问")
        answer = gr.Textbox(label="答案", lines=10)
        citations = gr.Code(label="引用证据", language="json")

    with gr.Tab("语音问答"):
        audio_input = gr.Audio(
            label="上传语音 / 录制语音",
            sources=["upload", "microphone"],
            type="filepath",
        )
        audio_language = gr.Dropdown(choices=["auto", "zh", "en"], value="auto", label="语言")
        audio_corpus_choice = gr.Radio(
            choices=[("默认招股书", "default"), ("当前上传 PDF", "uploaded")],
            value="default",
            label="查询范围",
        )
        audio_button = gr.Button("语音提问")
        transcript = gr.Textbox(label="识别文本", lines=4)
        audio_answer = gr.Textbox(label="答案", lines=10)
        audio_citations = gr.Code(label="引用证据", language="json")

    with gr.Row():
        feedback_yes = gr.Button("有帮助")
        feedback_no = gr.Button("需改进")
    feedback_result = gr.Textbox(label="反馈结果", lines=3)

    upload_button.click(upload_pdf, inputs=[upload_input], outputs=[upload_status, active_pdf])
    ask_button.click(ask_question_stream, inputs=[query, language, corpus_choice], outputs=[answer, citations])
    audio_button.click(
        ask_audio_question_stream,
        inputs=[audio_input, audio_language, audio_corpus_choice],
        outputs=[transcript, audio_answer, audio_citations],
    )
    feedback_yes.click(lambda text: submit_feedback(text, True), inputs=[answer], outputs=[feedback_result])
    feedback_no.click(lambda text: submit_feedback(text, False), inputs=[answer], outputs=[feedback_result])


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name=GRADIO_SERVER_NAME, server_port=_pick_port())
