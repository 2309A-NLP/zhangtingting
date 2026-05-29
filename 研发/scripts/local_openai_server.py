from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator
from threading import Thread

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


MODEL_PATH = os.getenv("MODEL_PATH", "/app/data/models/Qwen2.5-0.5B-Instruct")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "Qwen2.5-0.5B-Instruct")
HOST = os.getenv("OPENAI_COMPAT_HOST", "0.0.0.0")
PORT = int(os.getenv("OPENAI_COMPAT_PORT", "8001"))


app = FastAPI(title="Local OpenAI Compatible Server")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.3
    max_tokens: int = Field(default=512, alias="max_tokens")
    stream: bool = False


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=dtype,
)
model.to(device)
model.eval()


def generate_text(messages: list[ChatMessage], temperature: float, max_tokens: int) -> tuple[str, int, int]:
    prompt = tokenizer.apply_chat_template(
        [{"role": item.role, "content": item.content} for item in messages],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs = {
        **inputs,
        "max_new_tokens": min(max(1, max_tokens), 192),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = 0.9
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)

    prompt_tokens = int(inputs["input_ids"].shape[1])
    completion_ids = output_ids[0][prompt_tokens:]
    completion_tokens = int(completion_ids.shape[0])
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return text, prompt_tokens, completion_tokens


def build_generation_kwargs(messages: list[ChatMessage], temperature: float, max_tokens: int) -> tuple[dict, int]:
    prompt = tokenizer.apply_chat_template(
        [{"role": item.role, "content": item.content} for item in messages],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs = {
        **inputs,
        "max_new_tokens": min(max(1, max_tokens), 192),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = 0.9
    else:
        generation_kwargs["do_sample"] = False

    return generation_kwargs, int(inputs["input_ids"].shape[1])


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "model": SERVED_MODEL_NAME}


@app.get("/v1/models")
def list_models() -> dict[str, list[dict[str, str]]]:
    return {
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "owned_by": "local",
            }
        ]
    }


@app.post("/v1/chat/completions")
def chat_completions(payload: ChatCompletionRequest):
    started_at = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    if not payload.stream:
        response_text, prompt_tokens, completion_tokens = generate_text(
            payload.messages,
            payload.temperature,
            payload.max_tokens,
        )
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": started_at,
                "model": SERVED_MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

    def stream() -> Iterator[str]:
        generation_kwargs, _ = build_generation_kwargs(
            payload.messages,
            payload.temperature,
            payload.max_tokens,
        )
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_thread = Thread(
            target=model.generate,
            kwargs={**generation_kwargs, "streamer": streamer},
            daemon=True,
        )
        generation_thread.start()

        for piece in streamer:
            if not piece:
                continue
            yield (
                "data: "
                + json.dumps(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": started_at,
                        "model": SERVED_MODEL_NAME,
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
        yield (
            "data: "
            + json.dumps(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": started_at,
                    "model": SERVED_MODEL_NAME,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
