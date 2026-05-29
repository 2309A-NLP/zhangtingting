from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx


DEFAULT_EVAL_SET = Path("scripts/eval_set.json")


@dataclass(slots=True)
class RequestSample:
    index: int
    ok: bool
    status_code: int
    elapsed_ms: float
    server_latency_ms: int | None
    tokens_used: int | None
    role_id: str
    query: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple async load test for /api/v1/chat.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL.")
    parser.add_argument("--api-prefix", default="/api/v1", help="API prefix.")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent workers.")
    parser.add_argument("--requests", type=int, default=20, help="Total request count.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    parser.add_argument("--top-k", type=int, default=3, help="Chat retrieval top_k.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Chat temperature.")
    parser.add_argument("--stream", action="store_true", help="Use stream mode.")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET), help="JSON eval set path.")
    parser.add_argument("--username", default="demo_user", help="Login username.")
    parser.add_argument("--password", default="demo123456", help="Login password.")
    parser.add_argument("--token", default="", help="Bearer token. If provided, login is skipped.")
    parser.add_argument("--dev-user-id", default="", help="Use X-User-Id header instead of login.")
    parser.add_argument("--user-id", default="", help="Force chat request user_id.")
    parser.add_argument(
        "--auth-mode",
        choices=["auto", "login", "token", "dev-header"],
        default="auto",
        help="Authentication mode. auto will try token, then dev header, then login.",
    )
    parser.add_argument("--fixed-session", action="store_true", help="Reuse one session_id to include history/cache effects.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def load_eval_items(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Eval set must be a non-empty JSON list.")
    normalized: list[dict[str, Any]] = []
    for item in raw:
        question = str(item.get("question", "")).strip()
        role_id = str(item.get("role_id", "")).strip()
        if question and role_id:
            normalized.append({"question": question, "role_id": role_id})
    if not normalized:
        raise ValueError("No valid questions found in eval set.")
    return normalized


async def login_for_token(client: httpx.AsyncClient, base_api_url: str, username: str, password: str) -> tuple[str, str]:
    response = await client.post(
        f"{base_api_url}/auth/login",
        json={"username": username, "password": password},
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", {})
    token = str(data.get("access_token", "")).strip()
    user_id = str(data.get("user_id", "")).strip()
    if not token or not user_id:
        raise RuntimeError("Login succeeded but token/user_id missing.")
    return token, user_id


def resolve_auth_mode(args: argparse.Namespace) -> str:
    if args.auth_mode != "auto":
        return args.auth_mode
    if args.token:
        return "token"
    if args.dev_user_id:
        return "dev-header"
    return "login"


async def send_one(
    *,
    client: httpx.AsyncClient,
    chat_url: str,
    headers: dict[str, str],
    sample_index: int,
    item: dict[str, Any],
    user_id: str,
    top_k: int,
    temperature: float,
    stream: bool,
    session_id: str,
) -> RequestSample:
    body = {
        "user_id": user_id,
        "role_id": item["role_id"],
        "query": item["question"],
        "stream": stream,
        "top_k": top_k,
        "temperature": temperature,
        "session_id": session_id,
    }
    started_at = time.perf_counter()
    try:
        response = await client.post(chat_url, headers=headers, json=body)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        raw_text = response.text
        try:
            payload = response.json()
        except Exception:
            payload = None
        if response.status_code != 200:
            return RequestSample(
                index=sample_index,
                ok=False,
                status_code=response.status_code,
                elapsed_ms=elapsed_ms,
                server_latency_ms=None,
                tokens_used=None,
                role_id=item["role_id"],
                query=item["question"],
                error=(json.dumps(payload, ensure_ascii=False) if payload is not None else raw_text[:500]),
            )

        if payload is None:
            return RequestSample(
                index=sample_index,
                ok=False,
                status_code=response.status_code,
                elapsed_ms=elapsed_ms,
                server_latency_ms=None,
                tokens_used=None,
                role_id=item["role_id"],
                query=item["question"],
                error=f"HTTP 200 but response was not valid JSON: {raw_text[:500]}",
            )

        data = payload.get("data", {})
        return RequestSample(
            index=sample_index,
            ok=bool(payload.get("success", False)),
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
            server_latency_ms=_safe_int(data.get("latency_ms")),
            tokens_used=_safe_int(data.get("tokens_used")),
            role_id=item["role_id"],
            query=item["question"],
            error="" if payload.get("success", False) else str(payload.get("error", "")),
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        return RequestSample(
            index=sample_index,
            ok=False,
            status_code=0,
            elapsed_ms=elapsed_ms,
            server_latency_ms=None,
            tokens_used=None,
            role_id=item["role_id"],
            query=item["question"],
            error=str(exc),
        )


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def summarize(samples: list[RequestSample], wall_time_s: float) -> dict[str, Any]:
    elapsed_values = sorted(sample.elapsed_ms for sample in samples)
    server_values = sorted(sample.server_latency_ms for sample in samples if sample.server_latency_ms is not None)
    token_values = [sample.tokens_used for sample in samples if sample.tokens_used is not None]
    successes = [sample for sample in samples if sample.ok]
    failures = [sample for sample in samples if not sample.ok]

    return {
        "total_requests": len(samples),
        "success_count": len(successes),
        "failure_count": len(failures),
        "success_rate": round(len(successes) / len(samples), 4) if samples else 0.0,
        "wall_time_seconds": round(wall_time_s, 3),
        "throughput_rps": round(len(samples) / wall_time_s, 3) if wall_time_s > 0 else 0.0,
        "client_elapsed_ms": {
            "avg": round(statistics.mean(elapsed_values), 2) if elapsed_values else None,
            "min": round(min(elapsed_values), 2) if elapsed_values else None,
            "p50": round(percentile(elapsed_values, 0.50), 2) if elapsed_values else None,
            "p95": round(percentile(elapsed_values, 0.95), 2) if elapsed_values else None,
            "p99": round(percentile(elapsed_values, 0.99), 2) if elapsed_values else None,
            "max": round(max(elapsed_values), 2) if elapsed_values else None,
        },
        "server_latency_ms": {
            "avg": round(statistics.mean(server_values), 2) if server_values else None,
            "p50": round(percentile(server_values, 0.50), 2) if server_values else None,
            "p95": round(percentile(server_values, 0.95), 2) if server_values else None,
            "p99": round(percentile(server_values, 0.99), 2) if server_values else None,
            "max": round(max(server_values), 2) if server_values else None,
        },
        "tokens_used": {
            "avg": round(statistics.mean(token_values), 2) if token_values else None,
            "max": max(token_values) if token_values else None,
        },
        "by_role": build_role_summary(samples),
        "errors": [
            {
                "index": sample.index,
                "status_code": sample.status_code,
                "role_id": sample.role_id,
                "query": sample.query,
                "error": sample.error,
            }
            for sample in failures[:10]
        ],
    }


def build_role_summary(samples: list[RequestSample]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    grouped: dict[str, list[RequestSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.role_id, []).append(sample)

    for role_id, role_samples in grouped.items():
        elapsed_values = [sample.elapsed_ms for sample in role_samples]
        result[role_id] = {
            "count": len(role_samples),
            "success_rate": round(sum(1 for sample in role_samples if sample.ok) / len(role_samples), 4),
            "avg_elapsed_ms": round(statistics.mean(elapsed_values), 2),
            "p95_elapsed_ms": round(percentile(sorted(elapsed_values), 0.95), 2),
        }
    return result


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


async def main() -> None:
    args = parse_args()
    eval_items = load_eval_items(Path(args.eval_set))
    base_api_url = f"{args.base_url.rstrip('/')}{args.api_prefix}"
    chat_url = f"{base_api_url}/chat"

    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_keepalive_connections=max(args.concurrency, 1), max_connections=max(args.concurrency * 2, 2))
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        headers = {"Content-Type": "application/json"}
        auth_mode = resolve_auth_mode(args)
        login_user_id = ""

        if auth_mode == "dev-header":
            dev_user_id = args.dev_user_id or args.user_id
            if not dev_user_id:
                raise RuntimeError("dev-header mode requires --dev-user-id or --user-id.")
            user_id = args.user_id or dev_user_id
            headers["X-User-Id"] = dev_user_id
        elif auth_mode == "token":
            if not args.token:
                raise RuntimeError("token mode requires --token.")
            user_id = args.user_id
            if not user_id:
                raise RuntimeError("token mode requires --user-id because the script cannot decode your JWT locally.")
            headers["Authorization"] = f"Bearer {args.token}"
        else:
            try:
                token, login_user_id = await login_for_token(client, base_api_url, args.username, args.password)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 401:
                    raise SystemExit(
                        "Login failed with 401 Unauthorized.\n"
                        "Your current demo account is likely unavailable.\n"
                        "Use one of these instead:\n"
                        "1. --auth-mode dev-header --dev-user-id <your_user_id>\n"
                        "2. --auth-mode token --token <your_token> --user-id <your_user_id>\n"
                        "3. Pass a valid --username / --password"
                    ) from exc
                raise
            user_id = args.user_id or login_user_id
            headers["Authorization"] = f"Bearer {token}"

        if not user_id:
            raise RuntimeError("user_id is empty. Provide --user-id or use login/dev header mode.")

        fixed_session_id = uuid4().hex if args.fixed_session else ""
        plan = [
            {
                "index": index + 1,
                "item": random.choice(eval_items),
                "session_id": fixed_session_id or uuid4().hex,
            }
            for index in range(args.requests)
        ]

        semaphore = asyncio.Semaphore(args.concurrency)
        samples: list[RequestSample] = []

        async def worker(entry: dict[str, Any]) -> None:
            async with semaphore:
                sample = await send_one(
                    client=client,
                    chat_url=chat_url,
                    headers=headers,
                    sample_index=entry["index"],
                    item=entry["item"],
                    user_id=user_id,
                    top_k=args.top_k,
                    temperature=args.temperature,
                    stream=args.stream,
                    session_id=entry["session_id"],
                )
                samples.append(sample)
                status = "OK" if sample.ok else "FAIL"
                print(
                    f"[{status}] #{sample.index:03d} role={sample.role_id} "
                    f"status={sample.status_code} elapsed={sample.elapsed_ms:.0f}ms "
                    f"server={sample.server_latency_ms if sample.server_latency_ms is not None else '-'}"
                )

        started_at = time.perf_counter()
        await asyncio.gather(*(worker(entry) for entry in plan))
        wall_time_s = time.perf_counter() - started_at

    samples.sort(key=lambda item: item.index)
    summary = summarize(samples, wall_time_s)
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "base_api_url": base_api_url,
            "concurrency": args.concurrency,
            "requests": args.requests,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "stream": args.stream,
            "fixed_session": args.fixed_session,
            "eval_set": str(Path(args.eval_set)),
        },
        "summary": summary,
        "samples": [
            {
                "index": sample.index,
                "ok": sample.ok,
                "status_code": sample.status_code,
                "elapsed_ms": round(sample.elapsed_ms, 2),
                "server_latency_ms": sample.server_latency_ms,
                "tokens_used": sample.tokens_used,
                "role_id": sample.role_id,
                "query": sample.query,
                "error": sample.error,
            }
            for sample in samples
        ],
    }

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
