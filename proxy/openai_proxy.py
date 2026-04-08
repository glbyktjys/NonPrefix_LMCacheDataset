"""
Instrumented reverse proxy for OpenAI API traffic.

Sits between a local client (for example Codex/OpenClaw) and OpenAI's API,
forwarding requests transparently while optionally recording full
request/response traces to JSONL files when a session is active.

Operator endpoints (not forwarded to backend):
    POST   /session/start   {"name": "my_session"}
    POST   /session/end
    GET    /session/status

Everything else is reverse-proxied to OpenAI.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "https://api.openai.com").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8201"))
TRACE_DIR = Path(os.environ.get("TRACE_DIR", "./traces"))
TRACE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Terminal logging
# ---------------------------------------------------------------------------
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RED = "\033[31m"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("openai_proxy")


def _truncate(text: str, max_len: int = 120) -> str:
    return text if len(text) <= max_len else text[:max_len] + "..."


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _deep_get(data: Any, path: tuple[str, ...]) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _extract_text(value: Any) -> Optional[str]:
    if isinstance(value, str) and value:
        return value

    if isinstance(value, dict):
        for key in ("text", "output_text", "input_text"):
            field = value.get(key)
            if isinstance(field, str) and field:
                return field
        for key in ("content", "message", "input"):
            nested = _extract_text(value.get(key))
            if nested:
                return nested
        return None

    if isinstance(value, list):
        for item in value:
            nested = _extract_text(item)
            if nested:
                return nested

    return None


def _summarize_message(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    role = message.get("role", "?")
    content = message.get("content", message.get("input"))
    text = _extract_text(content)
    if text:
        return f"last={role}: {_truncate(text, 60)}"
    return f"last={role}"


def _summarize_input(input_value: Any) -> str:
    if isinstance(input_value, str):
        return f"input={_truncate(input_value, 60)}"
    if isinstance(input_value, list):
        if input_value and isinstance(input_value[-1], dict):
            summary = _summarize_message(input_value[-1])
            count = f"{len(input_value)} inputs"
            return f"{count} | {summary}" if summary else count
        text = _extract_text(input_value)
        if text:
            return f"{len(input_value)} inputs | {_truncate(text, 60)}"
        return f"{len(input_value)} inputs"
    text = _extract_text(input_value)
    if text:
        return f"input={_truncate(text, 60)}"
    return ""


def _extract_response_text(parsed_body: Any) -> Optional[str]:
    if not isinstance(parsed_body, dict):
        return None

    choices = parsed_body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message", choices[0].get("delta", {}))
        text = _extract_text(message)
        if text:
            return text

    output = parsed_body.get("output")
    text = _extract_text(output)
    if text:
        return text

    response = parsed_body.get("response")
    if isinstance(response, dict):
        text = _extract_response_text(response)
        if text:
            return text

    return None


def _cache_header_hints(headers: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if any(token in key.lower() for token in ("cache", "prefix", "hit", "lmcache"))
    }


def _extract_cache_metrics(parsed_body: Any, headers: dict[str, str]) -> dict[str, Any]:
    usage = parsed_body.get("usage") if isinstance(parsed_body, dict) else None
    header_hints = _cache_header_hints(headers)

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    cached_tokens = None
    cache_creation_tokens = None
    source = None

    if isinstance(usage, dict):
        prompt_tokens = _safe_int(usage.get("input_tokens"))
        if prompt_tokens is None:
            prompt_tokens = _safe_int(usage.get("prompt_tokens"))

        completion_tokens = _safe_int(usage.get("output_tokens"))
        if completion_tokens is None:
            completion_tokens = _safe_int(usage.get("completion_tokens"))

        total_tokens = _safe_int(usage.get("total_tokens"))
        cached_tokens = _safe_int(usage.get("cached_tokens"))
        if cached_tokens is None:
            cached_tokens = _safe_int(_deep_get(usage, ("input_tokens_details", "cached_tokens")))
        if cached_tokens is None:
            cached_tokens = _safe_int(_deep_get(usage, ("prompt_tokens_details", "cached_tokens")))
        if cached_tokens is None:
            cached_tokens = _safe_int(usage.get("cache_read_input_tokens"))

        cache_creation_tokens = _safe_int(usage.get("cache_creation_input_tokens"))
        if cache_creation_tokens is None:
            cache_creation_tokens = _safe_int(
                _deep_get(usage, ("input_tokens_details", "cache_creation_tokens"))
            )
        if cache_creation_tokens is None:
            cache_creation_tokens = _safe_int(
                _deep_get(usage, ("prompt_tokens_details", "cache_creation_tokens"))
            )

        if cached_tokens is not None or cache_creation_tokens is not None:
            source = "usage"

    if cached_tokens is None:
        for key, value in header_hints.items():
            key_lower = key.lower()
            if "cached" in key_lower and "token" in key_lower:
                cached_tokens = _safe_int(value)
                if cached_tokens is not None:
                    source = "header"
                    break

    cache_hit_rate = None
    if prompt_tokens and cached_tokens is not None and prompt_tokens > 0:
        cache_hit_rate = round(cached_tokens / prompt_tokens, 6)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_hit_rate": cache_hit_rate,
        "source": source,
        "header_hints": header_hints,
    }


def _format_cache_summary(cache_metrics: dict[str, Any]) -> str:
    prompt_tokens = cache_metrics.get("prompt_tokens")
    cached_tokens = cache_metrics.get("cached_tokens")
    cache_hit_rate = cache_metrics.get("cache_hit_rate")
    if prompt_tokens is None and cached_tokens is None:
        return ""
    if prompt_tokens is not None and cached_tokens is not None:
        if cache_hit_rate is not None:
            return f"cache={cached_tokens}/{prompt_tokens} ({cache_hit_rate:.1%})"
        return f"cache={cached_tokens}/{prompt_tokens}"
    return f"cache={cached_tokens if cached_tokens is not None else '?'}"


def _req_summary(parsed_body: Any) -> str:
    """One-line summary of an OpenAI API request body."""
    if not isinstance(parsed_body, dict):
        return ""

    parts: list[str] = []
    if "model" in parsed_body:
        parts.append(f"model={parsed_body['model']}")

    if "messages" in parsed_body and isinstance(parsed_body["messages"], list):
        messages = parsed_body["messages"]
        parts.append(f"{len(messages)} msgs")
        if messages:
            summary = _summarize_message(messages[-1])
            if summary:
                parts.append(summary)
    elif "input" in parsed_body:
        summary = _summarize_input(parsed_body["input"])
        if summary:
            parts.append(summary)

    if isinstance(parsed_body.get("tools"), list):
        parts.append(f"tools={len(parsed_body['tools'])}")

    max_tokens = parsed_body.get("max_output_tokens", parsed_body.get("max_tokens"))
    if max_tokens is not None:
        parts.append(f"max_tok={max_tokens}")

    if "stream" in parsed_body:
        parts.append(f"stream={parsed_body['stream']}")

    return " | ".join(parts)


def _resp_summary(parsed_body: Any, headers: dict[str, str]) -> str:
    """One-line summary of an OpenAI API response body."""
    if not isinstance(parsed_body, dict):
        return ""

    parts: list[str] = []
    text = _extract_response_text(parsed_body)
    if text:
        parts.append(_truncate(text, 80))

    cache_metrics = _extract_cache_metrics(parsed_body, headers)
    prompt_tokens = cache_metrics.get("prompt_tokens")
    completion_tokens = cache_metrics.get("completion_tokens")
    if prompt_tokens is not None or completion_tokens is not None:
        parts.append(
            f"tok={prompt_tokens if prompt_tokens is not None else '?'}/"
            f"{completion_tokens if completion_tokens is not None else '?'}"
        )

    cache_summary = _format_cache_summary(cache_metrics)
    if cache_summary:
        parts.append(cache_summary)

    return " | ".join(parts)


def log_request(method: str, path: str, parsed_body: Any, recording: bool):
    rec = f"{GREEN}[REC]{RESET} " if recording else ""
    summary = _req_summary(parsed_body)
    summary_str = f"  {DIM}{summary}{RESET}" if summary else ""
    log.info(f"{rec}{CYAN}>>>{RESET} {BOLD}{method} {path}{RESET}{summary_str}")


def log_response(
    status: int,
    path: str,
    parsed_body: Any,
    headers: dict[str, str],
    elapsed_ms: float,
    streaming: bool,
):
    color = GREEN if 200 <= status < 300 else YELLOW if 300 <= status < 400 else RED
    stream_tag = f" {MAGENTA}[stream]{RESET}" if streaming else ""
    summary = _resp_summary(parsed_body, headers)
    summary_str = f"  {DIM}{summary}{RESET}" if summary else ""
    log.info(f"{color}<<<{RESET} {status} {path} {DIM}({elapsed_ms:.0f}ms){RESET}{stream_tag}{summary_str}")


def log_session_event(action: str, name: str, extra: str = ""):
    log.info(f"{YELLOW}{'=' * 50}{RESET}")
    log.info(f"{YELLOW}[SESSION]{RESET} {BOLD}{action}{RESET}: {name}  {extra}")
    log.info(f"{YELLOW}{'=' * 50}{RESET}")


def _backend_error_payload(url: str, exc: Exception) -> dict[str, Any]:
    return {
        "error": "OpenAI backend unreachable",
        "backend_url": url,
        "exception_type": type(exc).__name__,
        "exception_repr": repr(exc),
        "exception_text": str(exc),
    }


def _parse_sse_events(raw_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current_event: Optional[str] = None
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal current_event, data_lines
        if not data_lines and current_event is None:
            return

        raw_data = "\n".join(data_lines).strip()
        if raw_data and raw_data != "[DONE]":
            parsed = _parse_json_body(raw_data.encode("utf-8"))
            events.append(
                {
                    "event": current_event,
                    "data": parsed if parsed is not None else raw_data,
                }
            )

        current_event = None
        data_lines = []

    for line in raw_text.splitlines():
        if not line:
            flush()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    flush()
    return events


def _assemble_chat_stream(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse chat-completion chunk events into a single response."""
    if not payloads:
        return {}

    choices_acc: dict[int, dict[str, Any]] = {}
    model = None
    created = None
    response_id = None
    usage = None

    for payload in payloads:
        model = model or payload.get("model")
        created = created or payload.get("created")
        response_id = response_id or payload.get("id")
        if "usage" in payload and payload["usage"]:
            usage = payload["usage"]

        for choice in payload.get("choices", []):
            index = choice.get("index", 0)
            if index not in choices_acc:
                choices_acc[index] = {
                    "index": index,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            acc = choices_acc[index]
            delta = choice.get("delta", {})
            if delta.get("content"):
                acc["message"]["content"] += delta["content"]
            if delta.get("role"):
                acc["message"]["role"] = delta["role"]
            if delta.get("tool_calls"):
                tool_calls = acc["message"].setdefault("tool_calls", [])
                for tool_call in delta["tool_calls"]:
                    tool_index = tool_call.get("index", 0)
                    while len(tool_calls) <= tool_index:
                        tool_calls.append(
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        )
                    existing = tool_calls[tool_index]
                    if tool_call.get("id"):
                        existing["id"] = tool_call["id"]
                    function = tool_call.get("function", {})
                    if function.get("name"):
                        existing["function"]["name"] += function["name"]
                    if function.get("arguments"):
                        existing["function"]["arguments"] += function["arguments"]
            if choice.get("finish_reason"):
                acc["finish_reason"] = choice["finish_reason"]

    result = {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [choices_acc[index] for index in sorted(choices_acc)],
    }
    if usage:
        result["usage"] = usage
    return result


def _assemble_streaming_response(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Best-effort collapse of OpenAI streaming events into one JSON body."""
    payloads = [event["data"] for event in events if isinstance(event.get("data"), dict)]
    if not payloads:
        return {}

    if any("choices" in payload for payload in payloads):
        return _assemble_chat_stream(payloads)

    for payload in reversed(payloads):
        response = payload.get("response")
        if payload.get("type") == "response.completed" and isinstance(response, dict):
            return response

    for payload in reversed(payloads):
        response = payload.get("response")
        if isinstance(response, dict):
            return response

    return payloads[-1]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
class SessionState:
    def __init__(self):
        self.active: bool = False
        self.name: Optional[str] = None
        self.start_time: Optional[float] = None
        self.trace_file = None
        self.request_count: int = 0
        self.finished_sessions: list[dict] = []
        self._lock = asyncio.Lock()

    async def start(self, name: str) -> dict:
        async with self._lock:
            if self.active:
                return {"error": f"Session '{self.name}' is already active. End it first."}

            TRACE_DIR.mkdir(parents=True, exist_ok=True)

            self.active = True
            self.name = name
            self.start_time = time.time()
            self.request_count = 0
            path = TRACE_DIR / f"{name}_trace.jsonl"
            self.trace_file = open(path, "a")
            return {
                "status": "started",
                "session_name": name,
                "trace_file": str(path),
            }

    async def end(self) -> dict:
        async with self._lock:
            if not self.active:
                return {"error": "No active session."}
            info = {
                "session_name": self.name,
                "requests_recorded": self.request_count,
                "duration_s": round(time.time() - self.start_time, 3),
                "trace_file": self.trace_file.name,
            }
            self.trace_file.close()
            self.trace_file = None
            self.finished_sessions.append(info)
            self.active = False
            self.name = None
            self.start_time = None
            return {"status": "ended", **info}

    async def write(self, record: dict):
        async with self._lock:
            if self.active and self.trace_file:
                self.request_count += 1
                self.trace_file.write(json.dumps(record, default=str) + "\n")
                self.trace_file.flush()

    def status(self) -> dict:
        saved_files = sorted(TRACE_DIR.glob("*_trace.jsonl"))
        return {
            "active_session": {
                "name": self.name,
                "elapsed_s": round(time.time() - self.start_time, 3) if self.start_time else None,
                "requests_recorded": self.request_count,
            }
            if self.active
            else None,
            "finished_sessions": self.finished_sessions,
            "saved_trace_files": [str(file) for file in saved_files],
        }


session = SessionState()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="OpenAI API Proxy")


@app.post("/session/start")
async def session_start(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)
    name = body.get("name")
    if not name:
        return JSONResponse({"error": "Provide a 'name' field."}, status_code=400)
    result = await session.start(name)
    if "status" in result:
        log_session_event("STARTED", name, f"-> {TRACE_DIR / f'{name}_trace.jsonl'}")
    status_code = 200 if "status" in result else 409
    return JSONResponse(result, status_code=status_code)


@app.post("/session/end")
async def session_end():
    result = await session.end()
    if "status" in result:
        log_session_event(
            "ENDED",
            result["session_name"],
            f"{result['requests_recorded']} requests in {result['duration_s']}s",
        )
    status_code = 200 if "status" in result else 409
    return JSONResponse(result, status_code=status_code)


@app.get("/session/status")
async def session_status():
    return JSONResponse(session.status())


async def _read_body(request: Request) -> bytes:
    return await request.body()


def _parse_json_body(raw: bytes) -> Optional[dict]:
    try:
        return json.loads(raw)
    except Exception:
        return None


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy(request: Request, path: str):
    raw_body = await _read_body(request)
    parsed_body = _parse_json_body(raw_body)

    request_id = f"{time.time_ns()}"
    request_record = {
        "type": "request",
        "request_id": request_id,
        "timestamp_rel_s": round(time.time() - session.start_time, 6) if session.start_time else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": request.method,
        "path": f"/{path}",
        "query_params": dict(request.query_params),
        "headers": dict(request.headers),
        "body": parsed_body if parsed_body else raw_body.decode("utf-8", errors="replace") if raw_body else None,
    }

    log_request(request.method, f"/{path}", parsed_body, session.active)
    if session.active:
        await session.write(request_record)

    req_start = time.time()
    url = f"{OPENAI_API_URL}/{path}"

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in ("host", "content-length")
    }
    if OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
    elif "authorization" not in {key.lower() for key in headers}:
        forwarded_key = request.headers.get("x-api-key")
        if forwarded_key:
            headers["Authorization"] = f"Bearer {forwarded_key}"

    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
    try:
        backend_request = client.build_request(
            method=request.method,
            url=url,
            headers=headers,
            content=raw_body,
            params=request.query_params,
        )
        backend_response = await client.send(backend_request, stream=True)
    except Exception as exc:
        error_payload = _backend_error_payload(url, exc)
        log.info(
            f"{RED}!!! Backend error contacting {url}: "
            f"{error_payload['exception_type']} {error_payload['exception_repr']}{RESET}"
        )
        if session.active:
            response_record = {
                "type": "response",
                "request_id": request_id,
                "timestamp_rel_s": round(time.time() - session.start_time, 6) if session.start_time else None,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status_code": 502,
                "headers": {},
                "body": error_payload,
                "cache_metrics": {},
            }
            await session.write(response_record)
        await client.aclose()
        return JSONResponse(error_payload, status_code=502)

    is_stream = "text/event-stream" in backend_response.headers.get("content-type", "")
    response_headers = {
        key: value
        for key, value in backend_response.headers.items()
        if key.lower() not in ("content-length", "transfer-encoding", "content-encoding")
    }
    raw_backend_headers = dict(backend_response.headers)

    if is_stream:

        async def streaming_generator():
            try:
                collected_chunks: list[str] = []
                async for chunk in backend_response.aiter_bytes():
                    collected_chunks.append(chunk.decode("utf-8", errors="replace"))
                    yield chunk

                full_text = "".join(collected_chunks)
                parsed_events = _parse_sse_events(full_text)
                assembled = _assemble_streaming_response(parsed_events) if parsed_events else None
                cache_metrics = (
                    _extract_cache_metrics(assembled, raw_backend_headers)
                    if assembled
                    else _extract_cache_metrics(None, raw_backend_headers)
                )

                if session.active:
                    response_record = {
                        "type": "response",
                        "request_id": request_id,
                        "timestamp_rel_s": round(time.time() - session.start_time, 6)
                        if session.start_time
                        else None,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "status_code": backend_response.status_code,
                        "headers": raw_backend_headers,
                        "body": assembled if assembled else full_text,
                        "raw_events": parsed_events if parsed_events else None,
                        "cache_metrics": cache_metrics,
                    }
                    await session.write(response_record)

                log_response(
                    backend_response.status_code,
                    f"/{path}",
                    assembled,
                    raw_backend_headers,
                    (time.time() - req_start) * 1000,
                    streaming=True,
                )
            finally:
                await backend_response.aclose()
                await client.aclose()

        return StreamingResponse(
            streaming_generator(),
            status_code=backend_response.status_code,
            headers=response_headers,
            media_type=backend_response.headers.get("content-type"),
        )

    response_body = await backend_response.aread()
    await backend_response.aclose()
    await client.aclose()

    parsed_response = _parse_json_body(response_body)
    cache_metrics = _extract_cache_metrics(parsed_response, raw_backend_headers)

    if session.active:
        response_record = {
            "type": "response",
            "request_id": request_id,
            "timestamp_rel_s": round(time.time() - session.start_time, 6) if session.start_time else None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status_code": backend_response.status_code,
            "headers": raw_backend_headers,
            "body": parsed_response if parsed_response else response_body.decode("utf-8", errors="replace"),
            "cache_metrics": cache_metrics,
        }
        await session.write(response_record)

    log_response(
        backend_response.status_code,
        f"/{path}",
        parsed_response,
        raw_backend_headers,
        (time.time() - req_start) * 1000,
        streaming=False,
    )

    return Response(
        content=response_body,
        status_code=backend_response.status_code,
        headers=response_headers,
        media_type=backend_response.headers.get("content-type"),
    )


if __name__ == "__main__":
    print(f"Proxying local traffic to OpenAI API: {OPENAI_API_URL}")
    if OPENAI_API_KEY:
        print("OpenAI Authorization: Bearer token injection enabled")
    print(f"Listening on: http://localhost:{PROXY_PORT}")
    print(f"Traces will be saved to: {TRACE_DIR.resolve()}")
    print("Operator endpoints: /session/start, /session/end, /session/status")
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
