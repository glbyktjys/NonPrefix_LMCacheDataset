"""
Instrumented reverse proxy for vLLM (or any OpenAI-compatible server).

Sits between external clients and the inference engine, forwarding all
requests transparently while optionally recording full request/response
traces to JSONL files when a session is active.

Operator endpoints (not forwarded to backend):
    POST   /session/start   {"name": "my_session"}
    POST   /session/end
    GET    /session/status

Everything else is reverse-proxied to the backend.
"""

import asyncio
import json
import logging
import sys
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8200")
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
BLUE = "\033[34m"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("proxy")


def _truncate(s: str, max_len: int = 120) -> str:
    return s if len(s) <= max_len else s[:max_len] + "..."


def _req_summary(parsed_body) -> str:
    """One-line summary of a request body."""
    if not isinstance(parsed_body, dict):
        return ""
    parts = []
    if "model" in parsed_body:
        parts.append(f"model={parsed_body['model']}")
    if "messages" in parsed_body:
        msgs = parsed_body["messages"]
        parts.append(f"{len(msgs)} msgs")
        if msgs:
            last = msgs[-1]
            role = last.get("role", "?")
            content = last.get("content", "")
            if isinstance(content, str):
                parts.append(f"last={role}: {_truncate(content, 60)}")
    if "stream" in parsed_body:
        parts.append(f"stream={parsed_body['stream']}")
    return " | ".join(parts)


def _resp_summary(parsed_body) -> str:
    """One-line summary of a response body."""
    if not isinstance(parsed_body, dict):
        return ""
    parts = []
    if "choices" in parsed_body:
        choices = parsed_body["choices"]
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message", choices[0].get("delta", {}))
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    parts.append(_truncate(content, 80))
    if "usage" in parsed_body and isinstance(parsed_body["usage"], dict):
        u = parsed_body["usage"]
        parts.append(f"tok={u.get('prompt_tokens','?')}/{u.get('completion_tokens','?')}")
    return " | ".join(parts)


def log_request(method: str, path: str, parsed_body, recording: bool):
    rec = f"{GREEN}[REC]{RESET} " if recording else ""
    summary = _req_summary(parsed_body)
    summary_str = f"  {DIM}{summary}{RESET}" if summary else ""
    log.info(f"{rec}{CYAN}>>>{RESET} {BOLD}{method} {path}{RESET}{summary_str}")


def log_response(status: int, path: str, parsed_body, elapsed_ms: float, streaming: bool):
    color = GREEN if 200 <= status < 300 else YELLOW if 300 <= status < 400 else RED
    stream_tag = f" {MAGENTA}[stream]{RESET}" if streaming else ""
    summary = _resp_summary(parsed_body)
    summary_str = f"  {DIM}{summary}{RESET}" if summary else ""
    log.info(f"{color}<<<{RESET} {status} {path} {DIM}({elapsed_ms:.0f}ms){RESET}{stream_tag}{summary_str}")


def log_session_event(action: str, name: str, extra: str = ""):
    log.info(f"{YELLOW}{'='*50}{RESET}")
    log.info(f"{YELLOW}[SESSION]{RESET} {BOLD}{action}{RESET}: {name}  {extra}")
    log.info(f"{YELLOW}{'='*50}{RESET}")


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
            } if self.active else None,
            "finished_sessions": self.finished_sessions,
            "saved_trace_files": [str(f) for f in saved_files],
        }


session = SessionState()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Instrumented Proxy")


# ---- Operator endpoints ---------------------------------------------------

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
        log_session_event("ENDED", result["session_name"],
                          f"{result['requests_recorded']} requests in {result['duration_s']}s")
    status_code = 200 if "status" in result else 409
    return JSONResponse(result, status_code=status_code)


@app.get("/session/status")
async def session_status():
    return JSONResponse(session.status())


# ---- Reverse proxy --------------------------------------------------------

async def _read_body(request: Request) -> bytes:
    return await request.body()


def _parse_json_body(raw: bytes) -> Optional[dict]:
    try:
        return json.loads(raw)
    except Exception:
        return None


def _assemble_streaming_response(events: list[dict]) -> dict:
    """Collapse SSE chunk events into a single response matching the
    non-streaming chat completion format."""
    if not events:
        return {}

    # Accumulate per-choice content and tool calls
    choices_acc: dict[int, dict] = {}
    model = None
    created = None
    resp_id = None
    usage = None

    for evt in events:
        model = model or evt.get("model")
        created = created or evt.get("created")
        resp_id = resp_id or evt.get("id")
        if "usage" in evt and evt["usage"]:
            usage = evt["usage"]

        for choice in evt.get("choices", []):
            idx = choice.get("index", 0)
            if idx not in choices_acc:
                choices_acc[idx] = {
                    "index": idx,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            acc = choices_acc[idx]
            delta = choice.get("delta", {})
            if delta.get("content"):
                acc["message"]["content"] += delta["content"]
            if delta.get("role"):
                acc["message"]["role"] = delta["role"]
            if delta.get("tool_calls"):
                tc_list = acc["message"].setdefault("tool_calls", [])
                for tc in delta["tool_calls"]:
                    tc_idx = tc.get("index", 0)
                    # Extend or merge
                    while len(tc_list) <= tc_idx:
                        tc_list.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    existing = tc_list[tc_idx]
                    if tc.get("id"):
                        existing["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        existing["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        existing["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                acc["finish_reason"] = choice["finish_reason"]

    result = {
        "id": resp_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [choices_acc[i] for i in sorted(choices_acc)],
    }
    if usage:
        result["usage"] = usage
    return result



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

    # Forward to backend — always use stream=True so we can handle both cases
    url = f"{BACKEND_URL}/{path}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}

    try:
        client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        backend_req = client.build_request(
            method=request.method,
            url=url,
            headers=headers,
            content=raw_body,
            params=request.query_params,
        )
        backend_resp = await client.send(backend_req, stream=True)
    except Exception as e:
        log.info(f"{RED}!!! Backend error: {e}{RESET}")
        await client.aclose()
        return JSONResponse({"error": f"Backend unreachable: {e}"}, status_code=502)

    is_stream = "text/event-stream" in backend_resp.headers.get("content-type", "")
    resp_headers = {k: v for k, v in backend_resp.headers.items()
                    if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")}

    if is_stream:
        async def streaming_generator():
            try:
                collected_chunks: list[str] = []
                async for chunk in backend_resp.aiter_bytes():
                    collected_chunks.append(chunk.decode("utf-8", errors="replace"))
                    yield chunk

                # Parse SSE events for logging and tracing
                full_text = "".join(collected_chunks)
                parsed_events = []
                for line in full_text.splitlines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        evt = _parse_json_body(line[6:].encode())
                        if evt:
                            parsed_events.append(evt)

                assembled = _assemble_streaming_response(parsed_events) if parsed_events else None

                if session.active:
                    response_record = {
                        "type": "response",
                        "request_id": request_id,
                        "timestamp_rel_s": round(time.time() - session.start_time, 6) if session.start_time else None,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "status_code": backend_resp.status_code,
                        "headers": dict(backend_resp.headers),
                        "body": assembled if assembled else full_text,
                    }
                    await session.write(response_record)

                # Terminal log for streaming response
                log_response(backend_resp.status_code, f"/{path}", assembled,
                             (time.time() - req_start) * 1000, streaming=True)
            finally:
                await backend_resp.aclose()
                await client.aclose()

        return StreamingResponse(
            streaming_generator(),
            status_code=backend_resp.status_code,
            headers=resp_headers,
            media_type=backend_resp.headers.get("content-type"),
        )

    # Non-streaming: read full body, close, then respond
    resp_body = await backend_resp.aread()
    await backend_resp.aclose()
    await client.aclose()

    parsed_resp = _parse_json_body(resp_body)
    if session.active:
        response_record = {
            "type": "response",
            "request_id": request_id,
            "timestamp_rel_s": round(time.time() - session.start_time, 6) if session.start_time else None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status_code": backend_resp.status_code,
            "headers": dict(backend_resp.headers),
            "body": parsed_resp if parsed_resp else resp_body.decode("utf-8", errors="replace"),
        }
        await session.write(response_record)

    log_response(backend_resp.status_code, f"/{path}", parsed_resp,
                 (time.time() - req_start) * 1000, streaming=False)

    return Response(
        content=resp_body,
        status_code=backend_resp.status_code,
        headers=resp_headers,
        media_type=backend_resp.headers.get("content-type"),
    )


if __name__ == "__main__":
    print(f"Proxying to backend: {BACKEND_URL}")
    print(f"Traces will be saved to: {TRACE_DIR.resolve()}")
    print(f"Operator endpoints: /session/start, /session/end, /session/status")
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
