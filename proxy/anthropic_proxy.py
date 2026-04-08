"""
Instrumented reverse proxy for Anthropic API.

Sits between Claude Code and Anthropic's API, forwarding all
requests transparently while recording full request/response traces.

Operator endpoints (not forwarded to backend):
    POST   /session/start   {"name": "my_session"}
    POST   /session/end
    GET    /session/status

Everything else is reverse-proxied to Anthropic.
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
ANTHROPIC_API_URL = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com")
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
    """One-line summary of an Anthropic API request."""
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
            elif isinstance(content, list):
                # Content blocks (text, image, etc.)
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                if text_parts:
                    parts.append(f"last={role}: {_truncate(text_parts[0], 60)}")
    if "max_tokens" in parsed_body:
        parts.append(f"max_tok={parsed_body['max_tokens']}")
    if "stream" in parsed_body:
        parts.append(f"stream={parsed_body['stream']}")
    return " | ".join(parts)


def _resp_summary(parsed_body) -> str:
    """One-line summary of an Anthropic API response."""
    if not isinstance(parsed_body, dict):
        return ""
    parts = []
    
    # Anthropic response has "content" array with blocks
    if "content" in parsed_body:
        content_blocks = parsed_body["content"]
        if content_blocks and isinstance(content_blocks, list):
            text_parts = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
            if text_parts:
                parts.append(_truncate(text_parts[0], 80))
    
    # Usage information
    if "usage" in parsed_body and isinstance(parsed_body["usage"], dict):
        u = parsed_body["usage"]
        parts.append(f"tok={u.get('input_tokens','?')}/{u.get('output_tokens','?')}")
    
    # Stop reason
    if "stop_reason" in parsed_body:
        parts.append(f"stop={parsed_body['stop_reason']}")
    
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

    # async def start(self, name: str) -> dict:
    #     async with self._lock:
    #         if self.active:
    #             return {"error": f"Session '{self.name}' is already active. End it first."}
    #         self.active = True
    #         self.name = name
    #         self.start_time = time.time()
    #         self.request_count = 0
    #         path = TRACE_DIR / f"{name}_trace.jsonl"
    #         self.trace_file = open(path, "a")
    #         return {
    #             "status": "started",
    #             "session_name": name,
    #             "trace_file": str(path),
    #         }
    async def start(self, name: str) -> dict:
        async with self._lock:
            if self.active:
                return {"error": f"Session '{self.name}' is already active. End it first."}
            
            # Ensure directory exists
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
            } if self.active else None,
            "finished_sessions": self.finished_sessions,
            "saved_trace_files": [str(f) for f in saved_files],
        }


session = SessionState()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Anthropic API Proxy")


# ---- Operator endpoints ---------------------------------------------------

@app.post("/session/start")
async def session_start(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        print(str(e))
        return JSONResponse({"error": "Invalid JSON body!!!"+str(e)}, status_code=400)
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
    """Collapse Anthropic SSE events into a single response."""
    if not events:
        return {}

    # Anthropic streaming events have different structure
    content_blocks = []
    stop_reason = None
    model = None
    usage = None
    response_id = None

    for evt in events:
        evt_type = evt.get("type")
        
        if evt_type == "message_start":
            message = evt.get("message", {})
            model = message.get("model")
            response_id = message.get("id")
            usage = message.get("usage")
        
        elif evt_type == "content_block_start":
            block = evt.get("content_block", {})
            content_blocks.append(block)
        
        elif evt_type == "content_block_delta":
            delta = evt.get("delta", {})
            if delta.get("type") == "text_delta":
                # Append to last text block
                if content_blocks and content_blocks[-1].get("type") == "text":
                    content_blocks[-1]["text"] += delta.get("text", "")
        
        elif evt_type == "message_delta":
            delta = evt.get("delta", {})
            if "stop_reason" in delta:
                stop_reason = delta["stop_reason"]
            # Update usage if present
            if "usage" in evt:
                if usage:
                    usage.update(evt["usage"])
                else:
                    usage = evt["usage"]

    result = {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
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

    # Forward to Anthropic API
    url = f"{ANTHROPIC_API_URL}/{path}"
    
    # Extract and forward Anthropic-specific headers
    headers = {}
    for k, v in request.headers.items():
        k_lower = k.lower()
        if k_lower not in ("host", "content-length"):
            headers[k] = v
    
    # Ensure we have the API key header
    if "x-api-key" not in headers:
        # Check for Authorization header
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            headers["x-api-key"] = auth.replace("Bearer ", "")

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
        log.info(f"{RED}!!! Anthropic API error: {e}{RESET}")
        await client.aclose()
        return JSONResponse({"error": f"Anthropic API unreachable: {e}"}, status_code=502)

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

                # Parse Anthropic SSE events
                full_text = "".join(collected_chunks)
                parsed_events = []
                for line in full_text.splitlines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str and data_str != "[DONE]":
                            evt = _parse_json_body(data_str.encode())
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
                        "raw_events": parsed_events if parsed_events else None,
                    }
                    await session.write(response_record)

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

    # Non-streaming
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
    print(f"╔{'═'*78}╗")
    print(f"║ Anthropic API Proxy                                                       ║")
    print(f"║ Listening on: http://localhost:{PROXY_PORT}                                      ║")
    print(f"║ Forwarding to: {ANTHROPIC_API_URL:<55} ║")
    print(f"║ Traces: {str(TRACE_DIR.resolve()):<63} ║")
    print(f"╚{'═'*78}╝")
    print("\nOperator endpoints:")
    print("  POST /session/start  - Start recording (send {\"name\": \"session_name\"})")
    print("  POST /session/end    - Stop recording")
    print("  GET  /session/status - Check status\n")
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)