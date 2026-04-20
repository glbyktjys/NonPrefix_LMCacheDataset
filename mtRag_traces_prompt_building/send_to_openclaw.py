"""
Send inline-RAG prompts to OpenClaw as multi-turn chat sessions.

Reads JSONL files (clapnq.jsonl, cloud.jsonl, fiqa.jsonl, govt.jsonl),
groups turns by session_id, and sends each session as a multi-turn
conversation via /v1/chat/completions.

Each turn sends only the current prompt (not the growing history).
OpenClaw session state is maintained via the x-openclaw-session-key header.
A new session key is generated per session_id; /new is sent at the start
of each session to reset OpenClaw's internal conversation.

Automatically calls the proxy's /session/start and /session/end between
sessions so each session gets its own trace file.

Usage:
  python send_to_openclaw.py clapnq.jsonl --base-url http://127.0.0.1:18789
  python send_to_openclaw.py clapnq.jsonl --token YOUR_TOKEN

  # Or all files:
  python send_to_openclaw.py clapnq.jsonl cloud.jsonl fiqa.jsonl govt.jsonl
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:18789"
DEFAULT_PROXY_URL = "http://localhost:18790"
DEFAULT_MODEL = "openclaw"


def proxy_session_start(proxy_url: str, name: str) -> None:
    """Tell the recording proxy to start a new named session."""
    print(f"  [proxy] Starting session: {name}")
    resp = requests.post(
        f"{proxy_url}/session/start",
        json={"name": name},
        timeout=10,
    )
    resp.raise_for_status()


def proxy_session_end(proxy_url: str) -> None:
    """Tell the recording proxy to end the current session."""
    print(f"  [proxy] Ending session")
    resp = requests.post(f"{proxy_url}/session/end", timeout=10)
    resp.raise_for_status()


def load_sessions(path: Path) -> list[dict]:
    """Load JSONL file and return list of session objects."""
    sessions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))
    return sessions


def make_session_key(session_id: str) -> str:
    """Derive a stable session key from a session_id."""
    return hashlib.sha256(f"inline-rag-{session_id}".encode()).hexdigest()[:32]


def send_turn(
    base_url: str,
    model: str,
    messages: list[dict],
    token: str | None = None,
    session_key: str | None = None,
    timeout: int = 300,
) -> dict:
    """Send a chat completion request and return the response."""
    payload = {
        "model": model,
        "messages": messages,
    }
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if session_key:
        headers["x-openclaw-session-key"] = session_key
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    if resp.status_code >= 400:
        print(f"\n      [HTTP {resp.status_code}] {resp.text[:500]}")
    resp.raise_for_status()
    return resp.json()


def reset_openclaw_session(
    base_url: str,
    model: str,
    token: str | None = None,
    session_key: str | None = None,
) -> None:
    """Send /new to reset OpenClaw's internal conversation for this session."""
    messages = [{"role": "user", "content": "/new"}]
    try:
        send_turn(base_url, model, messages, token=token, session_key=session_key, timeout=30)
    except Exception:
        pass  # /new may return an error or empty response; that's fine


def run_session(base_url: str, model: str, session: dict, token: str | None = None, delay: float = 1.0):
    """Run a single multi-turn session.

    Each turn sends only the current prompt as a single user message.
    OpenClaw maintains conversation state via x-openclaw-session-key.
    """
    session_id = session["session_id"]
    turns = sorted(session["turns"], key=lambda t: t["turn"])
    session_key = make_session_key(session_id)

    print(f"  Session: {session_id} ({len(turns)} turns, key={session_key[:8]}...)")

    # Reset OpenClaw session to start fresh
    reset_openclaw_session(base_url, model, token=token, session_key=session_key)

    for turn in turns:
        # Send only the current turn's prompt
        messages = [{"role": "user", "content": turn["prompt"]}]

        print(f"    Turn {turn['turn']}: sending prompt ({len(turn['prompt'])} chars)...", end=" ", flush=True)
        try:
            resp = send_turn(base_url, model, messages, token=token, session_key=session_key)
            assistant_content = resp["choices"][0]["message"]["content"]
            print(f"ok ({len(assistant_content)} chars)")
        except Exception as e:
            print(f"ERROR: {e}")

        if delay > 0:
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Send inline-RAG sessions to OpenClaw")
    parser.add_argument("files", nargs="+", help="JSONL files to send")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"OpenClaw base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--proxy-url", default=DEFAULT_PROXY_URL,
                        help=f"Recording proxy URL (default: {DEFAULT_PROXY_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay in seconds between turns (default: 1.0)")
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="Max sessions per file (default: all)")
    parser.add_argument("--token", default=None,
                        help="OpenClaw gateway Bearer token")
    args = parser.parse_args()

    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            continue

        sessions = load_sessions(path)
        if args.max_sessions:
            sessions = sessions[:args.max_sessions]

        print(f"\n{'='*60}")
        print(f"File: {path.name} — {len(sessions)} sessions")
        print(f"{'='*60}")

        for i, session in enumerate(sessions, 1):
            print(f"\n[{i}/{len(sessions)}]")
            session_name = f"{path.stem}_{session['session_id']}"
            proxy_session_start(args.proxy_url, session_name)
            try:
                run_session(args.base_url, args.model, session, token=args.token, delay=args.delay)
            finally:
                proxy_session_end(args.proxy_url)

    print("\nDone.")


if __name__ == "__main__":
    main()