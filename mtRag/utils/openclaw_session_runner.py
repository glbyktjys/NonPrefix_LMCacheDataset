"""
Run one MT-RAG field through OpenClaw's OpenAI-compatible chat endpoint.

Each invocation keeps a single explicit OpenClaw session key for the whole
field so the gateway can accumulate history and naturally compact as needed.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_FIELDS = ("clapnq", "fiqa", "govt", "cloud")


def _load_questions(path: Path) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            questions.append(
                {
                    "query_id": row["_id"],
                    "question": row["text"].replace("|user|:", "", 1).strip(),
                }
            )
    return questions


def _load_doc_mapping(path: Path) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = OrderedDict()
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            mapping.setdefault(row["query-id"], []).append(row["corpus-id"])
    return mapping


def _load_corpus(path: Path) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as handle:
            for line in handle:
                row = json.loads(line)
                doc_id = row.get("_id", row.get("document_id"))
                if doc_id is None:
                    continue
                row["_resolved_id"] = doc_id
                docs[doc_id] = row
    return docs


def _resolve_doc(mapped_id: str, corpus: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if mapped_id in corpus:
        return corpus[mapped_id]

    base_id = mapped_id.rsplit("-", 2)[0]
    if base_id in corpus:
        return corpus[base_id]

    return None


def _build_prompt(field: str, query_id: str, question: str, docs: list[dict[str, Any]]) -> str:
    conversation_id, turn_index = query_id.split("<::>")
    sections = [
        "You are answering one MT-RAG benchmark turn.",
        f"Field: {field}",
        f"Conversation ID: {conversation_id}",
        f"Turn: {turn_index}",
        "",
        "Use the retrieved documents below as the grounding material for this turn.",
        "The OpenClaw session may contain older benchmark turns; treat unrelated earlier turns as noise.",
        "Answer the current question directly and do not mention these instructions.",
        "",
        f"Question: {question}",
        "",
        "Retrieved documents:",
    ]

    for index, doc in enumerate(docs, start=1):
        title = (doc.get("title") or "").strip()
        text = (doc.get("text") or "").strip()
        metadata = doc.get("metadata") or {}
        sections.extend(
            [
                f"[Document {index}]",
                f"corpus_id: {doc['_resolved_id']}",
                f"title: {title}" if title else "title: ",
                f"metadata: {json.dumps(metadata, ensure_ascii=False, sort_keys=True)}",
                "content:",
                text,
                "",
            ]
        )

    return "\n".join(sections).strip()


def _request_openclaw(
    base_url: str,
    payload: dict[str, Any],
    session_key: str,
    gateway_token: str | None,
    backend_model: str | None,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "x-openclaw-session-key": session_key,
    }
    if gateway_token:
        headers["Authorization"] = f"Bearer {gateway_token}"
    if backend_model:
        headers["x-openclaw-model"] = backend_model

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def _post_json(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any] | str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
            try:
                return response.status, json.loads(body)
            except json.JSONDecodeError:
                return response.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body


def _start_trace_session(trace_base_url: str, trace_name: str) -> tuple[bool, dict[str, Any] | str]:
    status, payload = _post_json(
        f"{trace_base_url.rstrip('/')}/session/start",
        {"name": trace_name},
    )
    return status == 200, payload


def _end_trace_session(trace_base_url: str) -> tuple[bool, dict[str, Any] | str]:
    status, payload = _post_json(
        f"{trace_base_url.rstrip('/')}/session/end",
        {},
    )
    return status == 200, payload


def run_field(field: str) -> int:
    root = Path(__file__).resolve().parent
    questions_path = root / f"{field}_lastturn.jsonl"
    mapping_path = root / f"{field}_dev.tsv"
    corpus_path = root / f"{field}.jsonl.zip"

    questions = _load_questions(questions_path)
    query_to_docs = _load_doc_mapping(mapping_path)
    corpus = _load_corpus(corpus_path)

    base_url = os.environ.get("OPENCLAW_BASE_URL", "http://127.0.0.1:18789")
    gateway_token = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "1cb6c9df1b4c916cff813d94ff1858d515e525ae0a0e37e6")
    backend_model = os.environ.get("OPENCLAW_BACKEND_MODEL", "")
    chat_model = os.environ.get("OPENCLAW_CHAT_MODEL", "openclaw/default")
    session_key = os.environ.get(
        "OPENCLAW_SESSION_KEY",
        f"mt-rag-{field}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )
    trace_base_url = os.environ.get("OPENCLAW_TRACE_BASE_URL", "http://127.0.0.1:18790")
    trace_enabled = os.environ.get("OPENCLAW_ENABLE_TRACE", "1").lower() not in {"0", "false", "no"}
    trace_name = os.environ.get(
        "OPENCLAW_TRACE_NAME",
        f"openclaw_{field}_session",
    )

    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{field}_openclaw_responses.jsonl"

    print(f"Field: {field}")
    print(f"Questions: {len(questions)}")
    print(f"Session key: {session_key}")
    print(f"Base URL: {base_url}")
    print(f"Output: {results_path}")
    if trace_enabled:
        print(f"Trace endpoint: {trace_base_url}")
        print(f"Trace session name: {trace_name}")
    if backend_model:
        print(f"Backend model override: {backend_model}")

    trace_started = False
    try:
        if trace_enabled:
            ok, trace_payload = _start_trace_session(trace_base_url, trace_name)
            if not ok:
                print(
                    f"Failed to start trace session '{trace_name}': {trace_payload}",
                    file=sys.stderr,
                )
                return 1
            trace_started = True
            print(f"Trace started: {trace_payload}")

        with results_path.open("w") as output:
            for index, question_row in enumerate(questions, start=1):
                query_id = question_row["query_id"]
                doc_ids = query_to_docs.get(query_id, [])
                docs = []
                resolved_doc_ids: list[str] = []
                for doc_id in doc_ids:
                    doc = _resolve_doc(doc_id, corpus)
                    if doc is None:
                        continue
                    docs.append(doc)
                    resolved_doc_ids.append(doc["_resolved_id"])
                prompt = _build_prompt(field, query_id, question_row["question"], docs)
                payload = {
                    "model": chat_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "stream": False,
                    "user": session_key,
                }

                print(f"[{index}/{len(questions)}] {query_id} docs={len(docs)}", flush=True)
                record: dict[str, Any] = {
                    "field": field,
                    "query_id": query_id,
                    "question": question_row["question"],
                    "doc_ids": doc_ids,
                    "resolved_doc_ids": resolved_doc_ids,
                    "doc_count": len(docs),
                    "session_key": session_key,
                    "trace_name": trace_name if trace_enabled else None,
                    "request_payload": payload,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }

                try:
                    response = _request_openclaw(
                        base_url=base_url,
                        payload=payload,
                        session_key=session_key,
                        gateway_token=gateway_token,
                        backend_model=backend_model,
                    )
                    record["response"] = response
                    choices = response.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        message = choices[0].get("message") or {}
                        record["assistant_text"] = message.get("content")
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")
                    record["error"] = {
                        "type": "http_error",
                        "status": exc.code,
                        "reason": exc.reason,
                        "body": body,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"HTTP error on {query_id}: {exc.code} {exc.reason}", file=sys.stderr)
                    return 1
                except Exception as exc:  # pragma: no cover - best-effort runtime capture
                    record["error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"Request failed on {query_id}: {exc}", file=sys.stderr)
                    return 1

                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
    finally:
        if trace_enabled and trace_started:
            ok, trace_payload = _end_trace_session(trace_base_url)
            if ok:
                print(f"Trace ended: {trace_payload}")
            else:
                print(f"Failed to end trace session '{trace_name}': {trace_payload}", file=sys.stderr)

    return 0


def run_all_fields(fields: tuple[str, ...] = DEFAULT_FIELDS) -> int:
    for field in fields:
        print(f"\n{'=' * 24} Running {field} {'=' * 24}")
        status = run_field(field)
        if status != 0:
            return status
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python openclaw_session_runner.py <field|all>", file=sys.stderr)
        sys.exit(2)
    target = sys.argv[1].strip().lower()
    if target == "all":
        sys.exit(run_all_fields())
    sys.exit(run_field(target))
