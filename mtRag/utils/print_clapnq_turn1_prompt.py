from pathlib import Path

from openclaw_session_runner import (
    _build_prompt,
    _load_corpus,
    _load_doc_mapping,
    _load_questions,
    _resolve_doc,
)


def main() -> int:
    root = Path(__file__).resolve().parent
    field = "clapnq"

    questions = _load_questions(root / f"{field}_lastturn.jsonl")
    mapping = _load_doc_mapping(root / f"{field}_dev.tsv")
    corpus = _load_corpus(root / f"{field}.jsonl.zip")

    first_question = questions[0]
    query_id = first_question["query_id"]
    mapped_doc_ids = mapping.get(query_id, [])
    docs = []
    for doc_id in mapped_doc_ids:
        doc = _resolve_doc(doc_id, corpus)
        if doc is not None:
            docs.append(doc)

    prompt = _build_prompt(
        field=field,
        query_id=query_id,
        question=first_question["question"],
        docs=docs,
    )

    print(f"query_id: {query_id}")
    print(f"question: {first_question['question']}")
    print(f"mapped_doc_ids: {mapped_doc_ids}")
    print(f"resolved_doc_ids: {[doc['_resolved_id'] for doc in docs]}")
    print()
    print(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
