import uuid
from collections.abc import Callable
from pathlib import Path

from custom_types import RAQQueryResult
from data_loader import load_and_chunk_pdf, embed_texts
from provider_config import get_llm_model, get_llm_provider, get_openai_compatible_client
from vector_db import QdrantStorage


ProgressCallback = Callable[[float, str], None]


def _emit_progress(callback: ProgressCallback | None, progress: float, message: str) -> None:
    if callback is not None:
        callback(progress, message)


def ingest_pdf(
    pdf_path: str,
    source_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> int:
    source = source_id or pdf_path
    _emit_progress(progress_callback, 0.05, "Preparing ingestion")

    def on_chunk_progress(progress: float, message: str) -> None:
        _emit_progress(progress_callback, 0.10 + (progress * 0.35), message)

    chunks = load_and_chunk_pdf(pdf_path, progress_callback=on_chunk_progress)
    _emit_progress(progress_callback, 0.50, f"Created {len(chunks)} chunks")

    def on_embed_progress(progress: float, message: str) -> None:
        _emit_progress(progress_callback, 0.55 + (progress * 0.30), message)

    vecs = embed_texts(chunks, role="document", progress_callback=on_embed_progress)
    _emit_progress(progress_callback, 0.90, "Writing vectors to Qdrant")
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{i}")) for i in range(len(chunks))]
    payloads = [{"source": source, "text": chunks[i]} for i in range(len(chunks))]
    QdrantStorage().upsert(ids, vecs, payloads)
    _emit_progress(progress_callback, 1.0, f"Ingestion complete: {len(chunks)} chunks indexed")
    return len(chunks)


def search_contexts(question: str, top_k: int = 5) -> tuple[list[str], list[str]]:
    query_vec = embed_texts([question], role="query")[0]
    found = QdrantStorage().search(query_vec, top_k)
    return found["contexts"], found["sources"]


def _answer_style_instruction(answer_style: str) -> str:
    if answer_style == "concise":
        return "Answer briefly in a few sentences using only the provided context."
    if answer_style == "balanced":
        return (
            "Answer with moderate detail using only the provided context. "
            "Explain the main points clearly and include important nuances when relevant."
        )
    return (
        "Answer in detail using only the provided context. "
        "Provide a complete explanation, organize the answer clearly, and include relevant nuances and supporting points."
    )


def generate_answer(
    question: str,
    contexts: list[str],
    sources: list[str],
    answer_style: str = "detailed",
) -> RAQQueryResult:
    context_block = "\n\n".join(f"[Context {i + 1}] {c}" for i, c in enumerate(contexts))
    source_block = "\n".join(f"- {source}" for source in sources) if sources else "- Unknown source"
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Sources:\n{source_block}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        f"{_answer_style_instruction(answer_style)}\n"
        "When relevant, reference the source names in the answer."
    )

    client = get_openai_compatible_client(get_llm_provider())
    response = client.chat.completions.create(
        model=get_llm_model(),
        max_tokens=1024,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You answer questions using only the provided context. Prefer complete, well-structured answers over brief summaries."},
            {"role": "user", "content": user_content},
        ],
    )
    answer = (response.choices[0].message.content or "").strip()
    return RAQQueryResult(answer=answer, sources=sources, num_contexts=len(contexts))


def answer_question(question: str, top_k: int = 8, answer_style: str = "detailed") -> RAQQueryResult:
    contexts, sources = search_contexts(question, top_k)
    return generate_answer(question, contexts, sources, answer_style=answer_style)


def get_source_chunk_count(source_id: str) -> int:
    return QdrantStorage().count_by_source(source_id)


def list_uploaded_files_with_status(upload_dir: str = "uploads") -> list[dict]:
    uploads_path = Path(upload_dir)
    if not uploads_path.exists():
        return []

    store = QdrantStorage()
    rows = []
    for path in sorted(uploads_path.glob("*.pdf")):
        chunk_count = store.count_by_source(path.name)
        rows.append(
            {
                "file_name": path.name,
                "indexed": chunk_count > 0,
                "chunk_count": chunk_count,
                "file_size_kb": round(path.stat().st_size / 1024, 1),
            }
        )
    return rows


def get_total_indexed_chunks() -> int:
    return QdrantStorage().count_points()
