import json
import time
from collections.abc import Callable
from urllib import error, request

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from provider_config import (
    get_embedding_dimension,
    get_embedding_model,
    get_google_api_base,
    get_google_api_key,
    get_google_embed_batch_size,
    get_google_embed_delay_seconds,
    get_google_embed_max_retries,
    get_embedding_provider,
    get_openai_compatible_client,
)

load_dotenv()

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

ProgressCallback = Callable[[float, str], None]


def _emit_progress(callback: ProgressCallback | None, progress: float, message: str) -> None:
    if callback is not None:
        callback(progress, message)


def load_and_chunk_pdf(path: str, progress_callback: ProgressCallback | None = None):
    _emit_progress(progress_callback, 0.0, "Loading PDF")
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    total_texts = max(len(texts), 1)
    for index, t in enumerate(texts):
        chunks.extend(splitter.split_text(t))
        _emit_progress(
            progress_callback,
            (index + 1) / total_texts,
            f"Chunking document content ({index + 1}/{total_texts})",
        )
    return chunks


def _google_role_to_task_type(role: str) -> str:
    if role == "query":
        return "RETRIEVAL_QUERY"
    return "RETRIEVAL_DOCUMENT"


def _google_role_to_embedding2_prefix(text: str, role: str) -> str:
    if role == "query":
        return f"task: search result | query: {text}"
    return f"title: none | text: {text}"


def _post_json(url: str, api_key: str, payload: dict, retries: int) -> dict:
    last_error_message = None
    for attempt in range(retries):
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace").strip()
            detail = body or exc.reason or "Unknown HTTP error"
            last_error_message = f"HTTP {exc.code}: {detail}"
            if exc.code != 429 or attempt == retries - 1:
                raise RuntimeError(f"Google embeddings request failed: {last_error_message}") from exc
            retry_after = exc.headers.get("Retry-After")
            sleep_seconds = float(retry_after) if retry_after else min(2 ** attempt, 8)
            time.sleep(sleep_seconds)

    if last_error_message is not None:
        raise RuntimeError(f"Google embeddings request failed: {last_error_message}")
    raise RuntimeError("Embedding request failed without returning an HTTP response")


def _normalize_vector(vector: list[float]) -> list[float]:
    magnitude = sum(value * value for value in vector) ** 0.5
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _extract_embeddings(response: dict) -> list[list[float]]:
    if "embeddings" in response:
        return [embedding["values"] for embedding in response["embeddings"]]
    if "embedding" in response:
        return [response["embedding"]["values"]]
    raise KeyError(f"Unexpected embedding response shape. Keys: {sorted(response.keys())}")


def _should_fallback_to_single_requests(exc: Exception) -> bool:
    message = str(exc)
    return "HTTP 400" in message or "HTTP 404" in message


def _google_embed_texts(
    texts: list[str],
    role: str,
    progress_callback: ProgressCallback | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    model = get_embedding_model()
    api_key = get_google_api_key()
    api_base = get_google_api_base().rstrip("/")
    output_dimensionality = get_embedding_dimension()
    batch_size = get_google_embed_batch_size()
    delay_seconds = get_google_embed_delay_seconds()
    retries = get_google_embed_max_retries()
    batches = _chunked(texts, batch_size)

    if model == "gemini-embedding-001":
        embeddings = []
        try:
            for index, batch in enumerate(batches):
                _emit_progress(
                    progress_callback,
                    index / len(batches),
                    f"Embedding chunks with Google ({index + 1}/{len(batches)})",
                )
                payload = {
                    "requests": [
                        {
                            "model": f"models/{model}",
                            "taskType": _google_role_to_task_type(role),
                            "content": {
                                "parts": [{"text": text}],
                            },
                            "output_dimensionality": output_dimensionality,
                        }
                        for text in batch
                    ]
                }
                response = _post_json(
                    f"{api_base}/models/{model}:batchEmbedContents",
                    api_key,
                    payload,
                    retries=retries,
                )
                embeddings.extend(_extract_embeddings(response))
                if delay_seconds > 0 and index < len(batches) - 1:
                    time.sleep(delay_seconds)
        except Exception as exc:
            if not _should_fallback_to_single_requests(exc):
                raise

            embeddings = []
            for index, batch in enumerate(batches):
                for text in batch:
                    _emit_progress(
                        progress_callback,
                        index / len(batches),
                        f"Embedding chunks with Google ({index + 1}/{len(batches)})",
                    )
                    payload = {
                        "taskType": _google_role_to_task_type(role),
                        "content": {
                            "parts": [{"text": text}],
                        },
                        "output_dimensionality": output_dimensionality,
                    }
                    response = _post_json(
                        f"{api_base}/models/{model}:embedContent",
                        api_key,
                        payload,
                        retries=retries,
                    )
                    embeddings.extend(_extract_embeddings(response))
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)

        if output_dimensionality != 3072:
            return [_normalize_vector(embedding) for embedding in embeddings]
        _emit_progress(progress_callback, 1.0, "Embedding complete")
        return embeddings

    if model == "gemini-embedding-2":
        embeddings = []
        for index, batch in enumerate(batches):
            _emit_progress(
                progress_callback,
                index / len(batches),
                f"Embedding chunks with Google ({index + 1}/{len(batches)})",
            )
            payload = {
                "requests": [
                    {
                        "model": f"models/{model}",
                        "content": {
                            "parts": [{"text": _google_role_to_embedding2_prefix(text, role)}],
                        },
                        "output_dimensionality": output_dimensionality,
                    }
                    for text in batch
                ]
            }
            response = _post_json(
                f"{api_base}/models/{model}:batchEmbedContents",
                api_key,
                payload,
                retries=retries,
            )
            embeddings.extend(_extract_embeddings(response))
            if delay_seconds > 0 and index < len(batches) - 1:
                time.sleep(delay_seconds)
        _emit_progress(progress_callback, 1.0, "Embedding complete")
        return embeddings

    raise ValueError(
        f"Unsupported Google embedding model '{model}'. "
        "Supported values: ['gemini-embedding-001', 'gemini-embedding-2']"
    )


def embed_texts(
    texts: list[str],
    role: str = "document",
    progress_callback: ProgressCallback | None = None,
) -> list[list[float]]:
    provider = get_embedding_provider()
    if provider == "google":
        return _google_embed_texts(texts, role, progress_callback=progress_callback)

    _emit_progress(progress_callback, 0.0, "Embedding chunks")
    client = get_openai_compatible_client(provider)
    response = client.embeddings.create(
        model=get_embedding_model(),
        input=texts,
        dimensions=get_embedding_dimension(),
    )
    _emit_progress(progress_callback, 1.0, "Embedding complete")
    return [item.embedding for item in response.data]
