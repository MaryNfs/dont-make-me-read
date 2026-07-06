import os
from functools import lru_cache

from openai import OpenAI

DEFAULT_OPENAI_LLM_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_OPENAI_EMBED_DIM = 3072
DEFAULT_GOOGLE_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GOOGLE_EMBED_BATCH_SIZE = 8
DEFAULT_GOOGLE_EMBED_DELAY_SECONDS = 1.0
DEFAULT_GOOGLE_EMBED_MAX_RETRIES = 8
DEFAULT_GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

SUPPORTED_LLM_PROVIDERS = {"openai", "groq"}
SUPPORTED_EMBEDDING_PROVIDERS = {"google", "openai"}


def _normalized_env(name: str, default: str) -> str:
    return os.getenv(name, default).strip().lower()


@lru_cache(maxsize=1)
def get_llm_provider() -> str:
    provider = _normalized_env("LLM_PROVIDER", os.getenv("MODEL_PROVIDER", "openai"))
    if provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider}'. Supported values: {sorted(SUPPORTED_LLM_PROVIDERS)}"
        )
    return provider


@lru_cache(maxsize=1)
def get_embedding_provider() -> str:
    provider = _normalized_env("EMBEDDING_PROVIDER", "openai")
    if provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        raise ValueError(
            "Unsupported EMBEDDING_PROVIDER "
            f"'{provider}'. Supported values: {sorted(SUPPORTED_EMBEDDING_PROVIDERS)}"
        )
    return provider


@lru_cache(maxsize=1)
def get_llm_model() -> str:
    override = os.getenv("LLM_MODEL")
    if override:
        return override
    if get_llm_provider() == "groq":
        return os.getenv("GROQ_LLM_MODEL", DEFAULT_GROQ_LLM_MODEL)
    return os.getenv("OPENAI_LLM_MODEL", DEFAULT_OPENAI_LLM_MODEL)


@lru_cache(maxsize=1)
def get_embedding_model() -> str:
    override = os.getenv("EMBEDDING_MODEL")
    if override:
        return override
    provider = get_embedding_provider()
    if provider == "openai":
        return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBED_MODEL)
    if provider == "google":
        return os.getenv(
            "GOOGLE_EMBEDDING_MODEL",
            os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_GOOGLE_EMBED_MODEL),
        )
    raise ValueError(f"Unsupported embedding provider '{provider}'")


@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    return int(os.getenv("EMBEDDING_DIM", str(DEFAULT_OPENAI_EMBED_DIM)))


def _require_api_key(env_name: str) -> str:
    api_key = os.getenv(env_name)
    if not api_key:
        raise RuntimeError(f"Missing required environment variable: {env_name}")
    return api_key


def get_google_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing required environment variable: GEMINI_API_KEY")
    return api_key


def get_google_api_base() -> str:
    return os.getenv("GOOGLE_API_BASE", DEFAULT_GOOGLE_API_BASE)


def get_google_embed_batch_size() -> int:
    return max(1, int(os.getenv("GOOGLE_EMBED_BATCH_SIZE", str(DEFAULT_GOOGLE_EMBED_BATCH_SIZE))))


def get_google_embed_delay_seconds() -> float:
    return max(0.0, float(os.getenv("GOOGLE_EMBED_DELAY_SECONDS", str(DEFAULT_GOOGLE_EMBED_DELAY_SECONDS))))


def get_google_embed_max_retries() -> int:
    return max(1, int(os.getenv("GOOGLE_EMBED_MAX_RETRIES", str(DEFAULT_GOOGLE_EMBED_MAX_RETRIES))))


@lru_cache(maxsize=2)
def get_openai_compatible_client(provider: str) -> OpenAI:
    if provider == "openai":
        return OpenAI(api_key=_require_api_key("OPENAI_API_KEY"))
    if provider == "groq":
        return OpenAI(
            api_key=_require_api_key("GROQ_API_KEY"),
            base_url=os.getenv("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL),
        )
    raise ValueError(f"Unsupported provider '{provider}'")
