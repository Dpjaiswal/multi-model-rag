from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class AppConfig:
    project_root: Path
    data_dir: Path
    chat_history_dir: Path
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    hybrid_fetch_k: int
    rerank_top_k: int
    upload_batch_size: int
    qdrant_timeout_seconds: int
    qdrant_upsert_retries: int
    ollama_base_url: str
    ollama_fetch_model: str
    embedding_model: str
    sparse_embedding_model: str
    dense_vector_name: str
    sparse_vector_name: str
    hybrid_fusion: str
    reranker_model: str
    enable_reranking: bool
    qdrant_url: str
    qdrant_api_key: str
    llm_model: str
    llm_api_key: str
    llm_base_url: str

    def validate_indexing(self) -> None:
        missing = []
        if not self.qdrant_url:
            missing.append("QDRANT_URL")
        if not self.qdrant_api_key:
            missing.append("QDRANT_API_KEY")
        if not self.collection_name:
            missing.append("COLLECTION_NAME or QDRANT_COLLECTION_NAME")
        if not self.embedding_model:
            missing.append("EMBEDDING_MODEL")
        if not self.sparse_embedding_model:
            missing.append("SPARSE_EMBEDDING_MODEL")
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"Missing required environment values for indexing: {names}")

    def validate_querying(self) -> None:
        self.validate_indexing()
        if not self.llm_api_key:
            raise ValueError("Missing required environment values for querying: GROQ_API_KEY or GROK_API_KEY")
        if self.enable_reranking and not self.reranker_model:
            raise ValueError("Missing required environment values for reranking: RERANKER_MODEL")


def _clean_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip().strip('"').strip("'")


def _clean_bool(name: str, default: bool) -> bool:
    value = _clean_env(name, "true" if default else "false").lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_data_dir(project_root: Path) -> Path:
    candidates = [
        Path(_clean_env("PDF_DATA_DIR")),
        Path(_clean_env("DATA_BASE_DIR")) / "pdfs" if _clean_env("DATA_BASE_DIR") else Path(),
        project_root / "Data",
    ]
    for candidate in candidates:
        if str(candidate).strip() and candidate.exists():
            return candidate.resolve()
    return (project_root / "Data").resolve()


def load_config() -> AppConfig:
    load_dotenv()
    project_root = Path(__file__).resolve().parent.parent
    llm_model = _clean_env("GROQ_MODEL") or _clean_env("GROK_MODEL", "llama-3.1-8b-instant")
    llm_api_key = _clean_env("GROQ_API_KEY") or _clean_env("GROK_API_KEY")
    llm_base_url = (
        _clean_env("GROQ_BASE_URL")
        or _clean_env("GROK_BASE_URL")
        or _clean_env("GROK_BASR_URL")
        or "https://api.groq.com/openai/v1"
    )
    collection_name = _clean_env("COLLECTION_NAME") or _clean_env("QDRANT_COLLECTION_NAME", "langgraph-rag")
    retrieval_k = int(_clean_env("RETRIEVAL_K", "5"))
    hybrid_fetch_k = int(_clean_env("HYBRID_FETCH_K", str(max(retrieval_k * 3, 12))))
    rerank_top_k = int(_clean_env("RERANK_TOP_K", str(retrieval_k)))

    return AppConfig(
        project_root=project_root,
        data_dir=_resolve_data_dir(project_root),
        chat_history_dir=Path(_clean_env("CHAT_HISTORY_DIR", str(project_root / ".chat_history"))).resolve(),
        collection_name=collection_name,
        chunk_size=int(_clean_env("CHUNK_SIZE", "1000")),
        chunk_overlap=int(_clean_env("CHUNK_OVERLAP", "200")),
        retrieval_k=retrieval_k,
        hybrid_fetch_k=hybrid_fetch_k,
        rerank_top_k=rerank_top_k,
        upload_batch_size=int(_clean_env("UPLOAD_BATCH_SIZE", "8")),
        qdrant_timeout_seconds=int(_clean_env("QDRANT_TIMEOUT_SECONDS", "180")),
        qdrant_upsert_retries=int(_clean_env("QDRANT_UPSERT_RETRIES", "3")),
        ollama_base_url=_clean_env("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_fetch_model=_clean_env("OLLAMA_FETCH_MODEL", "llama3.1"),
        embedding_model=_clean_env("EMBEDDING_MODEL", "nomic-embed-text"),
        sparse_embedding_model=_clean_env("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25"),
        dense_vector_name=_clean_env("QDRANT_DENSE_VECTOR_NAME", "dense"),
        sparse_vector_name=_clean_env("QDRANT_SPARSE_VECTOR_NAME", "sparse"),
        hybrid_fusion=_clean_env("QDRANT_HYBRID_FUSION", "rrf").lower(),
        reranker_model=_clean_env("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        enable_reranking=_clean_bool("ENABLE_RERANKING", False),
        qdrant_url=_clean_env("QDRANT_URL"),
        qdrant_api_key=_clean_env("QDRANT_API_KEY"),
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
    )
