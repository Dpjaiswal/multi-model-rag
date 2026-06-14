from __future__ import annotations

from sentence_transformers import CrossEncoder

from rag_app.config import AppConfig
from rag_app.schemas import RetrievedChunk


_RERANKER_CACHE: dict[str, CrossEncoder] = {}
_RERANKER_FAILURES: set[str] = set()


def _get_reranker(config: AppConfig) -> CrossEncoder | None:
    model_name = config.reranker_model
    if model_name in _RERANKER_FAILURES:
        return None
    if model_name not in _RERANKER_CACHE:
        try:
            _RERANKER_CACHE[model_name] = CrossEncoder(model_name)
        except Exception as exc:
            print(f"Reranker unavailable, falling back to retrieval order. Reason: {exc}")
            _RERANKER_FAILURES.add(model_name)
            return None
    return _RERANKER_CACHE[model_name]


def rerank_results(query: str, documents: list[RetrievedChunk], top_k: int, config: AppConfig) -> list[RetrievedChunk]:
    if not documents:
        return []

    reranker = _get_reranker(config)
    if reranker is None:
        selected = documents[:top_k]
        if selected:
            print("Reranking skipped, using retrieval order:")
            for chunk in selected[: min(5, len(selected))]:
                score = chunk.retrieval_score if chunk.retrieval_score is not None else 0.0
                print(
                    f"- {chunk.metadata.get('document_name', 'unknown')} | "
                    f"{chunk.metadata.get('company_name', 'unknown')} | retrieval={score:.4f}"
                )
        return selected

    pairs = [(query, document.page_content) for document in documents]
    scores = reranker.predict(pairs)

    reranked = []
    for document, score in zip(documents, scores, strict=False):
        reranked.append(
            RetrievedChunk(
                page_content=document.page_content,
                metadata=document.metadata,
                retrieval_score=document.retrieval_score,
                rerank_score=float(score),
            )
        )

    reranked.sort(key=lambda item: item.rerank_score if item.rerank_score is not None else float("-inf"), reverse=True)
    selected = reranked[:top_k]

    if selected:
        print("Reranked sources:")
        for chunk in selected[: min(5, len(selected))]:
            print(
                f"- {chunk.metadata.get('document_name', 'unknown')} | "
                f"{chunk.metadata.get('company_name', 'unknown')} | rerank={chunk.rerank_score:.4f}"
            )
    return selected
