from __future__ import annotations

import re
from typing import cast

from langchain_openai import ChatOpenAI
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_app.config import AppConfig
from rag_app.schemas import FilterExtractionResponse, QueryFilters, RetrievedChunk
from rag_app.vectorstore import build_dense_fallback_vectorstore, build_sparse_fallback_vectorstore, build_vectorstore


YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
QUARTER_PATTERN = re.compile(r"\b(Q[1-4])\b", re.IGNORECASE)
DOCUMENT_PATTERN = re.compile(r"\b(10-K|10-Q|8-K)\b", re.IGNORECASE)
STOPWORDS = {
    "what", "did", "report", "reports", "in", "for", "the", "a", "an", "of", "and", "to", "about",
    "from", "on", "show", "tell", "me", "was", "were", "is", "are", "reported", "summarize",
    "company", "document", "documents", "financial", "results", "performance", "annual", "quarterly",
    "revenue", "operating", "income", "summary", "overview", "explain", "describe", "write", "email",
    "draft", "concise", "summarizing", "summarize", "team", "finance",
}


def _build_structured_llm(config: AppConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        temperature=0,
    )


def _available_companies(config: AppConfig) -> list[str]:
    if not config.data_dir.exists():
        return []
    return sorted(
        {
            path.name.lower()
            for path in config.data_dir.iterdir()
            if path.is_dir()
        }
    )


def _fallback_company_candidate(query: str, config: AppConfig) -> str | None:
    lowered = query.lower()
    for company in _available_companies(config):
        if re.search(rf"\b{re.escape(company)}\b", lowered):
            return company

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9&.-]*", query)
    candidates = []
    for token in tokens:
        lowered = token.lower()
        if lowered in STOPWORDS:
            continue
        if YEAR_PATTERN.fullmatch(lowered):
            continue
        if DOCUMENT_PATTERN.fullmatch(token):
            continue
        if len(lowered) < 3:
            continue
        candidates.append(lowered)
    return candidates[0] if candidates else None


def _infer_document_type(query: str) -> str | None:
    explicit_match = DOCUMENT_PATTERN.search(query)
    if explicit_match:
        return explicit_match.group(1).upper()

    lowered = query.lower()
    if "annual report" in lowered or "annual" in lowered or "10-k" in lowered:
        return "10-K"
    if "quarterly report" in lowered or "quarterly" in lowered or "10-q" in lowered:
        return "10-Q"
    if "8-k" in lowered or "current report" in lowered:
        return "8-K"
    return None


def _fallback_filters(query: str, config: AppConfig) -> FilterExtractionResponse:
    year_match = YEAR_PATTERN.search(query)
    quarter_match = QUARTER_PATTERN.search(query)
    document_type = _infer_document_type(query)
    company = _fallback_company_candidate(query, config)

    rewritten_parts = [query]
    if company:
        rewritten_parts.append(company)
    if year_match:
        rewritten_parts.append(year_match.group(1))
    if document_type:
        rewritten_parts.append(document_type)
    if company and year_match and document_type:
        rewritten_parts.append(f"{company} {document_type} {year_match.group(1)}")

    return FilterExtractionResponse(
        company_name=company,
        document_type=document_type,
        fiscal_year=int(year_match.group(1)) if year_match else None,
        fiscal_quarter=quarter_match.group(1).upper() if quarter_match else None,
        rewritten_query=" | ".join(dict.fromkeys(part for part in rewritten_parts if part)),
    )


def extract_filters(query: str, config: AppConfig) -> tuple[QueryFilters, str]:
    structured_llm = _build_structured_llm(config).with_structured_output(FilterExtractionResponse)
    try:
        response = cast(
            FilterExtractionResponse,
            structured_llm.invoke(
                (
                    "Extract structured retrieval filters from the user query. "
                    "Populate company_name, document_type, fiscal_year, and fiscal_quarter only when clearly present. "
                    "Also provide a concise rewritten_query optimized for hybrid retrieval over financial filings."
                    f"\n\nUser query: {query}"
                )
            ),
        )
    except Exception as exc:
        print(f"Structured filter extraction unavailable, using heuristic fallback. Reason: {exc}")
        response = _fallback_filters(query, config)

    filters = QueryFilters(
        company_name=response.company_name.lower() if response.company_name else None,
        document_type=response.document_type.upper() if response.document_type else None,
        fiscal_year=response.fiscal_year,
        fiscal_quarter=response.fiscal_quarter.upper() if response.fiscal_quarter else None,
    )
    rewritten_query = response.rewritten_query.strip() if response.rewritten_query else query
    return filters, rewritten_query


def _build_metadata_filter(filters: QueryFilters) -> models.Filter | None:
    conditions: list[models.FieldCondition] = []

    if filters.company_name:
        conditions.append(
            models.FieldCondition(
                key="metadata.company_name",
                match=models.MatchValue(value=filters.company_name),
            )
        )
    if filters.document_type:
        conditions.append(
            models.FieldCondition(
                key="metadata.report_type",
                match=models.MatchValue(value=filters.document_type),
            )
        )
    if filters.fiscal_year:
        conditions.append(
            models.FieldCondition(
                key="metadata.report_year",
                match=models.MatchValue(value=filters.fiscal_year),
            )
        )
    if filters.fiscal_quarter:
        conditions.append(
            models.FieldCondition(
                key="metadata.report_quarter",
                match=models.MatchValue(value=filters.fiscal_quarter),
            )
        )

    if not conditions:
        return None
    return models.Filter(must=conditions)


def _to_retrieved_chunks(results) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            page_content=doc.page_content,
            metadata=doc.metadata,
            retrieval_score=score,
        )
        for doc, score in results
    ]


def _run_dense_fallback(query: str, top_k: int, config: AppConfig, metadata_filter: models.Filter | None) -> list[RetrievedChunk]:
    dense_store = build_dense_fallback_vectorstore(config)
    print("Hybrid collection layout not available yet, falling back to dense-only retrieval on the existing collection.")
    try:
        if metadata_filter is not None:
            results = dense_store.similarity_search_with_score(query=query, k=top_k, filter=metadata_filter)
        else:
            results = dense_store.similarity_search_with_score(query=query, k=top_k)
    except UnexpectedResponse:
        results = dense_store.similarity_search_with_score(query=query, k=top_k)
    return _to_retrieved_chunks(results)


def _run_sparse_fallback(query: str, top_k: int, config: AppConfig, metadata_filter: models.Filter | None) -> list[RetrievedChunk]:
    sparse_store = build_sparse_fallback_vectorstore(config)
    print("Dense query embedding unavailable, falling back to sparse-only retrieval.")
    try:
        if metadata_filter is not None:
            results = sparse_store.similarity_search_with_score(query=query, k=top_k, filter=metadata_filter)
        else:
            results = sparse_store.similarity_search_with_score(query=query, k=top_k)
    except UnexpectedResponse:
        results = sparse_store.similarity_search_with_score(query=query, k=top_k)
    return _to_retrieved_chunks(results)


def hybrid_search(query: str, top_k: int, config: AppConfig, filters: QueryFilters) -> list[RetrievedChunk]:
    vectorstore = build_vectorstore(config)
    metadata_filter = _build_metadata_filter(filters)
    fusion_mode = models.FusionQuery(fusion=models.Fusion(config.hybrid_fusion))

    try:
        if metadata_filter is not None:
            print(f"Applying metadata filter for retrieval: {metadata_filter}")
            results = vectorstore.similarity_search_with_score(
                query=query,
                k=max(top_k, config.hybrid_fetch_k),
                filter=metadata_filter,
                hybrid_fusion=fusion_mode,
            )
        else:
            results = vectorstore.similarity_search_with_score(
                query=query,
                k=max(top_k, config.hybrid_fetch_k),
                hybrid_fusion=fusion_mode,
            )
        retrieved = _to_retrieved_chunks(results)
    except UnexpectedResponse as exc:
        error_text = str(exc)
        if "Not existing vector name error" in error_text or "sparse" in error_text.lower():
            retrieved = _run_dense_fallback(query, top_k, config, metadata_filter)
        else:
            print(f"Metadata-filtered hybrid retrieval failed, retrying without filter. Reason: {exc}")
            results = vectorstore.similarity_search_with_score(
                query=query,
                k=max(top_k, config.hybrid_fetch_k),
                hybrid_fusion=fusion_mode,
            )
            retrieved = _to_retrieved_chunks(results)
    except Exception as exc:
        error_text = str(exc).lower()
        if "memory layout cannot be allocated" in error_text or "ollama" in error_text or "embed" in error_text:
            retrieved = _run_sparse_fallback(query, top_k, config, metadata_filter)
        else:
            raise

    if retrieved:
        print("Retrieved sources:")
        for chunk in retrieved[: min(5, len(retrieved))]:
            score_text = f"{chunk.rerank_score:.4f}" if chunk.rerank_score is not None else f"{(chunk.retrieval_score or 0):.4f}"
            print(
                f"- {chunk.metadata.get('document_name', 'unknown')} | "
                f"{chunk.metadata.get('company_name', 'unknown')} | score={score_text}"
            )
    return retrieved
