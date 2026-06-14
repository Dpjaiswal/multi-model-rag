from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryFilters(BaseModel):
    company_name: str | None = Field(default=None)
    document_type: str | None = Field(default=None)
    fiscal_year: int | None = Field(default=None)
    fiscal_quarter: str | None = Field(default=None)


class RetrievedChunk(BaseModel):
    page_content: str
    metadata: dict
    retrieval_score: float | None = None
    rerank_score: float | None = None


class PipelineState(BaseModel):
    question: str
    top_k: int
    filters: QueryFilters = Field(default_factory=QueryFilters)
    search_query: str = ""
    retrieved_docs: list[RetrievedChunk] = Field(default_factory=list)
    reranked_docs: list[RetrievedChunk] = Field(default_factory=list)
    answer: str = ""
    citations: list[str] = Field(default_factory=list)


class FilterExtractionResponse(BaseModel):
    company_name: str | None = Field(default=None, description="Company or issuer name mentioned in the user query.")
    document_type: str | None = Field(default=None, description="SEC filing type like 10-K, 10-Q, or 8-K.")
    fiscal_year: int | None = Field(default=None, description="Fiscal year mentioned in the query, if any.")
    fiscal_quarter: str | None = Field(default=None, description="Quarter like Q1, Q2, Q3, or Q4, if any.")
    rewritten_query: str | None = Field(default=None, description="A concise retrieval-ready search query.")


FusionMode = Literal["rrf", "dbsf"]


CopilotIntent = Literal[
    "comparison",
    "forecast",
    "trend",
    "risk",
    "report",
    "letter",
    "suggestion",
    "qa",
]


QueryMode = Literal["auto", "legacy", "copilot"]


class CopilotRouteResponse(BaseModel):
    intent: CopilotIntent = Field(description="Detected user intent for the query.")
    fallback: bool = Field(
        default=False,
        description="True when the query should bypass the seven tools and use the fallback strategy.",
    )
    fallback_reason: str | None = Field(
        default=None,
        description="Short reason why the query was routed to fallback, if any.",
    )
    matched_companies: list[str] = Field(
        default_factory=list,
        description="Company names selected from the available data folders, if any.",
    )
    rewritten_query: str | None = Field(
        default=None,
        description="A concise retrieval-ready query tailored for the detected intent.",
    )
