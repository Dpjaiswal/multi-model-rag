from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    collection_name: str
    data_dir: str
    embedding_model: str
    sparse_embedding_model: str
    generation_model: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User query to ask the RAG application.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of reranked chunks to use for answer generation.")
    mode: Literal["auto", "legacy", "copilot"] = Field(
        default="auto",
        description="auto uses legacy logic for QA and copilot logic for specialized finance intents.",
    )


class QueryResponse(BaseModel):
    question: str
    intent: str
    answer: str
    citations: list[str]
    extracted_filters: dict
    search_query: str
    suggestions: list[str] = Field(default_factory=list)
    matched_companies: list[str] = Field(default_factory=list)
