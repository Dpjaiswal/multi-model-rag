from __future__ import annotations

import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rag_app.api_schemas import HealthResponse, QueryRequest, QueryResponse
from rag_app.config import load_config
from rag_app.copilot import run_financial_copilot_with_mode


app = FastAPI(title="Financial RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Financial RAG API is running.",
        "docs": "/docs",
        "health": "/health",
        "query": "/application/query",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    config = load_config()
    return HealthResponse(
        status="ok",
        collection_name=config.collection_name,
        data_dir=str(config.data_dir),
        embedding_model=config.embedding_model,
        sparse_embedding_model=config.sparse_embedding_model,
        generation_model=config.llm_model,
    )


@app.post("/application/query", response_model=QueryResponse)
def application_query(request: QueryRequest) -> QueryResponse:
    config = load_config()
    try:
        config.validate_querying()
        result = run_financial_copilot_with_mode(config, request.question, request.top_k, request.mode)

        return QueryResponse(
            question=request.question,
            intent=result["intent"],
            answer=result["answer"],
            citations=result["citations"],
            extracted_filters=result["filters"].model_dump(),
            search_query=result["search_query"],
            suggestions=result.get("suggestions", []),
            matched_companies=result.get("matched_companies", []),
        )
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Application query failed.",
                "error": str(exc),
            },
        ) from exc
