from __future__ import annotations

import time
from uuid import uuid4

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from rag_app.config import AppConfig
from rag_app.documents import load_pdf_documents


PAYLOAD_INDEXES = (
    ("metadata.company_name", models.PayloadSchemaType.KEYWORD),
    ("metadata.document_name", models.PayloadSchemaType.KEYWORD),
    ("metadata.report_year", models.PayloadSchemaType.INTEGER),
    ("metadata.report_quarter", models.PayloadSchemaType.KEYWORD),
    ("metadata.report_type", models.PayloadSchemaType.KEYWORD),
    ("metadata.age_bucket", models.PayloadSchemaType.KEYWORD),
)


def build_dense_embeddings(config: AppConfig) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=config.embedding_model,
        base_url=config.ollama_base_url,
    )


def build_sparse_embeddings(config: AppConfig) -> FastEmbedSparse:
    return FastEmbedSparse(model_name=config.sparse_embedding_model)


def build_qdrant_client(config: AppConfig) -> QdrantClient:
    return QdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        timeout=config.qdrant_timeout_seconds,
        check_compatibility=False,
    )


def build_vectorstore(config: AppConfig) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=build_qdrant_client(config),
        collection_name=config.collection_name,
        embedding=build_dense_embeddings(config),
        sparse_embedding=build_sparse_embeddings(config),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=config.dense_vector_name,
        sparse_vector_name=config.sparse_vector_name,
        validate_collection_config=False,
    )


def build_dense_fallback_vectorstore(config: AppConfig) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=build_qdrant_client(config),
        collection_name=config.collection_name,
        embedding=build_dense_embeddings(config),
        retrieval_mode=RetrievalMode.DENSE,
        vector_name="",
        validate_collection_config=False,
    )


def build_sparse_fallback_vectorstore(config: AppConfig) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=build_qdrant_client(config),
        collection_name=config.collection_name,
        sparse_embedding=build_sparse_embeddings(config),
        retrieval_mode=RetrievalMode.SPARSE,
        sparse_vector_name=config.sparse_vector_name,
        validate_collection_config=False,
    )


def _create_payload_indexes(client: QdrantClient, config: AppConfig) -> None:
    for field_name, field_schema in PAYLOAD_INDEXES:
        client.create_payload_index(
            collection_name=config.collection_name,
            field_name=field_name,
            field_schema=field_schema,
            timeout=config.qdrant_timeout_seconds,
        )
        print(f"Created payload index: {field_name} ({field_schema})")


def _recreate_collection(client: QdrantClient, config: AppConfig, vector_size: int) -> None:
    print(f"Recreating hybrid Qdrant collection '{config.collection_name}' with dense vector size {vector_size}...")
    client.recreate_collection(
        collection_name=config.collection_name,
        vectors_config={
            config.dense_vector_name: models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            config.sparse_vector_name: models.SparseVectorParams()
        },
        timeout=config.qdrant_timeout_seconds,
    )
    _create_payload_indexes(client, config)


def _upsert_batch_with_retry(
    vectorstore: QdrantVectorStore,
    batch,
    batch_ids,
    config: AppConfig,
    start: int,
    total_documents: int,
) -> None:
    last_error = None
    for attempt in range(1, config.qdrant_upsert_retries + 1):
        try:
            vectorstore.add_documents(documents=batch, ids=batch_ids)
            end = start + len(batch)
            print(f"Stored chunks {start + 1}-{end} / {total_documents}")
            return
        except ResponseHandlingException as exc:
            last_error = exc
            print(
                f"Qdrant upsert timeout on batch starting {start + 1}. "
                f"Retry {attempt}/{config.qdrant_upsert_retries}..."
            )
            time.sleep(min(attempt * 2, 10))
    raise last_error


def index_documents(config: AppConfig) -> int:
    documents = load_pdf_documents(config)
    client = build_qdrant_client(config)
    dense_embeddings = build_dense_embeddings(config)

    probe_vector = dense_embeddings.embed_query("hybrid vector size probe")
    _recreate_collection(client, config, len(probe_vector))

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=config.collection_name,
        embedding=dense_embeddings,
        sparse_embedding=build_sparse_embeddings(config),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=config.dense_vector_name,
        sparse_vector_name=config.sparse_vector_name,
        validate_collection_config=False,
    )

    total_documents = len(documents)
    print(
        f"Uploading {total_documents} chunks to Qdrant in batches of {config.upload_batch_size} "
        f"with timeout {config.qdrant_timeout_seconds}s..."
    )

    for start in range(0, total_documents, config.upload_batch_size):
        batch = documents[start:start + config.upload_batch_size]
        batch_ids = [str(uuid4()) for _ in batch]
        _upsert_batch_with_retry(vectorstore, batch, batch_ids, config, start, total_documents)

    client.close()
    return total_documents
