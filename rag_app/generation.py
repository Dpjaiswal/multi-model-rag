from __future__ import annotations

from rag_app.config import AppConfig
from rag_app.schemas import RetrievedChunk

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def _build_generation_llm(config: AppConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        temperature=0,
        timeout=60,
        max_retries=1,
    )


def _page_reference(metadata: dict) -> str:
    page_label = metadata.get("page_label")
    if page_label not in (None, ""):
        return str(page_label)
    page_number = metadata.get("page")
    if isinstance(page_number, int):
        return str(page_number + 1)
    if page_number not in (None, ""):
        return str(page_number)
    return "unknown"


def _dedupe_documents(documents: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[tuple[str, str]] = set()
    unique_documents: list[RetrievedChunk] = []

    for document in documents:
        key = (
            str(document.metadata.get("document_name", "")),
            _page_reference(document.metadata),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_documents.append(document)
    return unique_documents


def _build_context_and_citations(documents: list[RetrievedChunk]) -> tuple[str, list[str]]:
    citations: list[str] = []
    context_blocks: list[str] = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        document_name = document.metadata.get("document_name", "unknown")
        company_name = document.metadata.get("company_name", "unknown")
        report_type = document.metadata.get("report_type", "unknown")
        report_year = document.metadata.get("report_year", "unknown")
        page_ref = _page_reference(document.metadata)

        citations.append(f"[{index}] {document_name} (page {page_ref})")
        context_blocks.append(
            f"[{index}] Source: {source}\n"
            f"Company: {company_name}\n"
            f"Document: {document_name}\n"
            f"Report type: {report_type}\n"
            f"Report year: {report_year}\n"
            f"Page: {page_ref}\n"
            f"Content:\n{document.page_content}"
        )

    context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."
    return context, citations


def _fallback_answer(query: str, documents: list[RetrievedChunk], citations: list[str], reason: str) -> str:
    if not documents:
        return (
            "Final Answer:\n"
            "The retrieved context is insufficient to answer the question.\n\n"
            "Reasoning:\n"
            f"- No relevant chunks were available for synthesis.\n"
            f"- LLM generation fallback was triggered because: {reason}.\n\n"
            "Outcome:\n"
            "insufficient_context"
        )

    top_lines: list[str] = []
    for index, document in enumerate(documents[:3], start=1):
        snippet = " ".join(document.page_content.split())[:280].strip()
        if snippet:
            top_lines.append(f"- Evidence from [{index}]: {snippet}...")

    reasoning_lines = [
        "- A fallback response was used because the hosted generation call did not complete successfully.",
        "- The answer below is limited to retrieved evidence and should be treated as a grounded summary of the top chunks.",
    ]

    final_answer = "\n".join(top_lines) if top_lines else "The retrieved context is insufficient to answer the question."
    reasoning = "\n".join(reasoning_lines)
    return (
        "Final Answer:\n"
        f"{final_answer}\n\n"
        "Reasoning:\n"
        f"{reasoning}\n"
        f"- Generation fallback reason: {reason}.\n\n"
        "Outcome:\n"
        "answered"
    )


def generate_response(
    query: str,
    documents: list[RetrievedChunk],
    config: AppConfig,
) -> tuple[str, list[str]]:
    unique_documents = _dedupe_documents(documents)
    context, citations = _build_context_and_citations(unique_documents)

    if not unique_documents:
        return _fallback_answer(query, unique_documents, citations, "No retrieved context."), citations

    llm = _build_generation_llm(config)
    prompt = [
        SystemMessage(
            content=(
                "You are a grounded financial RAG assistant. "
                "Answer only from the retrieved context. "
                "Do not use prior knowledge, world knowledge, or training cutoff assumptions. "
                "Never mention knowledge cutoff dates. "
                "If the context is insufficient, explicitly say that the retrieved context is insufficient. "
                "If the request is unsafe or disallowed, refusal is a successful safety outcome, not a failure. "
                "Do not apologize for refusing; state the boundary clearly and briefly. "
                "Use concise evidence-based reasoning only from the provided context. "
                "Use inline citations like [1], [2]."
            )
        ),
        HumanMessage(
            content=(
                f"User query: {query}\n\n"
                "Return the answer in this fixed structure exactly:\n"
                "Final Answer:\n"
                "<1 short paragraph or 2-4 bullets with direct answer and citations>\n\n"
                "Reasoning:\n"
                "<2-4 short bullets explaining how the retrieved context supports the answer, each with citations>\n\n"
                "Outcome:\n"
                "<one of: answered | insufficient_context | safe_refusal>\n\n"
                "Rules:\n"
                "- If context is insufficient, say so under Final Answer and set Outcome to insufficient_context.\n"
                "- If refusal is required, keep it short and set Outcome to safe_refusal.\n"
                "- Do not invent facts.\n"
                "- Every factual point must include citations.\n"
                "- Keep the response concise and professional.\n\n"
                f"Retrieved context:\n{context}"
            )
        ),
    ]

    try:
        print(f"Calling generation model: {config.llm_model}")
        response = llm.invoke(prompt)
        content = str(response.content).strip()
        if not content:
            raise ValueError("Generation model returned an empty response.")
        return content, citations
    except Exception as exc:
        print(f"Generation unavailable, using grounded fallback. Reason: {exc}")
        return _fallback_answer(query, unique_documents, citations, str(exc)), citations
