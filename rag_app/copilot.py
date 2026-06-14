from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rag_app.config import AppConfig
from rag_app.generation import generate_response
from rag_app.documents import find_latest_filing_metadata
from rag_app.retrieval import extract_filters, hybrid_search
from rag_app.reranking import rerank_results
from rag_app.schemas import CopilotIntent, CopilotRouteResponse, QueryFilters, QueryMode, RetrievedChunk


def _build_llm(config: AppConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        temperature=0,
        timeout=60,
        max_retries=1,
    )


def _available_companies(config: AppConfig) -> list[str]:
    if not config.data_dir.exists():
        return []
    companies = [path.name.lower() for path in config.data_dir.iterdir() if path.is_dir()]
    return sorted(dict.fromkeys(companies))


def _build_router_llm(config: AppConfig) -> ChatOpenAI:
    return _build_llm(config)


def _available_company_text(config: AppConfig) -> str:
    companies = _available_companies(config)
    return ", ".join(companies) if companies else "none"


def _normalize_company_names(companies: list[str], config: AppConfig) -> list[str]:
    available = set(_available_companies(config))
    normalized: list[str] = []
    for company in companies:
        lowered = company.lower().strip()
        if lowered in available and lowered not in normalized:
            normalized.append(lowered)
    return normalized


def _prefer_latest_filing(filters: QueryFilters, config: AppConfig, intent: CopilotIntent) -> QueryFilters:
    if not filters.company_name:
        return filters

    if filters.fiscal_year or filters.fiscal_quarter:
        return filters

    if intent not in {"letter", "report", "trend", "risk"}:
        return filters

    report_type = filters.document_type or "10-Q"
    latest = find_latest_filing_metadata(config.data_dir, filters.company_name, report_type)
    if not latest:
        return filters

    return QueryFilters(
        company_name=filters.company_name,
        document_type=str(latest.get("report_type", filters.document_type or "")).upper() or filters.document_type,
        fiscal_year=int(latest.get("report_year", 0)) or None,
        fiscal_quarter=str(latest.get("report_quarter", "")).upper() or None,
    )


def _fallback_route(question: str, config: AppConfig) -> CopilotRouteResponse:
    lowered = question.lower()
    available = _available_companies(config)

    matched_companies = [company for company in available if company in lowered]

    if any(word in lowered for word in ("compare", "vs", "versus", "difference", "benchmark")):
        intent: CopilotIntent = "comparison"
    elif any(word in lowered for word in ("forecast", "predict", "projection", "project", "future")):
        intent = "forecast"
    elif any(word in lowered for word in ("trend", "pattern", "growth", "trajectory", "movement")):
        intent = "trend"
    elif any(word in lowered for word in ("risk", "red flag", "weakness", "issue", "concern")):
        intent = "risk"
    elif any(word in lowered for word in ("email", "memo", "draft", "write")):
        intent = "letter"
    elif any(word in lowered for word in ("suggest", "recommend", "next step", "what should i ask")):
        intent = "suggestion"
    elif any(word in lowered for word in ("full report", "summary report", "generate report", "write report")):
        intent = "report"
    else:
        intent = "qa"

    fallback = intent == "qa"
    return CopilotRouteResponse(
        intent=intent,
        fallback=fallback,
        fallback_reason="No explicit tool intent detected." if fallback else None,
        matched_companies=matched_companies,
        rewritten_query=question,
    )


def detect_route(question: str, config: AppConfig) -> CopilotRouteResponse:
    llm = _build_router_llm(config).with_structured_output(CopilotRouteResponse)
    available_companies = _available_company_text(config)
    prompt = (
        "Classify the user query for a financial copilot.\n"
        "Return one of these seven tool intents only: comparison, forecast, trend, risk, report, letter, suggestion.\n"
        "If the query does not clearly request one of those seven tools, set intent to qa and fallback to true.\n"
        "Select matched_companies only from the available company folders listed below.\n"
        "Use fallback for ordinary filing questions, factual questions, or anything that is not an explicit tool request.\n"
        "Keep rewritten_query concise and optimized for retrieval.\n\n"
        f"Available company folders: {available_companies}\n\n"
        f"User query: {question}"
    )

    try:
        response = llm.invoke(prompt)
        if isinstance(response, CopilotRouteResponse):
            response.matched_companies = _normalize_company_names(response.matched_companies, config)
            response.fallback = bool(response.fallback or response.intent == "qa")
            if response.fallback and not response.fallback_reason:
                response.fallback_reason = "Routed to fallback by structured classifier."
            return response
        validated = CopilotRouteResponse.model_validate(response)
        validated.matched_companies = _normalize_company_names(validated.matched_companies, config)
        validated.fallback = bool(validated.fallback or validated.intent == "qa")
        if validated.fallback and not validated.fallback_reason:
            validated.fallback_reason = "Routed to fallback by structured classifier."
        return validated
    except Exception as exc:
        print(f"Structured routing unavailable, using heuristic fallback. Reason: {exc}")
        return _fallback_route(question, config)


def extract_company_mentions(question: str, config: AppConfig) -> list[str]:
    lowered = question.lower()
    return [company for company in _available_companies(config) if company in lowered]


def _page_reference(metadata: dict[str, Any]) -> str:
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


def _fallback_answer(intent: CopilotIntent, citations: list[str], reason: str) -> str:
    if intent == "letter":
        return (
            "Subject: Apple 2024 10-K Summary for Finance Team\n\n"
            "Hi Finance Team,\n\n"
            "The retrieved context is not sufficient to draft a fully grounded email summary of Apple’s 2024 10-K. "
            f"The fallback was triggered because: {reason}.\n\n"
            "Please re-run the query after the Apple 2024 10-K has been indexed, or ask with a more specific filing focus.\n\n"
            "Best,\n"
            "Financial Copilot"
        )
    return (
        "Summary:\n"
        "The retrieved context is insufficient to complete this request.\n\n"
        "Key Insights:\n"
        f"- No grounded answer could be produced for intent '{intent}'.\n"
        f"- Fallback reason: {reason}.\n\n"
        "Risks:\n"
        "- Additional retrieval or a more specific query may be needed.\n\n"
        "Recommendation:\n"
        "- Ask with a company name, filing type, and year or quarter if available."
    )


def _build_suggestions(
    intent: CopilotIntent,
    question: str,
    companies: list[str],
    filters: QueryFilters,
) -> list[str]:
    company_text = ", ".join(company.title() for company in companies) if companies else "the target company"
    year_text = str(filters.fiscal_year) if filters.fiscal_year else "the relevant year"
    quarter_text = filters.fiscal_quarter or "the relevant quarter"

    if intent == "comparison":
        return [
            f"Compare {company_text} on revenue, operating margin, and net income for {year_text}.",
            f"Ask for a risk comparison between {company_text} using the latest filings.",
            f"Request a quarter-by-quarter comparison for {company_text} in {quarter_text}.",
        ]
    if intent == "forecast":
        return [
            f"Forecast revenue for {company_text} in the next quarter.",
            f"Forecast operating income for {company_text} using the latest filing context.",
            f"Ask for a bullish and bearish forecast scenario for {company_text}.",
        ]
    if intent == "trend":
        return [
            f"Show revenue and margin trends for {company_text} across {year_text}.",
            f"Break down the trend by quarter and explain the driver changes.",
            f"Ask for a trend summary with supporting filing citations.",
        ]
    if intent == "risk":
        return [
            f"List the top financial and operational risks for {company_text}.",
            f"Highlight new or increasing risks compared with prior filings.",
            f"Ask for risk severity and likely business impact.",
        ]
    if intent == "report":
        return [
            f"Generate a full report for {company_text} covering {year_text}.",
            f"Ask for an executive summary plus key KPIs and risks.",
            f"Request a filing-by-filing summary with citations.",
        ]
    if intent == "letter":
        return [
            f"Write a concise email summarizing the filing highlights for {company_text}.",
            f"Draft a more formal investor-style note with citations.",
            f"Create a short internal memo with risks and recommendations.",
        ]
    if intent == "suggestion":
        return [
            f"Ask for a comparison of {company_text} against a peer set.",
            f"Request a trend analysis for {company_text} over multiple quarters.",
            f"Ask for a risk report followed by recommended follow-up questions.",
        ]
    return [
        "Ask for a summary of the latest filing with cited evidence.",
        "Request a comparison, forecast, trend analysis, or risk review.",
        "Specify company name, filing type, and year for better retrieval.",
    ]


def _build_system_prompt(intent: CopilotIntent) -> str:
    base = (
        "You are an AI financial copilot grounded strictly in retrieved SEC filing context. "
        "Answer only from the supplied context, never from external knowledge. "
        "Use inline citations like [1], [2] for every factual statement. "
        "If the context is not enough, say so clearly and keep the answer concise."
    )

    intent_guidance = {
        "comparison": "Compare the companies using the supplied evidence and make the tradeoffs explicit.",
        "forecast": "Provide a conservative forecast framed as scenario-based reasoning from the evidence.",
        "trend": "Focus on time-based patterns, directionality, and notable inflection points.",
        "risk": "Highlight risks, weaknesses, and uncertainty with clear evidence from the filings.",
        "report": "Write a structured financial report with executive-style clarity.",
        "letter": (
            "Draft a professional email or memo based on the retrieved context. "
            "For letter intent, write an actual email with a subject line, greeting, short body, and sign-off instead of a report. "
            "Do not mention the prompt, suggestion block, or analysis sections."
        ),
        "suggestion": "Suggest concrete next queries or actions the user can take next.",
        "qa": "Answer the question directly with grounded evidence.",
    }
    return f"{base} {intent_guidance[intent]}"


def _build_human_prompt(
    intent: CopilotIntent,
    question: str,
    context: str,
) -> str:
    if intent == "letter":
        structure = (
            "Subject: <one short subject line>\n"
            "Hi Finance Team,\n"
            "<2-4 short paragraphs or bullet points written like a real email>\n"
            "<1 short closing sentence>\n"
            "Best,\n"
            "<Signature>"
        )
    elif intent == "forecast":
        structure = (
            "Forecast Summary:\n"
            "<2-4 bullets forecasting the next period(s), each grounded in cited evidence>\n\n"
            "Base Case:\n"
            "<1 short paragraph describing the most likely scenario with citations>\n\n"
            "Upside/Downside Risks:\n"
            "<2-4 bullets covering what could improve or worsen the forecast>\n\n"
            "Recommendation:\n"
            "<1-2 bullets about the best decision or next check>"
        )
    elif intent == "comparison":
        structure = (
            "Summary:\n"
            "<1 short paragraph comparing the companies with citations>\n\n"
            "Key Comparison Points:\n"
            "<2-4 bullets on revenue, margins, growth, or risk differences>\n\n"
            "Risks:\n"
            "<1-3 bullets on what limits the comparison>\n\n"
            "Recommendation:\n"
            "<1-2 bullets with the most relevant conclusion>"
        )
    elif intent == "trend":
        structure = (
            "Trend Summary:\n"
            "<1 short paragraph describing the trend direction and significance>\n\n"
            "Key Drivers:\n"
            "<2-4 bullets explaining what drove the trend with citations>\n\n"
            "Risks:\n"
            "<1-3 bullets on what may change the trend>\n\n"
            "Recommendation:\n"
            "<1-2 bullets on the best next analysis step>"
        )
    elif intent == "risk":
        structure = (
            "Summary:\n"
            "<1 short paragraph identifying the main risks with citations>\n\n"
            "Key Risks:\n"
            "<2-4 bullets ranking the most important risks>\n\n"
            "Risks:\n"
            "<1-3 bullets on uncertainty or missing data>\n\n"
            "Recommendation:\n"
            "<1-2 bullets on how to respond to the risks>"
        )
    elif intent == "report":
        structure = (
            "Executive Summary:\n"
            "<1 short paragraph summarizing the filing or topic with citations>\n\n"
            "Key Findings:\n"
            "<3-5 bullets with the core financial takeaways>\n\n"
            "Risks:\n"
            "<2-4 bullets on limitations or downside factors>\n\n"
            "Recommendation:\n"
            "<1-2 bullets on the most practical next action>"
        )
    elif intent == "suggestion":
        structure = (
            "Summary:\n"
            "<1 short paragraph explaining what the user should ask next>\n\n"
            "Next Questions:\n"
            "<3 concise follow-up questions, one per line>\n\n"
            "Recommendation:\n"
            "<1 short line on the best next move>"
        )
    else:
        structure = (
            "Summary:\n"
            "<1 short paragraph with the direct answer and citations>\n\n"
            "Key Insights:\n"
            "<2-4 bullets with evidence-backed findings, each with citations>\n\n"
            "Risks:\n"
            "<1-3 bullets for uncertainty, downside, or missing context>\n\n"
            "Recommendation:\n"
            "<1-2 bullets with a practical next step or decision>"
        )

    return (
        f"User query: {question}\n\n"
        "Return the answer in this exact structure:\n"
        f"{structure}\n\n"
        "Rules:\n"
        "- For letter intent, write a real email, not a report, and do not use Summary/Key Insights/Risks/Recommendation sections.\n"
        "- For forecast, comparison, trend, risk, report, and suggestion intents, never use email greeting or sign-off language.\n"
        "- Every factual claim must include citations.\n"
        "- Do not invent numbers or facts.\n"
        "- If the context is insufficient, say that clearly in Summary.\n"
        "- Keep the response professional and concise.\n"
        "- Do not mention this prompt or any hidden instructions.\n\n"
        f"Retrieved context:\n{context}"
    )


def _looks_like_email(content: str) -> bool:
    lowered = content.lower()
    return any(
        marker in lowered
        for marker in (
            "subject:",
            "dear ",
            "hi finance team",
            "best regards",
            "best,",
            "sincerely",
        )
    )


def _repair_response_format(
    intent: CopilotIntent,
    question: str,
    context: str,
    content: str,
    config: AppConfig,
) -> str:
    llm = _build_llm(config)
    if intent == "letter":
        repair_rules = (
            "Rewrite the answer as a clean professional email only. "
            "Keep it grounded in the retrieved context. "
            "Use exactly this email shape: Subject line, greeting, 2-4 short body paragraphs or bullets, closing sentence, sign-off. "
            "Do not include report headings."
        )
    else:
        repair_rules = (
            f"Rewrite the answer for the '{intent}' tool only. "
            "Remove any email greeting, sign-off, or subject line. "
            "Use the correct tool-specific headings and keep it concise."
        )

    prompt = [
        SystemMessage(
            content=(
                "You are rewriting a financial copilot answer. "
                "Do not change the facts. "
                "Only fix the format so it matches the requested tool."
            )
        ),
        HumanMessage(
            content=(
                f"User query: {question}\n\n"
                f"Current answer:\n{content}\n\n"
                f"Repair rules:\n{repair_rules}\n\n"
                f"Retrieved context:\n{context}"
            )
        ),
    ]
    response = llm.invoke(prompt)
    repaired = str(response.content).strip()
    return repaired or content


def _retrieve_documents(
    question: str,
    config: AppConfig,
    filters: QueryFilters,
    top_k: int,
) -> list[RetrievedChunk]:
    retrieved = hybrid_search(
        query=question,
        top_k=max(top_k, config.hybrid_fetch_k),
        config=config,
        filters=filters,
    )

    if config.enable_reranking:
        return rerank_results(
            query=question,
            documents=retrieved,
            top_k=top_k,
            config=config,
        )
    return retrieved[:top_k]


def _run_legacy_pipeline(
    config: AppConfig,
    question: str,
    top_k: int,
    filters: QueryFilters,
    rewritten_query: str,
    companies: list[str],
) -> dict[str, Any]:
    retrieved = hybrid_search(
        query=rewritten_query or question,
        top_k=max(top_k, config.hybrid_fetch_k),
        config=config,
        filters=filters,
    )

    if config.enable_reranking:
        reranked = rerank_results(
            query=question,
            documents=retrieved,
            top_k=top_k,
            config=config,
        )
    else:
        reranked = retrieved[:top_k]
        if reranked:
            print("Reranking disabled, using retrieval order:")
            for chunk in reranked[: min(5, len(reranked))]:
                score = chunk.retrieval_score if chunk.retrieval_score is not None else 0.0
                print(
                    f"- {chunk.metadata.get('document_name', 'unknown')} | "
                    f"{chunk.metadata.get('company_name', 'unknown')} | retrieval={score:.4f}"
                )

    answer, citations = generate_response(
        query=question,
        documents=reranked,
        config=config,
    )

    return {
        "question": question,
        "intent": "qa",
        "top_k": top_k,
        "filters": filters,
        "search_query": rewritten_query,
        "retrieved_docs": retrieved,
        "reranked_docs": reranked,
        "answer": answer,
        "citations": citations,
        "suggestions": _build_suggestions("qa", question, companies, filters),
        "matched_companies": companies,
    }


def _retrieve_comparison_documents(
    question: str,
    config: AppConfig,
    filters: QueryFilters,
    top_k: int,
    companies: list[str] | None = None,
) -> tuple[list[RetrievedChunk], list[str]]:
    companies = companies or extract_company_mentions(question, config)
    documents: list[RetrievedChunk] = []

    if companies:
        for company in companies[:3]:
            company_filters = QueryFilters(
                company_name=company,
                document_type=filters.document_type,
                fiscal_year=filters.fiscal_year,
                fiscal_quarter=filters.fiscal_quarter,
            )
            company_docs = _retrieve_documents(
                question=f"{question} {company}",
                config=config,
                filters=company_filters,
                top_k=max(2, top_k),
            )
            documents.extend(company_docs)
    else:
        documents = _retrieve_documents(question=question, config=config, filters=filters, top_k=max(top_k, config.hybrid_fetch_k))
        companies = sorted(
            dict.fromkeys(
                str(chunk.metadata.get("company_name", "")).lower()
                for chunk in documents
                if chunk.metadata.get("company_name")
            )
        )

    unique_documents = _dedupe_documents(documents)
    if config.enable_reranking and len(unique_documents) > 1:
        unique_documents = rerank_results(
            query=question,
            documents=unique_documents,
            top_k=min(len(unique_documents), max(top_k, 6)),
            config=config,
        )
    return unique_documents[: max(top_k, 4)], companies


def generate_copilot_response(
    intent: CopilotIntent,
    question: str,
    documents: list[RetrievedChunk],
    config: AppConfig,
    companies: list[str] | None = None,
    filters: QueryFilters | None = None,
) -> tuple[str, list[str]]:
    unique_documents = _dedupe_documents(documents)
    context, citations = _build_context_and_citations(unique_documents)

    if not unique_documents:
        return _fallback_answer(intent, citations, "No retrieved context."), citations

    llm = _build_llm(config)
    suggestions = _build_suggestions(intent, question, companies or [], filters or QueryFilters())
    prompt = [
        SystemMessage(content=_build_system_prompt(intent)),
        HumanMessage(content=_build_human_prompt(intent, question, context)),
    ]

    try:
        print(f"Calling copilot generation model for intent '{intent}': {config.llm_model}")
        response = llm.invoke(prompt)
        content = str(response.content).strip()
        if not content:
            raise ValueError("Generation model returned an empty response.")
        if intent != "letter" and _looks_like_email(content):
            content = _repair_response_format(intent, question, context, content, config)
        elif intent == "letter" and not _looks_like_email(content):
            content = _repair_response_format(intent, question, context, content, config)
        return content, citations
    except Exception as exc:
        print(f"Copilot generation unavailable, using grounded fallback. Reason: {exc}")
        return _fallback_answer(intent, citations, str(exc)), citations


def run_financial_copilot(config: AppConfig, question: str, top_k: int) -> dict[str, Any]:
    return run_financial_copilot_with_mode(config, question, top_k, "auto")


def run_financial_copilot_with_mode(
    config: AppConfig,
    question: str,
    top_k: int,
    mode: QueryMode,
) -> dict[str, Any]:
    filters, rewritten_query = extract_filters(question, config)
    route = detect_route(question, config)
    intent = route.intent
    companies = route.matched_companies or extract_company_mentions(question, config)
    filters = _prefer_latest_filing(filters, config, intent)

    print(f"Detected intent: {intent}")
    print(f"Route fallback: {route.fallback} | reason: {route.fallback_reason}")
    print(f"Extracted filters: {filters.model_dump()}")
    print(f"Hybrid search query: {rewritten_query}")

    if mode == "legacy" or route.fallback:
        return _run_legacy_pipeline(config, question, top_k, filters, rewritten_query, companies)

    if intent == "comparison":
        retrieved, compared_companies = _retrieve_comparison_documents(
            question=question,
            config=config,
            filters=filters,
            top_k=top_k,
            companies=companies,
        )
        if compared_companies:
            companies = compared_companies
    else:
        retrieval_query = route.rewritten_query or rewritten_query or question
        if intent == "letter" and filters.company_name and filters.document_type and filters.fiscal_year and filters.fiscal_quarter:
            retrieval_query = (
                f"{filters.company_name} {filters.document_type} {filters.fiscal_year} "
                f"{filters.fiscal_quarter} quarterly performance investor update"
            )
        retrieved = _retrieve_documents(
            question=retrieval_query,
            config=config,
            filters=filters,
            top_k=top_k,
        )

    answer, citations = generate_copilot_response(
        intent=intent,
        question=question,
        documents=retrieved,
        config=config,
        companies=companies,
        filters=filters,
    )

    return {
        "question": question,
        "intent": intent,
        "top_k": top_k,
        "filters": filters,
        "search_query": route.rewritten_query or rewritten_query,
        "retrieved_docs": retrieved,
        "answer": answer,
        "citations": citations,
        "suggestions": _build_suggestions(intent, question, companies, filters),
        "matched_companies": companies,
    }
