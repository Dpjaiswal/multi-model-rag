from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_app.config import AppConfig


YEAR_PATTERN = re.compile(r"(20\d{2})")
QUARTER_PATTERN = re.compile(r"\bq([1-4])\b", re.IGNORECASE)
REPORT_TYPE_PATTERN = re.compile(r"\b(10-k|10-q|8-k)\b", re.IGNORECASE)


def discover_pdf_paths(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.rglob("*.pdf") if path.is_file())


def _parse_report_metadata(pdf_path: Path) -> dict[str, str | int]:
    file_name = pdf_path.stem.lower()
    year_match = YEAR_PATTERN.search(file_name)
    quarter_match = QUARTER_PATTERN.search(file_name)
    report_type_match = REPORT_TYPE_PATTERN.search(file_name)

    report_year = int(year_match.group(1)) if year_match else 0
    report_quarter = f"Q{quarter_match.group(1)}" if quarter_match else ""
    report_type = report_type_match.group(1).upper() if report_type_match else "PDF"
    current_year = datetime.now().year
    age_years = current_year - report_year if report_year else -1

    if age_years < 0:
        age_bucket = "unknown"
    elif age_years <= 1:
        age_bucket = "recent"
    elif age_years <= 3:
        age_bucket = "mid"
    else:
        age_bucket = "archive"

    return {
        "company_name": pdf_path.parent.name,
        "document_name": pdf_path.name,
        "report_year": report_year,
        "report_quarter": report_quarter,
        "report_type": report_type,
        "age_years": age_years,
        "age_bucket": age_bucket,
        "source": str(pdf_path),
    }


def _sort_key(pdf_path: Path) -> tuple[int, str]:
    metadata = _parse_report_metadata(pdf_path)
    return (int(metadata["report_year"]), pdf_path.name.lower())


def find_latest_filing_metadata(
    data_dir: Path,
    company_name: str,
    report_type: str | None = None,
) -> dict[str, str | int] | None:
    company = company_name.strip().lower()
    report = report_type.strip().upper() if report_type else None
    candidates = []

    for pdf_path in discover_pdf_paths(data_dir):
        metadata = _parse_report_metadata(pdf_path)
        if metadata["company_name"].lower() != company:
            continue
        if report and str(metadata["report_type"]).upper() != report:
            continue
        candidates.append(metadata)

    if not candidates:
        return None

    quarter_order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "": 0}
    candidates.sort(
        key=lambda item: (
            int(item.get("report_year", 0)),
            quarter_order.get(str(item.get("report_quarter", "")).upper(), 0),
            str(item.get("document_name", "")).lower(),
        ),
        reverse=True,
    )
    return candidates[0]


def load_pdf_documents(config: AppConfig) -> list[Document]:
    pdf_paths = sorted(discover_pdf_paths(config.data_dir), key=_sort_key, reverse=True)
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under {config.data_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    chunks: list[Document] = []

    print(f"Discovered {len(pdf_paths)} PDF files in {config.data_dir}")

    for pdf_number, pdf_path in enumerate(pdf_paths, start=1):
        report_metadata = _parse_report_metadata(pdf_path)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            page.metadata.update(report_metadata)

        split_pages = splitter.split_documents(pages)
        for index, chunk in enumerate(split_pages, start=1):
            chunk.metadata["chunk_id"] = f"{pdf_path.stem}-chunk-{index}"
            chunk.metadata["chunking_strategy"] = "age-aware-recursive"
            chunk.metadata["sort_priority"] = (
                f"{chunk.metadata.get('report_year', 0):04d}-{chunk.metadata.get('report_quarter', 'Q0')}"
            )
        chunks.extend(split_pages)

        print(
            f"[{pdf_number}/{len(pdf_paths)}] {pdf_path.name}: {len(pages)} pages -> {len(split_pages)} chunks"
        )

    print(f"Total chunks prepared: {len(chunks)}")
    return chunks
