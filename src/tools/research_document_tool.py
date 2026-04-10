from __future__ import annotations

from typing import Any, Dict, List

from src.core.models import PaperAsset, ParsedDocument
from src.tools.contracts import (
    PaperFetchToolRequest,
    PDFParseToolRequest,
    SectionReadToolRequest,
    VisualExtractToolRequest,
)
from src.tools.document_components import (
    PDFReadingService,
    PaperAcquisitionService,
    serialize_asset as _serialize_asset,
    serialize_document as _serialize_document,
)
from src.tools.harness import DocumentToolHarness
from src.tools.registry import ToolDefinition, ToolParameter, register_tool


DOCUMENT_SERVICE = PDFReadingService()
ACQUISITION_SERVICE = PaperAcquisitionService()
DOCUMENT_TOOL_HARNESS = DocumentToolHarness(ACQUISITION_SERVICE, DOCUMENT_SERVICE)


@register_tool(
    ToolDefinition(
        name="fetch_paper_asset",
        description="Fetch a paper PDF or HTML by DOI, arXiv ID, PMID or PMCID.",
        parameters=[
            ToolParameter("identifier", "str", "DOI、arXiv ID、PMID 或 PMCID", True),
            ToolParameter("identifier_type", "str", "标识符类型", False),
            ToolParameter("prefer", "str", "优先返回 pdf 或 html", False),
            ToolParameter("download_dir", "str", "下载目录", False),
        ],
        tags=["paper", "fetch"],
    )
)
def fetch_paper_asset(
    identifier: str,
    identifier_type: str = "auto",
    prefer: str = "pdf",
    download_dir: str = "",
) -> Dict[str, Any]:
    return _serialize_asset(
        DOCUMENT_TOOL_HARNESS.fetch(
            PaperFetchToolRequest(
                identifier=identifier,
                identifier_type=identifier_type,
                prefer=prefer,
                download_dir=download_dir,
            )
        )
    )


@register_tool(
    ToolDefinition(
        name="parse_pdf_document",
        description="Parse PDF text, section hierarchy, tables and formula-like lines.",
        parameters=[
            ToolParameter("pdf_path", "str", "PDF路径", True),
            ToolParameter("target_section", "str", "目标章节名", False),
        ],
        tags=["pdf", "parse"],
    )
)
def parse_pdf_document(pdf_path: str, target_section: str = "") -> Dict[str, Any]:
    return _serialize_document(
        DOCUMENT_TOOL_HARNESS.parse_pdf(
            PDFParseToolRequest(
                pdf_path=pdf_path,
                target_section=target_section,
            )
        )
    )


@register_tool(
    ToolDefinition(
        name="extract_paper_visuals",
        description="Extract figures, tables and formulas from PDF, optionally with OCR.",
        parameters=[
            ToolParameter("pdf_path", "str", "PDF路径", True),
            ToolParameter("page_numbers", "list", "页码列表", False),
            ToolParameter("output_dir", "str", "图片输出目录", False),
        ],
        tags=["pdf", "vision"],
    )
)
def extract_paper_visuals(
    pdf_path: str,
    page_numbers: List[int] | None = None,
    output_dir: str = "",
) -> Dict[str, Any]:
    return DOCUMENT_TOOL_HARNESS.extract_visuals(
        VisualExtractToolRequest(
            pdf_path=pdf_path,
            page_numbers=page_numbers,
            output_dir=output_dir,
        )
    )


@register_tool(
    ToolDefinition(
        name="read_paper_section",
        description="Read a specific PDF section such as Methodology or Conclusion.",
        parameters=[
            ToolParameter("pdf_path", "str", "PDF路径", True),
            ToolParameter("section_name", "str", "章节名", True),
            ToolParameter("max_chunks", "int", "返回块数", False),
        ],
        tags=["pdf", "read"],
    )
)
def read_paper_section(pdf_path: str, section_name: str, max_chunks: int = 5) -> Dict[str, Any]:
    return DOCUMENT_TOOL_HARNESS.read_section(
        SectionReadToolRequest(
            pdf_path=pdf_path,
            section_name=section_name,
            max_chunks=max_chunks,
        )
    )


def fetch_paper(identifier: str, *, identifier_type: str = "auto", prefer: str = "pdf", download_dir: str = "") -> PaperAsset:
    return DOCUMENT_TOOL_HARNESS.fetch(
        PaperFetchToolRequest(
            identifier=identifier,
            identifier_type=identifier_type,
            prefer=prefer,
            download_dir=download_dir,
        )
    )


def parse_pdf(pdf_path: str, *, target_section: str = "") -> ParsedDocument:
    return DOCUMENT_TOOL_HARNESS.parse_pdf(
        PDFParseToolRequest(
            pdf_path=pdf_path,
            target_section=target_section,
        )
    )


__all__ = [
    "ACQUISITION_SERVICE",
    "DOCUMENT_SERVICE",
    "DOCUMENT_TOOL_HARNESS",
    "fetch_paper",
    "parse_pdf",
    "fetch_paper_asset",
    "parse_pdf_document",
    "extract_paper_visuals",
    "read_paper_section",
]
