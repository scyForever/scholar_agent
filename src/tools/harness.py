from __future__ import annotations

from typing import Any, Dict, List

from src.core.models import Paper
from src.tools.contracts import (
    LiteratureSearchToolRequest,
    LiteratureSourceSearchRequest,
    PaperFetchToolRequest,
    PDFParseToolRequest,
    SectionReadToolRequest,
    VisualExtractToolRequest,
    WebSearchToolRequest,
)


class LiteratureSearchToolHarness:
    """学术检索工具入口 harness，统一承接多源检索请求。"""

    def __init__(self, service: Any) -> None:
        self.service = service

    def search(self, request: LiteratureSearchToolRequest) -> Dict[str, Any]:
        return self.service.search(request)

    def search_source(self, request: LiteratureSourceSearchRequest) -> List[Paper]:
        return self.service.search_by_source(
            request.source_name,
            request,
        )


class DocumentToolHarness:
    """论文获取与阅读工具入口 harness。"""

    def __init__(self, acquisition_service: Any, document_service: Any) -> None:
        self.acquisition_service = acquisition_service
        self.document_service = document_service

    def fetch(self, request: PaperFetchToolRequest) -> Any:
        return self.acquisition_service.fetch(
            request.identifier,
            identifier_type=request.identifier_type,
            prefer=request.prefer,
            download_dir=request.download_dir,
        )

    def parse_pdf(self, request: PDFParseToolRequest) -> Any:
        return self.document_service.parse_pdf(
            request.pdf_path,
            target_section=request.target_section,
        )

    def extract_visuals(self, request: VisualExtractToolRequest) -> Dict[str, Any]:
        return self.document_service.extract_visuals(
            request.pdf_path,
            page_numbers=request.page_numbers,
            output_dir=request.output_dir,
        )

    def read_section(self, request: SectionReadToolRequest) -> Dict[str, Any]:
        return self.document_service.read_section(
            request.pdf_path,
            section_name=request.section_name,
            max_chunks=request.max_chunks,
        )


class WebSearchToolHarness:
    """通用网页补充检索 harness。"""

    def __init__(self, search_callable: Any) -> None:
        self.search_callable = search_callable

    def search(self, request: WebSearchToolRequest) -> List[Dict[str, str]]:
        return self.search_callable(
            request.query,
            max_results=request.max_results,
        )
