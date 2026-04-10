from __future__ import annotations

from typing import Dict, List

from src.tools.contracts import (
    PaperFetchToolRequest,
    PDFParseToolRequest,
    SectionReadToolRequest,
    VisualExtractToolRequest,
)
from src.tools.research_document_tool import DOCUMENT_TOOL_HARNESS


class DeepReadingComponent:
    def __init__(self) -> None:
        self.tool_harness = DOCUMENT_TOOL_HARNESS

    def fetch_full_text(
        self,
        identifier: str,
        *,
        identifier_type: str = "auto",
        prefer: str = "pdf",
        download_dir: str = "",
    ):
        return self.tool_harness.fetch(
            PaperFetchToolRequest(
                identifier=identifier,
                identifier_type=identifier_type,
                prefer=prefer,
                download_dir=download_dir,
            )
        )

    def parse_pdf(self, pdf_path: str, *, target_section: str = ""):
        return self.tool_harness.parse_pdf(
            PDFParseToolRequest(
                pdf_path=pdf_path,
                target_section=target_section,
            )
        )

    def targeted_read(
        self,
        pdf_path: str,
        *,
        section_name: str,
        max_chunks: int = 5,
    ) -> Dict[str, object]:
        return self.tool_harness.read_section(
            SectionReadToolRequest(
                pdf_path=pdf_path,
                section_name=section_name,
                max_chunks=max_chunks,
            )
        )

    def extract_visuals(
        self,
        pdf_path: str,
        *,
        page_numbers: List[int] | None = None,
        output_dir: str = "",
    ) -> Dict[str, object]:
        return self.tool_harness.extract_visuals(
            VisualExtractToolRequest(
                pdf_path=pdf_path,
                page_numbers=page_numbers,
                output_dir=output_dir,
            )
        )
