from __future__ import annotations

from typing import Dict, List

from src.core.models import ParsedDocument
from src.tools.document_components.reading_components import (
    PDFParsingComponent,
    PDFSectionReadingComponent,
    PDFVisualExtractionComponent,
)


class PDFReadingService:
    def __init__(self) -> None:
        self.visual_component = PDFVisualExtractionComponent()
        self.parsing_component = PDFParsingComponent(self.visual_component)
        self.section_reader_component = PDFSectionReadingComponent(self.parsing_component)

    def parse_pdf(
        self,
        pdf_path: str,
        *,
        target_section: str = "",
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> ParsedDocument:
        return self.parsing_component.parse_pdf(
            pdf_path,
            target_section=target_section,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def extract_visuals(
        self,
        pdf_path: str,
        *,
        page_numbers: List[int] | None = None,
        output_dir: str = "",
        enable_ocr: bool = True,
    ) -> Dict[str, Any]:
        return self.visual_component.extract_visuals(
            pdf_path,
            page_numbers=page_numbers,
            output_dir=output_dir,
            enable_ocr=enable_ocr,
        )

    def read_section(
        self,
        pdf_path: str,
        *,
        section_name: str,
        max_chunks: int = 5,
    ) -> Dict[str, Any]:
        return self.section_reader_component.read_section(
            pdf_path,
            section_name=section_name,
            max_chunks=max_chunks,
        )
