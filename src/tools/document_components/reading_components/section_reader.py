from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict


class PDFSectionReadingComponent:
    def __init__(self, parsing_component: Any) -> None:
        self.parsing_component = parsing_component

    def read_section(
        self,
        pdf_path: str,
        *,
        section_name: str,
        max_chunks: int = 5,
    ) -> Dict[str, Any]:
        document = self.parsing_component.parse_pdf(pdf_path, target_section=section_name)
        selected_sections = document.sections[:max_chunks]
        return {
            "pdf_path": pdf_path,
            "title": document.title,
            "matched_sections": [asdict(item) for item in selected_sections],
            "chunks": document.chunks[:max_chunks],
            "abstract": document.abstract,
            "metadata": document.metadata,
        }
