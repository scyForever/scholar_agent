from __future__ import annotations

from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List

from src.core.models import PaperSection, ParsedDocument
from src.tools.document_components.common import (
    SECTION_HEADING_RE,
    chunk_text,
    clean_text,
    looks_like_heading,
    normalize_heading,
    normalize_inline_whitespace,
    slug,
)

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None


class PDFParsingComponent:
    def __init__(self, visual_component: Any) -> None:
        self.visual_component = visual_component

    def parse_pdf(
        self,
        pdf_path: str,
        *,
        target_section: str = "",
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> ParsedDocument:
        path = Path(pdf_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if fitz is None:
            return ParsedDocument(
                document_id=self._document_id(path),
                source_path=str(path),
                metadata={
                    "parser_backend": "unavailable",
                    "warnings": ["PyMuPDF 未安装，无法执行版面感知解析。"],
                },
            )

        with fitz.open(path) as doc:
            page_texts: List[str] = []
            section_candidates: List[Dict[str, Any]] = []
            full_lines: List[str] = []
            for page_index, page in enumerate(doc):
                page_payload = page.get_text("dict")
                page_lines, page_candidates = self._extract_page_lines(page_payload, page_index + 1)
                page_text = "\n".join(line["text"] for line in page_lines if line["text"])
                page_texts.append(clean_text(page_text))
                full_lines.extend(item["text"] for item in page_lines if item["text"])
                section_candidates.extend(page_candidates)

            full_text = clean_text("\n\n".join(page_texts))
            sections = self._build_sections(section_candidates, full_lines)
            if target_section:
                sections = self._filter_sections(sections, target_section)

            figures, formulas = self.visual_component.extract_visual_placeholders(path)
            tables = self.visual_component.extract_tables(path)
            chunks = self._build_chunks_from_sections(sections, chunk_size=chunk_size, overlap=overlap)

            return ParsedDocument(
                document_id=self._document_id(path),
                source_path=str(path),
                title=self._infer_title(page_texts),
                abstract=self._infer_abstract(full_text, sections),
                full_text=full_text,
                page_texts=page_texts,
                sections=sections,
                chunks=chunks,
                figures=figures,
                tables=tables,
                formulas=formulas,
                metadata={
                    "parser_backend": "pymupdf",
                    "page_count": len(page_texts),
                    "double_column_detected": any(item.metadata.get("layout") == "double_column" for item in sections),
                    "target_section": target_section,
                },
            )

    def _document_id(self, path: Path) -> str:
        digest = md5(str(path).encode("utf-8"), usedforsecurity=False).hexdigest()
        return f"{slug(path.stem)}-{digest[:12]}"

    def _extract_page_lines(self, page_payload: Dict[str, Any], page_no: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        lines: List[Dict[str, Any]] = []
        font_sizes: List[float] = []
        blocks = [block for block in page_payload.get("blocks", []) if block.get("type") == 0]
        if not blocks:
            return [], []
        page_width = max(float(block.get("bbox", [0, 0, 0, 0])[2]) for block in blocks)
        left_count = sum(1 for block in blocks if float(block.get("bbox", [0, 0, 0, 0])[0]) < page_width * 0.45)
        right_count = sum(1 for block in blocks if float(block.get("bbox", [0, 0, 0, 0])[0]) > page_width * 0.55)
        layout = "double_column" if left_count >= 2 and right_count >= 2 else "single_column"

        for block in blocks:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(span.get("text", "") for span in spans).strip()
                if not text:
                    continue
                size = max(float(span.get("size") or 0.0) for span in spans) if spans else 0.0
                bbox = line.get("bbox", block.get("bbox", [0, 0, 0, 0]))
                font_sizes.append(size)
                lines.append(
                    {
                        "text": normalize_inline_whitespace(text),
                        "font_size": size,
                        "bbox": bbox,
                        "page": page_no,
                        "layout": layout,
                    }
                )

        if not lines:
            return [], []

        median_font = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 10.0
        if layout == "double_column":
            lines.sort(key=lambda item: (item["page"], 0 if item["bbox"][0] < page_width / 2 else 1, item["bbox"][1], item["bbox"][0]))
        else:
            lines.sort(key=lambda item: (item["page"], item["bbox"][1], item["bbox"][0]))

        section_candidates = [
            item
            for item in lines
            if looks_like_heading(item["text"], item["font_size"], median_font)
        ]
        return lines, section_candidates

    def _build_sections(self, candidates: List[Dict[str, Any]], full_lines: List[str]) -> List[PaperSection]:
        normalized_lines = [normalize_inline_whitespace(line) for line in full_lines if normalize_inline_whitespace(line)]
        if not normalized_lines:
            return []
        headings = {normalize_inline_whitespace(item["text"]): item for item in candidates}
        sections: List[PaperSection] = []
        current_heading = "全文"
        current_level = 1
        current_page = 1
        buffer: List[str] = []
        for line in normalized_lines:
            heading_meta = headings.get(line)
            if heading_meta is not None and buffer:
                sections.append(
                    PaperSection(
                        heading=current_heading,
                        level=current_level,
                        text=clean_text("\n".join(buffer)),
                        page_start=current_page,
                        page_end=heading_meta["page"],
                        metadata={"layout": heading_meta.get("layout", "single_column")},
                    )
                )
                current_heading = line
                current_level = 1
                current_page = heading_meta["page"]
                buffer = []
                continue
            if heading_meta is not None and not buffer:
                current_heading = line
                current_level = 1
                current_page = heading_meta["page"]
                continue
            buffer.append(line)
        if buffer:
            sections.append(
                PaperSection(
                    heading=current_heading,
                    level=current_level,
                    text=clean_text("\n".join(buffer)),
                    page_start=current_page,
                    page_end=current_page,
                    metadata={"layout": "single_column"},
                )
            )
        if not sections:
            sections.append(PaperSection(heading="全文", level=1, text=clean_text("\n".join(normalized_lines))))
        return [section for section in sections if section.text.strip()]

    def _filter_sections(self, sections: List[PaperSection], target_section: str) -> List[PaperSection]:
        target = normalize_heading(target_section)
        matched = [section for section in sections if target in normalize_heading(section.heading)]
        if matched:
            return matched
        keyword_map = {
            "methodology": ["method", "methods", "methodology", "approach"],
            "method": ["method", "methods", "methodology", "approach"],
            "conclusion": ["conclusion", "discussion", "limitations"],
            "result": ["result", "results", "experiment", "experiments"],
        }
        aliases = keyword_map.get(target, [target])
        return [
            section
            for section in sections
            if any(alias in normalize_heading(section.heading) for alias in aliases)
        ]

    def _build_chunks_from_sections(self, sections: List[PaperSection], *, chunk_size: int, overlap: int) -> List[str]:
        chunks: List[str] = []
        for section in sections:
            section_chunks = chunk_text(section.text, chunk_size=chunk_size, overlap=overlap)
            if section_chunks:
                chunks.extend([f"[{section.heading}] {chunk}" for chunk in section_chunks])
        return chunks

    def _infer_title(self, page_texts: List[str]) -> str:
        first_page = page_texts[0] if page_texts else ""
        for line in first_page.splitlines():
            cleaned = normalize_inline_whitespace(line)
            if 6 <= len(cleaned) <= 220 and not SECTION_HEADING_RE.match(cleaned):
                return cleaned
        return ""

    def _infer_abstract(self, full_text: str, sections: List[PaperSection]) -> str:
        for section in sections:
            if normalize_heading(section.heading).startswith("abstract"):
                return section.text[:2000]
        lowered = full_text.lower()
        marker = lowered.find("abstract")
        if marker >= 0:
            return full_text[marker : marker + 2000]
        return full_text[:1200]
