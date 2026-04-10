from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List
from uuid import uuid4

from src.core.models import VisualElement
from src.tools.document_components.common import (
    FIGURE_CAPTION_RE,
    FORMULA_LINE_RE,
    clean_text,
    normalize_inline_whitespace,
    slug,
)

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


class PDFVisualExtractionComponent:
    def extract_visuals(
        self,
        pdf_path: str,
        *,
        page_numbers: List[int] | None = None,
        output_dir: str = "",
        enable_ocr: bool = True,
    ) -> Dict[str, Any]:
        path = Path(pdf_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if fitz is None:
            return {
                "pdf_path": str(path),
                "figures": [],
                "tables": [],
                "formulas": [],
                "metadata": {
                    "ocr_enabled": False,
                    "warnings": ["PyMuPDF 未安装，无法提取图表。"],
                },
            }
        target_pages = set(page_numbers or [])
        image_dir = Path(output_dir).expanduser() if output_dir else Path("cache/figures") / slug(path.stem)
        image_dir.mkdir(parents=True, exist_ok=True)

        figures: List[VisualElement] = []
        formulas: List[VisualElement] = []
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc):
                page_no = page_index + 1
                if target_pages and page_no not in target_pages:
                    continue
                page_text = clean_text(page.get_text("text"))
                figure_captions = self._find_caption_lines(page_text, FIGURE_CAPTION_RE)
                for image_index, image_info in enumerate(page.get_images(full=True)):
                    xref = image_info[0]
                    image_path = ""
                    try:
                        pixmap = fitz.Pixmap(doc, xref)
                        if pixmap.alpha:
                            pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
                        image_path = str(image_dir / f"page-{page_no:03d}-img-{image_index:02d}.png")
                        pixmap.save(image_path)
                    except Exception:
                        image_path = ""
                    ocr_text = self._ocr_image(image_path) if enable_ocr else ""
                    markdown = self._table_like_markdown(ocr_text)
                    latex = self._formula_to_latex(ocr_text) if ocr_text else ""
                    figures.append(
                        VisualElement(
                            element_id=str(uuid4()),
                            kind="figure",
                            page=page_no,
                            caption=figure_captions[image_index] if image_index < len(figure_captions) else "",
                            text=ocr_text,
                            latex=latex,
                            markdown=markdown,
                            image_path=image_path,
                            metadata={"ocr_used": bool(ocr_text)},
                        )
                    )
                formulas.extend(self._extract_formula_lines(page_text, page_no))
        tables = self.extract_tables(path, page_numbers=page_numbers, enable_ocr=enable_ocr)
        return {
            "pdf_path": str(path),
            "figures": [asdict(item) for item in figures],
            "tables": [asdict(item) for item in tables],
            "formulas": [asdict(item) for item in formulas],
            "metadata": {
                "ocr_enabled": enable_ocr and pytesseract is not None and Image is not None,
                "image_dir": str(image_dir),
            },
        }

    def extract_visual_placeholders(self, path: Path) -> tuple[List[VisualElement], List[VisualElement]]:
        figures: List[VisualElement] = []
        formulas: List[VisualElement] = []
        if fitz is None:
            return figures, formulas
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc):
                page_no = page_index + 1
                page_text = clean_text(page.get_text("text"))
                for line in self._find_caption_lines(page_text, FIGURE_CAPTION_RE):
                    figures.append(
                        VisualElement(
                            element_id=str(uuid4()),
                            kind="figure",
                            page=page_no,
                            caption=line,
                            metadata={"source": "caption"},
                        )
                    )
                formulas.extend(self._extract_formula_lines(page_text, page_no))
        return figures, formulas

    def extract_tables(
        self,
        path: Path,
        *,
        page_numbers: List[int] | None = None,
        enable_ocr: bool = True,
    ) -> List[VisualElement]:
        tables: List[VisualElement] = []
        selected_pages = set(page_numbers or [])
        if pdfplumber is not None:
            with pdfplumber.open(path) as pdf:
                for page_index, page in enumerate(pdf.pages):
                    page_no = page_index + 1
                    if selected_pages and page_no not in selected_pages:
                        continue
                    for table in page.extract_tables() or []:
                        markdown = self._rows_to_markdown(table)
                        tables.append(
                            VisualElement(
                                element_id=str(uuid4()),
                                kind="table",
                                page=page_no,
                                markdown=markdown,
                                text=markdown,
                                metadata={"source": "pdfplumber"},
                            )
                        )
        return tables

    def _rows_to_markdown(self, rows: Iterable[Iterable[Any]]) -> str:
        cleaned_rows = [[normalize_inline_whitespace(str(cell or "")) for cell in row] for row in rows]
        cleaned_rows = [row for row in cleaned_rows if any(cell for cell in row)]
        if not cleaned_rows:
            return ""
        header = cleaned_rows[0]
        divider = ["---" for _ in header]
        body = cleaned_rows[1:] or [[]]
        markdown_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(divider) + " |",
        ]
        markdown_lines.extend("| " + " | ".join(row) + " |" for row in body if row)
        return "\n".join(markdown_lines)

    def _find_caption_lines(self, page_text: str, pattern: re.Pattern[str]) -> List[str]:
        return [
            normalize_inline_whitespace(line)
            for line in page_text.splitlines()
            if pattern.match(normalize_inline_whitespace(line))
        ]

    def _extract_formula_lines(self, page_text: str, page_no: int) -> List[VisualElement]:
        formulas: List[VisualElement] = []
        for line in page_text.splitlines():
            cleaned = normalize_inline_whitespace(line)
            if len(cleaned) < 6:
                continue
            if FORMULA_LINE_RE.search(cleaned) and any(char.isalpha() for char in cleaned):
                formulas.append(
                    VisualElement(
                        element_id=str(uuid4()),
                        kind="formula",
                        page=page_no,
                        text=cleaned,
                        latex=self._formula_to_latex(cleaned),
                        metadata={"source": "text_line"},
                    )
                )
        return formulas

    def _formula_to_latex(self, text: str) -> str:
        normalized = str(text or "")
        replacements = {
            "≤": r"\leq ",
            "≥": r"\geq ",
            "×": r"\times ",
            "∑": r"\sum ",
            "∏": r"\prod ",
            "α": r"\alpha ",
            "β": r"\beta ",
            "γ": r"\gamma ",
            "λ": r"\lambda ",
            "μ": r"\mu ",
            "σ": r"\sigma ",
        }
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)
        normalized = normalized.replace("^", "^{").replace("_", "_{")
        normalized = re.sub(r"([A-Za-z0-9}])\{(?=[^\s])", r"\1", normalized)
        return normalized

    def _ocr_image(self, image_path: str) -> str:
        if not image_path or pytesseract is None or Image is None:
            return ""
        try:
            with Image.open(image_path) as image:
                return clean_text(pytesseract.image_to_string(image))
        except Exception:
            return ""

    def _table_like_markdown(self, text: str) -> str:
        lines = [normalize_inline_whitespace(line) for line in text.splitlines() if normalize_inline_whitespace(line)]
        if len(lines) < 2:
            return ""
        rows = [re.split(r"\s{2,}", line) for line in lines]
        max_columns = max(len(row) for row in rows)
        if max_columns < 2:
            return ""
        normalized_rows = [row + [""] * (max_columns - len(row)) for row in rows[:8]]
        return self._rows_to_markdown(normalized_rows)
