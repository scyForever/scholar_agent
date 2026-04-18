from __future__ import annotations

from dataclasses import asdict
from hashlib import md5
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List
from uuid import uuid4

import requests

from src.core.models import PaperAsset, PaperSection, ParsedDocument, VisualElement
from src.tools.registry import ToolDefinition, ToolParameter, register_tool

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


REQUEST_TIMEOUT = 30
SECTION_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*)?\.?\s*(abstract|introduction|background|related work|method|methods|methodology|approach|experiment|experiments|results|discussion|conclusion|limitations|appendix)\b",
    flags=re.IGNORECASE,
)
FORMULA_LINE_RE = re.compile(r"[=+\-*/^_\\]|(?:\([0-9]{1,3}\)\s*$)")
FIGURE_CAPTION_RE = re.compile(r"^(figure|fig\.)\s*\d+", flags=re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r"^table\s*\d+", flags=re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r".+?(?:[。！？!?；;]+(?=\s|$)|\.(?=\s|$)|$)", flags=re.S)
CLAUSE_SPLIT_RE = re.compile(r"[^，,、；;：:]+[，,、；;：:]?", flags=re.S)


def _clean_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in str(text or "").replace("\r", "\n").splitlines()).strip()


def _normalize_inline_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", str(text or "")).strip()


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    return cleaned.strip("-") or "document"


def _normalize_chunk_source(text: str) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _split_paragraphs(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]


def _split_lines(text: str) -> List[str]:
    return [part.strip() for part in text.splitlines() if part.strip()]


def _split_sentences(text: str) -> List[str]:
    return [
        _normalize_inline_whitespace(match.group(0))
        for match in SENTENCE_SPLIT_RE.finditer(text)
        if match.group(0).strip()
    ]


def _split_clauses(text: str) -> List[str]:
    return [
        _normalize_inline_whitespace(match.group(0))
        for match in CLAUSE_SPLIT_RE.finditer(text)
        if match.group(0).strip()
    ]


def _split_words(text: str) -> List[str]:
    return [part for part in text.split() if part]


def _split_characters(text: str) -> List[str]:
    return [char for char in text if char]


def _merge_splits(splits: List[str], *, separator: str, chunk_size: int, overlap: int) -> List[str]:
    if not splits:
        return []

    chunks: List[str] = []
    current_parts: List[str] = []
    current_length = 0
    separator_length = len(separator)

    for split in splits:
        split_length = len(split)
        additional_length = split_length if not current_parts else separator_length + split_length

        if current_parts and current_length + additional_length > chunk_size:
            chunk = separator.join(current_parts).strip()
            if chunk:
                chunks.append(chunk)
            while current_parts and (
                current_length > overlap or current_length + additional_length > chunk_size
            ):
                removed = current_parts.pop(0)
                current_length -= len(removed)
                if current_parts:
                    current_length -= separator_length
            additional_length = split_length if not current_parts else separator_length + split_length

        current_parts.append(split)
        current_length += additional_length

    if current_parts:
        chunk = separator.join(current_parts).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def _recursive_split_text(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    level: int = 0,
) -> List[str]:
    # 按段落、换行、句子、分句、词和字符逐层细化，避免优先打断自然语义边界。
    normalized = _normalize_chunk_source(text)
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    splitters = [
        (_split_paragraphs, "\n\n"),
        (_split_lines, "\n"),
        (_split_sentences, " "),
        (_split_clauses, " "),
        (_split_words, " "),
        (_split_characters, ""),
    ]

    if level >= len(splitters):
        return [normalized[index : index + chunk_size] for index in range(0, len(normalized), chunk_size)]

    splitter, separator = splitters[level]
    parts = splitter(normalized)
    if len(parts) <= 1:
        return _recursive_split_text(normalized, chunk_size=chunk_size, overlap=overlap, level=level + 1)

    chunks: List[str] = []
    buffered_parts: List[str] = []
    for part in parts:
        if len(part) > chunk_size:
            if buffered_parts:
                chunks.extend(
                    _merge_splits(buffered_parts, separator=separator, chunk_size=chunk_size, overlap=overlap)
                )
                buffered_parts = []
            chunks.extend(_recursive_split_text(part, chunk_size=chunk_size, overlap=overlap, level=level + 1))
            continue
        buffered_parts.append(part)

    if buffered_parts:
        chunks.extend(_merge_splits(buffered_parts, separator=separator, chunk_size=chunk_size, overlap=overlap))

    return [_normalize_chunk_source(chunk) for chunk in chunks if _normalize_chunk_source(chunk)]


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    normalized = _normalize_chunk_source(text)
    if not normalized:
        return []
    safe_chunk_size = max(int(chunk_size), 1)
    safe_overlap = max(min(int(overlap), safe_chunk_size - 1), 0)
    return _recursive_split_text(normalized, chunk_size=safe_chunk_size, overlap=safe_overlap)


def _normalize_heading(text: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", str(text or "").strip()).strip().lower()


def _looks_like_heading(text: str, font_size: float, median_font_size: float) -> bool:
    cleaned = _normalize_inline_whitespace(text)
    if len(cleaned) > 140 or not cleaned:
        return False
    if SECTION_HEADING_RE.match(cleaned):
        return True
    if font_size >= median_font_size + 1.2 and len(cleaned.split()) <= 12:
        return True
    if cleaned.isupper() and len(cleaned.split()) <= 10:
        return True
    if re.match(r"^\d+(?:\.\d+)*\s+[A-Z][\w-]+", cleaned):
        return True
    return False


def _serialize_document(document: ParsedDocument) -> Dict[str, Any]:
    return asdict(document)


def _serialize_asset(asset: PaperAsset) -> Dict[str, Any]:
    return asdict(asset)


class PaperAcquisitionService:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/123.0 Safari/537.36 ScholarAgent/1.0"
                )
            }
        )

    def fetch(
        self,
        identifier: str,
        *,
        identifier_type: str = "auto",
        prefer: str = "pdf",
        download_dir: str = "",
    ) -> PaperAsset:
        resolved_type = self._resolve_identifier_type(identifier, identifier_type)
        if resolved_type == "arxiv":
            return self._fetch_arxiv(identifier, prefer=prefer, download_dir=download_dir)
        if resolved_type == "doi":
            return self._fetch_doi(identifier, prefer=prefer, download_dir=download_dir)
        if resolved_type == "pmcid":
            return self._fetch_pmc(identifier, prefer=prefer, download_dir=download_dir)
        if resolved_type == "pmid":
            return self._fetch_pubmed(identifier, prefer=prefer, download_dir=download_dir)
        return PaperAsset(
            identifier=identifier,
            source="unknown",
            asset_type=prefer,
            available=False,
            metadata={"reason": f"unsupported_identifier_type:{resolved_type}"},
        )

    def _resolve_identifier_type(self, identifier: str, identifier_type: str) -> str:
        if identifier_type != "auto":
            return identifier_type.strip().lower()
        normalized = identifier.strip()
        if normalized.lower().startswith("10."):
            return "doi"
        if normalized.lower().startswith("pmc"):
            return "pmcid"
        if normalized.isdigit():
            return "pmid"
        return "arxiv"

    def _fetch_arxiv(self, arxiv_id: str, *, prefer: str, download_dir: str) -> PaperAsset:
        normalized = arxiv_id.replace("arXiv:", "").strip()
        pdf_url = f"https://arxiv.org/pdf/{normalized}.pdf"
        html_url = f"https://arxiv.org/abs/{normalized}"
        if prefer == "html":
            content = self._fetch_text(html_url)
            local_path = self._maybe_write_text(download_dir, f"{_slug(normalized)}.html", content)
            return PaperAsset(
                identifier=normalized,
                source="arXiv",
                asset_type="html",
                url=html_url,
                local_path=local_path,
                content=content,
                content_type="text/html",
                available=bool(content),
                metadata={"pdf_url": pdf_url},
            )
        local_path = self._download_binary(pdf_url, download_dir, f"{_slug(normalized)}.pdf")
        return PaperAsset(
            identifier=normalized,
            source="arXiv",
            asset_type="pdf",
            url=pdf_url,
            local_path=local_path,
            content_type="application/pdf",
            available=bool(local_path),
            metadata={"html_url": html_url},
        )

    def _fetch_doi(self, doi: str, *, prefer: str, download_dir: str) -> PaperAsset:
        normalized = doi.replace("https://doi.org/", "").strip()
        metadata = self._resolve_doi_metadata(normalized)
        if prefer == "html":
            html_url = str(metadata.get("html_url") or f"https://doi.org/{normalized}")
            content = self._fetch_text(html_url)
            local_path = self._maybe_write_text(download_dir, f"{_slug(normalized)}.html", content)
            return PaperAsset(
                identifier=normalized,
                source=str(metadata.get("source") or "DOI"),
                asset_type="html",
                url=html_url,
                local_path=local_path,
                content=content,
                content_type="text/html",
                available=bool(content),
                metadata=metadata,
            )
        pdf_url = str(metadata.get("pdf_url") or "")
        if pdf_url:
            local_path = self._download_binary(pdf_url, download_dir, f"{_slug(normalized)}.pdf")
            if local_path:
                return PaperAsset(
                    identifier=normalized,
                    source=str(metadata.get("source") or "DOI"),
                    asset_type="pdf",
                    url=pdf_url,
                    local_path=local_path,
                    content_type="application/pdf",
                    available=True,
                    metadata=metadata,
                )
        return PaperAsset(
            identifier=normalized,
            source=str(metadata.get("source") or "DOI"),
            asset_type="pdf",
            url=str(metadata.get("html_url") or f"https://doi.org/{normalized}"),
            content_type="application/pdf",
            available=False,
            metadata={**metadata, "reason": "pdf_not_resolved"},
        )

    def _fetch_pmc(self, pmcid: str, *, prefer: str, download_dir: str) -> PaperAsset:
        normalized = pmcid.upper().strip()
        html_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{normalized}/"
        pdf_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{normalized}/pdf/"
        if prefer == "html":
            content = self._fetch_text(html_url)
            local_path = self._maybe_write_text(download_dir, f"{_slug(normalized)}.html", content)
            return PaperAsset(
                identifier=normalized,
                source="PMC",
                asset_type="html",
                url=html_url,
                local_path=local_path,
                content=content,
                content_type="text/html",
                available=bool(content),
                metadata={"pdf_url": pdf_url},
            )
        local_path = self._download_binary(pdf_url, download_dir, f"{_slug(normalized)}.pdf")
        return PaperAsset(
            identifier=normalized,
            source="PMC",
            asset_type="pdf",
            url=pdf_url,
            local_path=local_path,
            content_type="application/pdf",
            available=bool(local_path),
            metadata={"html_url": html_url},
        )

    def _fetch_pubmed(self, pmid: str, *, prefer: str, download_dir: str) -> PaperAsset:
        html_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        content = self._fetch_text(html_url)
        local_path = self._maybe_write_text(download_dir, f"pubmed-{pmid}.html", content) if prefer == "html" else ""
        return PaperAsset(
            identifier=pmid,
            source="PubMed",
            asset_type="html" if prefer == "html" else prefer,
            url=html_url,
            local_path=local_path,
            content=content if prefer == "html" else "",
            content_type="text/html" if prefer == "html" else "",
            available=bool(content),
            metadata={"reason": "pubmed_only_html_without_pmc"},
        )

    def _resolve_doi_metadata(self, doi: str) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"doi": doi, "html_url": f"https://doi.org/{doi}"}
        try:
            crossref = self.session.get(
                f"https://doi.org/{doi}",
                headers={"Accept": "application/vnd.citationstyles.csl+json", **self.session.headers},
                timeout=REQUEST_TIMEOUT,
            )
            if crossref.ok:
                payload = crossref.json()
                metadata["title"] = payload.get("title", "")
                metadata["source"] = payload.get("container-title", "")
        except Exception:
            pass
        try:
            openalex = self.session.get(
                "https://api.openalex.org/works",
                params={"filter": f"doi:https://doi.org/{doi}", "per-page": 1},
                timeout=REQUEST_TIMEOUT,
            )
            if openalex.ok:
                payload = openalex.json()
                result = ((payload or {}).get("results") or [{}])[0]
                location = result.get("primary_location") or {}
                metadata["source"] = metadata.get("source") or "OpenAlex"
                metadata["html_url"] = str(location.get("landing_page_url") or metadata["html_url"])
                metadata["pdf_url"] = str(location.get("pdf_url") or "")
                metadata["open_access"] = bool((result.get("open_access") or {}).get("is_oa"))
        except Exception:
            pass
        return metadata

    def _fetch_text(self, url: str) -> str:
        if not url:
            return ""
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except Exception:
            return ""
        return response.text

    def _download_binary(self, url: str, download_dir: str, filename: str) -> str:
        if not url:
            return ""
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except Exception:
            return ""
        target_dir = Path(download_dir).expanduser() if download_dir else Path("cache/papers")
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename
        target_path.write_bytes(response.content)
        return str(target_path)

    def _maybe_write_text(self, download_dir: str, filename: str, content: str) -> str:
        if not download_dir or not content:
            return ""
        target_dir = Path(download_dir).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename
        target_path.write_text(content, encoding="utf-8")
        return str(target_path)


class PDFReadingService:
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
                page_texts.append(_clean_text(page_text))
                full_lines.extend(item["text"] for item in page_lines if item["text"])
                section_candidates.extend(page_candidates)

            full_text = _clean_text("\n\n".join(page_texts))
            sections = self._build_sections(section_candidates, full_lines)
            if target_section:
                sections = self._filter_sections(sections, target_section)

            figures, formulas = self._extract_visual_placeholders(path)
            tables = self._extract_tables(path)
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
                    "chunk_strategy": "recursive_character_splitting",
                },
            )

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
        image_dir = Path(output_dir).expanduser() if output_dir else Path("cache/figures") / _slug(path.stem)
        image_dir.mkdir(parents=True, exist_ok=True)

        figures: List[VisualElement] = []
        formulas: List[VisualElement] = []
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc):
                page_no = page_index + 1
                if target_pages and page_no not in target_pages:
                    continue
                page_text = _clean_text(page.get_text("text"))
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
        tables = self._extract_tables(path, page_numbers=page_numbers, enable_ocr=enable_ocr)
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

    def read_section(
        self,
        pdf_path: str,
        *,
        section_name: str,
        max_chunks: int = 5,
    ) -> Dict[str, Any]:
        document = self.parse_pdf(pdf_path, target_section=section_name)
        selected_sections = document.sections[:max_chunks]
        return {
            "pdf_path": pdf_path,
            "title": document.title,
            "matched_sections": [asdict(item) for item in selected_sections],
            "chunks": document.chunks[:max_chunks],
            "abstract": document.abstract,
            "metadata": document.metadata,
        }

    def _document_id(self, path: Path) -> str:
        digest = md5(str(path).encode("utf-8"), usedforsecurity=False).hexdigest()
        return f"{_slug(path.stem)}-{digest[:12]}"

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
                        "text": _normalize_inline_whitespace(text),
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
            if _looks_like_heading(item["text"], item["font_size"], median_font)
        ]
        return lines, section_candidates

    def _build_sections(self, candidates: List[Dict[str, Any]], full_lines: List[str]) -> List[PaperSection]:
        normalized_lines = [_normalize_inline_whitespace(line) for line in full_lines if _normalize_inline_whitespace(line)]
        if not normalized_lines:
            return []
        headings = {_normalize_inline_whitespace(item["text"]): item for item in candidates}
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
                        text=_clean_text("\n".join(buffer)),
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
                    text=_clean_text("\n".join(buffer)),
                    page_start=current_page,
                    page_end=current_page,
                    metadata={"layout": "single_column"},
                )
            )
        if not sections:
            sections.append(PaperSection(heading="全文", level=1, text=_clean_text("\n".join(normalized_lines))))
        return [section for section in sections if section.text.strip()]

    def _filter_sections(self, sections: List[PaperSection], target_section: str) -> List[PaperSection]:
        target = _normalize_heading(target_section)
        matched = [section for section in sections if target in _normalize_heading(section.heading)]
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
            if any(alias in _normalize_heading(section.heading) for alias in aliases)
        ]

    def _build_chunks_from_sections(self, sections: List[PaperSection], *, chunk_size: int, overlap: int) -> List[str]:
        chunks: List[str] = []
        for section in sections:
            section_chunks = _chunk_text(section.text, chunk_size=chunk_size, overlap=overlap)
            if section_chunks:
                chunks.extend([f"[{section.heading}] {chunk}" for chunk in section_chunks])
        return chunks

    def _infer_title(self, page_texts: List[str]) -> str:
        first_page = page_texts[0] if page_texts else ""
        for line in first_page.splitlines():
            cleaned = _normalize_inline_whitespace(line)
            if 6 <= len(cleaned) <= 220 and not SECTION_HEADING_RE.match(cleaned):
                return cleaned
        return ""

    def _infer_abstract(self, full_text: str, sections: List[PaperSection]) -> str:
        for section in sections:
            if _normalize_heading(section.heading).startswith("abstract"):
                return section.text[:2000]
        lowered = full_text.lower()
        marker = lowered.find("abstract")
        if marker >= 0:
            return full_text[marker : marker + 2000]
        return full_text[:1200]

    def _extract_visual_placeholders(self, path: Path) -> tuple[List[VisualElement], List[VisualElement]]:
        figures: List[VisualElement] = []
        formulas: List[VisualElement] = []
        if fitz is None:
            return figures, formulas
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc):
                page_no = page_index + 1
                page_text = _clean_text(page.get_text("text"))
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

    def _extract_tables(
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
        cleaned_rows = [[_normalize_inline_whitespace(str(cell or "")) for cell in row] for row in rows]
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
            _normalize_inline_whitespace(line)
            for line in page_text.splitlines()
            if pattern.match(_normalize_inline_whitespace(line))
        ]

    def _extract_formula_lines(self, page_text: str, page_no: int) -> List[VisualElement]:
        formulas: List[VisualElement] = []
        for line in page_text.splitlines():
            cleaned = _normalize_inline_whitespace(line)
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
                return _clean_text(pytesseract.image_to_string(image))
        except Exception:
            return ""

    def _table_like_markdown(self, text: str) -> str:
        lines = [_normalize_inline_whitespace(line) for line in text.splitlines() if _normalize_inline_whitespace(line)]
        if len(lines) < 2:
            return ""
        rows = [re.split(r"\s{2,}", line) for line in lines]
        max_columns = max(len(row) for row in rows)
        if max_columns < 2:
            return ""
        normalized_rows = [row + [""] * (max_columns - len(row)) for row in rows[:8]]
        return self._rows_to_markdown(normalized_rows)


DOCUMENT_SERVICE = PDFReadingService()
ACQUISITION_SERVICE = PaperAcquisitionService()


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
        ACQUISITION_SERVICE.fetch(
            identifier,
            identifier_type=identifier_type,
            prefer=prefer,
            download_dir=download_dir,
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
    return _serialize_document(DOCUMENT_SERVICE.parse_pdf(pdf_path, target_section=target_section))


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
    return DOCUMENT_SERVICE.extract_visuals(pdf_path, page_numbers=page_numbers, output_dir=output_dir)


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
    return DOCUMENT_SERVICE.read_section(pdf_path, section_name=section_name, max_chunks=max_chunks)


def fetch_paper(identifier: str, *, identifier_type: str = "auto", prefer: str = "pdf", download_dir: str = "") -> PaperAsset:
    return ACQUISITION_SERVICE.fetch(identifier, identifier_type=identifier_type, prefer=prefer, download_dir=download_dir)


def parse_pdf(pdf_path: str, *, target_section: str = "") -> ParsedDocument:
    return DOCUMENT_SERVICE.parse_pdf(pdf_path, target_section=target_section)
