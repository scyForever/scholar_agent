from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from config.settings import settings
from src.tools.registry import ToolDefinition, ToolParameter, register_tool


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


@register_tool(
    ToolDefinition(
        name="extract_pdf_text",
        description="Extract text, tables and chunks from a PDF.",
        parameters=[ToolParameter("pdf_path", "str", "PDF路径", True)],
        tags=["pdf", "extract"],
    )
)
def extract_pdf_text(pdf_path: str) -> Dict[str, object]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text_parts: List[str] = []
    tables: List[str] = []
    images: List[Dict[str, object]] = []

    try:
        import fitz
    except ImportError:  # pragma: no cover
        fitz = None

    if fitz is not None:
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc):
                page_text = page.get_text("text")
                if page_text:
                    text_parts.append(page_text)
                for image_index, _ in enumerate(page.get_images(full=True)):
                    images.append(
                        {
                            "page": page_index + 1,
                            "index": image_index,
                            "description": f"Image extracted from page {page_index + 1}.",
                        }
                    )

    try:
        import pdfplumber
    except ImportError:  # pragma: no cover
        pdfplumber = None

    if pdfplumber is not None:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables() or []:
                    rows = [" | ".join(cell or "" for cell in row) for row in table]
                    tables.append("\n".join(rows))

    full_text = "\n".join(text_parts)
    chunks = _chunk_text(full_text, settings.rag_chunk_size, settings.rag_chunk_overlap)
    qa_pairs = []
    for idx, chunk in enumerate(chunks[: min(10, len(chunks))]):
        sentence = chunk.split("。")[0].split(".")[0].strip()
        if sentence:
            qa_pairs.append(
                {
                    "question": f"第 {idx + 1} 个文本块主要讨论什么？",
                    "answer": sentence,
                }
            )

    return {
        "pdf_path": str(path),
        "text": full_text,
        "chunks": chunks,
        "tables": tables,
        "images": images,
        "qa_pairs": qa_pairs,
    }
