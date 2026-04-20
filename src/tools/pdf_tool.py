from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from config.settings import settings
from src.tools.registry import ToolDefinition, ToolParameter, register_tool
from src.tools.research_document_tool import parse_pdf


@register_tool(
    ToolDefinition(
        name="extract_pdf_text",
        description="Extract text, tables and chunks from a PDF.",
        parameters=[ToolParameter("pdf_path", "str", "PDF路径", True)],
        tags=["pdf", "extract"],
    )
)
def extract_pdf_text(pdf_path: str) -> Dict[str, object]:
    parsed = parse_pdf(pdf_path)
    chunks = parsed.chunks or []
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
        "pdf_path": parsed.source_path,
        "text": parsed.full_text,
        "chunks": chunks,
        "tables": [item.markdown or item.text for item in parsed.tables if item.markdown or item.text],
        "images": [
            {
                "page": item.page,
                "index": index,
                "description": item.caption or f"Image extracted from page {item.page}.",
                "image_path": item.image_path,
            }
            for index, item in enumerate(parsed.figures)
        ],
        "sections": [asdict(item) for item in parsed.sections],
        "formulas": [asdict(item) for item in parsed.formulas],
        "metadata": parsed.metadata,
        "qa_pairs": qa_pairs,
    }
