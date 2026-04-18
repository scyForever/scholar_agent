from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_research_document_tool():
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(ROOT / "src")]
    sys.modules.setdefault("src", src_pkg)

    core_pkg = types.ModuleType("src.core")
    core_pkg.__path__ = [str(ROOT / "src" / "core")]
    sys.modules.setdefault("src.core", core_pkg)

    tools_pkg = types.ModuleType("src.tools")
    tools_pkg.__path__ = [str(ROOT / "src" / "tools")]
    sys.modules.setdefault("src.tools", tools_pkg)

    for module_name, relative_path in [
        ("src.core.models", "src/core/models.py"),
        ("src.tools.registry", "src/tools/registry.py"),
        ("src.tools.research_document_tool", "src/tools/research_document_tool.py"),
    ]:
        if module_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return sys.modules["src.tools.research_document_tool"]


class ResearchDocumentChunkingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_research_document_tool()

    def test_paragraph_boundary_is_preferred(self) -> None:
        text = "A" * 30 + "\n\n" + "B" * 30

        chunks = self.module._chunk_text(text, chunk_size=40, overlap=0)

        self.assertEqual(chunks, ["A" * 30, "B" * 30])

    def test_sentence_boundary_is_used_when_paragraph_is_too_large(self) -> None:
        text = "Sentence alpha long. Sentence beta long. Sentence gamma long."

        chunks = self.module._chunk_text(text, chunk_size=25, overlap=0)

        self.assertEqual(
            chunks,
            ["Sentence alpha long.", "Sentence beta long.", "Sentence gamma long."],
        )

    def test_character_level_overlap_is_used_for_long_token(self) -> None:
        text = "0123456789" * 5

        chunks = self.module._chunk_text(text, chunk_size=20, overlap=5)

        self.assertEqual([len(chunk) for chunk in chunks], [20, 20, 20])
        self.assertTrue(chunks[1].startswith(chunks[0][-5:]))


if __name__ == "__main__":
    unittest.main()
