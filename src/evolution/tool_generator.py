from __future__ import annotations

from pathlib import Path

from config.settings import settings
from src.core.llm import LLMManager


class ToolGenerator:
    def __init__(self, llm: LLMManager | None = None) -> None:
        self.llm = llm or LLMManager()

    def generate_tool(self, tool_name: str, description: str, output_dir: Path | None = None) -> Path:
        target_dir = output_dir or (settings.base_dir / "src" / "tools")
        target_dir.mkdir(parents=True, exist_ok=True)
        prompt = (
            "请生成一个 Python 工具函数文件，包含函数定义、文档字符串和基本参数校验。"
            f"\n工具名：{tool_name}\n描述：{description}"
        )
        code = self.llm.call(prompt)
        target_path = target_dir / f"{tool_name}.py"
        template = (
            f'"""Auto-generated tool: {tool_name}."""\n\n'
            "from __future__ import annotations\n\n\n"
            f"# Original generation summary:\n# {code[:400].replace(chr(10), ' ')}\n"
        )
        target_path.write_text(template, encoding="utf-8")
        return target_path
