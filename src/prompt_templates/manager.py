from __future__ import annotations

from pathlib import Path
from string import Formatter
from typing import Dict, Iterable

from config.settings import settings


DEFAULT_TEMPLATES: Dict[str, str] = {
    "intent_classification": (
        "你是学术助手的意图分类器。候选意图：{intent_options}。\n"
        "请阅读用户输入并返回JSON：{{\"intent\": \"...\", \"confidence\": 0-1, \"reason\": \"...\"}}。\n"
        "用户输入：{query}"
    ),
    "paper_analysis": (
        "请分析下列论文，输出四部分：摘要、核心贡献、方法、局限性。\n"
        "标题：{title}\n摘要：{abstract}\n额外上下文：{context}"
    ),
    "survey_writer": (
        "请根据给定材料生成一篇结构化综述，包含标题、摘要、正文和参考文献。\n"
        "主题：{topic}\n材料：{materials}"
    ),
    "paper_answer_writer": (
        "请基于给定材料回答用户关于单篇论文的问题。\n"
        "要求：直接回答这篇论文讲了什么、提出了什么新方法、核心贡献、关键实验结果和局限；"
        "不要写成领域综述，不要扩展到无关论文。\n"
        "问题：{topic}\n材料：{materials}"
    ),
    "compare_writer": (
        "请根据给定材料比较相关方法或论文。\n"
        "要求：按研究目标、核心方法、优缺点、适用场景四个维度组织答案，最后给出综合判断。\n"
        "问题：{topic}\n材料：{materials}"
    ),
    "concept_writer": (
        "请根据给定材料解释学术概念或方法。\n"
        "要求：先解释定义，再说明核心机制、典型应用和与相近概念的区别。\n"
        "问题：{topic}\n材料：{materials}"
    ),
    "daily_update_writer": (
        "请根据给定材料生成最近研究动态总结。\n"
        "要求：按时间或主题归纳最新进展，突出新增方法、代表论文和趋势判断。\n"
        "主题：{topic}\n材料：{materials}"
    ),
    "debate": (
        "请围绕问题进行正反视角分析，然后给出综合结论。\n"
        "问题：{question}\n材料：{materials}"
    ),
    "code_generation": (
        "请根据研究主题与方法生成可执行代码或伪代码，并说明关键设计。\n"
        "主题：{topic}\n参考材料：{materials}"
    ),
}


class PromptTemplateManager:
    def __init__(self, prompt_dir: Path | None = None) -> None:
        self.prompt_dir = prompt_dir or settings.prompt_dir
        self.prompt_dir.mkdir(parents=True, exist_ok=True)

    def ensure_default_templates(self) -> None:
        for name, template in DEFAULT_TEMPLATES.items():
            path = self.prompt_dir / f"{name}.txt"
            if not path.exists():
                path.write_text(template, encoding="utf-8")

    def list_templates(self) -> Iterable[str]:
        return sorted(path.stem for path in self.prompt_dir.glob("*.txt"))

    def load(self, name: str) -> str:
        self.ensure_default_templates()
        path = self.prompt_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {name}")
        return path.read_text(encoding="utf-8")

    def render(self, name: str, **kwargs: object) -> str:
        template = self.load(name)
        expected = {field for _, field, _, _ in Formatter().parse(template) if field}
        payload = {key: kwargs.get(key, "") for key in expected}
        return template.format(**payload)

    def save(self, name: str, content: str) -> None:
        (self.prompt_dir / f"{name}.txt").write_text(content, encoding="utf-8")
