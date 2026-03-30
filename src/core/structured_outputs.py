from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator

from config.intent_config import INTENT_SPECS


class IntentClassificationOutput(BaseModel):
    intent: str = Field(description="候选意图中的一个合法值。")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="意图判断置信度，范围 0 到 1。")
    reason: str = Field(default="", description="给出该意图判断的简要理由。")

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if normalized and normalized not in INTENT_SPECS:
            raise ValueError(f"Unsupported intent: {normalized}")
        return normalized


class QueryRewriteOutput(BaseModel):
    core_topic: str = Field(default="", description="提炼后的核心研究主题。")
    english_query: str = Field(default="", description="用于英文数据库检索的主检索式。")
    external_queries: List[str] = Field(default_factory=list, description="面向外部学术搜索引擎的检索式列表。")
    local_queries: List[str] = Field(default_factory=list, description="面向本地 RAG 的检索式列表。")


class SearchAgentExecutionStep(BaseModel):
    tool_name: str = Field(default="", description="计划使用或实际使用的检索工具名称。")
    query: str = Field(default="", description="该步骤对应的检索式。")
    purpose: str = Field(default="", description="为什么在该步骤调用这个工具。")


class SearchAgentAggregationOutput(BaseModel):
    summary: str = Field(default="", description="对聚合检索结果的总体总结。")
    key_findings: List[str] = Field(default_factory=list, description="从聚合结果中提炼出的关键发现。")
    representative_titles: List[str] = Field(
        default_factory=list,
        description="最具代表性的论文标题列表，只能填写真实返回结果中的标题。",
    )


class SearchAgentFinalOutput(BaseModel):
    selected_tools: List[str] = Field(default_factory=list, description="最终选择的检索工具列表。")
    tool_selection_reason: str = Field(default="", description="整体工具选择理由。")
    execution_plan: List[SearchAgentExecutionStep] = Field(
        default_factory=list,
        description="计划或已执行的检索步骤列表。",
    )
    aggregation: SearchAgentAggregationOutput = Field(
        default_factory=SearchAgentAggregationOutput,
        description="对聚合检索结果的结构化总结。",
    )
