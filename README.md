# ScholarAgent

ScholarAgent 是一个面向学术研究场景的智能助手，支持多源论文检索、论文分析、综述生成、代码生成、长期记忆、白盒追踪和 Gradio/CLI 双界面。

## 快速开始

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

## 主要能力

- 多源学术检索：arXiv、OpenAlex、Semantic Scholar
- Multi-Agent 协作：Search / Analyze / Debate / Write / Coder
- 任务分层与 LLM 分级调度
- RAG v3 风格查询链路：改写、路由、混合检索、融合、重排、验证
- 长期记忆、反馈学习、白盒追踪

## 文档

- 原始项目文档：`ScholarAgent_完整项目文档.md`
- RAG 流程图：`RAG_v3_完整流程图.html`
