# 完整项目文档入口

完整设计说明以根目录文档为准：

- `../ScholarAgent_完整项目文档.md`

## 1. 建议优先看的章节

### 如果你关心整体架构

优先看：

- 项目定位与目标
- 系统分层架构
- 端到端主链路

### 如果你关心本地 RAG

优先看：

- 本地 RAG 建库流程
- 本地 RAG 检索流程
- `HybridRetriever` 兼容入口与 `src/rag/components/` 组件拆分
- 数据落盘与索引结构

### 如果你关心研究场景能力

优先看：

- 统一学术搜索层
- 论文获取与深度阅读
- `src/skills/components/`、`src/tools/search_components/`、`src/tools/document_components/`
- 研究规划与研究记忆

### 如果你关心运行与验证

优先看：

- 运行方式
- 验证脚本
- 外部依赖与配置项

## 2. 当前实现要点速览

- 搜索工具层已统一接入 `arXiv / OpenAlex / Semantic Scholar / Web of Science / PubMed / IEEE Xplore / Google Scholar`，兼容入口位于 `research_search_tool.py`，底层组件位于 `src/tools/search_components/`
- 文档层支持 `DOI / arXiv ID / PMID / PMCID` 获取，以及 PDF 章节解析和 OCR；兼容入口位于 `research_document_tool.py`，底层组件位于 `src/tools/document_components/`
- `SearchAgent` 会先尝试本地 RAG，再补外部学术搜索
- 本地 RAG 对外仍通过 `HybridRetriever` 暴露，但索引和检索已经拆成 `src/rag/components/indexing.py`、`src/rag/components/retrieval.py` 与 `src/rag/components/retrieval_components/`
- `explain_concept` 可复用当前会话最近一次检索结果，并支持 `只用RAG / 不要检索` 控制
- 搜索规划优先尝试已验证成功的 `zhipu`；不可用时退回确定性搜索
- 本地 RAG 使用 `TF-IDF + BM25 + BGE-M3 + RRF + BGE Reranker + 相关性校验`
- 研究层增加了 `ResearchPlannerAgent / ResearchSearchAgent / ResearchReadingAgent / ResearchMemoryAgent`，其技能实现由 `ResearchSkillsHarness + src/skills/components/` 统一收口

## 3. 配套文档

- 主说明文档：`../README.md`
- 快速开始：`./QUICKSTART.md`
- 面试速记版：`./INTERVIEW_GUIDE.md`
- 文档总览：`./PROJECT_DOCUMENTATION.md`
- RAG 可视化流程图：`../RAG_v3_完整流程图.html`
