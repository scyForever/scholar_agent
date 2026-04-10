# 项目文档索引

当前文档已经按最新实现同步，核心口径如下：

- 架构对外建议表述为 `runtime -> agents -> skills -> tools/rag`，主执行链路由 `harness` 统一收口
- `skills / tools / rag` 三层已完成入口 harness 化，底层实现进一步拆成独立组件包
- 本地 RAG 基于 `SQLite + Chroma + BGE-M3 + BGE Reranker`
- 研究层新增了论文获取、深度阅读、研究规划、研究记忆
- OCR 已接入文档解析链路
- 搜索规划优先尝试已验证成功的 `zhipu`，否则自动降级为确定性搜索
- `explain_concept` 已支持会话内复用最近一次检索结果，并支持 `只用RAG / 不要检索` 这类显式控制

## 1. 文档阅读入口

### 1.1 主说明文档

- `../README.md`

适合：

- 快速了解项目是什么
- 查看能力边界、配置项和运行方式

### 1.2 快速开始

- `./QUICKSTART.md`

适合：

- 第一次运行项目
- 快速检查依赖、OCR 和验证脚本

### 1.3 完整项目文档

- `../ScholarAgent_完整项目文档.md`

适合：

- 系统理解整体架构
- 追踪从检索到写作的完整链路
- 查阅核心模块与数据流

### 1.4 完整文档入口

- `./COMPLETE_PROJECT_DOCUMENTATION.md`

适合：

- 快速定位完整文档中该看哪一章
- 从主题维度查找阅读路径

### 1.5 面试速记版

- `./INTERVIEW_GUIDE.md`

适合：

- 项目汇报
- 面试问答准备
- 快速复述设计取舍

### 1.6 可视化补充材料

- `../RAG_v3_完整流程图.html`

适合：

- 图形化理解本地 RAG 链路

## 2. 推荐阅读顺序

1. 先看 `../README.md`
2. 再看 `./QUICKSTART.md`
3. 然后看 `../ScholarAgent_完整项目文档.md`
4. 最后按需看 `./INTERVIEW_GUIDE.md`

## 3. 当前实现的几个关键提醒

- `research_search_tool.py`、`research_document_tool.py`、`rag/retriever.py` 现在主要是兼容入口，真实底层逻辑分别位于 `src/tools/search_components/`、`src/tools/document_components/`、`src/rag/components/`
- `src/skills/research_skills.py` 现在主要是兼容外观层，研究规划、搜索、阅读、记忆的具体实现位于 `src/skills/components/`
- 本地 RAG 和外部学术搜索是协同关系，不是同一个存储层
- 外部检索结果不会自动写入本地向量库
- 搜索规划不是固定依赖远程 LLM；远程不可用时会退回本地确定性策略
- OCR 已接入，但当前仍是基础 OCR，不等于高精度公式识别系统
- 文档解析层支持章节、表格、公式和图片抽取，但当前向量检索主链路仍以文本、表格、QA、KG 为主
- 上轮检索结果复用目前是会话内内存级能力，不是长期持久化能力

## 4. 后续维护建议

如果后续实现再变化，建议优先更新顺序：

1. `README.md`
2. `docs/QUICKSTART.md`
3. `ScholarAgent_完整项目文档.md`
4. `docs/INTERVIEW_GUIDE.md`
5. `docs/COMPLETE_PROJECT_DOCUMENTATION.md`
