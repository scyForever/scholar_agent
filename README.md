# ScholarAgent

ScholarAgent 是一个面向学术研究场景的多智能体助手。当前版本已经把 `tools -> skills -> agents -> runtime` 这条链路打通，支持多源学术检索、论文获取、PDF 精读、本地 RAG、研究规划、研究记忆和可观测执行追踪，并同时提供 Web 界面与 CLI 入口。

## 1. 当前能力

- 多源学术检索：支持 `arXiv`、`OpenAlex`、`Semantic Scholar`、`Web of Science Starter API`、`PubMed`、`IEEE Xplore`、`Google Scholar` 的统一检索与聚合。
- 论文获取：支持按 `DOI`、`arXiv ID`、`PMID`、`PMCID` 获取论文 PDF 或 HTML。
- PDF 解析：支持正文抽取、章节识别、双栏版面感知、表格抽取、公式行提取和定向章节阅读。
- OCR 与视觉提取：支持论文图片导出、基础 OCR、公式/表格候选内容提取，并输出 LaTeX 或 Markdown 近似表示。
- 本地 RAG：支持 PDF 建库到 `SQLite + Chroma`，检索链路为“对话增强 -> 查询改写 -> 路由 -> 词法召回 + 稠密召回 -> RRF 融合 -> BGE 重排 -> 相关性校验 -> 补充检索”。
- 多智能体协作：主链路包含 `search / analyze / debate / write / coder`，并补充 `research planner / research search / research reading / research memory`。
- 研究规划：可把“大主题”拆解为检索、筛选、精读、综合、成文等多阶段任务。
- 持久记忆：可记住用户偏好、读过的论文与研究笔记，降低重复推荐。
- 可观测性：每次对话都会记录 trace，前端可查看执行时间线、阶段摘要和模型调用信息。

## 2. 当前架构

当前实现按四层组织：

1. `tools`
   负责外部能力接入，例如学术搜索、论文获取、PDF 解析、OCR、网页补充检索。
2. `skills`
   负责把原子工具组合成研究动作，例如文献搜索、深度阅读、研究规划、研究记忆。
3. `agents`
   负责按角色编排执行，例如 `SearchAgent`、`AnalyzeAgent`、`ResearchReadingAgent`。
4. `runtime`
   由 `AgentV2 + RuntimeGraph + MultiAgentCoordinator` 统一调度意图识别、任务规划、RAG、写作与质量增强。

高层入口如下：

- 对话主入口：`src/core/agent_v2.py`
- 运行图：`src/core/runtime_graph.py`
- 多代理编排：`src/agents/multi_agent.py`
- 研究层技能：`src/skills/research_skills.py`
- 学术搜索工具：`src/tools/research_search_tool.py`
- 文档获取与解析工具：`src/tools/research_document_tool.py`
- 本地 RAG：`src/rag/retriever.py`

## 3. 快速开始

### 3.1 环境准备

推荐直接使用现有 `conda` 环境：

```bash
source /home/a1/miniconda3/etc/profile.d/conda.sh
conda activate agent
pip install -r requirements.txt
```

如果要新建虚拟环境：

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3.2 OCR 依赖

项目的 OCR 依赖分两部分：

- Python 包：`pytesseract`
- 系统可执行文件：`tesseract`

要求：

- `pip install -r requirements.txt` 会安装 `pytesseract`
- 还需要本机存在 `tesseract` 命令

检查方式：

```bash
tesseract --version
```

如果你使用 conda，可参考：

```bash
conda install -c conda-forge tesseract
```

### 3.3 最小配置

项目在没有真实 LLM provider 时也能运行，但会退化为本地规则/启发式链路。建议至少配置一组真实 provider：

- `SCNET_API_KEY`
- `SILICONFLOW_API_KEY`
- `ZHIPU_API_KEY`
- `DEEPSEEK_API_KEY`
- `DASHSCOPE_API_KEY`

学术源相关可选配置：

- `WOS_STARTER_API_KEY`
- `IEEE_XPLORE_API_KEY`
- `SERPAPI_API_KEY`
- `NCBI_API_KEY`

本地 RAG 模型相关配置：

```bash
export BGE_M3_MODEL_PATH="/你的本地/bge-m3"
export BGE_RERANKER_MODEL_PATH="/你的本地/bge-reranker-v2-m3"
export RAG_VECTOR_COLLECTION="rag_chunks"
export RAG_PARALLEL_WORKERS="8"
```

### 3.4 启动项目

```bash
python run.py
```

启动后可选择：

1. Web 界面
2. 命令行
3. 功能验证

## 4. 验证命令

基础回归：

```bash
python verify_features.py
```

这个脚本会检查：

- Prompt/Trace/Memory 基础模块
- `AgentV2` 核心入口
- 研究规划接口
- 论文获取接口
- 统一工具注册与白名单

搜索 agent 路径验证：

```bash
python verify_agentic_search.py
```

这个脚本用于验证：

- 查询改写是否正常
- 外部学术搜索工具能否返回结果
- `LangChain agent` 搜索规划是否命中

说明：

- 该脚本依赖至少一个真实可用的 provider
- 搜索规划当前优先尝试 `zhipu`
- 若没有已验证成功的规划 provider，运行时会自动降级为确定性搜索，不会阻塞整条链路

## 5. 使用示例

### 5.1 命令行问答

```text
搜索近三年关于多智能体强化学习的论文
```

```text
写一篇关于大模型幻觉的综述
```

```text
总结这篇论文的方法、实验设计和局限
```

### 5.2 代码方式调用

```python
from src.core.agent_v2 import AgentV2

agent = AgentV2()

plan = agent.plan_research(
    "写一篇关于大模型幻觉的综述",
    slots={"time_range": "2023-2024"},
)

asset = agent.fetch_paper("2401.14805", identifier_type="arxiv", prefer="pdf")

document = agent.read_paper(asset["local_path"], target_section="method")

response = agent.chat("搜索近三年关于多智能体强化学习的论文", session_id="demo")
print(response.answer)
```

## 6. 本地 RAG 完整流程

### 6.1 建库流程

1. `AgentV2.index_pdf()` 调用 `HybridRetriever.index_pdf()`
2. `pdf_tool.extract_pdf_text()` 调用 `research_document_tool` 解析 PDF
3. 提取正文块、表格块、QA 块、知识关系块
4. 写入 `data/memory/rag_index.db`
5. 同步写入 `data/vector_db/<collection>`

### 6.2 检索流程

1. `chat()` 先执行记忆召回、意图识别、槽位填充、任务规划
2. `SearchAgent` 先查本地 RAG
3. 检索器执行对话增强、查询改写和路由
4. 按 `text_chunk / table_chunk / qa_chunk / kg_chunk` 并行检索
5. 词法侧使用 `TF-IDF + BM25`，向量侧使用 `BGE-M3 + Chroma`
6. 结果经 `RRF` 融合后再由 `bge-reranker-v2-m3` 重排
7. 通过相关性判断做 CRAG 风格校验
8. 若本地结果不足，再补外部网页或学术搜索结果

### 6.3 写作流程

写作阶段会同时消费：

- 本地 RAG 命中的片段
- 外部学术搜索结果
- 论文分析结果
- 研究计划
- 历史记忆

最终由 `write` 节点生成回答；如果是 `FULL` 模式，还会再做一轮质量增强。

## 7. 数据落盘

- Trace：`logs/traces/`
- 长期记忆：`data/memory/memory.db`
- 本地 RAG 元数据：`data/memory/rag_index.db`
- 本地向量库：`data/vector_db/`
- 白名单：`data/whitelist.json`
- 反馈：`data/feedback/feedback.jsonl`
- 下载论文与中间文件：`cache/`

## 8. 当前边界

- 外部搜索得到的新论文不会自动写回本地 RAG，当前仍然是“外部检索”和“本地知识库”分层。
- 搜索规划不是始终强依赖远程模型。系统会优先尝试已验证成功的 provider；若不可用，则退回确定性搜索。
- 当前搜索规划阶段只把 `zhipu` 视为 agentic planning provider；未验证成功时直接退回确定性策略。
- OCR 已可用，但当前还是 `tesseract + pytesseract` 的基础能力，对复杂公式、扫描件和高密度图表的精度有限。
- `Google Scholar` 和 `IEEE Xplore` 依赖外部 key 与接口可用性，可能返回空结果。
- 没有真实 provider 时，项目仍可运行，但意图识别、查询改写、分析写作的效果会明显下降。

## 9. 文档索引

- 快速开始：`docs/QUICKSTART.md`
- 文档总览：`docs/PROJECT_DOCUMENTATION.md`
- 完整项目文档：`ScholarAgent_完整项目文档.md`
- 面试速记版：`docs/INTERVIEW_GUIDE.md`
- 完整文档入口：`docs/COMPLETE_PROJECT_DOCUMENTATION.md`
- RAG 可视化流程图：`RAG_v3_完整流程图.html`
