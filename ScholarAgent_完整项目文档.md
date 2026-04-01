# ScholarAgent 完整项目文档

> 面向学术研究场景的多智能体研究助手，覆盖文献检索、论文获取、PDF 精读、本地 RAG、研究规划与研究记忆。

---

## 目录

1. 项目定位
2. 当前功能概览
3. 系统分层架构
4. 端到端执行链路
5. 本地 RAG 设计
6. 研究层能力
7. 关键模块与目录
8. 配置、依赖与运行
9. 数据落盘与可观测性
10. 当前边界与后续方向

---

## 1. 项目定位

ScholarAgent 的目标不是只做“论文搜索框”，而是提供一条完整的研究工作链路：

- 检索候选论文
- 获取论文全文
- 解析 PDF 与关键章节
- 抽取图表、公式、表格
- 建立本地知识库
- 在问答、综述和方法对比任务中复用这些材料
- 记住用户偏好和已读论文，降低重复劳动

当前版本已经把这条链路拆分为 `tools -> skills -> agents -> runtime` 四层，便于维护、验证和扩展。

## 2. 当前功能概览

### 2.1 学术搜索

统一搜索层位于 `src/tools/research_search_tool.py`，支持：

- `arXiv`
- `OpenAlex`
- `Semantic Scholar`
- `Web of Science Starter API`
- `PubMed`
- `IEEE Xplore`
- `Google Scholar`

输出统一为 `Paper` 数据结构，包含：

- 标题
- 摘要
- 作者
- 年份
- 来源
- 引用数
- DOI / arXiv ID / PMID / PMCID
- PDF / HTML / Full Text URL

### 2.2 论文获取

文档获取层位于 `src/tools/research_document_tool.py`，支持：

- 按 `DOI` 获取 HTML 或 PDF
- 按 `arXiv ID` 获取 HTML 或 PDF
- 按 `PMID` 获取 PubMed HTML
- 按 `PMCID` 获取 PMC HTML 或 PDF

返回统一为 `PaperAsset`。

### 2.3 PDF 解析与深度阅读

当前解析能力包括：

- 全文提取
- 章节识别
- 双栏版面识别
- 表格提取
- 公式行提取
- 图像导出
- 基础 OCR
- 按章节定向阅读

解析结果统一为 `ParsedDocument`。

### 2.4 本地 RAG

当前本地 RAG 由 `src/rag/retriever.py` 负责，特征如下：

- 元数据索引：`SQLite`
- 向量索引：`Chroma`
- 向量模型：`BGE-M3`
- 重排序模型：`bge-reranker-v2-m3`
- 召回类型：`text_chunk / table_chunk / qa_chunk / kg_chunk`
- 召回策略：`TF-IDF + BM25 + 稠密检索 + RRF + rerank + 相关性校验`

### 2.5 研究层能力

当前新增了研究场景专用 skill 与 agent：

- `ResearchPlanningSkill`：拆解研究任务
- `LiteratureSearchSkill`：统一多源文献搜索
- `DeepReadingSkill`：论文获取、解析、定向精读、图表提取
- `ResearchMemorySkill`：保存偏好、读过的论文和研究笔记

对应 agent：

- `ResearchPlannerAgent`
- `ResearchSearchAgent`
- `ResearchReadingAgent`
- `ResearchMemoryAgent`

## 3. 系统分层架构

### 3.1 总体分层

```text
用户输入
  -> AgentV2
    -> 预处理层（记忆 / 意图 / 槽位 / 规划）
    -> RuntimeGraph
      -> MultiAgentCoordinator
        -> Search / Analyze / Debate / Write / Coder
      -> ReasoningEngine
      -> QualityEnhancer
```

并行的研究层能力作为专门服务存在：

```text
ResearchSkillset
  -> search
  -> reading
  -> planning
  -> memory
```

### 3.2 各层职责

#### tools

原子能力接入层，负责：

- 学术搜索
- 论文抓取
- PDF 解析
- OCR
- 网页补充搜索

#### skills

研究动作封装层，负责把多个工具拼成可复用流程，例如：

- 统一文献搜索
- 深度阅读
- 研究任务拆解
- 研究记忆排序与回忆

#### agents

角色执行层，负责按任务类型组合 skill 和 prompt，例如：

- 搜索代理
- 分析代理
- 写作代理
- 研究规划代理
- 深度阅读代理

#### runtime

总控与调度层，负责：

- 会话状态
- 任务规划
- runtime graph
- trace
- 质量增强

## 4. 端到端执行链路

### 4.1 chat 主流程

`AgentV2.chat()` 的当前流程是：

1. 召回相关历史记忆
2. 做意图识别
3. 填充槽位
4. 生成任务配置
5. 把状态交给 `RuntimeGraph`
6. `RuntimeGraph` 调度 `multi_agent -> reasoning -> quality`
7. 落盘 trace
8. 把本轮对话写入长期记忆

### 4.2 SearchAgent 的当前流程

`SearchAgent` 的处理逻辑是：

1. 根据槽位确定检索主题、时间范围和结果上限
2. 生成统一查询改写计划
3. 先调用本地 RAG
4. 若当前意图是 `analyze_paper` 且本地已命中，直接走 `local_rag_only`
5. 否则再补外部学术搜索
6. 聚合、去重、排序
7. 用研究记忆把“未读论文”排在前面

### 4.3 查询改写与降级逻辑

当前查询改写器支持两条路径：

- 有已验证成功的 provider 时：走结构化改写
- 没有已验证成功的 provider 时：走本地启发式改写

启发式改写会做：

- 中文主题提炼
- 英文学术术语翻译
- survey / comparison 扩展
- 本地 RAG 查询和外部检索查询分流

### 4.4 搜索规划逻辑

当前搜索规划并不是“无脑固定用远程 LLM”。实际逻辑是：

1. 只把已验证成功的 `zhipu` 视为 agentic planning provider
2. 如果满足 `LangChain agent + 工具 + provider` 条件，就尝试 agentic search
3. 否则直接走确定性多源搜索

这样做的原因是：

- 避免 provider 不稳定导致整条链路阻塞
- 避免不同 provider 在 tool-calling 兼容性上的不一致

## 5. 本地 RAG 设计

### 5.1 建库流程

PDF 建库入口：

```text
AgentV2.index_pdf
  -> HybridRetriever.index_pdf
    -> pdf_tool.extract_pdf_text
      -> research_document_tool.parse_pdf
```

建库时会生成：

- `text_chunk`
- `table_chunk`
- `qa_chunk`
- `kg_chunk`

然后写入：

- `data/memory/rag_index.db`
- `data/vector_db/<collection>`

### 5.2 检索流程

检索链路如下：

1. 对话增强
2. 查询改写
3. 路由到 `text/table/qa/kg`
4. 并行执行检索
5. 词法支路：`TF-IDF + BM25`
6. 稠密支路：`BGE-M3 + Chroma`
7. `RRF` 融合
8. `BGE Reranker` 重排
9. 相关性判断
10. 若高质量结果不足，则补 `search_web`

### 5.3 为什么这样设计

原因是当前研究场景里不同问题对证据类型的要求不同：

- 方法细节更适合查 `text_chunk`
- benchmark 和指标更适合查 `table_chunk`
- “为什么/如何”类问题更适合查 `qa_chunk`
- “关系/演化”类问题更适合查 `kg_chunk`

因此，当前实现不是“一套 embedding 查到底”，而是先路由，再并行检索。

### 5.4 当前边界

- 图像 OCR 已接入文档解析层，但当前向量检索主链路还没有把图像内容作为独立向量源接入
- 外部搜索返回的新论文不会自动入本地库
- OCR 当前是基础能力，不是专门的公式 OCR 系统

## 6. 研究层能力

### 6.1 研究规划

`ResearchPlanningSkill.plan()` 会把一个大课题拆成 6 类任务：

1. 界定研究范围
2. 执行多源检索
3. 初筛与聚类
4. 精读核心论文
5. 提炼共识与分歧
6. 输出最终交付

每个任务都带有：

- 目标
- 交付物
- 依赖关系
- 推荐工具

### 6.2 深度阅读

`DeepReadingSkill` 当前支持：

- 获取全文
- 解析 PDF
- 读取指定章节
- 提取图表与公式候选

适合支撑这些场景：

- 针对 `Methodology` 定向问答
- 只阅读 `Conclusion`
- 抽取图表文字
- 形成结构化阅读笔记

### 6.3 研究记忆

`ResearchMemorySkill` 当前支持：

- 记住用户研究偏好
- 记住已读论文摘要和高亮
- 召回相关研究上下文
- 在搜索结果里优先排未读论文

这让系统具备：

- 避免重复推荐
- 记住用户常用时间范围和来源偏好
- 复用既有研究上下文

## 7. 关键模块与目录

### 7.1 关键代码目录

```text
src/
  agents/         多代理与研究代理
  core/           AgentV2、RuntimeGraph、LLM 管理、数据模型
  memory/         长期记忆
  preprocessing/  意图识别、槽位、查询改写、对话状态
  rag/            检索器、向量库、embedding、reranker
  reasoning/      推理引擎
  skills/         研究技能层
  tools/          学术搜索、文档获取、PDF 解析、网页搜索
  ui/             Gradio 前端
  whitebox/       trace 记录
```

### 7.2 当前重要文件

- `src/core/agent_v2.py`
- `src/core/runtime_graph.py`
- `src/agents/multi_agent.py`
- `src/agents/research_agents.py`
- `src/skills/research_skills.py`
- `src/rag/retriever.py`
- `src/rag/vector_store.py`
- `src/tools/research_search_tool.py`
- `src/tools/research_document_tool.py`
- `src/ui/gradio_app.py`

## 8. 配置、依赖与运行

### 8.1 Python 依赖

关键依赖包括：

- `gradio`
- `langchain`
- `langgraph`
- `requests`
- `pdfplumber`
- `PyMuPDF`
- `pytesseract`
- `chromadb`
- `FlagEmbedding`
- `transformers`

### 8.2 外部配置

#### LLM provider

- `SCNET_API_KEY`
- `SILICONFLOW_API_KEY`
- `ZHIPU_API_KEY`
- `DEEPSEEK_API_KEY`
- `DASHSCOPE_API_KEY`

#### 学术搜索

- `WOS_STARTER_API_KEY`
- `IEEE_XPLORE_API_KEY`
- `SERPAPI_API_KEY`
- `NCBI_API_KEY`

#### 本地模型

- `BGE_M3_MODEL_PATH`
- `BGE_RERANKER_MODEL_PATH`
- `RAG_VECTOR_COLLECTION`
- `RAG_PARALLEL_WORKERS`

### 8.3 OCR 依赖

OCR 需要两层依赖：

- Python：`pytesseract`
- 系统命令：`tesseract`

检查方式：

```bash
tesseract --version
```

### 8.4 运行入口

```bash
python run.py
```

支持：

1. Web 界面
2. CLI
3. 功能验证

### 8.5 验证脚本

基础回归：

```bash
python verify_features.py
```

搜索 agent 验证：

```bash
python verify_agentic_search.py
```

## 9. 数据落盘与可观测性

### 9.1 主要落盘位置

- `logs/traces/`：每轮对话的 trace
- `data/memory/memory.db`：长期记忆
- `data/memory/rag_index.db`：RAG 元数据索引
- `data/vector_db/`：Chroma 向量库
- `data/whitelist.json`：工具白名单
- `data/feedback/feedback.jsonl`：反馈数据
- `cache/`：论文下载、图片导出、中间产物

### 9.2 前端可观测性

Gradio 前端可看到：

- 执行时间线
- 每步摘要
- 模型用途、provider、model、耗时、错误
- 阶段模型摘要
- 原始 trace JSON

## 10. 当前边界与后续方向

### 10.1 当前边界

- 外部搜索结果不会自动写入本地 RAG
- 图像内容尚未作为独立向量源进入主 RAG 检索链路
- OCR 目前是基础能力，对扫描件、复杂公式和复杂表格仍有局限
- `Google Scholar`、`IEEE Xplore` 受外部 key、配额和接口稳定性影响较大
- 搜索规划阶段当前只把 `zhipu` 视为 agentic planning provider

### 10.2 更适合下一步演进的方向

- 把外部检索到的高质量论文自动下载并纳入本地索引
- 为图表和公式引入更强的专用 OCR / 结构化模型
- 给本地 RAG 增加章节级和图片级独立向量源
- 增加自动去重、引用链跟踪和阅读笔记生成
- 为研究规划补充执行状态与任务级缓存

---

如果只想先理解整体行为，先看 `README.md`。如果要做汇报或面试准备，再看 `docs/INTERVIEW_GUIDE.md`。
