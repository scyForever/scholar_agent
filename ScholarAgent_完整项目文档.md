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

如果从模块依赖看，当前代码确实已经具备 `tools / skills / agents / runtime` 四层基础；但如果从“执行控制”视角写项目概述，更准确的表达应当是：

`runtime 决策 -> agent 执行 -> skill 编排 -> tool 落地`

原因是当前系统真正的控制入口在 `AgentV2` 与 `AgentRuntimeGraph`，而不是工具层。一次对话的执行顺序，本质上是：

1. runtime 先做任务识别、预算约束、流程裁剪和状态治理
2. agent 再按角色接管检索、分析、综合和写作
3. skill 负责把研究动作拆成可复用流程
4. tool 最终落到检索源、PDF、RAG、网页与 OCR 能力

基于当前代码结构，项目概述建议改写为“已具备双层 LangGraph 编排基础，并可进一步演进为 runtime 决策驱动的四层架构”：

- 已有基础：
  - `AgentV2 + ScholarAgentHarness + RuntimeHarness + AgentRuntimeGraph` 负责会话入口、任务分级和 runtime 调度
  - `MultiAgentCoordinator + MultiAgentHarness + Search/Analyze/Debate/Write/Coder` 负责角色执行
  - `ResearchSkillsHarness + src/skills/components/*` 负责研究规划、文献搜索、深度阅读与研究记忆
  - `research_search_tool / research_document_tool / web_search_tool / retriever` 负责兼容入口，真实能力落地已拆到 `src/tools/*_components/` 与 `src/rag/components/`
- 需要在项目概述中突出但不能误写成“已完全落地”的部分：
  - `LangGraph` 目前已经用于 runtime 图和 multi-agent 图
  - `Self-MoA` 与 `MPSC` 已在 `QualityEnhancer` 中实现，但当前更偏向末端质量增强，还没有被明确前移成 runtime 主路由节点

因此，下面这份概述采用“当前代码映射 + 目标演进表达”并行书写：既贴近现状，也对齐你希望强调的架构方向。

## 2. 当前功能概览

### 2.1 学术搜索

统一搜索层对外兼容入口位于 `src/tools/research_search_tool.py`，底层实现已拆为：

- `src/tools/search_components/service.py`：统一搜索服务与平台归一化
- `src/tools/search_components/adapters/`：按平台拆分的适配器目录，包含 `arxiv / openalex / semantic_scholar / web_of_science / pubmed / ieee_xplore / google_scholar`
- `src/tools/search_components/common.py`：查询请求、时间范围解析、去重排序等公共逻辑
- `src/tools/search_components/web_snippet_search.py`：网页补充搜索组件

支持：

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

文档获取与解析层对外兼容入口位于 `src/tools/research_document_tool.py`，底层实现已拆为：

- `src/tools/document_components/acquisition.py`：论文获取组件
- `src/tools/document_components/reading.py`：阅读服务编排入口
- `src/tools/document_components/reading_components/parsing.py`：PDF 解析、章节识别、分块与摘要抽取
- `src/tools/document_components/reading_components/visuals.py`：图表、表格、公式与 OCR 提取
- `src/tools/document_components/reading_components/section_reader.py`：定向章节阅读
- `src/tools/document_components/common.py`：序列化与公共工具函数

支持：

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

当前本地 RAG 对外仍由 `src/rag/retriever.py` 暴露兼容接口，但底层已经拆成：

- `src/rag/components/indexing.py`：建库、分块、KG 三元组构造、SQLite 与向量库写入
- `src/rag/components/retrieval.py`：RAG 检索编排入口
- `src/rag/components/retrieval_components/query_preparation.py`：对话增强、查询改写、路由准备
- `src/rag/components/retrieval_components/route_retrieval.py`：词法召回、稠密召回与并行检索
- `src/rag/components/retrieval_components/fusion_validation.py`：RRF 融合、重排、相关性校验与补检
- `src/rag/components/common.py`：分词与公共辅助逻辑

特征如下：

- 元数据索引：`SQLite`
- 向量索引：`Chroma`
- 向量模型：`BGE-M3`
- 重排序模型：`bge-reranker-v2-m3`
- 召回类型：`text_chunk / table_chunk / qa_chunk / kg_chunk`
- 召回策略：`TF-IDF + BM25 + 稠密检索 + RRF + rerank + 相关性校验`

### 2.5 研究层能力

当前新增了研究场景专用 skill 与 agent。skill 层对外由 `src/skills/research_skills.py` 与 `src/skills/harness.py` 提供兼容入口，对内已拆为 `src/skills/components/` 下的独立组件：

- `ResearchPlanningComponent / ResearchPlanningSkill`：拆解研究任务
- `LiteratureSearchComponent / LiteratureSearchSkill`：统一多源文献搜索
- `DeepReadingComponent / DeepReadingSkill`：论文获取、解析、定向精读、图表提取
- `ResearchMemoryComponent / ResearchMemorySkill`：保存偏好、读过的论文和研究笔记

对应 agent：

- `ResearchPlannerAgent`
- `ResearchSearchAgent`
- `ResearchReadingAgent`
- `ResearchMemoryAgent`

### 2.6 会话级检索结果复用

当前会话状态会保存最近一次 `search_result`。这让系统在处理 follow-up 问题时，能够识别：

- `根据之前查找到的资料`
- `根据之前的检索结果`
- `只用 RAG / 只用本地`
- `不要检索`

并据此决定：

- 直接复用上轮搜索结果
- 只查本地 RAG
- 完全跳过检索

### 2.7 质量增强与验证

当前质量增强模块位于 `src/quality/enhancer.py`，已经提供两类关键能力：

- `self_moa()`：对多个候选答案做自聚合，适合做末端回答增强
- `mpsc_verify()`：从理论、证据、边界三个路径做一致性校验

从“现状”看，这两者主要挂在 runtime 的质量节点，用于结果后处理；从“目标架构”看，更适合在项目概述中表述为：

- `Self-MoA` 负责 runtime 路由决策与末端答案聚合
- `MPSC` 负责关键阶段或最终答案的多路径验证

也就是说，代码里已经有机制雏形，概述里可以把它写成明确的演进方向，但不应把“runtime 路由前移”写成既成事实。

## 3. 系统分层架构

### 3.1 建议采用的总体表达

```text
用户输入
  -> 预处理（记忆 / 意图 / 槽位 / 任务分级）
  -> Runtime 决策层
     -> 决定走 direct / retrieve / analyze / synthesize / write 哪条链路
     -> 决定是否复用 previous_search、是否只查本地 RAG、是否补外部检索
     -> 决定是否启用质量增强与验证
  -> Agent 执行层
     -> Planner / Search / Analyze / Debate(可视作 Synthesize) / Write / Coder
  -> Skill 编排层
     -> planning / search / reading / memory
  -> Tool 落地层
     -> 学术搜索 / 文档获取 / PDF 解析 / OCR / RAG / 网页补充搜索
```

如果映射到当前代码文件，这四层分别对应：

- runtime：
  - `src/core/agent_v2.py`
  - `src/planning/task_hierarchy.py`
  - `src/pipeline/runtime_graph.py`
  - `src/quality/enhancer.py`
- agents：
  - `src/agents/multi_agent.py`
  - `src/agents/search_agent.py`
  - `src/agents/analyze_agent.py`
  - `src/agents/debate_agent.py`
  - `src/agents/write_agent.py`
- skills：
  - `src/skills/research_skills.py`
  - `src/skills/harness.py`
  - `src/skills/components/*.py`
  - `src/agents/research_agents.py`
- tools：
  - `src/tools/research_search_tool.py`
  - `src/tools/research_document_tool.py`
  - `src/tools/web_search_tool.py`
  - `src/tools/search_components/*.py`
  - `src/tools/document_components/*.py`
  - `src/rag/retriever.py`
  - `src/rag/harness.py`
  - `src/rag/components/*.py`
  - `src/tools/registry.py`

### 3.2 四层职责与当前代码映射

#### runtime 决策层

职责：

- 维护会话状态、任务等级、预算和 trace
- 选择执行流是“直接写作”还是“检索 -> 分析 -> 综合 -> 写作”
- 决定是否复用历史检索结果，是否只走本地 RAG，是否需要质量增强
- 为后续 agent 执行提供统一状态对象

当前代码映射：

- `AgentV2.chat()` 负责入口、记忆、意图、槽位与任务分级
- `TaskHierarchyPlanner` 负责复杂度分类与执行预算
- `AgentRuntimeGraph` 负责 runtime 图调度
- `QualityEnhancer` 负责 `Self-MoA / MPSC` 的质量处理

说明：

- 当前 runtime 已经存在，但“显式 runtime 路由决策节点”还没有单独抽象成一个专门对象
- 因而项目概述里更适合写成“runtime 决策驱动”，并说明这是对现有 runtime 的强化表达

#### agent 执行层

职责：

- 根据 runtime 给出的任务意图与流转顺序执行角色分工
- 把复杂研究任务拆成检索、分析、综合、写作等阶段
- 在不同执行模式下裁剪流程深度

当前代码映射：

- `MultiAgentCoordinator`
- `SearchAgent`
- `AnalyzeAgent`
- `DebateAgent`
- `WriteAgent`
- `CoderAgent`
- `ResearchPlannerAgent / ResearchSearchAgent / ResearchReadingAgent / ResearchMemoryAgent`

说明：

- 当前 `src/pipeline/graph.py` 的标准链路是 `plan -> search -> analyze -> debate -> write -> coder`
- 若按你希望的项目概述表达，建议把这里写成：
  - `retrieve` 对应当前的 `search`
  - `synthesize` 对应当前的 `debate`
  - 因此主研究链路可描述为 `retrieve -> analyze -> synthesize -> write`

#### skill 编排层

职责：

- 把“研究动作”抽象为可复用能力单元
- 对 agent 层屏蔽底层工具差异
- 管理研究规划、文献搜索、深度阅读与研究记忆等通用动作

当前代码映射：

- `ResearchPlanningSkill / ResearchPlanningSkillHarness / ResearchPlanningComponent`
- `LiteratureSearchSkill / LiteratureSearchSkillHarness / LiteratureSearchComponent`
- `DeepReadingSkill / DeepReadingSkillHarness / DeepReadingComponent`
- `ResearchMemorySkill / ResearchMemorySkillHarness / ResearchMemoryComponent`
- `ResearchSkillsHarness`

#### tool 落地层

职责：

- 直接接入外部平台、本地索引和解析工具
- 提供真实检索、抓取、解析、OCR、网页补充能力
- 通过统一注册和白名单机制暴露给上层

当前代码映射：

- 学术搜索兼容入口：`research_search_tool.py`
- 学术搜索底层组件：`search_components/service.py`、`search_components/adapters/*.py`、`search_components/common.py`
- 文档获取与解析兼容入口：`research_document_tool.py`、`pdf_tool.py`
- 文档获取与解析底层组件：`document_components/acquisition.py`、`document_components/reading.py`、`document_components/reading_components/*.py`
- 工具治理：`registry.py`、`whitelist/manager.py`
- 本地证据库兼容入口：`rag/retriever.py`
- 本地证据库底层组件：`rag/components/indexing.py`、`rag/components/retrieval.py`、`rag/components/retrieval_components/*.py`
- 向量存储：`vector_store.py`

### 3.3 LangGraph 在当前项目中的位置

当前项目实际上已经有两张图：

1. runtime 图：`src/pipeline/runtime_graph.py`
   当前主要负责 `multi_agent -> reasoning -> quality`
2. multi-agent 图：`src/pipeline/graph.py`
   当前主要负责 `plan -> search -> analyze -> debate -> write -> coder`

因此，如果要在项目概述里强调“基于 LangGraph 编排检索、分析、综述与写作流程”，最贴近现状的写法是：

- ScholarAgent 已经使用 LangGraph 构建 runtime 图与 agent 图
- 其中 agent 图承担研究主链路编排
- 当前的 `search` 节点可解释为“检索”
- 当前的 `debate` 节点可解释为“综述综合 / synthesis”
- 因而对外概述可以写成：系统通过 LangGraph 串起 `检索 -> 分析 -> 综述 -> 写作` 流程

### 3.4 Self-MoA 与 MPSC 的合理放置

如果结合当前实现和你的目标，项目概述建议这样写：

- 当前实现：
  - `Self-MoA` 已用于多候选答案聚合
  - `MPSC` 已用于最终回答的一致性验证
  - 两者都位于 `QualityEnhancer`，更偏向结果后处理
- 推荐升级方向：
  - 把 `Self-MoA` 从“末端润色”前移到 runtime，用于路线选择、深浅链路切换和候选执行方案聚合
  - 把 `MPSC` 从“最终回答校验”扩展为关键阶段或最终阶段的多路径证据验证

换句话说，项目概述可以把它写成：

`LangGraph 负责主流程编排，Self-MoA 负责 runtime 路由与候选聚合，MPSC 负责关键结论验证。`

但需要在文档里保留一句说明：

`当前代码已具备 QualityEnhancer 雏形，runtime 前移路由属于下一步强化方向。`

## 4. 端到端执行链路

### 4.1 建议在项目概述中使用的主流程表述

如果把当前实现抽象成对外更清晰的架构概述，建议写成：

1. 召回相关历史记忆
2. 做意图识别
3. 填充槽位
4. 做任务分级，形成 runtime 约束
5. 由 runtime 决定走哪条执行链路
6. 交给 LangGraph 编排的 agent 流程完成检索、分析、综述与写作
7. 在质量节点执行 `Self-MoA / MPSC` 增强或校验
8. 落盘 trace，并把关键结果写入长期记忆

补充：

- 当前会把最近一次 `search_result` 保存在会话状态里，供后续追问复用
- 这个状态是进程内内存态，不会跨重启持久化
- 当前代码中第 5 步和第 7 步还没有完全拆成独立“runtime 路由节点”和“阶段化验证节点”，但已经有足够的模块基础支撑这种表述

### 4.2 检索阶段：SearchAgent 负责 evidence retrieval

`SearchAgent` 的处理逻辑是：

1. 根据槽位确定检索主题、时间范围和结果上限
2. 读取 `context_source / rag_mode`
3. 若 `context_source=previous_search` 且会话里已有上轮 `search_result`，直接复用，`search_mode=reuse_previous_search`
4. 若 `rag_mode=off`，直接返回空检索结果，`search_mode=disabled_by_instruction`
5. 若 `rag_mode=local_only`，只执行本地 RAG，`search_mode=local_rag_only_by_instruction`
6. 否则生成统一查询改写计划并先调用本地 RAG
7. 若当前意图是 `analyze_paper` 且本地已命中，直接走 `local_rag_only`
8. 否则再补外部学术搜索
9. 聚合、去重、排序
10. 用研究记忆把“未读论文”排在前面

这一段在项目概述中建议写成：

- runtime 先决定检索范围与来源约束
- SearchAgent 再统一执行“历史结果复用、本地 RAG、外部多源搜索”的证据收集
- skill 层负责把检索动作和研究记忆拼起来
- tool 层负责真正打到学术源、网页源和本地向量库

### 4.3 分析与综述阶段：AnalyzeAgent + DebateAgent

当前分析与综合能力可以概括为：

- `AnalyzeAgent` 负责对候选论文做摘要、贡献、方法、发现和局限提炼
- 若用户提供 `pdf_path` 且可深读，会调用 `ResearchReadingAgent` 生成更强分析上下文
- `DebateAgent` 负责基于多篇分析结果做综合判断

如果从项目概述角度表达，建议把 `DebateAgent` 描述成“综述综合节点”而不是只写“辩论”：

- `analyze` 对应证据抽取与论文级比较
- `debate` 对应综述综合与观点收束
- 因而整体中段可以写成 `analysis -> synthesis`

### 4.4 写作与质量闭环

`WriteAgent` 当前会消费以下材料：

- 研究计划
- 检索结果
- 本地 RAG 片段
- 论文分析结果
- `DebateAgent` 的综合结论

因此，对外概述可以写成：

- 写作不是直接对用户问题一次生成，而是消费前序阶段沉淀下来的结构化研究材料
- 质量闭环位于写作之后，负责候选答案聚合与一致性校验

### 4.5 查询改写、路由控制与降级逻辑

当前查询改写器支持两条路径：

- 有已验证成功的 provider 时：走结构化改写
- 没有已验证成功的 provider 时：走本地启发式改写

启发式改写会做：

- 中文主题提炼
- 英文学术术语翻译
- survey / comparison 扩展
- 本地 RAG 查询和外部检索查询分流
- 显式控制短语剥离，例如“根据之前查找到的资料”“只用RAG”“不要检索”

此外，`explain_concept` 已支持基于自然语言的显式控制：

- `根据之前查找到的资料...`
  系统会优先复用最近一次搜索结果，而不是重新跑一轮外部搜索。
- `只用 RAG / 只用本地 / 只查知识库...`
  系统会保留 `search -> write`，但只跑本地 RAG。
- `不要检索 / 不用查资料...`
  系统会直接进入 `write`，不跑 `search`。

这一层通过槽位提取完成，当前对应的槽位是：

- `context_source=previous_search`
- `rag_mode=local_only`
- `rag_mode=off`

### 4.6 搜索规划逻辑

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
    -> RAGHarness.index_pdf
      -> RAGIndexingComponent.index_pdf
        -> pdf_tool.extract_pdf_text
          -> research_document_tool.parse_pdf
            -> PDFReadingService / document_components/reading_components/parsing.py
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

补充：

- 在进入这条标准链路前，`SearchAgent` 还会先判断是否应该复用上轮 `search_result`，或根据用户指令跳过/收缩检索范围

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

当前该能力由 `ResearchPlanningSkillHarness` 统一对外，底层实际执行位于 `src/skills/components/planning.py`。

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

当前该能力由 `DeepReadingSkillHarness` 统一对外，底层实际执行位于 `src/skills/components/reading.py`，并进一步调用 `DocumentToolHarness + src/tools/document_components/`。

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

当前该能力由 `ResearchMemorySkillHarness` 统一对外，底层实际执行位于 `src/skills/components/memory.py`。

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
  harness/        顶层依赖装配、请求契约与执行 harness
  memory/         长期记忆
  preprocessing/  意图识别、槽位、查询改写、对话状态
  rag/            检索器兼容层、harness、向量库、embedding、reranker、components
  reasoning/      推理引擎
  skills/         研究技能兼容层、harness、底层 components
  tools/          学术搜索兼容入口、文档入口、网页搜索、底层 components
  ui/             Gradio 前端
  whitebox/       trace 记录
```

### 7.2 当前重要文件

- `src/core/agent_v2.py`
- `src/pipeline/runtime_graph.py`
- `src/pipeline/graph.py`
- `src/agents/multi_agent.py`
- `src/agents/research_agents.py`
- `src/skills/research_skills.py`
- `src/skills/harness.py`
- `src/skills/components/planning.py`
- `src/skills/components/search.py`
- `src/skills/components/reading.py`
- `src/skills/components/memory.py`
- `src/rag/retriever.py`
- `src/rag/harness.py`
- `src/rag/components/indexing.py`
- `src/rag/components/retrieval.py`
- `src/rag/components/retrieval_components/query_preparation.py`
- `src/rag/components/retrieval_components/route_retrieval.py`
- `src/rag/components/retrieval_components/fusion_validation.py`
- `src/rag/vector_store.py`
- `src/tools/research_search_tool.py`
- `src/tools/search_components/service.py`
- `src/tools/search_components/adapters/__init__.py`
- `src/tools/search_components/adapters/arxiv.py`
- `src/tools/search_components/adapters/openalex.py`
- `src/tools/search_components/adapters/semantic_scholar.py`
- `src/tools/search_components/adapters/web_of_science.py`
- `src/tools/search_components/adapters/pubmed.py`
- `src/tools/search_components/adapters/ieee_xplore.py`
- `src/tools/search_components/adapters/google_scholar.py`
- `src/tools/research_document_tool.py`
- `src/tools/document_components/acquisition.py`
- `src/tools/document_components/reading.py`
- `src/tools/document_components/reading_components/parsing.py`
- `src/tools/document_components/reading_components/visuals.py`
- `src/tools/document_components/reading_components/section_reader.py`
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
- “复用之前检索结果”当前只依赖会话内内存状态，服务重启后不会保留

### 10.2 更适合下一步演进的方向

- 把外部检索到的高质量论文自动下载并纳入本地索引
- 为图表和公式引入更强的专用 OCR / 结构化模型
- 给本地 RAG 增加章节级和图片级独立向量源
- 增加自动去重、引用链跟踪和阅读笔记生成
- 为研究规划补充执行状态与任务级缓存

---

如果只想先理解整体行为，先看 `README.md`。如果要做汇报或面试准备，再看 `docs/INTERVIEW_GUIDE.md`。
