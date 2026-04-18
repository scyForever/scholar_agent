# ScholarAgent 完整项目文档

> 面向学术研究场景的多智能体研究助手。当前实现已经打通从“问题理解 -> 任务规划 -> 本地 RAG / 外部学术检索 -> 分析综合 -> 写作输出 -> 记忆沉淀 -> 执行追踪”的完整链路。

---

## 目录

1. 项目定位与设计目标
2. 系统整体结构
3. 核心设计思路
4. 端到端执行链路
5. RAG 详细设计
6. 研究能力与多智能体协作
7. 数据落盘、状态管理与可观测性
8. 关键目录与模块说明
9. 配置、运行与验证
10. 当前边界与后续演进方向

---

## 1. 项目定位与设计目标

ScholarAgent 的目标不是只做一个“论文搜索框”，而是把学术研究中的高频动作整合成一条连续工作链路：

- 理解用户当前任务到底是搜论文、写综述、解释概念、分析单篇论文，还是做方法对比
- 把自然语言请求转成结构化意图与槽位
- 先利用本地知识库与会话上下文，尽量复用已有检索结果
- 在必要时补充外部学术数据库搜索
- 对候选论文做摘要级分析，必要时结合 PDF 章节做深读
- 组织成综述、对比、解释或研究计划
- 把本轮结论、读过的论文和偏好写入长期记忆
- 保留完整 trace，方便回放执行路径

从代码实现上看，这个目标被拆成四层：

`tools -> skills -> agents -> runtime`

这种分层的作用不是为了“看起来整齐”，而是为了明确边界：

- `tools` 负责原子能力接入
- `skills` 负责把原子能力组合成研究动作
- `agents` 负责按角色执行某一步任务
- `runtime` 负责整条链路的调度、状态、回退和追踪

---

## 2. 系统整体结构

### 2.1 总体架构图

```text
用户输入
  -> AgentV2
    -> DialogueManager                  # 会话状态、历史、上轮 search_result
    -> MemoryManager                    # 长期记忆召回
    -> IntentClassifier                 # 意图识别
    -> SlotFiller                       # 槽位提取
    -> TaskHierarchyPlanner             # 任务等级与预算配置
    -> AgentRuntimeGraph
         -> MultiAgentCoordinator
              -> plan / search / analyze / debate / write / coder
         -> ReasoningEngine             # 多模式推理兜底
         -> QualityEnhancer             # FULL 模式下质量增强
    -> WhiteboxTracer                   # 全程追踪
    -> MemoryManager.store()            # 对话与研究记忆沉淀
```

### 2.2 核心入口

- 对话总入口：`src/core/agent_v2.py`
- 运行图：`src/pipeline/runtime_graph.py`
- 多智能体流水线：`src/pipeline/graph.py`
- 多智能体协调器：`src/agents/multi_agent.py`
- 检索代理：`src/agents/search_agent.py`
- 本地 RAG：`src/rag/retriever.py`
- 向量存储：`src/rag/vector_store.py`
- 研究技能集合：`src/skills/research_skills.py`
- 论文获取与解析：`src/tools/research_document_tool.py`
- 学术搜索聚合：`src/tools/research_search_tool.py`

### 2.3 各层职责

#### tools 层

负责接外部世界和底层能力，包含：

- 学术搜索：`arXiv`、`OpenAlex`、`Semantic Scholar`、`Web of Science`、`PubMed`、`IEEE Xplore`、`Google Scholar`
- 文档获取：按 `DOI / arXiv ID / PMID / PMCID` 获取 HTML 或 PDF
- PDF 解析：正文、章节、表格、图像、公式、OCR
- Web 补充搜索：当本地 RAG 证据不足时补网页摘要

特点是：输入输出尽量统一、单职责、方便被注册到工具注册表与白名单体系。

#### skills 层

负责把多个 tool 拼成“研究动作”：

- `LiteratureSearchSkill`：统一多源文献搜索
- `DeepReadingSkill`：论文获取、PDF 解析、章节阅读、图表提取
- `ResearchPlanningSkill`：把研究主题拆成阶段任务
- `ResearchMemorySkill`：偏好、已读论文、研究笔记的读写与排序

这一层的价值是避免在 agent 里散落重复流程。

#### agents 层

负责按角色组织具体执行：

- `ResearchPlannerAgent`：出研究任务分解
- `SearchAgent`：做本地 RAG、外部搜索、结果聚合与排序
- `AnalyzeAgent`：对候选论文做结构化分析
- `DebateAgent`：做观点综合与冲突消解
- `WriteAgent`：把材料组织成最终回答
- `CoderAgent`：面向代码生成类任务

#### runtime 层

负责全局控制：

- 会话状态
- 任务等级与预算
- 流程图编排
- 失败降级与兜底
- trace 持久化
- 质量增强

---

## 3. 核心设计思路

### 3.1 先理解任务，再决定是否检索

系统不会一上来就把用户问题丢给“搜索”。`AgentV2.chat()` 先做：

1. 长期记忆召回
2. 意图识别
3. 槽位提取
4. 任务等级分类
5. 再进入运行图

这样做的原因是：

- 用户可能只是要解释概念，不一定需要外部搜索
- 用户可能明确说了“只用本地 RAG”或“不要检索”
- 用户可能在追问上一轮的检索结果，系统应该复用，而不是重复搜索

### 3.2 RAG 不是附加插件，而是检索决策层的一部分

`SearchAgent` 的实现表明，本地 RAG 被放在外部搜索之前：

- 如果用户要求复用上轮检索，直接复用 `prior_search_result`
- 如果用户要求 `rag_mode=off`，直接跳过检索
- 如果用户要求 `rag_mode=local_only`，只查本地
- 如果是 `analyze_paper` 且本地已命中，直接用本地 RAG
- 否则先跑本地 RAG，再决定是否补外部学术搜索

这意味着项目的检索设计不是“外部搜索为主，RAG 为辅”，而是“本地证据优先，外部数据补全”。

### 3.3 同时保留可解释性与检索效果

本地 RAG 采用双索引：

- `SQLite`：保存文档元数据和 chunk，方便词法检索、审计和调试
- `Chroma`：保存向量索引，负责稠密召回

原因很直接：

- 纯向量检索效果不透明，调试困难
- 纯 BM25/TF-IDF 召回对同义改写和跨语言检索不够稳
- 双索引可以兼顾可解释性、效果和故障定位

### 3.4 默认允许降级，但不允许静默失真

项目里大量地方采用“可降级但要留痕”的策略：

- 意图识别：规则命中 -> TF-IDF -> LLM
- 查询改写：有已验证 provider 时走结构化改写，否则走本地启发式改写
- 搜索规划：优先尝试 `zhipu` 的 LangChain agent；失败后退回确定性搜索
- RuntimeGraph：无 LangGraph 时退回顺序执行
- RAG：本地没有建库时返回空命中和明确 trace，而不是假装命中

这样做的目的是让系统在依赖不完整时仍可运行，但结果路径必须可追踪。

### 3.5 会话态和长期记忆分开管理

项目里有两种“记忆”：

- 会话态：`DialogueManager` 管理，包含历史消息、缺失槽位、上轮 `SearchResult`
- 长期记忆：`MemoryManager` 管理，落盘到 `data/memory/memory.db`

这样分层的意义：

- 会话态适合处理当前对话中的追问和上下文继承
- 长期记忆适合保存偏好、已读论文、研究笔记和对话摘要

---

## 4. 端到端执行链路

### 4.1 `AgentV2.chat()` 主流程

`src/core/agent_v2.py` 中的 `chat()` 基本流程如下：

1. 读取当前 `session_id` 的会话状态
2. 记录用户消息并启动 trace
3. 召回长期记忆，形成 `memory_context`
4. 识别意图
5. 解析槽位
6. 若槽位不完整，直接返回追问，不进入后续流水线
7. 用 `TaskHierarchyPlanner` 生成任务等级和运行配置
8. 把状态交给 `AgentRuntimeGraph.execute()`
9. 从运行结果中提取 `artifacts`
10. 若本轮产生新的 `SearchResult`，写回会话状态
11. 把“用户问题 + 系统回答”写入长期记忆
12. 结束 trace 并返回 `AgentResponse`

### 4.2 RuntimeGraph 的职责

`src/pipeline/runtime_graph.py` 的运行图只保留三段主链路：

1. `multi_agent`
2. `reasoning`
3. `quality`

路由逻辑很清楚：

- 如果 `multi_agent` 已经产出答案，通常直接结束
- 如果没产出答案，则进入 `reasoning`
- 只有在 `FULL` 模式且开启质量增强时，才进入 `quality`

这意味着：

- 多智能体流水线是默认主路径
- 推理引擎是兜底路径
- 质量增强是高成本后处理，不默认总开

### 4.3 MultiAgent 流水线

`src/agents/multi_agent.py` 为不同意图定义了不同 flow。

例如：

- `generate_survey`：`plan -> search -> analyze -> debate -> write`
- `search_papers`：`search -> write`
- `analyze_paper`：`search -> analyze -> write`
- `explain_concept`
  - `FULL/STANDARD` 下通常是 `search -> write`
  - `FAST` 下默认可直接 `write`
  - 若用户显式要求复用历史或本地 RAG，仍会保留 `search`

### 4.4 SearchAgent 在主流程中的位置

`SearchAgent` 是项目里最关键的“检索决策器”，它同时连接：

- 会话上下文
- 查询改写
- 本地 RAG
- 外部学术搜索
- 记忆排序
- 写作材料组织

最终它返回 `SearchResult`，里面既有：

- 候选论文列表
- 来源分布
- 改写后的查询
- 本地 RAG trace
- 外部搜索路径
- 工具调用信息

后续 `WriteAgent` 会从 `SearchResult.trace.local_rag` 中抽取本地片段和网页补充片段，直接写进生成提示词。

---

## 5. RAG 详细设计

### 5.1 RAG 在项目中的角色

本地 RAG 的目标不是替代外部学术搜索，而是解决三个问题：

- 让系统能基于用户本地已收集论文做高质量问答
- 在追问时复用已有证据，而不是重复联网搜索
- 在分析单篇论文或解释概念时，把回答尽量锚定到本地材料

从实现上，本地 RAG 主要由 `src/rag/retriever.py` 和 `src/rag/vector_store.py` 负责。

### 5.2 建库流程

#### 5.2.1 入口

建库入口是 `AgentV2.index_pdf()`，实际调用 `HybridRetriever.index_pdf()`。

#### 5.2.2 PDF 解析

`index_pdf()` 内部通过 `extract_pdf_text(pdf_path)` 解析 PDF，这一步依赖：

- `src/tools/pdf_tool.py`
- `src/tools/research_document_tool.py`

解析结果包含：

- `chunks`：正文分块，采用递归层次分割，优先按段落切分；若块仍过大，再按换行、句子、分句、词和字符逐层细化
- `tables`：表格块
- `qa_pairs`：从前若干文本块构造的问答块
- `sections`：章节信息
- `formulas`：公式行
- `images`：图像信息

其中真正进入 RAG 索引的主要是四类 chunk：

- `text_chunk`
- `table_chunk`
- `qa_chunk`
- `kg_chunk`

#### 5.2.3 四类 chunk 的意义

`HybridRetriever.index_pdf()` 会把解析结果转换成四类索引记录：

1. `text_chunk`
   面向正文语义检索，是默认主路由。
2. `table_chunk`
   面向指标、表格、benchmark 查询。
3. `qa_chunk`
   通过“问题-答案”形式增强“是什么 / 如何 / 为什么”类问法的命中率。
4. `kg_chunk`
   由文本块中提取的实体关系三元组近似构成，用于补充关系型查询。

这里的 `kg_chunk` 不是完整知识图谱系统，而是轻量关系表达，核心目的是增强“关系 / 关联 / 演化”类问题的检索能力。

#### 5.2.4 双索引落盘

每个 chunk 会同时写入两处：

- `SQLite`
  - 表：`documents`
  - 表：`chunks`
  - 默认文件：`data/memory/rag_index.db`
- `Chroma`
  - 默认目录：`data/vector_db/<collection_name>`
  - 默认 collection：`rag_chunks`

两者保存的是同一批 chunk，只是用途不同：

- `SQLite` 用于元数据、词法检索、审计
- `Chroma` 用于向量召回

#### 5.2.5 一致性与回滚

建库时先写 SQLite，再写 Chroma。若向量入库失败，代码会回滚 SQLite 中对应文档和 chunk 记录。

这样做的目的是避免出现：

- 元数据里显示“已建库”
- 但向量库实际上没有该文档

后续检索时，系统还会额外检查：

- SQLite 中 chunk 数量
- Chroma 中向量数量

如果向量数量少于 SQLite 记录，会直接报出“本地向量索引不完整，需要重新建库”，而不是继续返回不可靠结果。

### 5.3 检索前的查询理解

本地 RAG 不直接拿原始 query 检索，而是先做三步预处理。

#### 5.3.1 对话增强

`_conversation_enhance()` 会把最近几轮用户消息拼接到当前查询前面，解决追问类表达的问题，例如：

- “它的局限是什么”
- “那方法部分呢”

这样能把省略主语的追问补成带上下文的完整检索输入。

#### 5.3.2 查询改写

`QueryRewriter.plan()` 会同时产出两组查询：

- `external_queries`
- `local_queries`

设计原因是：

- 外部学术数据库更适合标准英文术语和规范检索式
- 本地 RAG 可以同时接受中文、英文和双语变体

当没有可验证的远程 provider 时，系统会退回本地启发式改写，包含：

- 主题提炼
- 中英术语转换
- `survey / comparison / recent advances` 等扩展词
- 去除“根据之前查找到的资料”“只用 RAG”“不要检索”等控制性短语

#### 5.3.3 路由到不同 chunk 类型

`_route_sources()` 会根据问题内容决定查哪些 chunk 类型：

- 默认：`text_chunk`
- 涉及“表 / benchmark / 指标”：增加 `table_chunk`
- 涉及“是什么 / 如何 / 为什么 / how / why”：增加 `qa_chunk`
- 涉及“关系 / 演化 / 关联 / graph”：增加 `kg_chunk`

这一步的设计思路是：先把查询意图和内容结构对应起来，再做召回，而不是让所有问题都走同一套 chunk。

### 5.4 检索主流程

`HybridRetriever.retrieve()` 的主流程可以概括为：

```text
对话增强
  -> 查询改写
  -> 路由 source_type
  -> 针对每个 (rewritten_query, source_type) 并行召回
       -> SQLite 词法召回
       -> Chroma 向量召回
  -> RRF 融合
  -> BGE rerank
  -> 相关性校验
  -> 证据不足时补网页摘要
```

#### 5.4.1 并行召回

系统会把每个“改写查询 + chunk 类型”视作一个任务，并通过 `ThreadPoolExecutor` 并行执行。

并行度由 `RAG_PARALLEL_WORKERS` 控制，默认 8。

这样做的原因：

- 多个改写查询本身就是互相补充的
- 不同 chunk 类型之间没有顺序依赖
- 并行化可以显著降低总等待时间

#### 5.4.2 词法召回

词法召回来自 SQLite 中的 `chunks` 表，具体做法是：

- 用 `TfidfVectorizer` 计算 query 与 chunk 的相似度
- 用本地 BM25 计算关键词匹配分数
- 按 `rag_cc_alpha * BM25 + (1 - rag_cc_alpha) * TF-IDF cosine` 合成分数

默认 `rag_cc_alpha=0.6`，说明当前实现更偏向关键词匹配，但仍保留语义相似度贡献。

#### 5.4.3 稠密召回

稠密召回由 `LocalChromaVectorStore.search()` 完成：

- 向量模型：`BGE-M3`
- 向量库：`Chroma PersistentClient`
- 距离空间：`cosine`

向量结果会被转回统一的 `IndexedChunk` 结构，和词法结果进入同一套融合流程。

#### 5.4.4 RRF 融合

多个召回列表不会直接拼接，而是使用 `Reciprocal Rank Fusion`：

- 配置项：`rag_rrf_k`
- 默认值：`60`

采用 RRF 的原因是：

- 不同召回器的原始分数尺度不同
- 排名比绝对分数更稳定
- RRF 对“多路弱相关但共同支持”的结果更友好

#### 5.4.5 BGE 重排序

融合后的 chunk 会交给 `BGEReranker.rerank()` 进一步重排。

这一步的作用不是“再召回一次”，而是把已经进入候选集的 chunk 重新按查询相关性排序，减少前排噪声。

#### 5.4.6 相关性校验与 CRAG 风格补强

`_crag_validate()` 会对重排后的 chunk 做二次判断：

- 对每个 chunk，调用 LLM 输出 `correct / incorrect / ambiguous`
- `correct` 和 `ambiguous` 会保留
- `incorrect` 会过滤
- 如果结构化判断失败，则退回分数阈值 `score >= 0.18`

如果最终有效 chunk 少于 3 个，则自动触发 `search_web()` 获取网页摘要作为补充证据。

这里体现的是一种轻量 CRAG 思路：

- 先检索
- 再判断检索结果是否足够可信
- 不够时再外扩，而不是一开始就联网放大搜索范围

### 5.5 RAG 与外部搜索的协同

本地 RAG 并不总是独立输出，它会和 `SearchAgent` 的外部搜索路径协同工作。

#### 5.5.1 只查本地

当用户说：

- `只用 RAG`
- `只查本地`
- `只查知识库`

`SlotFiller` 会提取 `rag_mode=local_only`，`SearchAgent` 只返回本地命中，不补外部搜索。

#### 5.5.2 完全跳过检索

当用户说：

- `不要检索`
- `不用查资料`

会提取 `rag_mode=off`。此时：

- `SearchAgent` 直接返回空检索结果
- `MultiAgentCoordinator` 对 `explain_concept` 可直接走 `write`

#### 5.5.3 复用上轮搜索结果

当用户说：

- `根据之前查找到的资料`
- `根据之前的检索结果`

会提取 `context_source=previous_search`。如果当前会话里存在上一轮 `SearchResult`，系统直接复用，不再重新跑搜索。

这项设计非常重要，因为很多学术对话天然是多轮的：

- 第一轮找论文
- 第二轮比较方法
- 第三轮解释某个概念
- 第四轮继续问局限

如果每轮都重新检索，成本高，而且上下文不稳定。

#### 5.5.4 先本地后外部

默认情况下，`SearchAgent.run()` 的策略是：

1. 先跑本地 RAG
2. 如果是 `analyze_paper` 且本地已命中，直接停止
3. 否则再走外部学术搜索
4. 再做聚合、去重、排序和记忆重排

所以整个系统的“搜索”实际上是混合搜索，不是单纯外网搜索。

### 5.6 外部搜索部分的设计衔接

当需要补外部搜索时，`SearchAgent` 会：

- 根据主题领域做工具优先级排序
- 白名单内优先选 3 个工具
- 如果 `zhipu` 已验证可用，则用 LangChain agent 做工具规划
- 否则退回确定性搜索

当前的设计重点不是把外部搜索做得极其复杂，而是让它和本地 RAG 使用同一套查询改写结果，并把最终结果统一装进 `SearchResult`。

### 5.7 RAG 结果如何进入最终回答

`WriteAgent._compose_materials()` 会主动读取：

- `search_result.trace.local_rag.results`
- `search_result.trace.local_rag.supplement`

并把它们写入最终提示词中的两部分：

- `本地论文片段`
- `补充网页片段`

这意味着 RAG 不是只在后台“命中过”，而是会显式成为最终答案的写作材料。

### 5.8 RAG 的落盘位置与关键配置

默认配置来自 `config/settings.py`：

- `data/memory/rag_index.db`：RAG 元数据和 chunk
- `data/vector_db/<collection>`：Chroma 向量库
- `RAG_VECTOR_COLLECTION`：向量 collection 名称，默认 `rag_chunks`
- `RAG_PARALLEL_WORKERS`：并行召回线程数
- `BGE_M3_MODEL_PATH`：BGE-M3 本地模型路径
- `BGE_RERANKER_MODEL_PATH`：重排模型路径
- `rag_rrf_k`：RRF 融合参数
- `rag_cc_alpha`：BM25 与 TF-IDF 融合权重
- `rag_top_k`：默认返回上限

### 5.9 当前 RAG 的优势与边界

#### 优势

- 本地证据优先，适合论文私有库场景
- 同时支持词法、向量、问答式和关系式检索
- 检索 trace 完整，便于调试
- 与会话复用、外部搜索和写作链路耦合紧密

#### 边界

- `kg_chunk` 是启发式三元组，不是严格图谱构建
- 相关性校验依赖 LLM，可带来额外延迟
- PDF 章节识别、公式抽取和 OCR 仍以启发式规则为主
- 上轮 `search_result` 只保存在会话内存中，进程重启后不会保留
- 当本地文档尚未建库时，RAG 只能返回空结果，不会自动索引外部论文

---

## 6. 研究能力与多智能体协作

### 6.1 研究层 skill

`src/skills/research_skills.py` 提供四组研究场景能力：

- `ResearchPlanningSkill`
  把一个大主题拆成“范围界定 -> 多源检索 -> 初筛聚类 -> 精读 -> 综合 -> 写作”六阶段任务。
- `LiteratureSearchSkill`
  调用统一学术搜索层，并结合记忆把“未读论文”优先排前。
- `DeepReadingSkill`
  负责取全文、解析 PDF、按章节读取、抽图表与公式。
- `ResearchMemorySkill`
  保存偏好、已读论文和研究上下文，用于后续排序与召回。

### 6.2 研究层 agent

这些 skill 被进一步封装成：

- `ResearchPlannerAgent`
- `ResearchSearchAgent`
- `ResearchReadingAgent`
- `ResearchMemoryAgent`

它们的作用不是替代主链路 agent，而是为主链路提供领域化能力。

### 6.3 Analyze / Debate / Write 的关系

- `AnalyzeAgent`
  对候选论文做摘要级结构化分析；如果提供了 `pdf_path`，还能借助 `ResearchReadingAgent` 追加章节级上下文。
- `DebateAgent`
  负责把多篇论文中的支持观点、反对观点和分歧做综合。
- `WriteAgent`
  负责把研究计划、论文列表、RAG 片段、分析结论和辩论综合组织成最终回答。

这个设计对应的是典型研究流程：

先找，再读，再比，再写。

---

## 7. 数据落盘、状态管理与可观测性

### 7.1 会话状态

`DialogueManager` 维护每个 `session_id` 的：

- `history`
- `intent`
- `current_slots`
- `missing_slots`
- `last_trace_id`
- `last_search_result`

其中 `last_search_result` 是多轮检索复用的关键。

### 7.2 长期记忆

`MemoryManager` 把长期记忆存到 `data/memory/memory.db`，支持：

- 对话记忆
- 用户偏好
- 论文摘要
- 研究笔记
- 通用知识

召回策略综合考虑：

- TF-IDF 相似度
- BM25
- 重要度
- 时间衰减

### 7.3 Whitebox Trace

`WhiteboxTracer` 会把每次执行落盘到 `logs/traces/<trace_id>.json`。

trace 里会记录：

- memory recall
- intent
- slots
- planning
- search
- analyze
- debate
- write
- quality
- error

这使得系统具备比较强的“白盒可解释性”，尤其适合调试复杂问答路径。

### 7.4 Web 界面如何展示 trace

`src/ui/gradio_app.py` 会读取 trace，并把关键阶段展示成：

- 时间线
- 步骤摘要
- LLM 调用情况
- 本地 RAG 命中情况

所以 trace 不是只给开发者看的日志，而是 UI 层的一等展示对象。

---

## 8. 关键目录与模块说明

```text
config/
  settings.py                    全局路径、RAG、模型与 provider 配置

src/core/
  agent_v2.py                    系统总入口
  llm.py                         多 provider LLM 管理
  models.py                      核心数据结构

src/pipeline/
  runtime_graph.py               总运行图
  graph.py                       多智能体流水线图
  state.py                       流程状态定义

src/agents/
  multi_agent.py                 多智能体协调器
  search_agent.py                本地 RAG + 外部搜索决策
  analyze_agent.py               论文分析
  debate_agent.py                综合推理
  write_agent.py                 最终写作
  research_agents.py             研究层代理封装

src/rag/
  retriever.py                   本地 RAG 主流程
  vector_store.py                Chroma 向量存储
  bge_m3_embedder.py             向量编码
  bge_reranker.py                重排

src/tools/
  research_search_tool.py        多源学术搜索聚合
  research_document_tool.py      论文获取、PDF 解析、视觉提取
  pdf_tool.py                    面向 RAG 的 PDF 抽取适配
  web_search_tool.py             网页补充检索

src/skills/
  research_skills.py             研究场景技能集

src/memory/
  manager.py                     长期记忆管理

src/preprocessing/
  intent_classifier.py           意图识别
  slot_filler.py                 槽位提取
  query_rewriter.py              查询改写
  dialogue_manager.py            会话状态管理

src/whitebox/
  tracer.py                      trace 记录与持久化

src/ui/
  gradio_app.py                  Web 界面
```

---

## 9. 配置、运行与验证

### 9.1 运行入口

项目入口为 `run.py`，支持：

1. Web 界面
2. CLI
3. 功能验证

启动命令：

```bash
python run.py
```

### 9.2 关键依赖

- LLM provider API key
- `chromadb`
- `PyMuPDF`
- `pdfplumber`
- `pytesseract`
- 本地 `BGE-M3` 与 `bge-reranker-v2-m3`

### 9.3 验证脚本

基础验证：

```bash
python verify_features.py
```

覆盖：

- Prompt/Trace/Memory 基础能力
- `AgentV2` 核心状态
- 研究规划
- 论文获取
- 工具注册与白名单

搜索 agent 验证：

```bash
python verify_agentic_search.py
```

覆盖：

- 查询改写
- 搜索工具连通性
- LangChain agent 搜索路径
- 真实 provider 的可用性

---

## 10. 当前边界与后续演进方向

### 10.1 当前边界

- 本地 RAG 目前主要面向 PDF 与纯文本建库，还不是通用知识仓库平台
- 会话内 `last_search_result` 不跨进程持久化
- 外部搜索结果仍以元数据和摘要级证据为主，未自动全文入库
- PDF 结构化解析仍偏规则驱动，对扫描件、极复杂版面和公式密集论文存在误差
- 搜索规划阶段当前固定优先 `zhipu`，agentic search 的 provider 选择还较保守

### 10.2 值得继续增强的方向

- 把“外部检索 -> 自动抓取全文 -> 自动建库”接成闭环
- 为 RAG 增加文档级、章节级、表格级的多粒度索引控制
- 增加引用链路与证据定位展示，而不只是返回 chunk 文本
- 把会话级检索复用扩展成可持久化的研究工作区
- 为 PDF 解析增加更强的版面模型和结构化表格识别

---

## 附：一句话理解当前系统

ScholarAgent 当前最核心的特点，不是“多接了几个学术 API”，而是把本地论文知识库、外部学术搜索、多轮会话状态、研究记忆和多智能体写作链路串成了一套可追踪、可降级、可扩展的研究工作流。
