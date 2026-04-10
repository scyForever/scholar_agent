# ScholarAgent 面试问答（基于当前仓库代码）

## 前置说明

- 本文只基于当前仓库代码、现有文档和验证脚本整理口径，不替代你的真实项目经历。
- 能被代码直接证明的，我按“当前实现”回答；不能被代码直接证明的，我会明确标记为“需补实验依据”或“需按你本人经历改写”。
- 当前仓库能直接证明四件事：有 `runtime -> agent -> skill -> tool` 四层组织；有本地 RAG；有研究记忆；有末端质量增强。
- 当前仓库不能直接证明这些数字：`5.97s`、`12 篇候选论文的平均时延表现`、`Context Recall / Precision > 92%`、`Faithfulness +8%`、`关键问题回答准确率 91%`。仓库里没有对应评测集、报告或 benchmark 文件。
- 当前仓库也不能证明“你本人到底负责哪几层、写了多少代码”。这部分只能给你模板，最终必须按你的真实经历改。

## 1. 先用 2 分钟介绍一下 Scholar-Agent

### 可直接说的版本

ScholarAgent 不是“一个大 agent 加一堆 tools”，而是按 `runtime 决策 -> agent 执行 -> skill 编排 -> tool 落地` 四层拆开的学术研究助手。

它的主入口在 `src/core/agent_v2.py`。用户问题进来以后，系统先做记忆召回、意图识别、槽位填充和任务分级，再把状态送进 runtime 图；runtime 决定是走多智能体链路、补 reasoning，还是做质量增强；多智能体层再按任务意图调度 `plan / search / analyze / debate / write / coder`；skill 层把研究规划、文献搜索、深度阅读、研究记忆这些动作封成稳定能力；最底层 tool 才去调用 arXiv、OpenAlex、Semantic Scholar、PubMed、IEEE、Google Scholar、PDF 解析和本地 RAG。

这样拆的原因是四层解决的问题不一样。runtime 解决预算、流程裁剪和状态治理；agent 解决角色分工和链路编排；skill 解决研究动作复用；tool 解决外部系统接入。如果做成一个大 agent，模型既要决定流程，又要拼研究动作，还要直接碰外部 API，最后很难做降级、排障和白盒追踪。

### 为什么一定拆成四层

- `runtime` 处理的是“这次任务该怎么跑”。`AgentV2.chat()` 会先产出 `intent / slots / task_config`，再把 `query、history、memory_context、prior_search_result、execution_mode` 交给 `AgentRuntimeGraph.execute()`。这层关心的是流程控制和状态治理，不关心某个具体工具怎么调用。
- `agent` 处理的是“这次任务由谁做、按什么顺序做”。`MultiAgentCoordinator` 按意图决定 flow，例如综述任务走 `plan -> search -> analyze -> debate -> write`，概念解释在 fast 模式下甚至可以直接只走 `write`。
- `skill` 处理的是“研究动作的可复用封装”。比如研究规划、文献搜索、深度阅读、记忆排序，本质上是跨 agent 也会复用的稳定动作，不应该散落在 prompt 里。
- `tool` 处理的是“跟外界交互”。学术搜索、抓全文、解析 PDF、本地向量检索、网页补充搜索都在这一层，便于白名单控制、单测和降级。

### 每一层的输入、输出、状态分别是什么

`runtime` 层：

- 输入：`query / intent / slots / session_id / trace_id / task_config / history / memory_context / prior_search_result / execution_mode`。
- 输出：`answer + artifacts`。
- 状态：`RuntimeState`，核心是流程级状态和跨节点 artifact。

`agent` 层：

- 输入：`query / intent / slots / flow / history / prior_search_result / task_config`。
- 输出：`research_plan / search_result / analyses / debate / answer`。
- 状态：`MultiAgentState`，核心是多节点中间产物。

`skill` 层：

- 输入：研究动作级参数，比如 `topic / time_range / platforms / pdf_path / user_id`。
- 输出：`ResearchPlan`、论文列表、PDF 结构化结果、记忆召回结果等。
- 状态：多数 skill 自身偏轻，真正的状态落在 `MemoryManager`、RAG 索引和文档缓存里。

`tool` 层：

- 输入：底层参数，比如 `query / max_results / time_range / identifier / pdf_path`。
- 输出：`Paper`、`PaperAsset`、`ParsedDocument`、网页 snippet、本地 chunk 列表等。
- 状态：外部 API 自身状态、本地 `SQLite`、`Chroma`、缓存目录。

### 哪一层最容易失控，为什么

如果按当前代码说，我会答“最容易失控的是 `agent` 层，准确说是 `search agent` 这一段”。

- 这层同时叠加了查询改写、本地 RAG、外部多源搜索、LangChain agent 工具选择、确定性降级、结果聚合和记忆排序。
- 外部数据噪声最大，模型不确定性也最大。比如 `SearchAgent` 既可能走 LangChain agent，也可能回退到确定性搜索；既可能只用本地 RAG，也可能补外部学术源。
- runtime 反而相对可控，因为 runtime 图只有 `multi_agent -> reasoning -> quality` 三段，且路由规则是显式的。

### 你在这个项目里真正负责哪几层，代码量和核心难点在哪

这部分代码不能替你证明，只能给你一个靠谱模板。

如果你确实主做主链路，建议按下面口径说，不要张口就说“我全做了”：

- “我主要负责的是 `runtime + agents + RAG/tool` 这三层，skill 层更多是把已有能力抽成研究动作接口。”
- “从仓库代码量看，重心确实也在这几块：`src/tools` 大约 2263 行，`src/core` 大约 1652 行，`src/agents` 大约 1564 行，`src/rag` 大约 778 行；`src/skills` 只有大约 315 行，更像封装层而不是主复杂度来源。”
- “真正的难点不是写几个 agent prompt，而是把查询改写、本地 RAG、外部搜索、记忆复用、质量增强和 trace 串成一条可降级、可排障的链路。”

如果你其实没写某层，就不要把那层说成自己主导。

## 2. “多路径验证机制”具体怎么做的

### 当前代码里“多路径”到底是什么

当前仓库里要分两类“多路径”，不要混着讲。

- 检索层的多路径：是“多 query + 多 source_type + 词法/稠密双路”。本地 RAG 会对多个重写 query、多个 chunk 类型、两类召回方式一起跑，再用 RRF 融合。
- 质量层的多路径：是 `Self-MoA + MPSC`。`Self-MoA` 会生成多个候选答案再聚合；`MPSC` 会从“理论 / 证据 / 边界”三条验证路径检查同一个答案的一致性。

所以如果面试官问“你说的多路径验证到底是什么”，最准确的回答是：

“在当前实现里，检索层是多 query 和多召回路径，生成后的质量层是多视角验证路径；严格说，`MPSC` 本身不是多 agent，也不是多数据源，而是对同一答案做多视角验证。”

### 验证发生在检索后、生成前，还是生成后

当前仓库里两种验证都存在。

- 检索后、生成前：本地 RAG 在 `rerank` 之后会做相关性校验 `_crag_validate()`，过滤低相关 chunk；如果有效 chunk 太少，再补网页 snippet。
- 生成后：`AgentRuntimeGraph` 在 `quality` 节点里先做 `self_moa()`，再做 `mpsc_verify()`。这一步已经是 answer 生成之后了。

如果你说“多路径验证机制”指的是简历里那套质量增强，那当前代码口径应回答为“主要发生在生成后”。

### 验证打分用的是什么信号

`MPSC` 的一致性分数不是 cross-encoder，也不是规则系统，更不是纯 LLM-as-a-judge。

- 第一层信号：LLM 先分别从“理论 / 证据 / 边界”三个角度生成三条验证路径。
- 第二层信号：再用 `TfidfVectorizer + cosine_similarity` 计算这三条路径之间的文本一致性，产出 `consistency_score`。
- 最终判定：按阈值分成 `high_consistency / medium_consistency / low_consistency`。

本地 RAG 的相关性校验则是另一套逻辑：

- 先让 LLM 输出 `correct / incorrect / ambiguous` 的 JSON。
- 如果 JSON 不可靠，再回退到分数阈值 `score >= 0.18`。

### 如果多路径结果互相冲突，最终怎么仲裁

这恰好是当前实现的一个边界，不要夸大。

- `Self-MoA` 负责聚合多个候选答案，输出一个 `moa_result.answer`。
- `MPSC` 只做一致性打分和给建议，不会在低一致性时自动重写答案。
- 也就是说，当前版本的仲裁是“打标和提示”，不是“强制回滚到更保守答案”。

如果面试官追问，你可以坦白说：

“当前版已经有多路径校验，但仲裁还是软约束。低一致性时会打标并建议补证据，不会自动重生成。这是我认为下一步最值得加强的点。”

### Context Recall / Precision / Faithfulness 这些指标怎么来的

当前仓库没有评测集、标注文件或离线评测脚本，所以这些数字不能当成“代码可证明事实”。

正确说法应该是：

- “当前仓库实现了相关机制，但这几个具体指标不在仓库里，面试里如果要讲，必须配套拿出评测集构造方式、标注标准、跑分脚本和结果报表。”
- “如果现在拿不出来，就不要在面试里硬说具体百分比。”

## 3. “检索 5.97s 返回 12 篇候选论文”的完整时延拆解

### 当前代码能直接证明什么

- `12 篇` 可以证明，因为 `SearchAgent.run()` 默认 `max_papers = 12`，除非用户 query 显式指定别的数量。
- `5.97s` 不能证明，因为仓库没有端到端平均时延报告。
- 当前系统只对 LLM 调用做了细粒度 `latency_ms` 记录；搜索工具、PDF 获取、RAG 各子阶段没有内建统一耗时拆解。
- 验证脚本 `verify_agentic_search.py` 只演示了如何手动对“查询改写”和“单个工具调用”做 `perf_counter()` 计时，不是正式 benchmark。

### 如果被追问“那你怎么拆”

基于当前代码，你应该诚实回答成这样：

- 查询改写：代码里有独立模块，也能单独打点，但仓库没有固定平均值。
- 外部搜索：每个 tool 调用可单独打点，但当前搜索 fallback 是串行的，不是统一计时器。
- 本地检索：有并发执行，但没有把 `conversation_enhance / rewrite / route / lexical / vector / rerank / validate` 分段计时落盘。
- PDF 获取、清洗、切分、embedding、rerank：当前都没有标准化 latency trace，只能临时 profiling。

### 为什么是 12 篇，不是 5 篇或 20 篇

从代码角度，唯一能被直接证明的原因是“默认值就是 12”。

- `SearchAgent` 默认拿 12。
- `TaskHierarchyPlanner` 只有当 `max_papers >= 20` 时才会提高任务复杂度评分。
- `AnalyzeAgent` 最多只分析 5 篇，所以 12 更像是“候选池”，不是“全量精读量”。

更直白地说：

“当前代码能证明 12 是工程默认值，但不能证明 12 是经过实验严格优化出的最优点。”

### 如果要把平均响应从 5.97s 压到 3s，优先动哪三个点

即使没有正式 benchmark，按当前代码结构，我会先动这三个点：

- 第一，减少外部搜索串行开销。当前 `deterministic_search` 是“工具 × query”双层串行循环，优先把外部 tool 调用并行化。
- 第二，降低 query 改写和结构化总结的 LLM 依赖。当前查询改写在有 verified provider 时会走结构化 LLM；搜索 agent 没返回结构化结果时还会再补一次 follow-up structured call。
- 第三，缩小外部检索触发面。当前已经先查本地 RAG，再决定是否外部搜索；如果要压时延，应把“本地命中足够时直接停”的门槛做得更强，而不是默认继续补外部源。

## 4. Scholar-Agent 的搜索 Agent 是怎么工作的

### 搜索关键词是谁决定的

不是纯靠规划 agent 现想，主导权在 `QueryRewriter`。

- `SearchAgent` 一上来先调用 `rewriter.plan(topic, intent)`。
- `QueryRewriter` 会产出 `core_topic / english_query / external_queries / local_queries`。
- 如果有 verified provider，就走结构化 prompt 重写；没有的话就走启发式翻译和规则扩展。
- LangChain search agent 拿到的是这些“候选检索式”，系统 prompt 还明确要求“优先使用提供的 rewritten_queries，不要自行发明过长查询”。

所以更准确的说法是：

“检索式先由查询改写模块给出候选，搜索 agent 主要决定用哪些工具和哪几个候选检索式，不是完全放任 agent 自由生成关键词。”

### 搜索范围是谁决定的

当前实现里有两层路由。

- 第一层是“本地还是外部”。`SearchAgent` 默认先跑本地 RAG；如果用户说“只用本地”，就只跑本地；如果用户说“不要检索”，就直接跳过；如果用户要求复用上轮资料，就直接复用 `prior_search_result`。
- 第二层是“外部学术源选哪些”。`_prioritize_search_tools()` 会按主题做启发式路由：生医偏 `PubMed`，电气通信偏 `IEEE Xplore`，默认偏 `arXiv + OpenAlex + Semantic Scholar`，综述/日报还会补 `Web of Science`。

要注意两个边界：

- 当前代码里没有“公开视频”路由。
- 通用网页检索 `search_web` 不是主搜索源，它只在本地 RAG 相关性校验后做补充 snippet，用来兜底。

### 拿回来的原始数据是什么格式

不同源不一样，当前代码是多格式混合，不要答成“统一 API JSON”。

- arXiv：ATOM XML。
- OpenAlex、Semantic Scholar、IEEE Xplore、SerpAPI：JSON。
- PubMed：先 `esearch` JSON 拿 ID，再 `efetch` XML 拉详情。
- Google Scholar fallback：HTML 解析。
- 论文正文获取：可能是 PDF 二进制，也可能是 HTML 文本。

### 清洗是在搜索 agent 做，还是交给后面的分析 agent 做

两段都有，但职责不同。

- 搜索阶段负责轻清洗和结构化：各 adapter 会把原始响应转成统一的 `Paper` 对象，并做基础 query 匹配、年份过滤、去重、打分。
- 深度清洗和结构化阅读在文档工具和阅读链路里做：抓 PDF、解析 section、抽表格、抽公式、定向阅读，这不是 SearchAgent 自己干的。
- AnalyzeAgent 真正做的是基于摘要和可选的深读上下文生成论文分析，不负责原始网页清洗。

### 怎么避免搜索 agent 把垃圾内容喂给后面的分析 agent

当前代码里有几层防护，但也不是完美的。

- 有工具白名单，搜索 agent 不能任意调未知工具。
- 查询改写先把主题正规化，减少脏 query。
- 各搜索源 adapter 会做标题/摘要与 query 的浅匹配、年份过滤。
- 外部论文会去重并按 `score + citations + year` 排序。
- 本地 RAG 结果会先 RRF、再 rerank、再相关性校验，低相关 chunk 会被过滤。
- SearchAgent 的结构化 follow-up 明确要求“只能基于真实工具调用，不要补造论文”。

但也要承认限制：

- 外部论文层没有更强的全文级质量过滤。
- 最终分析前最多还是依赖标题、摘要和排序结果，所以垃圾召回只能说“被压低概率”，不是“绝对不会进来”。

## 5. 系统里的记忆怎么设计的

### 短期记忆和长期记忆分别存什么

短期记忆是会话态，长期记忆是持久化记录。

短期记忆：

- 存在 `DialogueState` 里。
- 包含 `intent / current_slots / missing_slots / history / last_trace_id / last_search_result`。
- 存储位置是进程内内存，不是数据库。

长期记忆：

- 存在 `MemoryManager` 的 `memory.db` 里。
- 支持 `conversation / knowledge / preference / paper_summary / research_note` 等类型。
- 当前聊天主入口每轮都会把“用户问题 + 系统回答”写成 `CONVERSATION` 记忆。
- 研究层还会把论文摘要和用户偏好写进长期记忆。

### 检索 key 是什么

当前实现不是单一 key，而是“`user_id` 过滤 + 文本相似度排序”。

- `AgentV2.chat()` 里把 `session_id` 当成 `user_id` 传给 `memory.recall()`。
- 召回时先按 `user_id` 和可选 `memory_type` 过滤。
- 然后对命中文档做 `TF-IDF 相似度 + BM25 + importance + recency` 混合打分。

所以它不是只靠 `user_id`，也不是只靠 embedding 相似度，更不是只靠标签。

### 如何判断“当前问题是否和历史对话相关”

严格说，当前仓库没有显式的“相关 / 不相关”二分类判断器。

- 现在做法是直接召回 top N。
- 然后把召回内容拼成 `memory_context` 送后续链路。
- 没有最小分数阈值，也没有“低于阈值则不进上下文”的硬门槛。

这意味着当前系统更像“相关性排序”，不是“先判相关再召回”。

### 长期记忆会不会进数据库 / 向量库

当前仓库里，长期记忆只进 `SQLite`，不进独立向量库。

- `MemoryManager` 只维护 `memory.db`。
- 召回时临时在内存里算 TF-IDF 和 BM25。
- 没有给记忆单独建 Chroma 或别的向量索引。

如果面试官问为什么，可以按当前实现口径回答：

“当前记忆规模默认按个人研究助手场景设计，先用 SQLite 做轻量持久化和结构化元数据，召回时用 TF-IDF/BM25 混合排序；如果后续记忆规模显著增大，再考虑把长期记忆向量化。”

### 如果一段历史只有 20% 相关，怎么处理

当前代码里没有“20% 相关就压缩或丢弃”的机制。

- 它可能仍然因为排在 top N 而被召回。
- 召回后也没有二次压缩模块。

所以正确回答是：

“当前版没有这套精细阈值控制，属于后续可以补的能力，不要说成已经做了召回后压缩。”

## 6. 上下文窗口满了，怎么压缩

### 当前仓库的真实情况

当前仓库没有成型的“上下文窗口压缩”实现。

- `DialogueManager` 只是持续把 `history` 追加到内存里。
- 本地 RAG 的对话增强只会取最近 4 条 user message 拼到 query 前面。
- 没有 tokenizer 级别的上下文长度管理。
- 也没有“先压 user 还是先压 assistant/tool message”的策略代码。
- LLM 预算控制的是 `max_llm_calls`，不是上下文 token 预算。

所以这道题在面试里最稳妥的答法不是“我做了摘要压缩”，而是：

“当前版本还没做完整的上下文压缩体系。现在只有两种轻量手段，一是检索侧只拿最近 4 条用户消息做 query enhance，二是会话内直接复用 `last_search_result`，避免每次都把整段历史喂回去。真正的 tokenizer 级压缩还没落地。”

### 如果面试官逼你讲设计策略

这部分只能讲“下一步设计”，不能冒充“已上线实现”。

更可信的设计口径是：

- 先压 `assistant/tool` 消息，再压 `user` 消息，因为用户约束通常更关键。
- 压缩时必须保留问答对齐关系，不能只保留 assistant 结论。
- token 计数要接真实 tokenizer，不该继续用字符数估算。
- 阈值应绑定具体模型上下文长度，而不是写死常量。
- 压缩效果要用 trace 里的答案质量和命中率做回归监控。

但请记住，这些是设计建议，不是当前仓库事实。

## 7. 为什么选 Chroma + SQLite + bge-m3 + bge-reranker-v2-m3 + RRF

### Chroma，不是 FAISS / Milvus / Elasticsearch

代码能直接证明的是：

- 当前实现要的是“本地持久化 + 内嵌式部署 + 简单 upsert/query + metadata 过滤”。
- `LocalChromaVectorStore` 直接用 `chromadb.PersistentClient`，按 collection 落地到本地目录。

代码不能直接证明的是“我严格比较过 FAISS / Milvus / Elasticsearch 后得出它最优”。

更稳妥的工程口径是：

“这个项目当前优先的是单机本地研究助手场景，需要轻部署、零服务治理成本、支持持久化和 metadata 过滤。Chroma 更贴合当前代码形态；如果要做更大规模分布式检索，再讨论 Milvus 或 ES。”

### SQLite 在这里存什么，为什么需要它

这个问题当前代码可以答得很硬。

- `SQLite` 存的是 `documents` 和 `chunks` 两张表。
- 它不仅存元数据，还支撑词法召回，因为本地 TF-IDF/BM25 检索直接从 `chunks` 表读取全文内容。
- 它还承担索引完整性校验的作用，比如对比 `indexed_chunk_count` 和 `vector_chunk_count`。

所以不是“Chroma 就够了”。Chroma 只负责向量检索；SQLite 负责结构化文档索引、词法召回和元数据持久化。

### 为什么 embedding 用 bge-m3

当前仓库能证明的只有两点：

- 稠密召回模型就是 `BGEM3Embedder`。
- 它和 reranker 都走 `FlagEmbedding` 生态，本地模型路径由环境变量配置。

当前仓库不能证明：

- 你比较过 e5、gte、jina 并且得出了量化结论。

所以面试里最安全的说法是：

“当前实现统一选了本地可部署的 BGE 系列，embedding 用 bge-m3，rerank 用 bge-reranker-v2-m3，工程上能减少模型栈切换成本。至于和 e5、gte、jina 的量化对比，这个仓库没有实验记录，不能硬说。”

### 为什么 reranker 要单独加一层

这个问题代码可以直接支撑。

- 本地召回先跑多 query、多 source_type、词法和向量双路，候选会很多，噪声也会累积。
- RRF 只解决“融合排序”，不解决“query 和 chunk 的精配”。
- 所以当前实现会在融合后再走一次 `BGEReranker.rerank()`，然后再做相关性校验。

一句话版：

“embedding 相似度适合粗召回，不适合直接当最终排序；reranker 是为了把 query-chunk 的精匹配再做一遍。”

### RRF 为什么适合你的场景，它融合的是哪几路

RRF 适合这里，是因为当前本地 RAG 天然就是多列表融合。

它融合的不是“多个向量库”，而是：

- 多个 rewritten query。
- 多个 `source_type`，比如 `text_chunk / table_chunk / qa_chunk / kg_chunk`。
- 每个 `(query, source_type)` 下的词法召回结果。
- 每个 `(query, source_type)` 下的语义召回结果。

所以这里的 RRF 是“多 query、多 chunk 类型、词法/稠密双路”的统一融合器。

## 8. “查询改写 -> 检索路由 -> TF-IDF/BM25 召回 -> 融合重排 -> 相关性校验”的实现细节

### 查询改写是 prompt、few-shot，还是训练模块

当前不是训练模块，也不是 few-shot 训练器，而是“结构化 prompt + 规则降级”。

- 有 verified provider 时，走结构化 prompt，要求输出 `core_topic / english_query / external_queries / local_queries`。
- 没有 verified provider 时，回退到启发式规则：术语翻译、survey/comparison 变体、去重清洗。

### 路由怎么决定走语义检索、关键词检索，还是混合检索

当前本地 RAG 不做“二选一”，而是默认混合检索。

- 对每个 route source，都同时做词法检索和向量检索。
- 再把所有 ranked list 交给 RRF。

真正的“路由”发生在 chunk 类型上，不发生在“只走语义还是只走关键词”上。

- 默认查 `text_chunk`。
- 问表格/指标，就补 `table_chunk`。
- 问 why/how/是什么，就补 `qa_chunk`。
- 问关系/演化，就补 `kg_chunk`。

### Top K 怎么设

当前各阶段的 K 并不一样。

- `SearchAgent` 调本地 RAG 时把 `top_k` 设成 5。
- `HybridRetriever` 默认 `rag_top_k` 是 10，但被上层传入时会覆盖。
- 每个路由子任务实际取的是 `top_k * 2`，也就是给融合和 rerank 留余量。
- rerank 后只拿前 `top_k` 个去做相关性校验。
- 最终输出是 `validated[:top_k]`。

外部论文搜索则是另一套 K：默认 `max_papers = 12`。

### “相关性校验”是在过滤 hallucination，还是过滤低相关 chunk

当前这一步主要是在过滤低相关 chunk，不是直接过滤最终 hallucination。

- 检索链路里的 `_judge_relevance()` 是对 chunk 做相关性判断。
- 最终答案层的 hallucination 风险，更多是靠后面的 `MPSC` 去做一致性校验。

### 如果 query 是多跳问题，哪一步最脆弱

按当前实现，我会说最脆弱的是“查询改写 + 首轮召回”。

- 当前没有多跳问题分解器。
- `_conversation_enhance()` 只是把最近 4 条 user message 拼接。
- query rewrite 还是单轮主题改写，不是显式的 hop-by-hop decomposition。
- 如果第一跳没把关键实体改写出来，后面 route、召回、rerank 都会一起偏。

## 9. “关键问题回答准确率提升至 91%”

### 当前仓库能不能证明 91%

不能。

- 没有评测集文件。
- 没有开发集 / 测试集划分。
- 没有跑分脚本。
- 没有报表。

所以面试里如果你还要讲 `91%`，必须自己补这四样证据。否则最好的做法是直接删掉这个数字。

### bad case 主要会集中在哪几类

这部分只能做“基于代码的风险推断”，不能说是已有统计。

从代码看，风险更可能集中在：

- query rewrite 没把多跳问题拆开，导致首轮召回缺失。
- PDF chunking 仍然有固定窗口，语义边界可能被切断。
- 外部搜索的 query 匹配主要还是标题/摘要浅匹配。
- rerank 之后的相关性校验依赖 LLM JSON 判断，稳定性受 provider 影响。
- 生成后虽然有 `MPSC`，但它不会强制改写答案，所以低一致性时仍可能保留原回答。

### 如果要从 91% 提到 95%，先改什么

如果没有可信评测集，我会先改评测，不会先改 prompt。

原因很简单：

- 你现在连 91% 是怎么算出来的都证明不了，就没法知道哪一类问题真正拖分。
- 先补评测集，才能知道该优先改 query rewrite、召回、rerank 还是生成。

如果评测集已经可信，再按当前代码结构，我会优先改检索，不会先改 prompt。

- 因为这个项目的主瓶颈更像“证据进不来或排不对”，不是“写作模板不够花”。

## 10. PDF 解析怎么做

### 多栏 PDF、表格、公式、图片、页眉页脚怎么处理

当前实现已经做了基础结构处理，但不是文档智能解析大模型。

多栏：

- 用 PyMuPDF 读取页面 block。
- 通过左右 block 分布粗判 `double_column / single_column`。
- 双栏页面按“左列优先，再右列”的顺序重排文本行。

表格：

- 用 `pdfplumber.extract_tables()` 抽表格。
- 再把表格转成 Markdown。

公式：

- 先用正则识别公式样式文本行。
- 再做一个非常轻量的 LaTeX 近似替换。
- 另外对导出的图片 OCR 文本也会尝试转 LaTeX。

图片：

- 用 PyMuPDF 把嵌入图片导出来。
- 如果开了 OCR，就用 `pytesseract` 识别文字。

页眉页脚：

- 当前没有专门的页眉页脚消噪逻辑。
- 也就是说，页眉页脚可能混进正文，这是当前 PDF 解析链路的真实短板。

### chunk 是怎么切的，overlap 怎么设

PDF 解析链路是“先按 section，再做固定窗口切块”。

- 先根据标题候选构建 section。
- 再按 section 内文本做定长切块。
- `parse_pdf()` 默认 `chunk_size=1200`、`overlap=200`。
- 切出来的 chunk 会带 section heading 前缀。

要注意：

- `index_pdf()` 用的是 PDF 解析出来的 chunk，所以 PDF 建库默认跟的是 `1200/200`。
- `HybridRetriever.index_text()` 另一条纯文本建库链路才用 `settings.rag_chunk_size=500`、`rag_chunk_overlap=50`。

### 文档更新后，索引怎么增量刷新

当前没有真正的增量刷新机制。

- `index_pdf()` 每次都会生成新的 `document_id`。
- 没有按 `pdf_path` 做 upsert。
- 没有旧版本清理逻辑。

所以如果同一篇 PDF 重复建库，当前更像“新增一份索引”，不是“覆盖更新”。

### 连续追问时，上一轮答案是否进入检索条件，怎么避免 query drift

当前实现里，上一轮“用户消息”会部分进入检索条件，上一轮“assistant 答案”基本不会。

- 本地 RAG 的对话增强只拼最近 4 条 user message。
- `last_search_result` 会保存在会话状态里，用户明确说“根据之前查找到的资料”时可以直接复用。
- assistant 上一轮长答案不会直接整段塞进检索 query，所以这在一定程度上避免了 query drift。

但要承认两点：

- 这不是严格的 drift control，只是轻量约束。
- 当前没有显式的 topic anchoring 或 query drift 检测器。

## 最后给你的面试建议

- 第一，不要把当前代码里“已经有的机制”和“你准备下一步做的设计”混着讲。
- 第二，不要再口头硬背 `5.97s`、`92%+`、`91%` 这些仓库没有证据的数字。
- 第三，这个项目真正能打的点不是“会搜论文”，而是“把检索、阅读、分析、写作、记忆和质量增强串成了可降级、可追踪的完整链路”。
- 第四，如果面试官很强，反而会欣赏你明确说出当前边界，例如 `MPSC` 还只是软仲裁、上下文压缩还没落地、PDF 去页眉页脚还没做、索引刷新还不是增量更新。

## 主要证据文件

- `src/core/agent_v2.py`
- `src/pipeline/runtime_graph.py`
- `src/pipeline/graph.py`
- `src/agents/multi_agent.py`
- `src/agents/search_agent.py`
- `src/preprocessing/query_rewriter.py`
- `src/preprocessing/slot_filler.py`
- `src/rag/retriever.py`
- `src/rag/vector_store.py`
- `src/memory/manager.py`
- `src/tools/research_search_tool.py`
- `src/tools/research_document_tool.py`
- `src/quality/enhancer.py`
- `verify_agentic_search.py`

## 第二轮深挖问答

## 追问 1：runtime 到底怎么“决策”

### case：用户说“请帮我系统调研一下 2024 年以来多模态 RAG 的代表性工作，并给我一份结构化综述，还要指出哪些方法适合工业落地。”

### intent 怎么判

按当前实现，这条 query 大概率会被判成 `generate_survey`。

- 不是因为模型做了复杂理解，而是因为 `IntentClassifier` 先走规则短路。
- 规则表里只要命中“综述”或“写一篇”这类词，就会直接返回 `generate_survey`，置信度固定写成 `0.92`。
- 当前代码里，规则优先级高于词法相似度和远程 LLM 判别。

所以这里最稳妥的口径不是“runtime 理解出这是系统调研任务”，而是：

“当前版先靠规则把它判成 `generate_survey`，再由 task planner 根据 query 长度和多目标特征把复杂度抬高。”

### slots 至少会抽哪些字段

当前代码里，稳定能抽到的主要是：

- `topic`

可能抽不到但你直觉上以为应该抽到的：

- `time_range`

原因是 `SlotFiller` 只支持两类时间表达：

- `2023-2024` 这种区间
- “近三年 / 最近两年”这种模式

它不支持“2024 年以来”这种表达，所以这条 query 里的时间约束在当前版本大概率不会进 `slots["time_range"]`。

另外，下面这些在当前实现里也不会落成独立 slot：

- “结构化综述”
- “适合工业落地”

这些信息会停留在原始 query 文本里，后面更多是由 `write` 或 `debate` 阶段吸收，而不是 runtime 级结构化字段。

### task_config 里会有哪些关键参数

这条 query 在当前代码里很容易被打到高复杂度。

原因有三类：

- `intent=generate_survey` 的基础分本来就高
- query 很长
- query 有明显多目标标记，比如“并给我”“还要指出”

因此它大概率会落到 `L5_EXPERT`，对应关键参数是：

- `enable_multi_agent=True`
- `enable_quality_enhance=True`
- `reasoning_modes=["cot", "debate", "reflection"]`
- `max_llm_calls=30`
- `timeout_seconds=300`

如果只按代码口径说，这里的“deep”其实不是一个单独模式，而是“任务分级把预算和推理偏好抬高了”。

### execution_mode 什么时候会从 fast 变成 deep

当前实现里不会自动从 `fast` 切成 `deep`。

- `execution_mode` 只有 `FAST / STANDARD / FULL`
- 它只能通过 `AgentV2.set_mode()` 显式设置
- runtime 不会因为 query 复杂就自动把 `FAST` 升成 `FULL`

真正自动变化的是：

- `task_level`
- `task_config`

所以如果面试官问你“复杂 query 来了会不会自动从 fast 切 deep”，正确回答应该是：

“当前版不会自动切 execution_mode，只会自动提高 task_config。也就是说，深浅更多体现在预算、flow 和质量增强开关，而不是 execution_mode 的自动切换。”

### runtime 图里为什么会走 `multi_agent -> reasoning -> quality`，而不是只走 `multi_agent`

这里要先纠正一个容易说错的点。

对这条 query，当前实现更可能走的是：

- `multi_agent -> quality`

而不是：

- `multi_agent -> reasoning -> quality`

原因是 runtime 路由规则很直接：

- 只要 `multi_agent` 已经产出了 `answer`，就不会再补 `reasoning`
- 只有 `multi_agent` 没有产出答案时，才会进入 `reasoning`
- 如果 `enable_quality_enhance=True` 且 `execution_mode=FULL`，有答案后再进 `quality`

而这条 query 在当前多智能体 flow 下通常会走：

- `plan -> search -> analyze -> debate -> write`

`write` 会生成 `answer`，所以 runtime 没必要再补 reasoning。

### “补 reasoning”和“质量增强”分别解决什么问题，边界是什么

这两个不要混着讲。

`reasoning` 解决的是：

- 上游多智能体没产出答案
- 或者某些轻量请求直接需要一个 reasoning engine 来补回答

本质上它是“补生成”。

`quality` 解决的是：

- 已有答案后，做候选聚合和一致性校验

本质上它是“后处理”。

边界可以这么讲：

- `reasoning` 负责把答案生出来
- `quality` 负责对已有答案做增强和打标
- `quality` 不会回头改 plan、改 search、改 analyze
- `MPSC` 即使发现一致性差，也只是给出 `medium/low_consistency` 和建议，不会自动重跑整条链路

## 追问 2：你说 agent 层最容易失控，那失控的具体表现是什么

### 当前代码能直接支撑的“失控形态”

当前仓库没有线上事故报告，所以不能伪装成“我在线上看到过某某事故”。但从代码设计能反推出，作者显然在防这些问题：

- query 改写把检索主题改偏
- search agent 工具选错
- LangChain agent 空跑、乱调、调不到有效结果
- agent 虽然调用了工具，但结构化输出不可解析
- provider 不稳定导致 agentic search 失败
- 外部检索结果质量波动太大，必须回退到确定性搜索

也就是说，当前最可信的说法不是“我见过具体线上事故”，而是：

“从代码防线看，最担心的是搜索规划不稳定、工具调用不稳定和结构化总结不稳定。”

### 你是怎么监控到“失控”的

当前实现主要靠三类证据：

- `trace`
- 搜索步骤里的结构化产物
- LLM 事件日志

具体看：

- 搜索阶段会记录 `tool_strategy / agent_selected_tools / agent_tool_calls / final_output_source / provider_attempts / agent_errors`
- LLM 层会记录 `stage / purpose / provider / model / latency_ms / prompt_preview / response_preview / error`
- 整条请求还会落完整 trace JSON

所以如果我要判断 search agent 是否失控，优先看这些信号：

- 是否频繁出现 `deterministic_fallback`
- `agent_errors` 是否很多
- `provider_attempts` 是否反复失败
- `selected_tools` 和 `tool_calls` 是否为空
- `final_output_source` 是否经常不是理想路径

### 你做了哪些显式约束去收敛它

当前实现里收敛手段不少，而且都比较工程化：

- 工具白名单：search agent 只能调用白名单工具
- `search_web` 不进入主学术搜索工具集合，避免把通用网页搜索混进主检索
- 工具优先级路由：按主题只优先选 1 到 3 个工具
- 搜索规划 provider 固定只用 `zhipu`，避免多 provider 轮询放大不稳定性
- LangChain agent 设置了 `recursion_limit=6`
- 调用后只解析真实 `ToolMessage.artifact`，不信模型口头编结果
- 结构化输出要过 schema 校验
- schema 不合格就做 follow-up structured call
- follow-up 还不行就退到 deterministic structured fallback
- LangChain agent 本身不可用时，直接 deterministic fallback

一句话总结就是：

“当前版不是指望 agent 自觉收敛，而是加了一圈显式约束，保证它最差也能退回确定性检索。”

### 如果 search agent 这一层挂了，系统能不能降级

可以降级，但不是所有情况都优雅降级。

能降级的部分：

- LangChain agent 不可用
- 规划 provider 不可用
- agent 返回结果为空
- structured response 不可解析

这些都会退到：

- `deterministic_fallback`
- `deterministic_fallback_after_agent_failure`
- `deterministic_structured_fallback`

不能完全优雅降级的部分：

- 本地 RAG 索引已存在，但向量库不完整
- reranker 模型不可用

这类错误在当前实现里会直接抛异常，不是温和降级。

### 失控 bad case：search agent 调了，但 agentic search 没站稳，最后只能退回确定性搜索

如果面试官要求我讲一个更具体的 bad case，我会选这个，而不是泛泛说“模型不稳定”。

#### 1. 先讲现象

现象是：

- 搜索链路已经进入了 `SearchAgent`
- 也尝试了 LangChain agent 搜索规划
- 但最终没有拿回稳定、可用的 agentic search 结果
- 系统最后只能退回 `deterministic_fallback`、`deterministic_fallback_after_agent_failure` 或 `deterministic_structured_fallback`

这类 case 的特点不是“整条系统崩了”，而是：

- agent 路径不稳定
- 但系统靠降级仍然能给结果

这比完全报错更适合在面试里讲，因为它体现的是工程收敛能力。

#### 2. 再讲链路上具体哪里失控

失控不在 `IntentClassifier`，也不在 task planner，主要在 `SearchAgent` 这一段。

最常见的失控点有三层：

- 第一层：搜索规划不稳定。当前代码里搜索规划阶段固定只用 `zhipu`，这本身就说明作者已经观察到这个环节对 provider 很敏感。
- 第二层：agent 虽然调用了工具，但没有拿回有效聚合结果，代码里直接会记录成 `no_tool_results`。
- 第三层：agent 拿回了消息，但 `structured_response` 不可解析，最后还要补一次 follow-up structured call；再不行就退回确定性结构化摘要。

所以如果面试官逼我定位到组件级，我会这么说：

- 主要失控组件是 `SearchAgent._run_external_search()`
- 更具体地说，是 “LangChain agent 工具规划 + 工具结果聚合 + 结构化总结” 这三个环节叠加导致的不稳定

#### 3. 然后讲你怎么发现它

我会说我主要靠两个可观测信号发现它。

第一个信号是搜索 trace 里的策略字段异常：

- `tool_strategy`
- `final_output_source`
- `provider_attempts`
- `agent_errors`

如果我看到：

- `tool_strategy` 经常不是 `langchain_agent`
- `final_output_source` 经常落到 `deterministic_search_fallback`
- `agent_errors` 很多

那我就知道 agentic search 没站稳。

第二个信号是工具调用统计异常：

- `selected_tools` 不为空，但 `aggregated` 结果为空
- `tool_calls` 有记录，但单次 `count` 很低
- source 分布异常单一

这类信号说明不是“完全没执行”，而是“执行了但没产出有效结果”。

如果要再补一个更朴素的发现方式，我会坦白说：

- 当前仓库没有完整离线评测集
- 所以 bad case 很大一部分仍然是靠 trace 回放和人工看 case 发现的

#### 4. 最后讲你怎么收敛

这里我不会答成“优化 prompt”，而会强调工程约束。

我会讲这些收敛手段：

- 工具白名单，限制 search agent 的可调用范围
- 主学术搜索主动排除 `search_web`，避免把网页噪声混进主检索
- 工具优先级路由，按主题最多优先选 1 到 3 个工具
- 搜索规划 provider 固定只用 `zhipu`，不做多 provider 轮询
- LangChain agent 设置 `recursion_limit=6`
- 调用后只信真实 `ToolMessage.artifact`，不信模型口头描述
- `SearchAgentFinalOutput` 做 schema 校验
- schema 不合格就补一次 follow-up structured call
- 再失败就退到 deterministic fallback
- 本地 RAG 侧再加 rerank 和相关性校验，减少下游污染

一句话总结是：

“我不是试图让 agent 永远不出错，而是让它出错时更容易被识别、被约束、被降级。”

#### 5. 再补一句：如果这个模块挂了，怎么降级

search agent 挂了以后，系统通常还能回答，但会变保守。

当前可用的降级路径包括：

- 退回确定性多工具检索
- 在用户明确要求时退回本地 RAG
- 如果检索信息不足，则保留更保守的写作结果，并把不确定性交给质量层打标

用户体验上，当前实现不会特别显式地弹一句“搜索 agent 已降级”，但不确定性会通过两种方式暴露：

- trace 里能看到 fallback、agent_errors、provider_attempts
- 质量层在一致性不足时会给出“建议补充证据或缩小结论范围”

## 追问 3：state 落在 `RuntimeState` 和 `MultiAgentState`，那一致性怎么保证

### 如果 `plan` 已产出、`search` 已拿到 20 篇候选文献，但 `analyze` 中途失败

当前实现里，这些中间结果会进入运行时 state，但不会自动变成可恢复 checkpoint。

当前真实情况是：

- `research_plan` 会存在 `MultiAgentState`
- `search_result` 也会存在 `MultiAgentState`
- 这些会跟着 `artifacts` 在本轮执行中往下传
- 但失败后不会自动持久化成“可断点恢复快照”

### `search_result` 和 `research_plan` 要不要持久化

当前代码里：

- `search_result` 只有在整轮成功结束后，才会被写进 `DialogueState.last_search_result`
- `research_plan` 没有单独持久化入口

所以严格说：

- `search_result` 是“成功后会话级暂存”
- `research_plan` 是“仅本轮运行态存在”

如果 `analyze` 失败，当前这两个对象不会自动变成可恢复 checkpoint。

### 重试时是从 `analyze` 断点续跑，还是整条链路重跑

当前实现是整条链路重跑，不是断点续跑。

原因很简单：

- 没有 checkpoint store
- 没有节点级 resume 机制
- `AgentV2.chat()` 一旦异常，会 `fail_trace()` 然后直接抛出

所以用户再次发起同一请求时，本质上是新的一轮执行。

### 怎么避免重复调用外部 API

当前实现只能部分避免，主要靠两种手段：

- 会话成功后复用 `last_search_result`
- 用户显式要求“根据之前查找到的资料”时，直接走 `previous_search`

但如果上一次失败在 `analyze`，由于整轮没有成功返回：

- `last_search_result` 也不会被更新
- 所以下一次还是会重查外部源

也就是说，当前没有真正的失败后 API 级去重缓存。

### `trace_id` / `session_id` 分别起什么作用

`session_id`：

- 会话身份
- 用来取历史消息
- 用来做记忆召回
- 用来保存 `last_search_result`

`trace_id`：

- 一次执行的唯一标识
- 用来把本轮的步骤、LLM 调用和最终输出串成一条 trace
- 更偏“可观测性”，不偏“会话语义”

一句话就是：

- `session_id` 解决连续对话一致性
- `trace_id` 解决一次执行可追踪性

### 如果用户中途打断，再回来继续问“接着上次的分析做”，怎么恢复上下文

当前实现只能恢复一部分上下文，不能真的从 `analyze` 节点续跑。

能恢复的：

- `history`
- `memory_context`
- 成功轮次留下的 `last_search_result`

不能恢复的：

- 失败到一半的 `research_plan`
- 失败到一半的 `analyses`
- 某个 pipeline 节点的精确执行断点

所以更准确的说法是：

“当前版支持会话级资料复用，不支持严格意义上的断点恢复。”

### 用户可见层面：这类失败最后会造成什么问题

如果这条多智能体链跑到：

- `plan` 成功
- `search` 成功
- `analyze` 半途失败

当前用户可见问题通常不是“完全没有回答”，而是下面几类：

- 直接报错中断，本轮没有最终答案
- 下一次用户重试时，系统可能重新搜一遍文献，导致结果波动
- 即使成功重跑，综述覆盖范围也可能和上一次不同
- “工业落地建议”这类依赖完整分析链的结论最容易失真，因为它本来就依赖 `analyze -> debate -> write` 的后半段

这也是为什么在面试里要把“状态一致性”和“用户可见体验”连起来讲，而不是只讲内部 state。

### 当前代码现状 vs 我会怎么设计

这部分面试里最好分成两层回答：

- 先说当前代码现状
- 再说如果我要把它做成可恢复链路，会怎么定明确策略

这样既诚实，也能体现设计能力。

### 1. 哪些中间产物必须持久化

当前代码现状：

- `research_plan` 不持久化，只存在本轮 `MultiAgentState`
- `search_result` 只有在整轮成功后才会写进 `DialogueState.last_search_result`
- `analyze` 的中间结果不会分批保存
- 真正落盘的是 trace JSON，而不是节点 checkpoint

如果我要把它设计成可恢复链路，我会明确要求这三类 artifact 必须持久化：

- `research_plan`
- `search_result`
- `analyze` 的分批中间结果

我的明确策略是：

- `research_plan` 要存，因为它是后续节点的任务边界和约束，不应该每次失败都重算
- `search_result` 要存，因为它最贵，涉及外部 API 调用和结果波动
- `analyze` 要按论文粒度分批存，至少做到“分析完一篇，落一篇”

存储位置我会这样分层：

- 任务状态和 checkpoint：`SQLite`
- 下载的 PDF / HTML：文件系统缓存
- 解析结果和 chunk：沿用已有 `SQLite + Chroma`
- trace：继续 JSON 落盘

最简单可落地的工程实现，是新增一张 `task_checkpoints` 表，字段至少包含：

- `task_id`
- `session_id`
- `trace_id`
- `node_name`
- `status`
- `input_hash`
- `artifact_json`
- `updated_at`
- `version`

### 2. 失败后从哪里恢复

当前代码现状很明确：

- 失败后整条链路重跑
- 不支持节点级 resume

如果让我给一个明确恢复策略，我不会说“看情况”，而会这样定：

- 如果 `plan` 和 `search` 都成功，`analyze` 失败，那么默认从 `analyze` 节点恢复
- 不重跑 `plan`
- 只要已有 `search_result` 还有效，就不重跑 `search`

只有以下条件满足时，才允许重跑外部检索：

- `search_result` 不存在
- `search_result` 已过期
- 用户修改了 `topic / time_range / max_papers / sources`
- query rewrite 或 tool routing 版本发生变化
- 用户明确说“重新搜一遍”

所以我的明确恢复策略是：

- `plan` 成功就复用
- `search_result` 有效就强制复用
- `analyze` 只继续未完成论文
- `debate / write` 基于恢复后的完整分析结果重生成

### 3. 怎么避免重复调用外部 API

这里我不会用 `trace_id` 做缓存键，因为它是一次执行的唯一 ID，每次都会变，不适合跨轮复用。

我会拆成两层键：

- `task_id`
- `artifact_key`

我的建议设计是：

- `task_id = hash(intent + normalized_topic + time_range + max_papers + output_goal + user_constraints)`
- `search_cache_key = hash(tool_set + rewritten_queries + filters)`
- `document_key = DOI / arXiv ID / PMID / canonical URL`
- `parse_key = file_checksum + parser_version`
- `analysis_key = paper_id + focus + analysis_prompt_version`

缓存粒度我会拆成四层：

- 论文列表级：`search_result`
- 文档级：下载好的 PDF / HTML
- 解析级：`ParsedDocument` 和 chunk
- 单篇分析级：`PaperAnalysis`

外部搜索去重：

- 按 `search_cache_key` 复用论文列表
- 同一轮内部再按 `title / DOI / arXiv ID` 去重

PDF 解析去重：

- 不按文件路径去重
- 按文件 checksum 去重
- 同一 PDF 已解析过就直接复用解析结果

最终答案级我反而不会做强缓存，只会做会话内短缓存，因为答案受上下文和写作目标影响太大，误复用风险高。

### 4. 用户中断后再回来，怎么续上

如果用户说：

“接着上次的分析继续做，不要重新搜文献。”

我会先判断这是不是同一个 task，而不是只看是不是同一个 session。

我会用这几个条件判断：

- `session_id` 是否相同
- `task_id` 是否相同
- 用户是否显式要求“继续”“不要重新搜”
- 当前请求有没有修改核心约束

如果满足，就复用这些状态：

- `research_plan`
- `search_result`
- 已下载文档
- 已解析文档
- 已完成的单篇 `analysis`
- 未完成论文队列

但我会重新校验三类状态是否过期：

- `search_result` 是否超过 TTL
- 缓存文件是否还存在，checksum 是否匹配
- prompt / parser / rerank 版本是否发生变化，导致旧结果不再可靠

所以对这句用户输入，我的明确策略是：

- 不重跑 `search`
- 直接从未完成的 `analyze` 子任务继续
- `analyze` 完成后再重新做 `debate` 和 `write`

当前代码其实做不到这么完整。它只能基于：

- `session_id`
- `history`
- 成功轮次留下的 `last_search_result`

做会话级复用，不能恢复失败到一半的 `analyze` 状态。

### 5. 最后补一句设计权衡

如果让我解释为什么选择这种恢复策略，我会这样答：

我会优先做 “artifact-first 的断点恢复”，而不是一上来做“自动闭环自修复”。

原因是它更稳：

- 成本可控
- 行为可解释
- 不容易把一次错误放大成循环性错误
- 外部 API 压力更小
- trace 更容易读

它牺牲的东西也很明确：

- 需要更多状态管理和存储
- 可能复用到稍旧的 artifact
- 系统不会自动反思并重跑整条链路，智能感没那么强

如果以后升级成真正的闭环自修复系统，我会优先改 `runtime`，不是先改 prompt。

我会先补三样：

- 持久化 checkpoint 和 `task_id` 体系
- 节点级 retry / resume 策略
- 质量层触发的“定向重跑”，例如只重跑 `search` 或只补跑部分 `analyze`

一句话总结就是：

“先把可恢复做扎实，再把自修复做智能。”

## 追问 4：skill 层到底是薄封装，还是里面也带策略

### `plan / search / read / memory` 这些 skill，到底只是薄封装，还是带策略

当前 skill 层不是完全薄封装，要分开看。

偏薄封装的：

- `DeepReadingSkill`

它主要是把论文获取、PDF 解析、定向阅读、视觉提取这些工具做了一层统一接口。

带明显策略的：

- `ResearchPlanningSkill`
- `ResearchMemorySkill`

`ResearchPlanningSkill` 里有：

- 时间范围抽取
- 平台推荐
- 固定研究任务模板
- 里程碑和风险提示

`ResearchMemorySkill` 里有：

- 研究上下文召回
- 已读论文判定
- 未读优先排序
- 搜索偏好记忆

所以更准确的回答是：

“skill 层不是主复杂度中心，但不是纯薄封装，其中 planning 和 memory 明显带策略。”

### skill 和 agent 的边界怎么划

一个好用的说法是：

- skill 负责可复用研究动作
- agent 负责角色化执行和上下文 orchestration

比如：

- “研究规划”是一个稳定动作，所以适合作为 skill
- “是否现在先 plan 再 search，再 analyze，再 debate”是链路控制，所以属于 agent / runtime

### 比如“query 改写”应该放在 skill，还是 search agent 内部

按当前项目，放在 search agent 内部更合理。

原因是 query 改写直接服务两条搜索链路：

- 外部学术搜索的 `external_queries`
- 本地 RAG 的 `local_queries`

而且它跟：

- 工具路由
- 本地 RAG route
- LangChain search agent prompt

耦合都很强。

所以当前实现没有把 query rewrite 放进 `ResearchSkillset`，而是放在：

- `SearchAgent`
- `HybridRetriever`

内部各自使用。

### skill 是否可跨 agent 复用？举两个真实例子

可以，当前项目里有两个很典型的复用。

例子 1：`ResearchMemorySkill`

- 搜索阶段用它做 `rank_unseen_first` 和 `remember_search_preferences`
- 分析阶段用它存论文摘要和 highlights

例子 2：阅读能力

- `AnalyzeAgent` 通过 `ResearchReadingAgent.build_analysis_context()` 取 method/focus 段落
- `AgentV2.read_paper()` 和 `extract_paper_visuals()` 也直接复用同一套 reading skill

### 如果以后不用 LangGraph，skill 层能否保留

可以保留，而且应该保留。

原因是：

- 当前 skill 层本来就是普通 Python service，不依赖 LangGraph 图节点才能工作
- LangGraph 负责的是执行编排，不是 skill 本体

所以即使以后把 runtime 和 agent orchestration 换成别的框架：

- `ResearchPlanningSkill`
- `ResearchMemorySkill`
- `DeepReadingSkill`

这些都可以直接保留。

## 追问 4（补充）：短期 / 长期记忆与任务恢复如何打通

这类问题在面试里很容易和“状态恢复”“上下文恢复”“长期记忆设计”混成一团，所以回答时最好先把三层东西分开：

- 会话态
- 长期记忆
- 任务 checkpoint

当前仓库已经有前两层，第三层还没有正式做出来。

### 先讲当前实现里，短期记忆和长期记忆分别是什么

短期记忆在当前项目里，本质上是：

- `DialogueState`

里面现在实际存的是：

- `intent`
- `current_slots`
- `missing_slots`
- `history`
- `last_trace_id`
- `last_search_result`

这层的特点是：

- 会话内可用
- 进程内内存态
- 适合处理“上一轮刚做了什么”
- 不适合做强恢复或跨进程恢复

长期记忆在当前项目里，本质上是：

- `MemoryManager + memory.db`

里面现在实际存的是：

- 对话记忆
- 用户偏好
- 论文摘要
- 研究笔记 / 知识型记忆

这层的特点是：

- 持久化
- 能跨轮复用
- 更适合回答“这个用户长期关注什么”“读过哪些论文”“偏好什么输出风格”
- 不适合直接当执行状态机

所以如果面试官问“短期/长期记忆怎么打通任务恢复”，第一句最好先说：

“当前版里，短期记忆负责会话连续性，长期记忆负责研究上下文复用，但它们还不是严格的任务恢复系统。”

### 为什么长期记忆不能直接替代任务恢复

这是一个很关键的边界。

长期记忆现在存的是“语义上有价值的内容”，不是“执行时必须精确还原的 artifact”。

比如：

- “用户偏好多模态 RAG 和结构化回答”
- “这篇论文我之前总结过，核心贡献是什么”

这些适合进长期记忆。

但下面这些不应该只靠长期记忆恢复：

- 上一轮 search 的完整候选论文池
- 上一轮 analyze 已经完成到第几篇
- 哪些论文还没分析
- debate 是否已经跑完
- 当前 write 用的是哪一版 research_plan

原因是：

- 长期记忆是做相关性召回的，不是做强一致性恢复的
- 它会排序、裁剪、过滤
- 一旦把执行状态也混进去，恢复时就会不可靠

一句话就是：

“长期记忆适合帮你想起有用信息，不适合替你恢复精确执行现场。”

### 如果要把短期 / 长期记忆和任务恢复打通，我会怎么分层

我会把它拆成三层，而不是只靠一种 memory 结构硬扛。

第一层：会话态

- 仍然保留 `DialogueState`
- 负责最近交互、最近一次 `search_result`、当前缺失槽位
- 目标是提升交互连续性

第二层：长期语义记忆

- 继续用 `memory.db`
- 存用户偏好、论文摘要、研究主题偏好、已读论文、长期研究线索
- 目标是提升长期个性化和跨轮复用

第三层：任务恢复层

- 单独建 `task_checkpoints`
- 按 task 存节点级 artifact
- 目标是恢复执行，不参与语义召回排序

这是我认为最稳的分层方式，因为三层职责非常清楚：

- 会话态回答“刚才聊到哪”
- 长期记忆回答“这个用户长期在做什么”
- checkpoint 回答“这条链跑到哪一步了”

### 当前问题是否和历史对话相关，怎么判断

当前代码里，其实没有一个显式的“相关/不相关判别器”。

现在做法更像：

- 先按 `session_id` 找会话态
- 长期记忆按 `user_id + query` 做相似度召回
- 本地 RAG 对话增强只拼最近 4 条 user message

也就是说，当前系统的判断方式偏“相关性排序”，不偏“任务级识别”。

如果要和任务恢复真正打通，我会额外加一个：

- `task_matcher`

它不需要很复杂，但至少要判断：

- 这次 query 是不是在延续上一个 task
- 是延续 search 阶段，还是延续 analyze 阶段
- 是想复用旧资料，还是想重新搜

最简单可落地的判断信号可以是：

- 显式话术：如“继续上次的分析”“不要重新搜”
- `session_id` 是否相同
- 归一化 topic 是否相同
- output goal 是否相同
- 当前 query 是否只是在补充约束，而不是换题

### “接着上次做”时，短期记忆、长期记忆、checkpoint 分别起什么作用

这是面试里最该答清楚的一段。

假设用户说：

“接着上次的分析继续做，不要重新搜文献。”

我会这样用三层状态：

短期记忆：

- 从 `history` 里知道用户刚才在说哪条链
- 从 `last_search_result` 里知道最近一次搜索结果是否存在
- 从 `last_trace_id` 里快速定位上一次执行

长期记忆：

- 补充这个用户长期关注的主题偏好
- 判断某些论文是不是已经读过
- 在结果排序时优先未读论文

checkpoint：

- 精确恢复 `research_plan`
- 精确恢复 `search_result`
- 精确知道 `analyze` 已完成哪些论文
- 精确知道下一步应该从哪一篇继续

所以这三层不是互相替代，而是协同：

- 短期记忆告诉系统“你是在延续”
- 长期记忆告诉系统“哪些长期信息值得带进来”
- checkpoint 告诉系统“具体从哪一步继续”

### 当前实现里，记忆和恢复实际上打通到了什么程度

当前代码已经做到的：

- 用 `session_id` 保持会话态
- 用 `last_search_result` 支持 follow-up 复用
- 用长期记忆召回研究上下文
- 用长期记忆给搜索结果做“未读优先”排序

当前代码还没做到的：

- 用长期记忆驱动任务级恢复
- 用短期记忆恢复失败到一半的节点执行
- 用 checkpoint 避免 analyze 半途失败后整条链重跑

所以如果面试官逼你说“已经打通了吗”，最诚实的回答是：

“当前版打通的是会话连续性和研究上下文复用，还没打通成严格的任务恢复系统。”

### 如果要升级，我优先改哪一层

如果目标是把“短期 / 长期记忆”和“任务恢复”真的连起来，我不会先改长期记忆算法，也不会先改 prompt。

我会先补：

- `task_id`
- `task_checkpoints`
- `task_matcher`

原因是：

- 没有 task identity，就无法判断“这是不是上一个任务”
- 没有 checkpoint，就无法真正恢复
- 没有 matcher，就无法把自然语言里的“继续上次做”映射到具体 task

然后再往上接：

- 会话态里的 `last_active_task_id`
- 长期记忆里的 task-level 偏好和历史研究主题

这样系统才能真正做到：

- 知道你是谁
- 知道你长期在研究什么
- 知道你这次到底想继续哪条链

### 最后一句设计权衡

我会这样总结这个问题的设计权衡：

“短期记忆、长期记忆、任务恢复如果混成一层，系统会很难保证一致性；把它们拆开，工程复杂度会高一些，但可解释性、可恢复性和可控性会强很多。”

## 追问 5：你提到“白盒追踪”，那你怎么做 trace

### trace 到什么粒度

当前实现至少到三层粒度：

- 轮次级
- agent / pipeline 节点级
- LLM 调用级

轮次级：

- `trace_id`
- `session_id`
- 原始 query
- started / finished
- final_output

节点级：

- `memory_recall`
- `intent`
- `slots`
- `planning`
- `search`
- `analyze`
- `debate`
- `write`
- `quality`
- `error`

LLM 调用级：

- 用单独的 `type="llm"` step 写入 trace

所以不能只说“我有 trace”，而要说“我已经细到节点和单次 LLM 调用”。

### 一次完整请求里，至少会记录哪些字段

轮次级最少会有：

- `trace_id`
- `session_id`
- `query`
- `status`
- `metadata`
- `started_at`
- `finished_at`
- `final_output`

每个 step 最少会有：

- `type`
- `input`
- `output`
- `metadata`
- `timestamp`

LLM step 额外会带：

- `call_id`
- `stage`
- `purpose`
- `requested_provider`
- `provider`
- `model`
- `latency_ms`
- `response_format`
- `prompt_preview`
- `response_preview`
- `error`
- `http_status`
- `budget_remaining`

### 你怎么定位某次回答质量差，到底是检索问题、路由问题、模型问题，还是工具问题

当前 trace 足够做这件事，而且定位路径比较清晰。

如果是检索问题，通常看：

- `search` step 里的 `local_rag.trace`
- `validated_count`
- `agent_tool_calls`
- `source_breakdown`

如果是路由问题，通常看：

- `intent`
- `slots`
- `planning`
- 最终 `flow`

如果是模型问题，通常看：

- `llm` step 的 provider 失败
- latency 异常
- response_preview 明显跑偏

如果是工具问题，通常看：

- `agent_errors`
- 某个 tool call 的 `count=0`
- deterministic fallback 是否被触发

一句话版：

“我不是靠主观猜，而是先看 trace 里哪一层第一次偏了。”

### 你有没有做过 offline replay 或 case 回放

当前更准确的说法是：

- 有 case 回放
- 没有严格意义上的 offline deterministic replay

能做到的：

- 通过 `trace_id` 直接读取落盘 trace JSON
- 回看完整执行链路、节点输入输出和 LLM 调用

做不到的：

- 基于冻结工具结果和冻结状态做一次完全一致的重放执行

所以如果面试官问，你应该答：

“当前版支持 trace-based case 回放和人工复盘，但还没有做成严格的离线重演系统。”

## 追问 5（补充）：如何判断当前请求是在延续任务，还是开启新任务

这类问题本质上不是“记忆召回”问题，而是“任务识别”问题。

面试里我会先把这两件事分开：

- 相关对话召回：判断哪些历史内容值得带进来
- 任务延续判断：判断当前请求是不是在继续同一条执行链

当前仓库已经有前者的雏形，但后者还没有完整任务层实现。

### 当前实现里，系统实际上是怎么做的

当前代码里，延续性判断主要依赖三类弱信号：

- `session_id`
- `history`
- `last_search_result`

具体表现是：

- 同一个 `session_id` 下，会复用 `DialogueState.history`
- 同一个 `session_id` 下，会保留最近一次 `last_search_result`
- 如果用户显式说“根据之前查找到的资料”，`SlotFiller` 会把它抽成 `context_source=previous_search`
- `SearchAgent` 检测到这个 slot 且存在 `prior_search_result` 时，会直接复用上轮搜索结果

所以当前项目能做的是：

- 判断“这大概是在延续刚才那轮检索”

但还做不到严格判断：

- “这是不是同一个 task”
- “是在继续 analyze，还是重新开一个相似 topic 的新任务”

### 为什么不能只靠 `session_id`

`session_id` 只能说明“是同一个会话”，不能说明“是同一个任务”。

例如同一个 session 里，用户可能先问：

- “搜一下多模态 RAG 近两年论文”

然后又问：

- “顺便帮我比较一下 Agentic RAG 和 Graph RAG”

这两条都在一个 session 里，但它们很可能不是同一个 task。

所以如果只靠 `session_id`：

- 很容易把“同会话不同任务”误判成“同任务继续”
- 从而错误复用 `search_result` 或旧 plan

这也是为什么面试里不能只答“我们用 session_id 判断”。

### 我会怎么定义“延续任务”

如果让我补一套明确策略，我会把“延续任务”定义成同时满足下面几类信号。

第一类：显式语言信号

- “继续上次的分析”
- “接着刚才做”
- “不要重新搜文献”
- “基于前面的结果继续”

第二类：结构化任务相似性

- `intent` 相同或兼容
- `normalized_topic` 相同
- 时间范围约束没有发生关键变化
- 输出目标没有变化，例如都是“综述”或都是“分析”

第三类：可恢复 artifact 仍然有效

- `research_plan` 仍然存在
- `search_result` 仍然存在且未过期
- 相关文档缓存仍然有效

只有这三类信号同时满足，我才会把它判成“延续任务”。

### 如果让我给出明确判断逻辑，我会怎么做

我不会说“看情况”，我会给一个确定流程。

第 1 步：先看显式指令

- 如果用户明确说“重新搜”“重新开始”“换个角度”，直接判成新任务
- 如果用户明确说“继续”“不要重新搜”“沿用上次结果”，进入任务匹配流程

第 2 步：算 task-level 匹配

我会为每轮任务维护一个：

- `task_id`

同时从当前 query 里抽一个：

- `task_signature`

组成示例可以是：

- `intent`
- `normalized_topic`
- `time_range`
- `output_goal`
- `constraints`

如果当前 `task_signature` 和上一个活跃任务高度一致，就进入下一步。

第 3 步：校验 artifact 是否可复用

要检查：

- `search_result` 是否存在
- TTL 是否未过期
- 相关文档是否还在
- parser / prompt / reranker 版本是否发生重大变化

通过了，才判成“延续任务”。

否则就算用户说“继续”，也只能回复：

- “可以延续主题，但需要重新检索 / 重新解析”

### 当前请求和上一次“只有 20% 相关”怎么办

这是面试里很容易被追问的一点。

如果只靠历史文本相似度，很容易误把“弱相关 follow-up”当成“继续任务”。

所以我会明确说：

- 低相关历史不应直接驱动任务续跑
- 它最多只能作为长期记忆召回的参考
- 不能直接复用 `search_result` 或 `research_plan`

也就是说：

- 语义上 20% 相关，可能值得召回一点背景
- 但不代表执行上应该续上同一个 task

这就是“语义相关”与“任务一致”的区别。

### 当前代码里有哪些线索已经在为这件事打基础

虽然当前仓库还没有完整 `task_matcher`，但已经有几个雏形：

- `DialogueState` 已经有 `history` 和 `last_search_result`
- `SlotFiller` 已经能识别 `context_source=previous_search`
- `SearchAgent` 已经能在有 `prior_search_result` 时复用结果
- `MemoryManager` 已经能按 `user_id + query` 召回长期研究上下文

所以可以这么说：

“当前版已经实现了会话级延续和检索结果复用的雏形，但还没有抽象成正式的任务层判断器。”

### 如果用户说“接着上次的分析继续做，不要重新搜文献”，我会怎么处理

我会用下面这套明确策略：

先判断是不是同一个 task：

- `session_id` 相同
- 用户显式说“继续”“不要重新搜”
- 当前 query 的 topic 没变化
- 当前目标仍然是分析 / 综述，不是换成了别的任务

如果满足，再判断：

- 是否存在可用 `search_result`
- 是否存在任务 checkpoint

如果两个都在：

- 直接从 `analyze` 未完成部分继续

如果只有 `search_result` 在：

- 复用 search
- 重跑 analyze

如果两者都不在：

- 不能假装续跑
- 只能明确告诉用户需要重新检索或重新开始分析

### 最后一句设计权衡

我会这样总结这类问题：

“判断是否延续任务，不能只靠 session，也不能只靠语义相似度。最稳的做法是把显式用户意图、task identity 和 artifact 有效性三者结合起来；这样实现会更复杂，但能显著减少错误复用和状态污染。”

## 追问 6：上下文窗口满了时，任务状态、长期记忆和检索证据谁优先保留

### 当前代码现状

先把边界说清楚：当前仓库还没有真正落地“上下文窗口管理器”。

现在已经有的，只是几条轻量能力：

- `DialogueState` 会保留 `history / last_search_result / current_slots`
- 长期记忆会通过 `memory.recall()` 做 top N 召回
- 本地 RAG 的对话增强只会拼最近 4 条 user message

当前还没有的，是这些真正的窗口管理能力：

- 基于真实 tokenizer 的 token 统计
- 不同模型上下文长度的统一阈值系统
- 面向 `task state / evidence / memory` 的分层压缩器
- 压缩后的自动回放验证

所以这一节在面试里要分两层讲：

- “当前代码已经做到什么”
- “如果我要把它工程化落地，我会怎么设计保留优先级”

这道题如果只答“做摘要”基本会被继续追问，因为真正困难的不是压缩本身，而是：

- 压缩以后还能不能继续任务
- 压缩以后还能不能保住证据链
- 压缩以后会不会把用户硬约束压没

所以我在面试里会先给一个总原则：

“上下文窗口不够时，我优先保留能保证任务继续正确执行的信息，其次保留能支撑答案可信度的证据，最后才保留长期偏好和一般背景。”

也就是说，在任务状态、长期记忆和检索证据之间，我的优先级不是平均分配，而是分层保留。

### 1. 先给保留优先级

如果窗口不够，我的默认优先级会是：

1. 当前任务状态
2. 用户硬约束
3. 高价值检索证据
4. 错误恢复点和决策原因
5. 长期记忆
6. 一般历史消息和冗余生成

换成更直白的话就是：

- 先保“系统现在在做什么”
- 再保“为什么这么做”
- 最后才保“用户长期喜欢什么”

### 2. 为什么任务状态优先级最高

因为如果任务状态丢了，系统最容易出现的是执行错误，而不是回答风格变差。

必须优先保留的任务状态包括：

- 当前 `task_id`
- 当前 `intent`
- 当前 `topic`
- 当前用户硬约束，如 `time_range / source 限制 / 输出目标`
- 已完成子任务
- 未完成子任务
- 当前恢复点，例如“search 已完成，analyze 做到第 7 篇”

这些信息一旦丢了，最直接的后果不是“回答不优雅”，而是：

- 重新搜错文献
- 重新跑错节点
- 把延续任务误判成新任务
- debate / write 基于错误前提继续生成

所以在窗口紧张时，任务状态一定比长篇对话原文更值得保留。

### 3. 检索证据为什么排在长期记忆前面

因为对这类 ScholarAgent 任务来说，证据链直接决定答案是否可信。

高价值检索证据至少包括：

- 代表性论文标题、来源、年份
- 当前答案实际依赖的关键 chunk
- 本轮检索得到的 source 分布
- 哪些证据支持“工业可落地”这种判断

长期记忆更适合保留的是：

- 用户长期偏好
- 读过哪些论文
- 长期研究方向

这些信息当然有价值，但在上下文窗口极限时，它们的优先级应低于“当前结论的证据链”。

一句话就是：

“长期记忆能提高个性化，检索证据能保证可信度；在研究问答里，可信度优先级更高。”

### 4. 哪些长期记忆值得保留，哪些应该先丢

我不会把长期记忆整体保留，而会按价值再分层。

优先保留的长期记忆：

- 与当前主题直接相关的长期研究偏好
- 已读 / 未读论文状态
- 用户明确反复强调的输出偏好

优先丢弃的长期记忆：

- 与当前任务弱相关的历史兴趣
- 很早以前的普通对话记忆
- 只影响表达风格、不影响任务正确性的偏好

所以长期记忆在窗口紧张时，不是“全部带入”，而是只保留对当前 task 有直接价值的部分。

### 5. 我会怎么做压缩单位

为了不把问答对和推理链压断，我不会按“单条 message”独立压缩，而会按两类单位处理：

- 任务片段
- 证据块

任务片段包括：

- 一轮用户要求
- 系统基于该要求形成的关键决策
- 该轮产生的完成状态和未完成状态

证据块包括：

- 一个 search 结果集合的摘要
- 一组关键 chunk 的证据包
- 一次 analyze 的论文级结论

这样做的好处是：

- 不会把 user constraint 和 assistant 结论拆开
- 不会把 tool observation 和后续判断拆开
- 更容易在恢复时重新拼装上下文

### 6. 压缩后的表示怎么设计

我会用混合表示，不会只做单一摘要。

我会拆成四类：

第一类：任务状态块

- `task_id`
- 当前阶段
- 已完成 / 未完成子任务
- 当前硬约束

第二类：证据摘要块

- 代表性论文
- 关键证据 chunk
- 当前主结论和支撑来源

第三类：长期记忆块

- 当前主题相关的长期偏好
- 已读/未读线索

第四类：最近原始对话块

- 最近 1 到 2 轮未压缩的原始问答

重新拼回上下文时，我会按这个顺序：

1. `system`
2. 当前用户问题
3. 任务状态块
4. 证据摘要块
5. 必要长期记忆块
6. 最近原始问答块

这样保留的是“结构”，不是“所有原文”。

### 7. 如果三者必须三选一，谁先保，谁先丢

如果面试官逼我给一个非常明确的排序，我会这样答：

- 第一优先：任务状态
- 第二优先：检索证据
- 第三优先：长期记忆

原因分别是：

- 任务状态决定系统还能不能继续正确执行
- 检索证据决定系统答案还能不能自证
- 长期记忆更多是增强项，不是当前任务正确性的主支柱

所以当窗口满了时，我宁可先丢一些长期偏好，也不愿意丢：

- 当前任务恢复点
- 当前答案依赖的关键论文和 chunk

### 8. 错保和错丢的代价分别是什么

如果把长期记忆保太多、任务状态保太少，后果是：

- 系统更“像认识用户”
- 但任务会接错、恢复错、节点续跑错

如果把任务状态保住了、长期记忆保少了，后果是：

- 个性化变弱
- 但主链路仍然正确

如果把检索证据压掉了，后果是：

- 答案还能说
- 但引用链会断
- “为什么这个方法适合工业落地”这种结论会更像主观判断

所以我的策略偏置会很明确：

- 宁可牺牲一点个性化
- 也不牺牲任务连续性和证据可追溯性

### 9. 和前面 task 恢复逻辑怎么打通

这道题最重要的是，不要把“压缩”答成一个孤立模块。

我会明确说：

- 任务状态块直接服务 `task_matcher`
- 证据摘要块直接服务 `write / debate` 继续生成
- 长期记忆块只做补充，不承担恢复主职责

也就是说，压缩不是为了“省 token”本身，而是为了在 token 不够时，仍然保住：

- task identity
- 恢复点
- 证据链

这三件事一旦保住，系统就算上下文缩短了，也还能稳定延续任务。

### 10. 最后一句设计权衡

我会这样总结：

“上下文窗口满了时，不应该平均压缩所有信息，而应该优先保住任务状态和证据链；这样会牺牲一部分历史细节和个性化信息，但能最大限度保证任务连续性、答案可信度和错误恢复能力。”

### 11. 这套策略怎么和当前项目落地对齐

如果我把这套设计落到当前 ScholarAgent，我不会先做一个泛化摘要器，而会优先补三件事：

- `token_budget_manager`：接入真实 tokenizer，按模型上下文长度给出预算
- `context_compactor`：把 `task state / evidence / long-term memory / raw dialogue` 压成分层块
- `compression_eval`：做压缩前后的一致性回归，验证 task 恢复点和证据链有没有丢

原因是当前项目的核心不是普通聊天，而是研究任务链。

所以压缩是否成功，我不会只看 token 降了多少，而会重点看这几项：

- `task_state_recall`：任务阶段、硬约束、未完成子任务有没有保住
- `evidence_recall`：代表性论文、关键 chunk、引用来源有没有保住
- `resume_success_rate`：压缩后还能不能正确接着 `analyze / debate / write` 往下跑
- `answer_consistency`：压缩前后最终综述主结论是否发生明显漂移

如果这些指标没保住，即使 token 节省很多，我也不会认为压缩成功。

## 追问 7：评测体系和指标口径

### 当前代码现状

这题先不要说大话。当前仓库并没有完整的评测闭环实现。

当前能被代码直接证明的只有：

- 配置里预留了 `evaluation_dir`
- 有 `verify_features.py` 和 `verify_agentic_search.py` 两个验证脚本

但这两类脚本本质上是：

- 功能可用性验证
- 搜索链路探活和手工计时验证

它们不是严格意义上的：

- 评测集
- 标注集
- 离线批跑脚本
- 版本对比报表

所以如果面试官问“这些指标怎么来的”，当前代码口径必须先承认：

“仓库里有验证脚本，但没有完整 benchmark 闭环；如果要讲 `Context Recall / Precision / Faithfulness / 91%` 这类数字，必须额外补评测集、标注标准、打分脚本和结果报表。”

### 1. 先定义指标

我不会把 ScholarAgent 的效果评估混成一个总分，而会拆成四层：

- 检索是否召全：`Context Recall`
- 检索是否干净：`Context Precision`
- 生成是否忠实于证据：`Faithfulness`
- 任务是否真正答对：`Answer Accuracy`

#### `Context Recall` 怎么定义

我会把它定义成：

“金标准证据中，有多少最终被系统召回进可用上下文。”

公式可以写成：

- `Context Recall = 被召回的金标准证据数 / 金标准证据总数`

这里的“证据”不要只定义成 chunk，我会分两层：

- `paper-level recall`
- `chunk-level recall`

原因是 ScholarAgent 不只是文档问答，还涉及：

- 论文级候选池是否完整
- 论文内部关键段落是否被命中

#### `Context Precision` 怎么定义

我会定义成：

“系统最终送进生成阶段的上下文里，有多少是真正相关证据。”

公式是：

- `Context Precision = 相关证据数 / 上下文总证据数`

这个指标的核心不是“召回多不多”，而是：

- 有没有把低相关 chunk 一起喂进去
- 有没有把噪声论文混进综述写作

#### `Faithfulness` 怎么定义

我会定义成：

“最终答案里的关键事实和关键结论，有多少能被上下文证据直接支持。”

它不是看文风，不是看像不像对，而是看：

- 结论能不能在证据里找到支撑
- 是否出现超出证据边界的扩写

对于 ScholarAgent 这类系统，我会按“关键事实点”做标注，而不是整段 impression 打分。

#### “回答准确率 91%”里的“准确”是什么意思

这个项目如果真要讲“准确率”，我不会定义成 `exact match`。

更合理的口径是：

“答案是否命中用户要求的关键事实、关键结论和关键维度。”

具体可以拆成三个问题：

- 事实有没有错
- 结论有没有偏题
- 是否覆盖了用户要求的核心维度

例如对“多模态 RAG 代表性工作 + 工业落地建议”这类题，准确不等于“写得像综述”，而是要同时满足：

- 代表性工作覆盖到位
- 结论能被文献支撑
- 工业落地建议不是空泛主观判断

### 2. 再讲评测集怎么构造

如果我要把这套评测做完整，我不会只用一种来源，而会混三类数据：

- 真实用户问题
- 人工设计问题
- benchmark 改造问题

#### 数据从哪来

真实用户问题：

- 来自历史 query、trace 和 bad case
- 优点是真实、噪声足、能覆盖 follow-up 场景

人工设计问题：

- 用来专门覆盖复杂边界
- 比如多约束、多跳、时间过滤、source 限制、工业建议

benchmark 改造问题：

- 用公开 QA / RAG benchmark 的材料
- 但把问法改造成 ScholarAgent 的实际任务形式

我不会只用“自己编的问题”，因为那样很容易高估系统表现。

#### 怎么覆盖不同任务

我会按任务桶构造评测，而不是所有题混一起算一个平均分。

至少分这四类：

- 综述生成
- 事实问答
- 比较分析
- 工业落地建议

原因是这四类任务的瓶颈完全不同：

- 综述生成更吃证据覆盖和结构整合
- 事实问答更吃精确命中
- 比较分析更吃多文献对齐
- 工业落地建议更吃边界感和证据支撑

所以如果只给一个总分，其实会掩盖问题。

### 3. 标注怎么做

我不会只做答案级标注，最少做三层：

- 答案级
- 证据级
- 轨迹级

#### 答案级

看的是：

- 是否完成任务
- 是否偏题
- 是否有明显事实错误
- 是否覆盖了要求的关键维度

#### 证据级

看的是：

- 哪些论文是金标准候选
- 哪些 chunk 是关键证据
- 哪些引用必须出现
- 哪些证据属于噪声

#### 轨迹级

看的是：

- query rewrite 是否跑偏
- 检索是否漏召回
- rerank 是否把关键证据排后
- write 阶段是否把证据整合错了

#### 谁来标

理想情况是双人交叉标注：

- 一个偏技术内容
- 一个偏产品任务口径

如果资源有限，至少关键集做双标，普通集单标。

#### 多个标注人意见不一致怎么办

我不会简单多数投票，而会区分冲突类型。

如果冲突是：

- “这个 chunk 算不算关键证据”
- “这个建议算不算有文献支撑”

那就需要：

- 事先写清 adjudication 规则
- 保留仲裁记录

如果冲突只是：

- “这段话写得好不好”
- “表达是不是够漂亮”

这种主观项不应该进入主指标。

### 4. 评测怎么跑

我会分成两条线：

- 离线批跑
- 线上人工抽检

#### 离线批跑

适合做：

- 版本回归
- 检索策略对比
- prompt / rerank / query rewrite A/B

离线批跑必须固定这些东西：

- 评测集版本
- prompt 版本
- rerank 版本
- query rewrite 版本
- provider / model

不然分数没法比较。

#### 线上人工抽检

适合做：

- 抓真实 bad case
- 看用户追问链是否稳定
- 看长链路任务是否中途失控

因为有些问题离线集很难覆盖，比如：

- “继续上次分析”
- “不要重新搜，只基于前面结果改写”

这类多轮任务更适合配合 trace 抽检。

#### `LLM-as-a-judge` 用没用？用在什么环节

我会用，但只放在辅助环节，不做唯一裁决。

可以用在：

- `Faithfulness` 初筛
- 答案结构完整性初筛
- 批量 bad case 分类

不能直接完全交给 judge 的原因是：

- judge 模型本身有偏见
- 它很容易偏好“看起来更完整”的答案
- 但未必真的更忠实于证据

#### 如何避免 judge 偏见

我的做法会是：

- 生成模型和 judge 模型分离
- judge prompt 固定，标准显式写清楚
- 尽量让 judge 基于“证据对齐”打分，而不是基于文风打分
- 关键样本做人审复核，用来校准 judge 偏差

### 5. 最后讲 bad case

如果按 ScholarAgent 这类系统的结构，我预期最差的 case 往往集中在三类：

- 多跳、多约束 query 没被改写好
- 证据召回到了，但 rerank 或相关性校验没把关键证据排上来
- 写作整合阶段把多篇论文概括过头，导致结论超出证据边界

#### 哪一类 case 最差

我最担心的是：

- “检索看起来有结果，但写出来的结论证据不够”

因为这种 case 最不容易第一眼发现。

直接空结果反而容易排查；
最麻烦的是：

- 系统给了一版看起来像样的综述
- 但关键判断其实支撑不足

特别是在：

- 工业落地建议
- 方法优劣比较
- 趋势判断

这类需要跨文献综合判断的任务上，最容易出问题。

#### 最后发现瓶颈通常在哪

如果让我按优先级排，我通常会先怀疑：

1. 检索和 query rewrite
2. rerank 和相关性校验
3. 写作整合
4. prompt

原因很简单：

- 证据进不来，后面再怎么写都救不回来
- 证据排不对，写作阶段很容易拿错重点

所以我不会一上来就说“优化 prompt”。

#### 如果要把 91% 提到 95%，第一优先级改哪里

如果还没有一套可信评测集，我第一优先级先改评测，不先改模型和 prompt。

原因是：

- 你必须先知道哪一类 case 在拖分
- 不然所有优化都只是拍脑袋

如果评测集已经可信，再按这个项目结构，我会优先改：

- query rewrite
- retrieval
- rerank

不会先改：

- 写作 prompt

因为对 ScholarAgent 这类系统来说，主瓶颈通常更像“证据链质量”，不是“文风模板”。

### 最后一句总结

我会这样总结整套评测闭环：

“ScholarAgent 的评测不能只看最终答案好不好看，而要把检索、证据、忠实性和任务完成度拆开评；如果没有评测集、标注标准、批跑脚本和 bad case 分类，任何漂亮数字都不应该在面试里硬讲。”

### 主要证据文件

- `config/settings.py`
- `verify_features.py`
- `verify_agentic_search.py`
