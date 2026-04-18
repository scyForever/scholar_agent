# 面试速记版

## 1. 一句话介绍

ScholarAgent 是一个面向学术研究场景的多智能体助手，支持多源文献检索、论文获取、PDF 精读、本地 RAG、研究规划、研究记忆和可观测执行追踪。

## 2. 当前版本最值得讲的点

- 分层清晰：把能力拆成 `tools -> skills -> agents -> runtime`
- 研究场景完整：不仅能搜论文，还能抓 PDF、做章节级精读、抽图表、做研究规划
- 本地 RAG 已打通：从 PDF 解析、建库、混合检索到写作回答是一条完整链路
- 工程上可降级：没有稳定 provider 时，意图识别、查询改写和搜索规划都能回退到本地启发式流程
- 会话上下文更完整：概念解释可以复用上轮检索结果，也可以按指令只查本地 RAG 或跳过检索
- 可观测性强：trace 落盘，Web 界面能直接看到步骤时间线和模型调用

## 3. 端到端主链路

用户问题进入系统后，当前主流程是：

1. `memory_recall`
2. `intent`
3. `slots`
4. `planning`
5. `runtime_graph`
6. `multi_agent`
7. `reasoning`
8. `quality`
9. `trace` 落盘

其中真正的研究执行主要发生在：

- `SearchAgent`
- `AnalyzeAgent`
- `DebateAgent`
- `WriteAgent`
- `ResearchPlannerAgent`
- `ResearchReadingAgent`

## 4. 你可以重点讲的设计点

### 4.1 本地 RAG

- 上传 PDF 后，系统会解析正文、表格、QA 片段和知识关系
- 元数据写入 `SQLite`，向量写入 `Chroma`
- 检索链路是 `对话增强 -> 查询改写 -> 路由 -> TF-IDF/BM25 + 稠密检索 -> RRF -> rerank -> 相关性校验`
- 如果本地结果不足，会补外部搜索结果
- 在标准链路前，系统还支持 3 个显式控制：复用上轮搜索、只查本地 RAG、跳过检索

### 4.2 查询改写与降级策略

- 中文问题会先做主题抽取和英文检索式生成
- 有已验证成功的 provider 时，可以走结构化重写和搜索规划
- 没有可用 provider 时，自动退回本地规则和启发式改写，不会卡死在远程调用

### 4.3 研究层能力

- `ResearchPlanningSkill` 负责把综述写作这类大任务拆成可执行子任务
- `DeepReadingSkill` 负责论文获取、PDF 解析、定向章节阅读和图表提取
- `ResearchMemorySkill` 负责记住读过的论文和用户偏好，避免重复推荐

### 4.4 OCR 与多模态文档处理

- 当前可以抽取图片并做基础 OCR
- 能从公式行和表格候选里生成 LaTeX/Markdown 近似表示
- 但这还不是专门的高精度公式 OCR 系统，复杂扫描件仍有局限

## 5. 高频面试问答

### Q1：为什么要拆成 tools、skills 和 agents？

可回答为：

- `tools` 解决原子能力接入，比如搜索、抓论文、解析 PDF
- `skills` 解决研究动作编排，比如深度阅读、研究规划、记忆排序
- `agents` 负责角色化执行和上下文协作
- 这样做比把所有逻辑堆进一个大 agent 更容易维护、测试和扩展

### Q2：本地 RAG 和外部搜索是什么关系？

可回答为：

- 本地 RAG 负责“已经下载或建库过的文档”
- 外部搜索负责“找新论文和补充背景”
- 当前两者是协同关系，不是同一个存储层
- 外部结果不会自动写入本地向量库

### Q3：连续追问时，系统能复用上一轮资料吗？

可回答为：

- 可以，当前会话里会缓存最近一次 `search_result`
- 比如“根据之前查找到的资料，解释 ...”会直接复用上轮搜索结果
- 另外也支持“只用RAG”和“不要检索”这类显式控制
- 但这个复用目前是会话内内存级能力，服务重启后不会保留

### Q4：现在搜索规划是不是强依赖 zhipu？

可回答为：

- 规划阶段优先尝试已验证成功的 `zhipu`
- 但不是强绑定；如果不可用，会自动退回确定性搜索
- 这个设计是为了兼顾效果和稳定性，避免整条链路阻塞

### Q5：现在的 OCR 做到什么程度？

可回答为：

- 已经具备图片导出和基础 OCR 能力
- 能辅助抽取图表文字、公式候选和表格近似结构
- 但对扫描版 PDF、复杂公式和密集图表，精度仍有限

### Q6：这个项目最有工程价值的点是什么？

可回答为：

- 不是单点“会搜论文”，而是把研究任务拆成了完整的执行链路
- 研究规划、检索、精读、写作、记忆和 trace 都已经串起来了
- 出问题时还能从 trace 快速定位是检索、分析还是写作环节的问题

### Q7：详细介绍一下这个 agent 项目

可回答为：

- 这是一个面向学术研究任务的多智能体系统，不是“一个大模型 + 一堆工具”
- 主入口在 `src/core/agent_v2.py`，先做 `memory_recall -> intent -> slots -> planning`
- 然后交给 runtime 决定是否走多智能体、补 reasoning、做质量增强
- 多智能体层按任务意图编排 `plan -> search -> analyze -> debate -> write`
- `skills` 层把研究规划、文献搜索、深度阅读、研究记忆做成可复用能力
- `tools` 层真正调用 arXiv、OpenAlex、PubMed、IEEE、PDF 解析和本地 RAG

### Q8：上下文超限是怎么处理的？

可回答为：

- 当前版本没有完整的 tokenizer 级上下文压缩器
- 现在主要靠“局部截断 + 限制规模 + 最近历史增强”来控上下文
- 本地 RAG 只拼最近 4 条 user message 做对话增强
- reasoning 阶段会把上下文截到固定长度，例如 `3200/3600` 字符
- analyze 阶段最多分析 5 篇论文，深读材料也只截取少量 section 和 chunk
- write 阶段只拼有限数量的 chunk、论文列表和分析摘要
- 所以当前口径应说成“轻量限流”，不要说成“已经有完整上下文压缩系统”

### Q9：生成摘要的提示词有什么？

可回答为：

- 论文分析摘要模板是：
- “请分析下列论文，输出四部分：摘要、核心贡献、方法、局限性”
- 综述写作模板是：
- “请根据给定材料生成一篇结构化综述，包含标题、摘要、正文和参考文献”
- 单篇论文解读、方法对比、概念解释、研究动态也各有独立 writer 模板
- 搜索链路里还有一类“结构化汇总 prompt”，专门把真实工具调用结果整理成 `selected_tools / execution_plan / aggregation`
- 这些 prompt 分工明确，不是一个通用大 prompt 包打天下

### Q10：系统提示词是怎么设计的？

可回答为：

- 当前项目不是所有节点都用 system prompt
- 更常见的做法是：system prompt 定义角色和边界，user prompt 装业务材料
- 最典型的是 `SearchAgent`，system prompt 明确要求：
- 必须至少调用一个工具
- 只能用 `query / max_results / time_range`
- 优先使用 rewritten queries
- 不要编造论文，不要跳过工具直接生成结果
- 辩论推理也用了多组 system prompt，分别定义正方、反方、复辩和裁判的职责

### Q11：举一个例子，什么场景下用了什么提示词，解决了什么问题？

可回答为：

- 场景：用户要“系统调研近三年多模态 RAG，并写结构化综述，还要给工业落地建议”
- 第一步用查询改写 prompt，把自然语言需求改成适合数据库检索的 `core_topic / english_query / external_queries / local_queries`
- 第二步用搜索代理 system prompt，约束 agent 只在白名单工具里选 1 到 3 个最匹配工具，避免乱调工具
- 第三步用 `paper_analysis` prompt，对候选论文逐篇提炼摘要、贡献、方法和局限
- 第四步用 `survey_writer` prompt，把研究计划、检索结果、分析结果和辩论综合拼成综述
- 这一整套 prompt 分别解决的是：搜不到、搜偏了、读不深、写不成

### Q12：如果有两个相似工具，如何保证准确调用？

可回答为：

- 不能只靠模型自己判断，必须加工程约束
- 第一层是工具白名单，不同 agent 只能看到允许的工具
- 第二层是工具 schema，每个工具都有明确描述和参数约束
- 第三层是主题路由，例如生医优先 `PubMed`，电气通信优先 `IEEE Xplore`
- 第四层是 system prompt 限制，要求优先使用给定 query，不允许乱编参数
- 第五层是结果校验，只信真实工具返回的 artifact，不信模型口头说“我查到了什么”
- 如果 agent 路径不稳定，还会退回 deterministic search

### Q13：如果做一件事情需要调用十几个接口，怎么保证结果符合要求？

可回答为：

- 关键不是“把接口都调完”，而是把任务拆成有边界的阶段
- 当前项目就是显式 flow，不让 agent 无限自由发挥
- 每个阶段都定义清楚输入、输出和允许使用的工具
- 每一步调用都留 trace，便于回放和排障
- 搜索阶段有 fallback，provider 不稳定时可以退回确定性路径
- LLM 侧还有预算约束，避免链路无限膨胀
- 如果继续升级，我会再补 checkpoint、幂等 key、节点级重试和 artifact cache

### Q14：有了解过 skills 吗？

可回答为：

- 有，这个项目里就明确有 `skills` 层
- 当前主要有四类：`ResearchPlanningSkill`、`LiteratureSearchSkill`、`DeepReadingSkill`、`ResearchMemorySkill`
- 它们被收敛在 `ResearchSkillset` 里，再供多个 agent 复用
- skill 不是角色，而是稳定研究动作的封装

### Q15：skills 为解决什么问题而诞生的？

可回答为：

- skills 解决的是“高频动作复用”和“prompt 逻辑下沉”
- 如果没有 skills，很多研究动作会散落在各个 agent prompt 里，难维护、难测试、难复用
- tools 解决“能做什么”，skills 解决“这些动作怎么稳定复用”，agents 解决“谁在什么时候调这些动作”
- 在这个项目里，`plan / search / read / memory` 之所以能跨 agent 复用，就是因为先抽成了 skills
- 所以 skill 层的价值不在于多加一层抽象，而在于把研究动作从 agent 角色里解耦出来

## 6. 主动说明的边界

- 外部检索结果不会自动入本地 RAG
- 搜索规划优先用 `zhipu`，但会降级
- 会话检索结果复用目前不会跨进程持久化
- OCR 已可用，但不是高精度公式 OCR
- `Google Scholar`、`IEEE Xplore` 等源会受 key、配额和接口稳定性影响

## 7. 推荐配套阅读

- `../README.md`
- `./QUICKSTART.md`
- `../ScholarAgent_完整项目文档.md`
