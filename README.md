# ScholarAgent

ScholarAgent 是一个面向学术研究场景的多智能体助手，支持多源论文检索、论文分析、综述生成、代码生成、长期记忆和白盒追踪，并提供 Gradio Web 界面与 CLI 两种使用方式。

## 当前能力

- 多源学术检索：`arXiv`、`OpenAlex`、`Semantic Scholar`、`Web of Science Starter API`
- LLM 驱动查询改写：中文、英文、缩写和中英混合主题会先重写成结构化检索查询
- Multi-Agent 协作：`Search / Analyze / Debate / Write / Coder`
- 多执行模式：快速模式、标准模式、完整模式
- RAG 检索链路：对话增强、查询改写、路由、混合检索、融合、重排、相关性验证
- 长期记忆：`SQLite` 持久化会话记忆
- 白盒追踪：保存完整 trace，并在前端右侧实时展示执行时间线
- 模型调用可视化：前端可看到每次 LLM 调用的用途、provider、model、耗时和失败信息
- PDF 建库：上传 PDF 后建立本地知识库，参与后续问答和综述生成

## 近期更新

- 新增 `SCNet` provider，并统一兼容 OpenAI Chat Completions 风格 endpoint
- `Web of Science Starter API` 已接入统一搜索链路
- 查询重写从静态术语映射改为 `LLM` 结构化重写，输出 `english_query / external_queries / local_queries`
- Web 前端新增右侧实时执行时间线，`llm` 步骤会显示用途和实际命中的 `provider / model`
- 前端会把同一次 `llm started / completed` 合并成一张卡片展示，时间线区域支持滚动
- `MemoryManager` 与 `HybridRetriever` 改为按次创建 SQLite 连接，兼容 Gradio worker thread
- trace 写盘前会自动确保 [logs/traces](logs/traces) 目录存在
- 长文写作与质量增强使用单独的长输出 token 配置，减少综述中途截断

## 快速开始

### 方式一：使用现有 conda 环境

```bash
source /home/a1/miniconda3/etc/profile.d/conda.sh
conda activate agent
python run.py
```

### 方式二：新建虚拟环境

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py
```

启动后可选择：

- `1` Web 界面
- `2` 命令行
- `3` 功能验证

## 配置

### LLM API Key

项目支持多家兼容 OpenAI Chat Completions 的 provider。你可以通过环境变量或直接修改 [api_keys.py](api_keys.py) 提供密钥。

常用键包括：

- `SCNET_API_KEY`
- `SILICONFLOW_API_KEY`
- `ZHIPU_API_KEY`
- `DEEPSEEK_API_KEY`
- `DASHSCOPE_API_KEY`

### Web of Science

如果要启用 `Web of Science Starter API`，需要配置：

```bash
export WOS_STARTER_API_KEY="your-key"
```

可选配置：

```bash
export WOS_DOCUMENTS_URL="https://api.clarivate.com/apis/wos-starter/v1/documents"
export WOS_DATABASE="WOS"
```

对应设置位于 [settings.py](config/settings.py)。

## 使用示例

### 论文检索

```text
搜索近三年关于多智能体强化学习的论文
```

### 综述生成

```text
写一篇 SERF 效应的综述
```

### 概念解释

```text
解释一下 diffusion model 和 score matching 的关系
```

### 本地 PDF 建库

在 Web 界面上传 PDF 后，再提问：

```text
总结这篇论文的方法、实验设置和局限
```

## 执行模式

- 快速模式：`search -> write`
- 标准模式：`search -> analyze -> debate -> write`
- 完整模式：标准流程后再执行 `Self-MoA + MPSC`

说明：

- 只有在未开启快速模式时，`质量增强` 才会实际进入完整模式。
- trace 里的 `planning.task_config` 是任务规划器给出的建议配置，不等于最终实际运行模式。
- 实际运行模式由前端开关或 [AgentV2.set_mode](src/core/agent_v2.py) 决定。

## 前端可视化

Web 界面右侧提供三层可观测性：

- 执行时间线：实时滚动显示 `memory_recall / intent / slots / planning / search / llm / analyze / debate / write / quality`
- 步骤详情：查看任一步的 `input / output / metadata`
- 原始 Trace JSON：完整保留落盘 trace

其中 `llm` 步骤会展示：

- 本次调用用途，例如 `查询改写`、`论文分析`、`综述写作`
- 实际命中的 `provider / model`
- 调用耗时
- 失败信息

Trace 默认写入 [logs/traces](logs/traces)。

## 数据落盘

- 对话 trace： [logs/traces](logs/traces)
- 长期记忆库： [data/memory/memory.db](data/memory/memory.db)
- 用户反馈： [data/feedback/feedback.jsonl](data/feedback/feedback.jsonl)

其中：

- trace 保存完整执行步骤、最终输出和模型调用信息
- `memory.db` 保存会话级记忆条目
- `feedback.jsonl` 只在显式提交反馈时写入

## Provider 诊断

项目根目录提供了 [test_provider_access.py](test_provider_access.py)，可直接诊断某个 provider 的请求链路：

```bash
source /home/a1/miniconda3/etc/profile.d/conda.sh
conda activate agent
python test_provider_access.py --provider scnet
```

它会打印实际请求 URL、代理环境变量、HTTP 状态码和原始响应，适合排查 `SCNet` 或其他 provider 的连通性问题。

## 核心模块

- [AgentV2](src/core/agent_v2.py)：总控入口，串联意图识别、槽位填充、任务规划、多智能体执行、质量增强、记忆与 trace
- [LLMManager](src/core/llm.py)：provider 管理、重试、故障转移、模型调用 trace
- [QueryRewriter](src/preprocessing/query_rewriter.py)：LLM 驱动查询重写
- [MultiAgentCoordinator](src/agents/multi_agent.py)：Search / Analyze / Debate / Write / Coder 协作
- [HybridRetriever](src/rag/retriever.py)：本地 RAG 检索
- [WhiteboxTracer](src/whitebox/tracer.py)：trace 持久化
- [Gradio UI](src/ui/gradio_app.py)：实时执行时间线与步骤详情展示

## 相关文档

- 完整项目文档：[ScholarAgent_完整项目文档.md](ScholarAgent_完整项目文档.md)
- RAG 流程图：[RAG_v3_完整流程图.html](RAG_v3_完整流程图.html)

## 说明

- 当前 Web 前端是“步骤级实时刷新”，不是 token 级流式输出正文。
- 无真实 LLM provider 可用时，系统仍可运行，但查询改写和高质量生成能力会明显下降。
- `Web of Science` 真实可用性取决于 `WOS_STARTER_API_KEY` 与网络环境。
