# ScholarAgent V2 完整项目文档

> 智能学术研究助手 - 基于多Agent协作与高级推理的论文检索与分析系统

---

## 目录

1. [项目概述](#1-项目概述)
2. [核心功能](#2-核心功能)
3. [系统架构](#3-系统架构)
4. [项目结构详解](#4-项目结构详解)
5. [核心模块实现](#5-核心模块实现)
6. [关键技术与难点](#6-关键技术与难点)
7. [问题与解决方案](#7-问题与解决方案)
8. [面试讲解版本](#8-面试讲解版本)
9. [项目复现指南](#9-项目复现指南)
10. [API参考](#10-api参考)

---

## 1. 项目概述

### 1.1 项目背景

学术研究人员在进行文献调研时面临以下痛点：
- 需要在多个数据库（arXiv、Semantic Scholar、OpenAlex、Web of Science）之间切换搜索
- 论文数量庞大，筛选和分析耗时
- 综述撰写需要大量人工整理
- 中文术语、缩写和中英混合主题在英文数据库中的检索质量不稳定
- 缺乏智能化的研究辅助工具

### 1.2 项目目标

构建一个**智能学术研究助手**，实现：
- 🔍 多源论文检索与聚合
- 📊 智能论文分析与对比
- 📝 自动综述生成
- 💬 自然语言交互界面

### 1.3 技术栈

| 类别 | 技术 |
|------|------|
| 编程语言 | Python 3.11+ |
| LLM集成 | 多Provider统一接入（SCNet、硅基流动、智谱AI、DeepSeek 等） |
| 学术API | arXiv、OpenAlex、Semantic Scholar、Web of Science Starter API |
| Web框架 | Gradio 6.x |
| 数据库 | SQLite（长期记忆 + RAG chunk元数据） + ChromaDB（本地向量库） |
| 检索与RAG | LLM查询重写 + TF-IDF/BM25词法检索 + BGE-M3向量检索 + RRF融合 + BGE-Reranker + CRAG式验证 |

### 1.4 项目规模

```
总代码行数: ~5,000行（Python源码）
核心模块: 17个
Python文件: 53个
测试覆盖: 34/35项功能测试通过
```

---

## 2. 核心功能

### 2.1 七大高级特性

| 特性 | 描述 | 实现模块 |
|------|------|----------|
| **任务分层与LLM分级** | 5级任务复杂度 × 3级LLM能力匹配 | `planning/task_hierarchy.py` |
| **白名单管理** | 动态控制Agent可调用的工具范围 | `whitelist/manager.py` |
| **工具自演化** | LLM自动生成新工具代码 | `evolution/tool_generator.py` |
| **白盒过程追踪** | 完整记录推理链路，并在前端实时展示执行时间线 | `whitebox/tracer.py` |
| **反馈与人机协作** | 用户反馈收集与学习 | `feedback/collector.py` |
| **长期记忆系统** | 跨会话知识积累与检索 | `memory/manager.py` |
| **Prompt模板库** | 可复用的提示词管理 | `prompt_templates/manager.py` |

### 2.1.1 最近新增能力

- `Web of Science Starter API` 已接入统一搜索链路，与 `arXiv / OpenAlex / Semantic Scholar` 一起参与聚合排序。
- 查询重写已从“静态术语词表改写”升级为“LLM 输出结构化重写计划”，同时生成 `english_query / external_queries / local_queries`。
- 本地 RAG 已切换为 `BGE-M3 embedding + ChromaDB 持久化向量库 + BGE-Reranker`，不再使用 `FAISS`。
- Gradio 右侧新增实时执行时间线，不再只展示原始 JSON。
- 每次 LLM 调用会在 trace 中记录 `purpose / provider / model / latency / error`，前端可直接看到实际命中的模型。
- 规划器输出的 `reasoning_modes / enable_multi_agent / max_llm_calls` 已接入运行时执行，不再只是 trace 字段。
- OpenAI 兼容 provider 的 `base_url` 会统一规范到完整 `chat/completions` endpoint，减少接口地址配置错误。
- `MemoryManager` 与 `HybridRetriever` 已改为按次创建 SQLite 连接，兼容 Gradio worker thread。
- trace 每次写盘前都会确保 `logs/traces` 目录存在，避免目录缺失导致写盘失败。
- 执行时间线中的 `analyze / reasoning / debate / write` 阶段卡片会内联展示实际命中的模型。
- 前端新增浏览器侧历史持久化，主题切换、浏览器返回和刷新后可恢复最近对话记录。
- 长文写作与质量增强使用单独的长输出 token 配置，减少综述正文中途截断。
- 新增 `rebuild_chroma_index.py`，可从现有 `rag_index.db` 一键重建 `ChromaDB` 向量索引。
- `ReAct` 已升级为真实工具循环，可调用本地 RAG 与白名单工具。
- `ToT` 已升级为显式分支扩展、打分与剪枝的树搜索。
- `Debate` 已升级为正方、反方、复辩、主持四轮多代理对辩。
- 搜索工具规划阶段固定使用 `zhipu`；若 `zhipu` 不可用，则回退到确定性检索。

### 2.2 支持的任务类型

```python
SUPPORTED_INTENTS = [
    "search_papers",      # 论文搜索
    "explain_concept",    # 概念解释
    "compare_methods",    # 方法对比
    "generate_survey",    # 综述生成
    "generate_code",      # 代码生成
    "analyze_paper",      # 论文分析
    "daily_update",       # 每日更新
]
```

### 2.3 执行模式

| 模式 | 含义 | 预计时间 |
|------|------|----------|
| ⚡ 快速模式 | 走 `intent_flows_fast`，保留该意图的最小完整链路 | 15-30秒 |
| 📝 标准模式 | 走 `intent_flows_full`，不启用质量增强 | 30-60秒 |
| 📚 完整模式 | 走 `intent_flows_full`，并在答案生成后叠加 `MoA + MPSC` 质量增强 | 60-120秒 |

说明：

- `quality_mode=True` 只有在 `fast_mode=False` 时才会真正进入 `FULL` 执行模式。
- `AgentV2.set_mode()` 决定的是用户请求的执行模式上限；真正执行时还会再叠加规划器约束。
- 如果规划结果里 `enable_multi_agent=False`，即使用户选择了 `STANDARD / FULL`，基础 flow 也会收缩到 `intent_flows_fast`。
- `max_llm_calls` 现在是运行时真实预算，但统计范围限定为执行阶段的高层 LLM 调用；意图识别、查询改写、RAG 相关性判断不计入该预算。
- trace 中 `planning.task_config` 仍保留原始规划结果，实际运行模式继续写入 trace metadata 的 `mode`。

#### 2.3.1 各 Intent 的实际执行流程

| Intent | 快速模式 | 标准模式 | 完整模式 |
|--------|----------|----------|----------|
| `search_papers` | `search -> write` | `search -> write` | `search -> write -> quality` |
| `explain_concept` | `write` | `search -> write` | `search -> write -> quality` |
| `analyze_paper` | `search -> analyze -> write` | `search -> analyze -> write` | `search -> analyze -> write -> quality` |
| `daily_update` | `search -> analyze -> write` | `search -> analyze -> write` | `search -> analyze -> write -> quality` |
| `compare_methods` | `search -> analyze -> write` | `search -> analyze -> debate -> write` | `search -> analyze -> debate -> write -> quality` |
| `generate_survey` | `search -> analyze -> write` | `search -> analyze -> debate -> write` | `search -> analyze -> debate -> write -> quality` |
| `generate_code` | `search -> analyze -> coder` | `search -> analyze -> coder` | `search -> analyze -> coder -> quality` |

补充说明：

- `analyze_paper` 与 `generate_survey` 现在已经拆开：前者走“单篇论文解读”，后者走“领域综述写作”。
- `analyze_paper` 如果本地 RAG 已命中上传论文片段，会优先采用 `local_rag_only` 的搜索模式，不再先按综述思路去外部扩搜大量论文。
- `quality` 阶段发生在 `AgentV2` 主流程中，而不是 `MultiAgentCoordinator` 内部，所以完整模式是在基础 flow 之后额外叠加。
- 上表描述的是各模式的上限 flow；若规划器给出 `enable_multi_agent=False`，系统会把 `STANDARD / FULL` 的基础 flow 自动降到对应的快速 flow。

---

## 3. 系统架构

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户界面层                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Gradio UI  │  │   CLI模式   │  │    API接口 (预留)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Agent V2 核心层                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    AgentV2 主控制器                      │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │    │
│  │  │意图识别  │ │槽位填充  │ │任务规划  │ │执行调度  │   │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Multi-Agent   │ │  Reasoning层    │ │  Quality层      │
│   协作系统       │ │                 │ │                 │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │ SearchAgent │ │ │ │   Direct    │ │ │ │  Self-MoA   │ │
│ │AnalyzeAgent │ │ │ │   CoT       │ │ │ │  (多模型     │ │
│ │ DebateAgent │ │ │ │   ReAct     │ │ │ │   聚合)     │ │
│ │ WriteAgent  │ │ │ │   ToT       │ │ │ └─────────────┘ │
│ │ CoderAgent  │ │ │ │   Debate    │ │ │ ┌─────────────┐ │
│ └─────────────┘ │ │ │  Reflection │ │ │ │    MPSC     │ │
└─────────────────┘ │ │   CoVe      │ │ │ │ (多路径      │ │
                    │ └─────────────┘ │ │ │  自洽验证)   │ │
                    └─────────────────┘ │ └─────────────┘ │
                                        └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        工具与服务层                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ │
│  │  arXiv   │ │ OpenAlex │ │ Semantic │ │   WoS    │ │  PDF  │ │
│  │   API    │ │   API    │ │ Scholar  │ │ Starter  │ │Parser │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └───────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │   RAG    │ │ 长期记忆 │ │ 白盒追踪 │ │ 工具注册 │           │
│  │ Retriever│ │  SQLite  │ │  Tracer  │ │ Registry │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        LLM Provider层                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  SCNet   │ │ 硅基流动 │ │  智谱AI  │ │ DeepSeek │  ...      │
│  │MiniMax   │ │DeepSeek  │ │  GLM-4   │ │          │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│         智能故障转移 + 健康检查 + 模型调用Trace绑定             │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 数据流图

```
用户输入
    │
    ▼
┌──────────────────┐
│  意图识别        │ ──→ search_papers / generate_survey / ...
│  IntentClassifier│
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  槽位填充        │ ──→ {topic, time_range, max_papers, ...}
│  SlotFiller      │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  任务分层        │ ──→ L1~L5 任务等级
│  TaskHierarchy   │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Multi-Agent执行 │
│  ┌────┐┌────┐   │
│  │搜索││分析│   │ ──→ 论文列表 + 分析结果
│  └────┘└────┘   │
│  ┌────┐┌────┐   │
│  │辩论││写作│   │ ──→ 最终综述
│  └────┘└────┘   │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  质量增强        │ ──→ 优化后的输出
│  MoA + MPSC      │
└──────────────────┘
    │
    ▼
用户输出 + 白盒追踪记录
```

---

## 4. 项目结构详解

### 4.1 完整目录树

```
scholar-agent/
│
├── 📄 run.py                    # 程序入口（Web/CLI/测试三种模式）
├── 📄 api_keys.py               # API密钥配置（10种LLM提供商）
├── 📄 requirements.txt          # Python依赖
├── 📄 verify_features.py        # 功能验证测试脚本
├── 📄 rebuild_chroma_index.py   # 从 SQLite chunk 重建 ChromaDB 索引
├── 📄 README.md                 # 项目说明
│
├── 📁 config/                   # 配置文件
│   ├── __init__.py
│   ├── settings.py              # 全局设置（超时、路径、模型参数）
│   └── intent_config.py         # 意图配置（8种意图+槽位定义）
│
├── 📁 src/                      # 核心源码（~7,500行）
│   │
│   ├── 📁 core/                 # 核心模块（2,599行）
│   │   ├── __init__.py
│   │   ├── models.py            # 数据模型（Paper, SearchResult, etc.）
│   │   ├── llm.py               # LLM管理器（10个Provider + 故障转移）
│   │   ├── agent.py             # Agent V1（基础版，保留兼容）
│   │   └── agent_v2.py          # Agent V2（完整版，集成7大特性）
│   │
│   ├── 📁 preprocessing/        # 预处理模块（917行）
│   │   ├── __init__.py
│   │   ├── intent_classifier.py # 意图分类器
│   │   ├── slot_filler.py       # 槽位填充（一次性收集所有必需信息）
│   │   ├── query_rewriter.py    # 查询重写
│   │   └── dialogue_manager.py  # 多轮对话管理
│   │
│   ├── 📁 agents/               # Multi-Agent系统（649行）
│   │   ├── __init__.py
│   │   └── multi_agent.py       # 5个专业Agent协作
│   │       ├── SearchAgent      # 多源论文搜索
│   │       ├── AnalyzeAgent     # 论文分析
│   │       ├── DebateAgent      # 多视角辩论
│   │       ├── WriteAgent       # 综述撰写
│   │       └── CoderAgent       # 代码生成
│   │
│   ├── 📁 tools/                # 工具系统（1,579行）
│   │   ├── __init__.py
│   │   ├── registry.py          # 工具注册中心
│   │   ├── arxiv_tool.py        # arXiv API
│   │   ├── openalex_tool.py     # OpenAlex API
│   │   ├── semantic_scholar_tool.py  # Semantic Scholar API
│   │   ├── web_of_science_tool.py    # Web of Science Starter API
│   │   └── pdf_tool.py          # PDF解析工具
│   │
│   ├── 📁 reasoning/            # 推理引擎（571行）
│   │   ├── __init__.py
│   │   └── engine.py            # 7种推理模式
│   │       ├── Direct           # 直接回答
│   │       ├── CoT              # 链式思维
│   │       ├── ReAct            # 推理+行动
│   │       ├── ToT              # 思维树
│   │       ├── Debate           # 辩论式推理
│   │       ├── Reflection       # 反思优化
│   │       └── CoVe             # 验证链
│   │
│   ├── 📁 quality/              # 质量增强（327行）
│   │   ├── __init__.py
│   │   └── enhancer.py
│   │       ├── Self-MoA         # 多模型答案聚合
│   │       └── MPSC             # 多路径自洽验证
│   │
│   ├── 📁 planning/             # 任务规划（~500行）
│   │   ├── __init__.py
│   │   └── task_hierarchy.py    # 5级任务分层 + LLM分级调度
│   │
│   ├── 📁 memory/               # 长期记忆（710行）
│   │   ├── __init__.py
│   │   └── manager.py           # SQLite存储 + TF-IDF/BM25/重要性混合召回
│   │
│   ├── 📁 rag/                  # RAG检索（~400行）
│   │   ├── __init__.py
│   │   ├── bge_m3_embedder.py   # BGE-M3 向量编码
│   │   ├── bge_reranker.py      # BGE-Reranker 重排序
│   │   ├── vector_store.py      # ChromaDB 持久化向量库
│   │   └── retriever.py         # 词法+向量并行检索 + RRF + CRAG
│   │
│   ├── 📁 whitebox/             # 白盒追踪（~450行）
│   │   ├── __init__.py
│   │   └── tracer.py            # 完整推理链路记录
│   │
│   ├── 📁 whitelist/            # 白名单管理（~380行）
│   │   ├── __init__.py
│   │   └── manager.py           # 工具访问控制
│   │
│   ├── 📁 evolution/            # 自演化系统（~450行）
│   │   ├── __init__.py
│   │   └── tool_generator.py    # LLM自动生成新工具
│   │
│   ├── 📁 feedback/             # 反馈系统（~480行）
│   │   ├── __init__.py
│   │   └── collector.py         # 用户反馈收集与HITL
│   │
│   ├── 📁 prompt_templates/     # Prompt模板（~580行）
│   │   ├── __init__.py
│   │   └── manager.py           # 模板管理与渲染
│   │
│   └── 📁 ui/                   # 用户界面（~350行）
│       ├── __init__.py
│       └── gradio_app.py        # Gradio Web界面 + 实时执行时间线
│
├── 📁 data/                     # 数据目录
│   ├── prompts/                 # Prompt模板文件
│   ├── memory/                  # 记忆数据库
│   ├── feedback/                # 反馈数据
│   ├── evaluation/              # 评估结果
│   └── whitelist.json           # 白名单配置
│
├── 📁 docs/                     # 文档
│   ├── COMPLETE_PROJECT_DOCUMENTATION.md  # 本文档
│   ├── PROJECT_DOCUMENTATION.md # 简版文档
│   ├── INTERVIEW_GUIDE.md       # 面试指南
│   └── QUICKSTART.md            # 快速开始
│
├── 📁 cache/                    # 缓存目录
├── 📁 logs/                     # 日志目录
└── 📁 reports/                  # 报告输出
```

### 4.2 模块依赖关系

```
                    ┌─────────────┐
                    │   run.py    │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ gradio_  │ │  CLI     │ │  Tests   │
        │ app.py   │ │  Mode    │ │          │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             └────────────┼────────────┘
                          ▼
                   ┌──────────────┐
                   │  agent_v2.py │
                   └──────┬───────┘
                          │
     ┌────────────────────┼────────────────────┐
     ▼                    ▼                    ▼
┌─────────┐        ┌─────────────┐       ┌─────────┐
│ prepro- │        │ multi_agent │       │ planning│
│ cessing │        │    .py      │       │         │
└────┬────┘        └──────┬──────┘       └────┬────┘
     │                    │                   │
     │     ┌──────────────┼───────────────────┤
     │     ▼              ▼                   ▼
     │ ┌───────┐    ┌──────────┐        ┌─────────┐
     │ │tools/ │    │reasoning/│        │ quality/│
     │ └───┬───┘    └────┬─────┘        └────┬────┘
     │     │             │                   │
     │     │     ┌───────┴───────┐           │
     │     │     ▼               ▼           │
     │     │ ┌───────┐     ┌─────────┐       │
     │     │ │ rag/  │     │ memory/ │       │
     │     │ └───────┘     └─────────┘       │
     │     │                                 │
     └─────┴────────────────┬────────────────┘
                            ▼
                     ┌────────────┐
                     │  llm.py    │
                     │ LLMManager │
                     └────────────┘
```

---

## 5. 核心模块实现

### 5.1 LLM管理器 (`src/core/llm.py`)

#### 5.1.1 多Provider架构

```python
class LLMManager:
    """
    统一管理多家兼容 OpenAI Chat Completions 的 Provider
    - 智能故障转移
    - 健康检查与自动恢复
    - Trace 绑定与模型调用可观测性
    - 执行阶段 LLM 调用预算控制
    """

    def __init__(self):
        self.providers = {"mock": MockProvider()}
        self.provider_status = {"mock": ProviderStatus()}
        self._failure_threshold = 3
        self._recovery_time = 300
        self._trace_id_var = ContextVar("llm_trace_id", default="")
        self._tracer_var = ContextVar("llm_tracer", default=None)
```

#### 5.1.2 支持的LLM提供商

| 提供商 | 模型 | 免费额度 | 优先级 |
|--------|------|----------|--------|
| SCNet | MiniMax-M2.5 | 依账号配置 | ⭐⭐⭐⭐ |
| 硅基流动 | DeepSeek-V3 | 2000万Token | ⭐⭐⭐ |
| 智谱AI | GLM-4.7 | 完全免费 | ⭐⭐⭐ |
| 阿里百炼 | Qwen-Turbo | 100万Token | ⭐⭐ |
| 百度千帆 | ERNIE-Speed | 免费 | ⭐⭐ |
| 腾讯混元 | Hunyuan-Lite | 免费 | ⭐ |
| 讯飞星火 | Spark-Lite | 免费 | ⭐ |
| 字节豆包 | Doubao-Lite | 免费 | ⭐ |
| Moonshot | moonshot-v1 | 15元 | ⭐ |
| 百川智能 | Baichuan2 | 免费 | ⭐ |
| DeepSeek | deepseek-chat | 10元 | ⭐⭐ |

#### 5.1.3 故障转移机制

```python
def call_with_fallback(self, prompt: str, **kwargs) -> str:
    """
    智能故障转移流程：
    1. 获取健康Provider列表
    2. 按优先级依次尝试
    3. 记录成功/失败状态
    4. 超过阈值暂时禁用
    5. 5分钟后自动恢复
    6. 全部失败使用Mock
    """
    healthy_providers = self._get_healthy_providers()

    for provider_name in healthy_providers:
        try:
            result = self._invoke_provider(
                provider_name=provider_name,
                prompt=prompt,
                **kwargs,
            )
            self._record_success(provider_name)
            return result
        except Exception:
            self._record_failure(provider_name)
            continue

    return self._invoke_provider("mock", prompt=prompt, **kwargs)
```

#### 5.1.4 模型调用追踪

当前 `LLMManager` 不再只是“返回文本”，还会把每次真实模型调用写入 trace：

```python
tracer.trace_step(
    trace_id,
    "llm",
    {
        "call_id": call_id,
        "purpose": purpose,
        "requested_provider": requested_provider or "auto",
        "prompt_preview": prompt[:240],
    },
    {
        "status": "running|success|error",
        "provider": provider.name,
        "model": provider.model,
        "latency_ms": latency_ms,
        "error": error,
    },
)
```

这使前端可以实时展示：

- 当前这次调用要做什么，例如 `查询改写`、`论文分析`、`综述写作`
- 实际命中的 `provider / model`
- 调用耗时和失败信息

另外，`LLMManager` 现在还维护每轮请求的执行预算：

- `bind_budget(max_calls)`：在规划完成后绑定本轮预算
- `budgeted=True`：只有被标记为执行阶段高层调用的请求才会消耗预算
- 预算耗尽时会抛出明确错误：`LLM call budget exceeded: used x/y`
- `llm` trace 步骤会额外写入 `budget_limit / budget_used / budget_remaining`

当前纳入预算的调用包括：

- `AnalyzeAgent` 的论文分析
- `WriteAgent / CoderAgent` 的生成
- `ReasoningEngine` 的各类推理模式
- `QualityEnhancer` 的候选生成、聚合与验证

当前不纳入预算的调用包括：

- 意图识别
- 查询改写
- RAG 相关性判断

这样做的目的是让 `max_llm_calls` 真正约束主执行链，而不是在预处理阶段被提前耗尽。

#### 5.1.5 Provider配置与响应解析

为兼容不同 OpenAI 风格 provider 的返回体与 endpoint 习惯，`LLMManager` 还处理了两类运行时问题：

```python
def _resolve_chat_completions_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"

def _extract_response_text(body: Dict[str, Any]) -> Optional[str]:
    # 兼容 message.content / legacy text / output_text / output block list
```

这两部分分别解决：

- provider 只配置根 URL 时，自动补全到 `chat/completions`
- 某些模型返回 `content=None`、内容块列表或 `output_text` 时，仍能正确提取文本

因此 `Analyze / Write / Quality` 等上游模块拿到的是“已解析完成的文本”，而不是原始响应体。

### 5.2 Multi-Agent协作 (`src/agents/multi_agent.py`)

#### 5.2.1 Agent定义

```python
class SearchAgent:
    """多源论文搜索，聚合 arXiv / OpenAlex / Semantic Scholar / Web of Science"""
    
class AnalyzeAgent:
    """论文深度分析，提取核心贡献与方法"""
    
class DebateAgent:
    """多视角辩论，平衡不同观点"""
    
class WriteAgent:
    """综述撰写，支持快速/完整两种模式"""
    
class CoderAgent:
    """代码生成，基于论文方法实现"""
```

#### 5.2.2 协作流程

```python
class MultiAgentCoordinator:
    # 完整流程（默认）
    intent_flows_full = {
        "generate_survey": ["search", "analyze", "debate", "write"],
        "compare_methods": ["search", "analyze", "debate", "write"],
        "generate_code": ["search", "analyze", "coder"],
        "search_papers": ["search", "write"],
        "daily_update": ["search", "analyze", "write"],
        "analyze_paper": ["search", "analyze", "write"],
        "explain_concept": ["search", "write"],
    }

    # 快速流程
    intent_flows_fast = {
        "generate_survey": ["search", "analyze", "write"],
        "compare_methods": ["search", "analyze", "write"],
        "generate_code": ["search", "analyze", "coder"],
        "search_papers": ["search", "write"],
        "daily_update": ["search", "analyze", "write"],
        "analyze_paper": ["search", "analyze", "write"],
        "explain_concept": ["write"],
    }
```

当前设计不是“所有意图共用一条搜索-综述链路”，而是按意图拆成不同任务形态：

- `search_papers`：只负责搜和列结果，不做深度写作
- `explain_concept`：以解释为目标，快速模式下可直接写，标准模式会先补充检索材料
- `analyze_paper`：面向单篇论文解读，包含 `analyze`，但不走 `debate`
- `daily_update`：面向近期动态归纳，包含 `analyze`，不走 `debate`
- `compare_methods`：标准/完整模式包含 `debate`，因为需要显式比较和权衡
- `generate_survey`：标准/完整模式包含 `debate`，因为要综合多篇工作并组织成综述
- `generate_code`：最终阶段不是 `write`，而是 `coder`

`WriteAgent` 也不再统一使用综述模板，而是按意图切换：

- `generate_survey -> survey_writer`
- `compare_methods -> compare_writer`
- `analyze_paper -> paper_answer_writer`
- `daily_update -> daily_update_writer`
- `explain_concept -> concept_writer`

这部分的直接目的是避免“意图识别正确，但写作阶段仍然按综述生成”的结构性错误。

当前 `MultiAgentCoordinator` 还会把规划器配置接进执行期：

- `enable_multi_agent=False` 时，基础 flow 会自动降到 `intent_flows_fast`
- `max_llm_calls` 会通过 `LLMManager` 预算约束执行阶段调用数
- `AnalyzeAgent` 不再固定分析 5 篇论文，而是会先为后续 `debate / write / coder` 预留预算，再决定当前最多分析多少篇

#### 5.2.3 LLM驱动查询重写与多源检索

`SearchAgent` 现在不会直接把用户原始中文问题丢给英文数据库，而是先调用 `QueryRewriter` 生成结构化检索计划：

```python
rewritten_queries = self.rewriter.rewrite(
    topic,
    intent=intent,
    target="external",
)
```

`QueryRewriter` 会通过一次 `LLM.call_json()` 输出：

- `core_topic`
- `english_query`
- `external_queries`
- `local_queries`

其中：

- `external_queries` 用于 `arXiv / OpenAlex / Semantic Scholar / Web of Science`
- `local_queries` 用于本地 RAG 检索

这样中文术语、缩写和中英混合主题会优先被改写成更适合英文数据库的标准学术检索式。

外部搜索阶段还新增了一层运行时约束：

- 搜索工具规划通过 `LangChain agent` 执行时，固定只使用 `zhipu`
- 若 `zhipu` 不可用、LangChain agent 运行失败或无可用工具，则自动退回确定性多源检索
- 搜索到的外部论文结果只保留在本轮 `search_result` 与 trace 中，不会自动写入本地 RAG 向量库

#### 5.2.4 本地 RAG 检索链路

当前本地 RAG 采用“共享改写计划 + 词法/向量并行检索 + 融合重排”的流程：

```python
rewrite_plan = self.rewriter.plan(topic, intent=intent)

local_result = self.retriever.retrieve(
    topic,
    chat_history=history,
    rewritten_queries=rewrite_plan.local_queries,
    rewrite_plan=rewrite_plan,
)
```

`HybridRetriever.retrieve()` 的核心步骤是：

1. 对 query 做 `conversation_enhance`
2. 复用 `QueryRewritePlan.local_queries`
3. 根据问题类型路由 `text_chunk / table_chunk / qa_chunk / kg_chunk`
4. 对每个 `(rewritten_query, source_type)` 并行执行两路召回
5. 词法路：`TF-IDF + BM25`
6. 向量路：`BGE-M3 embedding -> ChromaDB query`
7. 多路结果用 `RRF` 融合
8. 用 `BGE-Reranker` 对融合结果重排
9. 用 LLM 做 `CRAG` 式相关性判断
10. 若验证通过的 chunk 少于 3 条，再补充网页搜索片段

简化后的实现轮廓如下：

```python
tasks = [(rewritten_query, source_type) for rewritten_query in rewrites for source_type in routes]
ranked_lists = parallel_lexical_and_vector_search(tasks)
fused = self._rrf_fusion(ranked_lists)
reranked = self.reranker.rerank(query, fused)
validated, supplement = self._crag_validate(query, reranked[:top_k])
```

对 `analyze_paper` 这个意图，还额外有一条分流规则：

```python
if intent == "analyze_paper" and local_context.get("results"):
    return SearchResult(
        papers=[],
        trace={"local_rag": local_context, "search_mode": "local_rag_only"},
    )
```

含义是：

- 如果用户上传了 PDF，且本地 RAG 已经召回到这篇论文的片段
- 那么 `SearchAgent` 会优先把这轮任务当作“单篇文档解析”
- 而不是继续按综述任务去外部扩搜大量论文

### 5.3 任务分层 (`src/planning/task_hierarchy.py`)

#### 5.3.1 5级任务复杂度

```python
class TaskLevel(Enum):
    L1_SIMPLE = "simple"       # 简单问答
    L2_MODERATE = "moderate"   # 中等复杂度
    L3_COMPLEX = "complex"     # 复杂任务
    L4_ADVANCED = "advanced"   # 高级任务
    L5_EXPERT = "expert"       # 专家级任务
```

#### 5.3.2 LLM分级调度

```python
class LLMTier(Enum):
    LITE = "lite"           # 轻量模型（GLM-4-Flash）
    STANDARD = "standard"   # 标准模型（DeepSeek-V3）
    PREMIUM = "premium"     # 高级模型（预留）
```

#### 5.3.3 任务配置

```python
@dataclass
class TaskConfig:
    llm_tier: LLMTier
    max_reasoning_depth: int
    enable_multi_agent: bool
    enable_quality_enhance: bool
    reasoning_modes: List[str]
    max_llm_calls: int
    timeout_seconds: int

# 示例配置
L5_EXPERT_CONFIG = TaskConfig(
    llm_tier=LLMTier.STANDARD,
    max_reasoning_depth=5,
    enable_multi_agent=True,
    enable_quality_enhance=True,
    reasoning_modes=["cot", "debate", "reflection"],
    max_llm_calls=30,
    timeout_seconds=300,
)
```

#### 5.3.4 意图复杂度与 LLM 能力匹配标准

当前任务分层不是只看 query 长度，而是“按意图给基线分，再按槽位和上下文修正”：

```python
INTENT_BASE_SCORES = {
    "search_papers": 1,
    "explain_concept": 1,
    "analyze_paper": 2,
    "daily_update": 2,
    "compare_methods": 3,
    "generate_code": 4,
    "generate_survey": 4,
}
```

基线分的语义是：

- `1`：单目标、信息组织要求低
- `2`：单目标，但需要抽取论文方法、贡献、时间线等结构化信息
- `3`：需要显式比较或权衡
- `4`：需要多论文综合、代码落地或长篇组织能力

在意图基线分之上，还会做这些修正：

```python
if slots.get("comparison_target"):
    score += 1
if slots.get("time_range"):
    score += 1
if int(slots.get("max_papers") or 0) >= 20:
    score += 1
if len(query.strip()) >= 60:
    score += 1
if any(marker in query for marker in ("并且", "同时", "以及", "还要", "并说明", "并比较", "并分析")):
    score += 1
score = min(score, 5)
```

再把总分映射到任务等级和 `LLM` 能力层级：

| 总分 | TaskLevel | LLM Tier | 推理配置 |
|------|-----------|----------|----------|
| `<=1` | `simple` | `lite` | `direct` |
| `2` | `moderate` | `lite` | `cot` |
| `3` | `complex` | `standard` | `cot + react` |
| `4` | `advanced` | `standard` | `cot + debate + reflection` |
| `>=5` | `expert` | `standard` | `cot + debate + reflection`，更高调用预算 |

例如：

- `搜索 Transformer 论文`
  `search_papers` 基线分是 `1`，通常落在 `simple / lite`

- `这篇文章讲了什么，提出了什么新方法？`
  `analyze_paper` 基线分是 `2`，通常落在 `moderate / lite`

- `比较近三年两种多智能体强化学习方法，并分析优缺点`
  `compare_methods` 基线分 `3`，再叠加 `time_range + comparison_target + 多目标标记`，会升到 `advanced` 或 `expert`

- `写一篇近五年多智能体强化学习综述，并给出代表论文和未来方向`
  `generate_survey` 基线分 `4`，再叠加 `time_range + 多目标标记`，通常落到 `expert`

这些配置现在与运行时的关系如下：

- `reasoning_modes`：已经真实接入推理引擎，`ReasoningEngine` 会在规划器允许的模式集合内自动选模。
- `enable_multi_agent`：已经真实接入执行器；若为 `False`，`MultiAgentCoordinator` 会把基础 flow 收缩到 `intent_flows_fast`。
- `max_llm_calls`：已经真实接入 `LLMManager` 的预算控制，并进一步限制分析步最多分析多少篇论文，以便为后续 `debate / write / coder` 预留调用数。
- `llm_tier`：目前仍主要作为规划层标签与 trace 字段，尚未直接改写 Provider/模型路由。
- `enable_quality_enhance`：目前仍主要由 UI 与 `AgentV2.set_mode()` 控制，规划器字段暂未直接改写质量增强开关。
- `FAST / STANDARD / FULL` 仍由 `AgentV2.set_mode()` 决定，但运行时会叠加上述规划器约束。

举例：

- `moderate` 级的 `analyze_paper` 预算为 `4`。系统会先为 `write` 预留 1 次调用，因此 `AnalyzeAgent` 最多只分析 3 篇论文，而不是固定分析 5 篇。
- `advanced / expert` 级的 `generate_survey` 通常仍允许 `debate + write + quality` 保留，因此预算和可用推理模式都会明显更高。

### 5.4 推理引擎 (`src/reasoning/engine.py`)

#### 5.4.1 7种推理模式

```python
class ReasoningEngine:
    def reason(self, query, context, mode="auto"):
        if mode == "direct":
            return self._direct_answer(query, context)
        elif mode == "cot":
            return self._chain_of_thought(query, context)
        elif mode == "react":
            return self._react_loop(query, context)
        elif mode == "tot":
            return self._tree_of_thought(query, context)
        elif mode == "debate":
            return self._debate_reasoning(query, context)
        elif mode == "reflection":
            return self._reflection_loop(query, context)
        elif mode == "cove":
            return self._chain_of_verification(query, context)
```

当前推理模式选择已经不是单纯的关键词启发式，而是“两阶段决策”：

1. 先根据 query 语义生成候选模式
2. 再用规划器给出的 `reasoning_modes` 作为允许集合做约束

例如：

- 比较类问题优先尝试 `debate`
- 综述类问题优先尝试 `reflection`
- 实现/步骤类问题优先尝试 `react`
- 但如果规划器只允许 `["cot"]`，最终就会收缩到 `cot`

另外，`ReasoningEngine` 现在还会估算不同推理模式的调用开销：

- `direct / cot`：1次
- `react`：4次预算预估值
- `tot`：6次预算预估值
- `debate`：4次预算预估值
- `reflection`：2次
- `cove`：3次

这个估算值会被执行器用于给 `AnalyzeAgent` 留出后续 `debate / write` 所需预算。

当前三种关键模式已经不是“单条提示词模拟”：

- `react`：每轮先让模型输出 JSON 决策，再真实调用工具，再根据观察决定下一步；可调用本地 `search_local_rag` 与 `reasoning_agent` 白名单工具
- `tot`：先扩展多条候选分支，再对分支打分并做 beam 剪枝，最后基于最佳路径综合输出
- `debate`：按“正方立论 -> 反方质询 -> 正方复辩 -> 主持裁决”执行多轮多代理对辩

#### 5.4.2 推理模式对比

| 模式 | 适用场景 | 特点 |
|------|----------|------|
| Direct | 简单问答 | 快速，单次调用 |
| CoT | 逻辑推理 | 逐步推理 |
| ReAct | 需要工具调用 | 真实工具循环，逐轮 `thought -> action -> observation` |
| ToT | 复杂决策 | 显式分支扩展、打分与剪枝 |
| Debate | 争议性问题 | 正方/反方/复辩/主持的多代理对辩 |
| Reflection | 需要优化 | 自我反思改进 |
| CoVe | 需要验证 | 生成+验证+修正 |

### 5.5 质量增强 (`src/quality/enhancer.py`)

#### 5.5.1 Self-MoA (Mixture of Agents)

```python
def self_moa(self, query: str, context: str) -> MoAResult:
    """
    多模型答案聚合：
    1. 优先选取非 scnet 的真实 Provider 生成候选答案
    2. 若只有单个候选成功，则直接保留该候选
    3. 若多个候选成功，则再调用一次聚合
    4. 若候选或聚合失败，则保留增强前答案，不中断主流程
    """
    candidates = []
    errors = []
    for provider in explicit_providers:
        try:
            candidates.append(self.llm.call(candidate_prompt, provider=provider))
        except Exception as exc:
            errors.append(str(exc))

    if not candidates:
        return MoAResult(answer=context, candidates=[], errors=errors)
```

#### 5.5.2 MPSC (Multi-Path Self-Consistency)

```python
def mpsc_verify(self, query: str, answer: str) -> VerificationResult:
    """
    多路径自洽验证：
    1. 生成多条推理路径
    2. 检查结论一致性
    3. 计算置信度分数
    """
    paths = [call_verify_path(prompt) for prompt in prompts if call_succeeds]
    if len(paths) < 2:
        return VerificationResult(
            answer=answer,
            consistency_score=0.0,
            verdict="skipped_due_to_llm_error",
        )
    consistency = tfidf_cosine_consistency(paths)
    return VerificationResult(answer, consistency, paths)
```

当前完整模式还有一个重要行为约束：

- `quality` 阶段失败不会再导致整轮请求中断
- 如果某个候选、某条校验路径或聚合调用失败，系统会保留 `write` 阶段已经生成的答案
- trace 中会记录 `quality` 阶段的错误，但前端仍会返回可用答案
- `quality` 阶段的 LLM 调用现在也会计入 `max_llm_calls` 执行预算；若预算不足，质量增强会失败并保留基础答案，而不会打断整轮流程

### 5.6 长期记忆 (`src/memory/manager.py`)

#### 5.6.1 记忆类型

```python
class MemoryType(Enum):
    CONVERSATION = "conversation"  # 对话历史
    KNOWLEDGE = "knowledge"        # 知识片段
    PREFERENCE = "preference"      # 用户偏好
    FEEDBACK = "feedback"          # 反馈记录
```

#### 5.6.2 存储结构

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    type TEXT,
    content TEXT,
    metadata TEXT,
    importance REAL,
    access_count INTEGER,
    created_at TEXT,
    updated_at TEXT
);
```

#### 5.6.3 记忆检索

```python
def recall(self, query: str, memory_type: MemoryType = None, 
           limit: int = 5) -> List[Memory]:
    """
    混合检索策略：
    1. TF-IDF 语义相似度
    2. BM25 关键词得分
    3. 重要性加权
    4. 时间衰减
    """
    similarities = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
    bm25 = _bm25_scores(query, contents)
    score = 0.45 * similarities + 0.25 * bm25 + 0.2 * importance + 0.1 * recency_bonus
```

#### 5.6.4 线程与落盘行为

当前长期记忆与本地 RAG 的 SQLite 访问都改成了“按次连接、按次提交”的模式：

```python
def _connect(self) -> sqlite3.Connection:
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    return conn
```

这样做的直接原因是：

- Gradio 的提交处理运行在 worker thread 中
- 如果复用主线程里创建的 SQLite 连接，会触发 `SQLite objects created in a thread can only be used in that same thread`

目前的落盘位置分别是：

- 长期记忆：`data/memory/memory.db`
- RAG 索引：`data/memory/rag_index.db`
- trace：`logs/traces/<trace_id>.json`

---

## 6. 关键技术与难点

### 6.1 核心技术挑战

#### 挑战1：多LLM提供商的统一管理

**问题**：10种不同的API格式、认证方式、错误处理

**解决方案**：
```python
class LLMProvider(ABC):
    """抽象基类，统一接口"""
    
    def _request_with_retry(self, url, headers, data):
        """通用重试机制"""
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(url, headers=headers, 
                                         json=data, timeout=self.timeout)
                if response.status_code == 200:
                    return self._extract_content(response.json())
            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep((attempt + 1) * 2)  # 指数退避
        raise last_error
```

#### 挑战2：中文学术搜索质量差

**问题**：中文术语、缩写和中英混合表达在英文数据库中的检索质量差，例如 `SERF效应`、`多智能体强化学习综述`

**解决方案**：
```python
raw = self.llm.call_json(
    rewrite_prompt,
    purpose="查询改写",
)

english_query = raw["english_query"]
external_queries = raw["external_queries"]
local_queries = raw["local_queries"]
```

当前链路不再依赖静态中英文词表，而是：

1. 用 LLM 提炼 `core_topic`
2. 生成英文主检索式 `english_query`
3. 生成外部数据库检索集合 `external_queries`
4. 生成本地 RAG 检索集合 `local_queries`
5. 结果返回后再做标题/摘要相关性检查

#### 挑战3：综述生成时间过长

**问题**：完整流程需要94秒，用户体验差

**解决方案**：
```python
# 多模式设计
class AgentV2:
    def set_mode(self, fast_mode=False, enable_quality_enhance=True):
        """
        快速模式: 15-30秒 (跳过analyze, debate)
        标准模式: 30-60秒 (完整流程，无质量增强)
        完整模式: 60-120秒 (完整流程 + MoA + MPSC)
        """
```

#### 挑战4：槽位填充的多轮交互

**问题**：每次只问一个问题，用户需要多次交互

**解决方案**：
```python
def fill_slots_once(self, query, intent):
    """一次性收集所有缺失槽位"""
    missing = self._get_missing_required_slots(intent, current_slots)
    if missing:
        prompt = f"请提供以下信息：{missing}"
        return {"ask": prompt}  # 一次性询问所有
```

### 6.2 架构设计难点

#### 难点1：任务复杂度与资源的平衡

```
简单问题 ──→ 轻量模型 + 直接回答 ──→ 快速响应
    │
复杂问题 ──→ 标准模型 + CoT推理 ──→ 中等延迟
    │
专家任务 ──→ 多模型 + 辩论+聚合 ──→ 高质量输出
```

#### 难点2：白盒可追踪性

```python
class WhiteboxTracer:
    """记录完整推理过程与模型调用过程"""
    
    def trace_step(self, step_type, input_data, output_data, metadata):
        self.steps.append({
            "type": step_type,
            "input": input_data,
            "output": output_data,
            "timestamp": datetime.now(),
            "metadata": metadata,
        })
    
    def get_reasoning_chain(self):
        """返回完整推理链，可视化展示"""
        return self.steps
```

当前白盒追踪已经分成两层：

- 业务步骤：`memory_recall / intent / slots / planning / search / analyze / debate / write / quality`
- 模型步骤：`llm`

其中 `llm` 步骤会额外记录：

- `purpose`
- `provider`
- `model`
- `latency_ms`
- `error`

Gradio 前端会实时轮询 trace，并将这些步骤渲染成右侧执行时间线，而不再只展示 JSON。前端展示层还会：

- 在运行中显示“这次模型调用要做什么”
- 完成后显示 `provider / model / latency`
- 在 `analyze / reasoning / debate / write` 阶段卡片中内联展示阶段实际使用的模型
- 用浏览器本地状态恢复最近对话，因此主题切换、返回和刷新后通常还能看到最近历史
- 将时间线放入固定高度滚动容器，便于长链路观察

---

## 7. 问题与解决方案

### 7.1 开发过程中遇到的问题

#### 问题1：Gradio 6.x 兼容性问题

**现象**：
```
TypeError: Chatbot.__init__() got an unexpected keyword argument 'show_copy_button'
```

**原因**：Gradio 6.x移除了多个参数

**解决**：
```python
# 修改前 (Gradio 4.x)
chatbot = gr.Chatbot(show_copy_button=True, type="messages")

# 修改后 (Gradio 6.x)
chatbot = gr.Chatbot()  # 移除不支持的参数
```

#### 问题2：OpenAlex API返回None导致崩溃

**现象**：
```
TypeError: 'NoneType' object is not subscriptable
```

**原因**：API返回的数据中部分字段为None

**解决**：
```python
# 添加完整的空值检查
def _parse_work(work: Dict) -> Optional[Paper]:
    if not work or not isinstance(work, dict):
        return None
    
    title = work.get("title") or ""
    if not title:
        return None
    
    # 安全获取嵌套字段
    authors = []
    authorships = work.get("authorships")
    if authorships and isinstance(authorships, list):
        for a in authorships:
            if a and isinstance(a, dict):
                author = a.get("author")
                if author and isinstance(author, dict):
                    name = author.get("display_name")
                    if name:
                        authors.append(str(name))
```

#### 问题3：LLM超时导致整体失败

**现象**：
```
HTTPSConnectionPool: Read timed out. (read timeout=120)
```

**解决**：
```python
# 1. 减少单次超时时间
self.timeout = 60  # 从120减少到60

# 2. 添加重试机制
self.max_retries = 2

# 3. 智能故障转移
def call_with_fallback(self, prompt):
    for provider in self._get_healthy_providers():
        try:
            return provider.call(prompt)
        except:
            self._record_failure(provider)
    return self.mock_response()
```

#### 问题4：Dataclass默认值问题

**现象**：
```
TypeError: non-default argument follows default argument
```

**原因**：Python dataclass要求有默认值的字段必须在无默认值字段之后

**解决**：
```python
# 修改前 (错误)
@dataclass
class DialogueState:
    current_slots: Dict = field(default_factory=dict)
    intent: str  # 错误：无默认值在有默认值之后

# 修改后 (正确)
@dataclass
class DialogueState:
    intent: str = ""  # 添加默认值
    current_slots: Dict = field(default_factory=dict)
```

#### 问题5：搜索结果不精准

**现象**：中文问题直接搜索英文数据库时，容易召回主题漂移的结果

**原因**：
1. 用户原始输入不是学术检索式
2. 中文/缩写/中英混合主题没有被结构化展开
3. 本地 RAG 需要同时兼顾词法召回与向量召回
4. 外部源返回后缺少统一相关性过滤

**解决**：
```python
# 1. 先做 LLM 查询重写
plan = self.rewriter.plan(query, intent=intent)

# 2. 用多个 external_queries 搜索外部学术源
for rewritten in plan.external_queries[:3]:
    papers = TOOL_REGISTRY.call(tool_name, query=rewritten, ...)

# 3. 本地 RAG 用 local_queries 并行跑词法/向量检索
ranked_lists = parallel_lexical_and_vector_search(plan.local_queries)

# 4. 对外部返回和本地 chunk 再做统一相关性过滤
validated = self._crag_validate(query, candidates)
```

#### 问题6：Gradio 中 SQLite 跨线程报错

**现象**：
```
sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread
```

**原因**：
`MemoryManager` 和 `HybridRetriever` 早期持有长生命周期连接，而 Gradio 事件处理运行在 worker thread。

**解决**：
```python
def _connect(self) -> sqlite3.Connection:
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    return conn

with self._connect() as conn:
    rows = conn.execute(sql, params).fetchall()
```

即每次数据库操作单独创建连接，不跨线程复用连接对象。

#### 问题7：模型返回 `content=None` 导致上游崩溃

**现象**：
```
TypeError: 'NoneType' object is not subscriptable
```

**原因**：
部分 provider 的返回不是简单字符串，而可能是：

- `message.content = null`
- 内容块列表
- `output_text`

**解决**：
```python
def _extract_response_text(body: Dict[str, Any]) -> Optional[str]:
    # 统一兼容多种响应格式
```

如果仍然提取不到文本，则把这次 provider 调用视为失败，由 `LLMManager` 继续上抛或切换 provider，而不是把 `None` 传给业务层。

#### 问题8：trace 目录缺失导致写盘失败

**现象**：
```
FileNotFoundError: .../logs/traces/<trace_id>.json
```

**解决**：
```python
def _persist(self, trace_id: str) -> None:
    self.storage_dir.mkdir(parents=True, exist_ok=True)
    path = self.storage_dir / f"{trace_id}.json"
    path.write_text(...)
```

每次写 trace 前都重新确保目录存在。

#### 问题9：综述长文本在 `final_output` 中被截断

**现象**：
正文写到中途停止，trace 中 `final_output.answer` 只保存到半句。

**原因**：
写作调用复用了默认 `max_tokens`，不足以支撑长篇综述输出。

**解决**：
```python
llm_long_output_max_tokens: int = 32000
```

并让 `WriteAgent` 与质量增强阶段的长文生成显式使用这组长输出配置。

### 7.2 性能优化历程

| 版本 | 综述生成时间 | 优化措施 |
|------|-------------|----------|
| v1.0 | 120秒+ | 完整流程，无优化 |
| v1.1 | 94秒 | 减少LLM调用次数 |
| v1.2 | 60秒 | 添加快速模式 |
| v2.0 | 15-30秒(快速) | 多模式设计 |

---

## 8. 面试讲解版本

### 8.1 极简版（30秒电梯演讲）

> ScholarAgent是一个**智能学术研究助手**，可以自动搜索论文、分析对比、生成综述。
> 
> **核心亮点**：
> - 多Agent协作（搜索、分析、写作专业分工）
> - 7种推理模式（CoT、辩论、反思等）
> - 多Provider LLM 智能切换，并实时展示模型调用
> 
> 项目代码1.5万行，实现了任务分层、白盒追踪、长期记忆等7大高级特性。

### 8.2 精炼版（2分钟介绍）

> **项目背景**：
> 学术研究需要在多个数据库搜索论文，筛选分析耗时，综述撰写更是需要大量人工。
> 
> **解决方案**：
> 我开发了ScholarAgent，一个基于多Agent协作的智能研究助手。
> 
> **技术架构**：
> 1. **预处理层**：意图识别 + 槽位填充，理解用户需求
> 2. **Agent层**：5个专业Agent分工协作（搜索、分析、辩论、写作、编码）
> 3. **推理层**：7种推理模式，根据任务复杂度自动选择
> 4. **LLM层**：多Provider统一接入，智能故障转移，并写入模型调用 trace
> 
> **核心技术**：
> - 任务分层：5级复杂度 × 3级LLM能力匹配
> - 质量增强：Self-MoA多模型聚合 + MPSC自洽验证
> - 白盒追踪：完整记录推理过程与模型调用，可解释可审计
> 
> **技术难点**：
> - 中文学术搜索质量差 → LLM结构化查询重写 + 相关性过滤
> - 多LLM统一管理 → 抽象接口 + 智能故障转移 + 模型调用追踪
> - 响应时间过长 → 多模式设计（快速/标准/完整）
> 
> **项目成果**：
> - 代码1.5万行，48个模块
> - 功能测试34/35项通过
> - 综述生成从120秒优化到15-30秒

### 8.3 详细版（5分钟深度讲解）

#### 第一部分：项目概述（1分钟）

> 这是我独立开发的智能学术研究助手ScholarAgent，解决研究人员文献调研的痛点。
> 
> **核心功能**：
> - 多源论文搜索（arXiv、OpenAlex、Semantic Scholar、Web of Science）
> - 智能分析对比
> - 自动综述生成
> - 代码实现生成
> 
> **技术规模**：
> - Python 3.11，15000行代码
> - 17个核心模块
> - 7大高级特性

#### 第二部分：系统架构（1.5分钟）

> 采用分层架构设计：
> 
> **用户界面层**：
> Gradio Web界面，支持多轮对话，右侧实时显示执行时间线与步骤详情
> 
> **预处理层**：
> - 意图分类：8种任务类型识别
> - 槽位填充：一次性收集所有必需信息
> - 查询重写：LLM输出结构化检索计划
> 
> **核心Agent层**：
> 5个专业Agent协作：
> - SearchAgent：并行调用4个学术源 + 本地 RAG
> - AnalyzeAgent：提取论文核心贡献
> - DebateAgent：多视角辩论
> - WriteAgent：综述撰写（支持outline→draft两阶段）
> - CoderAgent：基于论文方法生成代码
> 
> **推理引擎**：
> 7种模式自动选择：Direct、CoT、ReAct、ToT、Debate、Reflection、CoVe
> 
> **质量增强**：
> - Self-MoA：多模型答案聚合
> - MPSC：多路径自洽验证
> 
> **LLM管理**：
> 多Provider统一接入，智能故障转移，健康检查与自动恢复，并记录模型调用 trace

#### 第三部分：关键技术（1.5分钟）

> **技术亮点1：任务分层与LLM分级**
> 
> 设计了5级任务复杂度（L1简单~L5专家），匹配3级LLM能力。
> 简单问题用轻量模型快速响应，复杂任务用强模型+多轮推理。
> 
> **技术亮点2：中文学术搜索优化**
> 
> 遇到的问题：搜索"强化学习"返回"党建学习"。
> 解决方案：
> 1. LLM生成 `english_query / external_queries / local_queries`
> 2. 用英文主检索式搜索外部数据库
> 3. 返回结果后做相关性二次检查
> 
> **技术亮点3：智能故障转移**
> 
> 10种LLM可能同时有多个不可用，设计了：
> - 健康检查机制
> - 故障计数与自动禁用
> - 5分钟自动恢复
> - 最终降级到Mock保证可用
> 
> **技术亮点4：白盒可追踪**
> 
> 完整记录每一步推理过程，包括：
> - 输入输出
> - 时间戳
> - 推理模式
> - 模型调用用途、provider、model、耗时和错误信息
> 
> 支持可视化展示，便于调试和审计。

#### 第四部分：问题解决（1分钟）

> **问题1：Gradio 6.x兼容性**
> 升级后多个参数不支持，通过查阅文档逐一修复。
> 
> **问题2：API空值崩溃**
> OpenAlex返回数据结构不固定，添加了完整的空值检查链。
> 
> **问题3：响应时间过长**
> 120秒→15-30秒，通过多模式设计解决。
> 
> **问题4：Dataclass默认值顺序**
> Python语法要求有默认值在后，重新排序字段解决。

### 8.4 技术深挖版（针对技术面试官）

#### Q1: 为什么选择Multi-Agent架构？

> 单Agent难以处理复杂学术任务，存在以下问题：
> 1. 提示词过长导致LLM注意力分散
> 2. 不同子任务需要不同专业知识
> 3. 难以实现并行处理
> 
> Multi-Agent架构的优势：
> 1. **专业分工**：每个Agent专注单一任务，提示词精简
> 2. **模块解耦**：独立开发测试，易于扩展
> 3. **流程可控**：可视化协作流程，便于调试
> 
> 我设计了5个Agent：
> - SearchAgent：论文搜索，处理3个API的异步调用
> - AnalyzeAgent：深度分析，提取核心贡献
> - DebateAgent：多视角辩论，产生更全面的观点
> - WriteAgent：综述撰写，支持outline→draft两阶段
> - CoderAgent：代码生成，基于论文方法实现

#### Q2: 如何处理LLM故障？

> 设计了三层容错机制：
> 
> **第一层：单Provider重试**
> ```python
> max_retries = 2
> timeout = 60  # 秒
> backoff = exponential  # 2s, 4s
> ```
> 
> **第二层：跨Provider故障转移**
> ```python
> providers = ["siliconflow", "zhipu", "deepseek", ...]
> for provider in healthy_providers:
>     try:
>         return provider.call(prompt)
>     except:
>         record_failure(provider)
> ```
> 
> **第三层：健康检查与自动恢复**
> ```python
> if failure_count >= 3:
>     disable_provider(300)  # 禁用5分钟
> if time_since_failure > 300:
>     enable_provider()  # 自动恢复
> ```
> 
> **最终保底**：Mock Provider确保不会完全失败

#### Q3: 推理引擎的设计思路？

> 设计了7种推理模式，根据任务复杂度自动选择：
> 
> | 模式 | 适用场景 | LLM调用次数 |
> |------|----------|------------|
> | Direct | 简单问答 | 1 |
> | CoT | 逻辑推理 | 1 |
> | ReAct | 需要工具 | 预算预估 4；真实执行为“决策 -> 工具 -> 观察”循环 |
> | ToT | 复杂决策 | 预算预估 6；真实执行为树扩展 + 打分剪枝 |
> | Debate | 争议问题 | 预算预估 4；真实执行为四轮多代理对辩 |
> | Reflection | 需要优化 | 2 |
> | CoVe | 需要验证 | 3 |
> 
> 自动选择策略：
> ```python
> if "比较" in query:
>     mode = "debate"
> elif "综述" in query:
>     mode = "reflection"
> elif "实现" in query or "流程" in query:
>     mode = "react"
> elif "trade-off" in query or "path" in query:
>     mode = "tot"
> else:
>     mode = "cot" or "direct"
> ```

#### Q4: 质量增强如何实现？

> **Self-MoA (Mixture of Agents)**：
> 
> 受论文《Mixture-of-Agents Enhances Large Language Model Capabilities》启发，
> 让多个模型各自生成答案，再由另一个模型聚合：
> 
> ```python
> candidates = [
>     siliconflow.call(query),  # DeepSeek-V3
>     zhipu.call(query),        # GLM-4-Flash
>     dashscope.call(query),    # Qwen-Turbo
> ]
> 
> aggregation_prompt = f"""
> 有{len(candidates)}个答案，请综合分析，
> 保留准确部分，纠正错误，输出最优答案。
> """
> final_answer = aggregator.call(aggregation_prompt)
> ```
> 
> **MPSC (Multi-Path Self-Consistency)**：
> 
> 多条推理路径验证结论一致性：
> 
> ```python
> paths = [generate_reasoning_path() for _ in range(3)]
> conclusions = [extract_conclusion(p) for p in paths]
> consistency = calculate_consistency(conclusions)
> confidence = consistency_to_confidence(consistency)
> ```

---

## 9. 项目复现指南

### 9.1 环境准备

```bash
# 1. 克隆/解压项目
tar -xzf scholar-agent-v2.tar.gz
cd scholar-agent

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

如果本地已经有 `agent` conda 环境，也可以直接：

```bash
source /home/a1/miniconda3/etc/profile.d/conda.sh
conda activate agent
```

### 9.1.1 本地 RAG 模型与向量库准备

如果需要启用本地 PDF 建库与本地 RAG，请额外准备本地模型权重并确认 `chromadb` 可用：

```bash
export BGE_M3_MODEL_PATH=/your/path/to/bge-m3
export BGE_RERANKER_MODEL_PATH=/your/path/to/bge-reranker-v2-m3
```

本地 RAG 相关组件当前是：

- `BGEM3Embedder`：负责 chunk 与 query 的 dense embedding
- `LocalChromaVectorStore`：负责向量持久化与查询
- `BGEReranker`：负责对 `RRF` 融合结果做最终重排


### 9.2 配置API密钥

编辑 `api_keys.py`：

```python
API_KEYS = {
    # 必填（至少一个）
    "SCNET_API_KEY": "sk-xxx",
    "SILICONFLOW_API_KEY": "sk-xxx",  # 推荐，2000万Token免费
    "ZHIPU_API_KEY": "xxx",            # 完全免费
    
    # 可选
    "WOS_STARTER_API_KEY": "xxx",      # Web of Science Starter API
    "DASHSCOPE_API_KEY": "",
    "DEEPSEEK_API_KEY": "",
    # ...
}
```

`Web of Science` 相关可选环境变量：

```bash
export WOS_DOCUMENTS_URL="https://api.clarivate.com/apis/wos-starter/v1/documents"
export WOS_DATABASE="WOS"
```

### 9.3 启动服务

```bash
# Web界面（推荐）
python run.py
# 选择 [1] Web界面模式

# 命令行模式
python run.py
# 选择 [2] 命令行模式

# 运行测试
python run.py
# 选择 [3] 运行测试
```

### 9.4 使用示例

**示例1：论文搜索**
```
用户: 搜索Transformer相关论文
Agent: 正在搜索...找到25篇论文
       1. Attention Is All You Need (2017) - 引用: 50000+
       2. BERT: Pre-training of Deep Bidirectional... (2018)
       ...
```

**示例2：综述生成**
```
用户: 写一篇关于强化学习的综述
Agent: 请提供时间范围（如：2020-2024）
用户: 最近3年
Agent: [memory_recall] → [intent] → [slots] → [planning]
       → [search] → [llm: 查询改写] → [analyze]
       → [debate] → [write] → [llm: 综述写作]
       
       # 强化学习研究综述 (2022-2025)
       
       ## 1. 引言
       强化学习作为机器学习的重要分支...
```

### 9.5 目录权限

确保以下目录可写：
```
data/memory/    # 长期记忆数据库
data/feedback/  # 用户反馈
logs/           # 日志文件
cache/          # 缓存
```

---

## 10. API参考

### 10.1 AgentV2主类

```python
from src.core.agent_v2 import AgentV2

# 初始化
agent = AgentV2()

# 设置模式
agent.set_mode(fast_mode=True, enable_quality_enhance=False)

# 对话
response = agent.chat("搜索深度学习论文", session_id="user_123")
print(response)

# 若需要在前端/服务层拿到 trace_id
response = agent.chat(
    "写一篇 SERF 效应的综述",
    session_id="user_123",
    on_trace_start=lambda trace_id: print(trace_id),
)

# 获取状态
status = agent.get_status()
```

### 10.2 LLMManager

```python
from src.core.llm import LLMManager

manager = LLMManager()

# 直接调用
result = manager.call("你好", provider="zhipu")

# 带故障转移
result = manager.call_with_fallback("你好")

# 查看状态
status = manager.get_status()
print(status)
# {'siliconflow': {'available': True, 'failure_count': 0}, ...}

# 重置故障
manager.reset_failures("siliconflow")
```

### 10.3 工具注册

```python
from src.tools.registry import register_tool, ToolDefinition, ToolParameter

@register_tool(ToolDefinition(
    name="my_tool",
    description="自定义工具",
    parameters=[
        ToolParameter("query", "str", "搜索关键词", required=True)
    ]
))
def my_tool(query: str) -> str:
    return f"处理: {query}"
```

### 10.4 记忆系统

```python
from src.memory.manager import MemoryManager, MemoryType

memory = MemoryManager()

# 存储
memory.store("user_preference", "喜欢简洁的回答", 
             memory_type=MemoryType.PREFERENCE)

# 检索
results = memory.recall("用户偏好", limit=5)

# 遗忘（低重要性）
memory.forget(importance_threshold=0.3)
```

---

## 附录

### A. 完整依赖列表

```
gradio>=6.0.0
requests>=2.28.0
beautifulsoup4>=4.11.0
pdfplumber>=0.7.0
arxiv>=2.0.0
chromadb>=0.5.5
FlagEmbedding>=1.3.3
transformers>=4.44.2,<5
numpy>=1.24.0
scikit-learn>=1.2.0
```

### B. 测试覆盖

| 模块 | 测试项 | 通过率 |
|------|--------|--------|
| LLM管理 | 5 | 100% |
| 意图分类 | 4 | 100% |
| 槽位填充 | 3 | 100% |
| Multi-Agent | 4 | 100% |
| 推理引擎 | 5 | 100% |
| 质量增强 | 3 | 100% |
| 记忆系统 | 4 | 100% |
| 工具系统 | 3 | 100% |
| UI | 3 | 66% |

### C. 版本历史

| 版本 | 日期 | 主要更新 |
|------|------|----------|
| v1.0 | 2026-01-10 | 基础功能实现 |
| v1.1 | 2026-01-10 | 7大高级特性 |
| v1.2 | 2026-01-10 | Bug修复 + Gradio 6.x兼容 |
| v2.0 | 2026-01-11 | 性能优化 + 搜索质量提升 |
| v2.1 | 2026-03-26 | Web of Science 接入 + LLM查询改写 + 实时执行时间线 + 模型调用可视化 |
| v2.2 | 2026-03-27 | 本地 RAG 切换到 BGE-M3 + ChromaDB + BGE-Reranker，新增 ChromaDB 重建脚本 |

---

*文档版本: 2.2 | 最后更新: 2026-03-27*
