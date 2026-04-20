# 快速开始

本文档面向“先把项目跑起来，再逐步理解实现”的场景。

## 1. 前置条件

至少准备以下环境：

- Python 3.11+
- 可用的 `pip`
- 一个可用的本地终端环境
- 如果要启用 OCR，需要安装 `tesseract`
- 如果要启用本地 RAG，需要准备 `bge-m3` 和 `bge-reranker-v2-m3` 本地模型目录

## 2. 安装依赖

推荐直接使用现有 `conda` 环境：

```bash
source /home/a1/miniconda3/etc/profile.d/conda.sh
conda activate agent
pip install -r requirements.txt
```

如果你需要自己创建环境：

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. OCR 检查

项目中的图片 OCR 依赖 `tesseract` 可执行文件。确认方式：

```bash
tesseract --version
```

如果未安装，可按你的系统方式安装。使用 conda 时可参考：

```bash
conda install -c conda-forge tesseract
```

## 4. 最小配置

### 4.1 LLM Provider

建议至少配置一个真实 provider：

- `SCNET_API_KEY`
- `SILICONFLOW_API_KEY`
- `ZHIPU_API_KEY`
- `DEEPSEEK_API_KEY`
- `DASHSCOPE_API_KEY`

不配置也能运行，但会退化为本地规则和 mock provider。

### 4.2 学术搜索源

按需配置：

- `WOS_STARTER_API_KEY`
- `IEEE_XPLORE_API_KEY`
- `SERPAPI_API_KEY`
- `NCBI_API_KEY`

### 4.3 本地 RAG

如果要启用本地向量检索，至少确认：

```bash
export BGE_M3_MODEL_PATH="/你的本地/bge-m3"
export BGE_RERANKER_MODEL_PATH="/你的本地/bge-reranker-v2-m3"
```

## 5. 启动项目

```bash
python run.py
```

启动后可选择：

1. Web 界面
2. 命令行
3. 功能验证

## 6. 推荐验证顺序

### 6.1 基础功能验证

```bash
python verify_features.py
```

适合先确认：

- 核心模块能否初始化
- 工具注册和白名单是否正常
- 研究规划与论文获取是否可用

### 6.2 搜索 agent 验证

```bash
python verify_agentic_search.py
```

适合确认：

- 查询改写是否正常
- 外部学术源能否返回结果
- 搜索工具规划是否命中 `LangChain agent` 路径

说明：

- 这一步需要至少一个真实 provider
- 当前搜索规划优先使用已验证成功的 `zhipu`
- 若没有可用规划 provider，运行时会自动回退到确定性搜索

## 7. 首次使用建议

- 如果你要分析本地论文，先上传 PDF 或先调用 `index_pdf`
- 如果你要看执行过程，优先使用 Web 界面，右侧可以看 trace 时间线
- 如果你主要做研究型任务，可以优先用 `plan_research -> fetch_paper -> read_paper -> chat` 这条链路
- 如果你要做连续追问，可以直接说“根据之前查找到的资料...”；当前系统会优先复用本会话最近一次检索结果
- 如果你只想用本地知识库，可直接说“只用 RAG / 只用本地”；如果不想检索，可直接说“不要检索”

## 8. 继续阅读

- 主说明文档：`README.md`
- 文档总览：`PROJECT_DOCUMENTATION.md`
- 完整项目文档：`../ScholarAgent_完整项目文档.md`
- 面试速记版：`INTERVIEW_GUIDE.md`
