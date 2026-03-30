# 快速开始

本文档面向“先跑起来，再逐步理解项目”的场景。

## 1. 运行环境

推荐直接使用现有 `conda` 环境：

```bash
source /home/a1/miniconda3/etc/profile.d/conda.sh
conda activate agent
```

如果你需要自行创建环境：

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. 基础配置

至少准备以下 API Key 中的一部分：

- `ZHIPU_API_KEY`
- `SCNET_API_KEY`
- `SILICONFLOW_API_KEY`
- `DEEPSEEK_API_KEY`
- `DASHSCOPE_API_KEY`
- `WOS_STARTER_API_KEY`

说明：

- 当前运行时的“搜索工具规划”固定使用 `zhipu`
- 若 `zhipu` 不可用，搜索节点会退回确定性检索
- 上传 PDF 后才会进入本地 RAG；外部搜索结果不会自动写入向量库

## 3. 启动项目

```bash
python run.py
```

启动后可选择：

1. Web 界面
2. 命令行
3. 功能验证

## 4. 推荐的首次验证

### 4.1 验证 Web 或 CLI 是否可启动

```bash
python run.py
```

### 4.2 验证搜索规划链路

```bash
python verify_agentic_search.py
```

这个脚本会直接检查：

- 搜索工具规划是否能跑通
- 实际命中的 provider 是谁
- `agent_selected_tools / agent_tool_calls / agent_final_output` 是否产出

### 4.3 验证单个 provider 连通性

```bash
python test_provider_access.py --provider zhipu
```

## 5. 使用建议

- 如果你要分析本地论文，先上传 PDF，再提问
- 如果你要看系统内部过程，优先用 Web 界面，右侧有执行时间线
- 如果你发现改主题或刷新后历史不见了，先确认是否使用同一浏览器；当前前端会通过浏览器本地状态恢复最近对话

## 6. 继续阅读

- 主说明文档：[README.md](../README.md)
- 项目文档入口：[PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md)
- 完整项目文档：[ScholarAgent_完整项目文档.md](../ScholarAgent_完整项目文档.md)
- 面试速记版：[INTERVIEW_GUIDE.md](./INTERVIEW_GUIDE.md)
