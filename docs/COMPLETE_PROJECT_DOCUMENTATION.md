# 完整项目文档入口

完整设计说明以根目录文档为准：

- [ScholarAgent_完整项目文档.md](../ScholarAgent_完整项目文档.md)

## 建议关注的章节

如果你主要关注不同主题，可直接在完整项目文档中优先查这些部分：

- 架构总览：整体模块与执行链路
- LLM 管理：provider 初始化、故障转移、预算与 trace
- Multi-Agent：`search / analyze / debate / write / coder` 协作
- 推理引擎：`direct / cot / react / tot / debate / reflection / cove`
- RAG：本地索引、检索、重排、相关性验证
- 前端：执行时间线、阶段模型展示、最近历史恢复

## 相关补充材料

- 主说明文档：[README.md](../README.md)
- 快速开始：[QUICKSTART.md](./QUICKSTART.md)
- 面试速记版：[INTERVIEW_GUIDE.md](./INTERVIEW_GUIDE.md)
- RAG 流程图：[RAG_v3_完整流程图.html](../RAG_v3_完整流程图.html)

## 当前实现提醒

为避免把旧版本行为误认为当前实现，先记住这几个关键点：

- 搜索工具规划固定走 `zhipu`
- 外部搜索结果不会自动写入本地 RAG
- `ReAct` 已是带真实工具调用的循环
- `ToT` 已是显式树搜索
- `Debate` 已是多代理多轮对辩
