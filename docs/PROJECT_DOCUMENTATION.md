# 项目文档

`docs/` 目录面向不同阅读目标提供不同粒度的说明，当前文档已同步到最近一轮实现，包括：

- 搜索工具规划固定使用 `zhipu`
- `ReAct / ToT / Debate` 已升级为真实多步推理流程
- 前端执行时间线支持阶段模型展示与最近历史恢复

## 文档索引

### 1. 快速上手

- [QUICKSTART.md](./QUICKSTART.md)

适合：

- 第一次运行项目
- 只想知道怎么启动、怎么验证

### 2. 主说明文档

- [README.md](../README.md)

适合：

- 快速了解项目能力、配置、常用命令
- 查看当前实现的高层行为

### 3. 完整设计文档

- [ScholarAgent_完整项目文档.md](../ScholarAgent_完整项目文档.md)

适合：

- 需要系统理解架构、执行链路和模块边界
- 需要排查推理、检索、RAG、前端可观测性等细节

### 4. 面试速记版

- [INTERVIEW_GUIDE.md](./INTERVIEW_GUIDE.md)

适合：

- 准备项目汇报
- 准备面试问答
- 需要快速复述设计取舍

### 5. 可视化补充材料

- [RAG_v3_完整流程图.html](../RAG_v3_完整流程图.html)

适合：

- 需要图形化理解本地 RAG 流程

## 推荐阅读顺序

1. 先看 [README.md](../README.md)
2. 再看 [QUICKSTART.md](./QUICKSTART.md)
3. 然后看 [ScholarAgent_完整项目文档.md](../ScholarAgent_完整项目文档.md)
4. 最后按需看 [INTERVIEW_GUIDE.md](./INTERVIEW_GUIDE.md)

## 当前文档边界

- `README` 负责“是什么、怎么跑、当前行为”
- `docs/` 负责“怎么读、怎么汇报、从哪里进入”
- 根目录完整项目文档负责“实现细节与设计解释”

如果以后代码行为再调整，优先更新顺序建议是：

1. `README`
2. `docs/QUICKSTART.md`
3. `docs/INTERVIEW_GUIDE.md`
4. `ScholarAgent_完整项目文档.md`
