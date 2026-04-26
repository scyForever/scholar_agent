from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from config.settings import settings


@dataclass(frozen=True, slots=True)
class DomainSpec:
    slug: str
    zh: str
    en: str


@dataclass(frozen=True, slots=True)
class ThemeSpec:
    slug: str
    category: str
    title_template: str
    survey_label: str
    mechanism_template: str
    finding_template: str
    implication_template: str
    mechanism_keywords: tuple[str, ...]
    finding_keywords: tuple[str, ...]
    implication_keywords: tuple[str, ...]
    forbidden_keywords: tuple[str, ...]


DOMAINS: tuple[DomainSpec, ...] = (
    DomainSpec("biomedical", "生物医学", "Biomedical"),
    DomainSpec("robotics", "机器人", "Robotics"),
    DomainSpec("climate", "气候科学", "Climate"),
    DomainSpec("materials", "材料科学", "Materials"),
    DomainSpec("education", "教育技术", "Education"),
    DomainSpec("finance", "金融科技", "Finance"),
    DomainSpec("legal", "法律科技", "Legal"),
    DomainSpec("healthcare", "医疗健康", "Healthcare"),
    DomainSpec("energy", "能源系统", "Energy"),
    DomainSpec("agriculture", "农业科学", "Agriculture"),
)


THEMES: tuple[ThemeSpec, ...] = (
    ThemeSpec(
        slug="hybrid_retrieval",
        category="retrieval",
        title_template="Hybrid Scholar Retrieval for {domain_en} Literature with BM25, Dense Search and RRF",
        survey_label="混合学术检索与 RRF 融合",
        mechanism_template="作者将 BM25 术语匹配、dense retrieval 语义召回和 RRF 融合结合，用于减少 {domain_zh} 论文搜索中的相关文献遗漏。",
        finding_template="实验显示在 {domain_zh} 文献集合上，Recall@10 提高 {primary_gain} 个百分点，Precision@5 提高 {secondary_gain} 个百分点。",
        implication_template="结论是学术研究助手应同时保留术语匹配与语义召回，才能兼顾覆盖面和可解释性。",
        mechanism_keywords=("BM25", "dense retrieval", "RRF"),
        finding_keywords=("Recall@10", "Precision@5", "百分点"),
        implication_keywords=("覆盖面", "可解释性", "学术研究助手"),
        forbidden_keywords=("research memory", "faithfulness", "multi-agent debate"),
    ),
    ThemeSpec(
        slug="query_rewriting",
        category="retrieval",
        title_template="Query Rewriting for {domain_en} Scholarly Search with Terminology Normalization",
        survey_label="学术检索查询改写与术语规范化",
        mechanism_template="方法通过术语规范化、同义短语扩展和关键词重组来改写 {domain_zh} 论文检索查询。",
        finding_template="在 {domain_zh} 语料上，改写后查询使 Top-{top_k} 召回提高 {primary_gain} 个百分点，并减少冷门术语漏检。",
        implication_template="作者认为研究助手在写综述前应先做查询改写，以扩大候选论文覆盖范围。",
        mechanism_keywords=("术语规范化", "同义短语", "查询改写"),
        finding_keywords=("Top-", "召回", "漏检"),
        implication_keywords=("综述", "覆盖范围", "研究助手"),
        forbidden_keywords=("RRF", "research memory", "citation style"),
    ),
    ThemeSpec(
        slug="cross_encoder_reranking",
        category="retrieval",
        title_template="Cross-Encoder Reranking for {domain_en} Academic Search",
        survey_label="Cross-Encoder 学术搜索重排",
        mechanism_template="作者使用 cross-encoder 对第一阶段检索候选进行重打分，以提升 {domain_zh} 场景下的前排准确率和上下文相关性。",
        finding_template="实验结果表明 Precision@5 提高 {primary_gain} 个百分点，但平均响应延迟增加 {latency_gain} 毫秒。",
        implication_template="结论是 reranker 适合作为第二阶段压缩噪声，而不是替代第一阶段召回。",
        mechanism_keywords=("cross-encoder", "重打分", "上下文相关性"),
        finding_keywords=("Precision@5", "延迟", "毫秒"),
        implication_keywords=("第二阶段", "压缩噪声", "第一阶段召回"),
        forbidden_keywords=("research memory", "citation grounding", "planner"),
    ),
    ThemeSpec(
        slug="faithfulness_benchmark",
        category="rag",
        title_template="Faithfulness Benchmarking for {domain_en} Scholarly RAG",
        survey_label="学术 RAG faithfulness 评测",
        mechanism_template="论文提出用于 {domain_zh} 学术 RAG 的评测框架，同时检查检索上下文和生成回答之间的证据一致性。",
        finding_template="基准显式评估 Recall@k、Precision@k、answer relevance、answer correctness 和 faithfulness。",
        implication_template="作者强调不能只看回答流畅度，还要确认回答断言是否被论文证据支持。",
        mechanism_keywords=("学术 RAG", "评测框架", "证据一致性"),
        finding_keywords=("Recall@k", "Precision@k", "faithfulness"),
        implication_keywords=("流畅度", "论文证据", "断言"),
        forbidden_keywords=("RRF", "research memory", "APA"),
    ),
    ThemeSpec(
        slug="citation_grounded_writing",
        category="writing",
        title_template="Citation-Grounded Survey Writing for {domain_en} Research Assistants",
        survey_label="引用可追溯的综述写作",
        mechanism_template="系统先解析输出语言、章节结构和最少参考文献数量，再为 {domain_zh} 综述的每个章节单独收集证据。",
        finding_template="实验显示加入 section constraints 后，结构完整度提高 {primary_gain} 个百分点，citation grounding 质量提高 {secondary_gain} 个百分点。",
        implication_template="作者认为写作约束会直接影响检索规模、证据覆盖和最终综述质量。",
        mechanism_keywords=("输出语言", "章节结构", "最少参考文献"),
        finding_keywords=("section constraints", "结构完整度", "citation grounding"),
        implication_keywords=("检索规模", "证据覆盖", "综述质量"),
        forbidden_keywords=("research memory", "RRF", "cross-encoder"),
    ),
    ThemeSpec(
        slug="multi_agent_synthesis",
        category="agent",
        title_template="Multi-Agent Scientific Synthesis for {domain_en} Reviews",
        survey_label="多智能体科学综述综合",
        mechanism_template="系统把 {domain_zh} 综述任务分配给 planner、searcher、analyst 和 debater 四类角色，各自负责拆题、检索、证据抽取和综合。",
        finding_template="实验显示相较单智能体基线，任务成功率提高 {primary_gain} 个百分点，但延迟增加 {latency_gain} 毫秒。",
        implication_template="结论是多智能体更适合复杂综述和方法比较，而简单问答可以走更轻量的链路。",
        mechanism_keywords=("planner", "searcher", "analyst", "debater"),
        finding_keywords=("任务成功率", "延迟", "单智能体"),
        implication_keywords=("复杂综述", "方法比较", "轻量链路"),
        forbidden_keywords=("RRF", "GB/T 7714", "research memory"),
    ),
    ThemeSpec(
        slug="research_memory",
        category="memory",
        title_template="Research Memory for Long-Horizon {domain_en} Scholarly Assistants",
        survey_label="长任务研究记忆",
        mechanism_template="研究记忆模块保存 {domain_zh} 研究过程中的已读论文摘要、用户偏好、检索历史和高亮观点，而不是全文本身。",
        finding_template="实验发现未读优先排序使信息增量提高 {primary_gain} 个百分点，重复阅读率下降 {secondary_gain} 个百分点。",
        implication_template="作者认为研究记忆能增强多轮任务连续性，并降低长期研究中的重复劳动。",
        mechanism_keywords=("已读论文摘要", "用户偏好", "检索历史"),
        finding_keywords=("未读优先", "信息增量", "重复阅读率"),
        implication_keywords=("多轮任务", "连续性", "重复劳动"),
        forbidden_keywords=("RRF", "faithfulness", "APA"),
    ),
    ThemeSpec(
        slug="temporal_survey_planning",
        category="planning",
        title_template="Temporal Survey Planning for {domain_en} Literature Reviews",
        survey_label="按时间线组织的综述规划",
        mechanism_template="系统会为 {domain_zh} 综述显式抽取时间范围、发展阶段和阶段性代表论文，以构建时间线式写作大纲。",
        finding_template="结果表明时间线规划让历史脉络覆盖率提高 {primary_gain} 个百分点，章节衔接错误减少 {secondary_gain} 个百分点。",
        implication_template="作者建议需要解释技术演化时优先使用时间线组织，而不是单纯按主题堆叠材料。",
        mechanism_keywords=("时间范围", "发展阶段", "时间线"),
        finding_keywords=("历史脉络覆盖率", "章节衔接", "百分点"),
        implication_keywords=("技术演化", "时间线组织", "主题堆叠"),
        forbidden_keywords=("cross-encoder", "research memory", "faithfulness"),
    ),
    ThemeSpec(
        slug="method_taxonomy",
        category="analysis",
        title_template="Method Taxonomy Construction for {domain_en} Scholarly Reviews",
        survey_label="方法分类与 taxonomy 构建",
        mechanism_template="论文提出将 {domain_zh} 领域文献按问题定义、模型结构和证据类型三条轴线构建 taxonomy。",
        finding_template="实验中 taxonomy 方案让方法对比覆盖率提高 {primary_gain} 个百分点，冗余分类降低 {secondary_gain} 个百分点。",
        implication_template="作者认为 taxonomy 能帮助研究助手生成更可比较、更可复用的综述结构。",
        mechanism_keywords=("问题定义", "模型结构", "taxonomy"),
        finding_keywords=("方法对比覆盖率", "冗余分类", "百分点"),
        implication_keywords=("可比较", "可复用", "综述结构"),
        forbidden_keywords=("RRF", "research memory", "faithfulness"),
    ),
    ThemeSpec(
        slug="evidence_verification",
        category="verification",
        title_template="Evidence Verification for {domain_en} Literature Grounding",
        survey_label="文献证据验证与答案真实性",
        mechanism_template="方法在 {domain_zh} 研究助手中引入 claim-to-evidence 对齐，逐条检查回答断言是否能映射到文献证据。",
        finding_template="实验显示错误断言率下降 {primary_gain} 个百分点，答案真实性提高 {secondary_gain} 个百分点。",
        implication_template="作者强调生成系统需要单独建模证据验证，而不能只依赖语言流畅性。",
        mechanism_keywords=("claim-to-evidence", "断言", "文献证据"),
        finding_keywords=("错误断言率", "答案真实性", "百分点"),
        implication_keywords=("证据验证", "语言流畅性", "生成系统"),
        forbidden_keywords=("RRF", "planner", "research memory"),
    ),
    ThemeSpec(
        slug="domain_adaptive_retrieval",
        category="retrieval",
        title_template="Domain-Adaptive Retrieval for {domain_en} Scholarly Search",
        survey_label="领域自适应学术检索",
        mechanism_template="论文通过领域词表和自适应负采样优化 {domain_zh} 学术搜索模型，使检索器更好理解专业术语。",
        finding_template="在 {domain_zh} 任务上，领域自适应使 nDCG@10 提高 {primary_gain} 个百分点，低频术语命中率提高 {secondary_gain} 个百分点。",
        implication_template="作者认为不同学科的研究助手应具备领域自适应检索能力，而不是使用完全统一的召回器。",
        mechanism_keywords=("领域词表", "自适应负采样", "专业术语"),
        finding_keywords=("nDCG@10", "低频术语", "百分点"),
        implication_keywords=("不同学科", "领域自适应", "召回器"),
        forbidden_keywords=("faithfulness", "research memory", "planner"),
    ),
    ThemeSpec(
        slug="benchmark_alignment",
        category="evaluation",
        title_template="Benchmark Alignment for {domain_en} Research Assistants",
        survey_label="研究助手 benchmark 对齐",
        mechanism_template="作者把 {domain_zh} 研究助手评测拆成检索、分析、写作和引用四个阶段，并分别设计对齐指标。",
        finding_template="结果显示阶段化 benchmark 能让失败定位效率提高 {primary_gain} 个百分点，功能回归检测提前 {secondary_gain} 个百分点。",
        implication_template="结论是复杂学术助手必须把检索过程和生成过程分开评测，才能准确定位瓶颈。",
        mechanism_keywords=("检索", "分析", "写作", "引用"),
        finding_keywords=("失败定位效率", "功能回归检测", "百分点"),
        implication_keywords=("分开评测", "生成过程", "瓶颈"),
        forbidden_keywords=("research memory", "RRF", "GB/T 7714"),
    ),
)


def _primary_gain(theme_index: int, domain_index: int) -> int:
    return 6 + ((theme_index * 5 + domain_index * 3) % 9)


def _secondary_gain(theme_index: int, domain_index: int) -> int:
    return 4 + ((theme_index * 4 + domain_index * 2) % 7)


def _latency_gain(theme_index: int, domain_index: int) -> int:
    return 18 + ((theme_index * 17 + domain_index * 11) % 55)


def _top_k(theme_index: int, domain_index: int) -> int:
    return 5 + ((theme_index + domain_index) % 4)


def build_corpus_documents() -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    for theme_index, theme in enumerate(THEMES):
        for domain_index, domain in enumerate(DOMAINS):
            title = theme.title_template.format(domain_en=domain.en)
            primary_gain = _primary_gain(theme_index, domain_index)
            secondary_gain = _secondary_gain(theme_index, domain_index)
            latency_gain = _latency_gain(theme_index, domain_index)
            top_k = _top_k(theme_index, domain_index)
            text = " ".join(
                [
                    f"论文《{title}》面向 {domain.zh} 学术研究助手场景。",
                    theme.mechanism_template.format(domain_zh=domain.zh),
                    theme.finding_template.format(
                        domain_zh=domain.zh,
                        primary_gain=primary_gain,
                        secondary_gain=secondary_gain,
                        latency_gain=latency_gain,
                        top_k=top_k,
                    ),
                    theme.implication_template.format(domain_zh=domain.zh),
                    f"作者还指出，在 {domain.zh} 场景下需要同时关注可追溯引用、上下文相关性和答案真实性。",
                ]
            )
            documents.append(
                {
                    "doc_id": f"{theme.slug}-{domain.slug}",
                    "title": title,
                    "category": theme.category,
                    "theme": theme.slug,
                    "domain": domain.slug,
                    "text": text,
                }
            )
    return documents


def _reference_points(
    *,
    theme: ThemeSpec,
    domain: DomainSpec,
    theme_index: int,
    domain_index: int,
) -> List[Dict[str, Any]]:
    primary_gain = _primary_gain(theme_index, domain_index)
    secondary_gain = _secondary_gain(theme_index, domain_index)
    latency_gain = _latency_gain(theme_index, domain_index)
    top_k = _top_k(theme_index, domain_index)
    return [
        {
            "fact_id": f"{theme.slug}-{domain.slug}-mechanism",
            "statement": theme.mechanism_template.format(domain_zh=domain.zh),
            "keywords": list(theme.mechanism_keywords),
        },
        {
            "fact_id": f"{theme.slug}-{domain.slug}-finding",
            "statement": theme.finding_template.format(
                domain_zh=domain.zh,
                primary_gain=primary_gain,
                secondary_gain=secondary_gain,
                latency_gain=latency_gain,
                top_k=top_k,
            ),
            "keywords": list(theme.finding_keywords),
        },
        {
            "fact_id": f"{theme.slug}-{domain.slug}-implication",
            "statement": theme.implication_template.format(domain_zh=domain.zh),
            "keywords": list(theme.implication_keywords),
        },
    ]


def _combined_keywords(theme: ThemeSpec) -> List[str]:
    return [
        *theme.mechanism_keywords,
        *theme.finding_keywords,
        *theme.implication_keywords,
    ]


def _title_prefix(theme: ThemeSpec) -> str:
    return theme.title_template.split("{domain_en}")[0].strip()


def build_retrieval_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    exact_domain_indices = (0, 2, 4, 6)
    pair_groups = ((0, 1), (3, 4), (6, 7))
    triple_groups = ((0, 1, 2), (3, 4, 5), (7, 8, 9))
    for theme_index, theme in enumerate(THEMES):
        keywords = _combined_keywords(theme)
        title_prefix = _title_prefix(theme)

        for case_offset, domain_index in enumerate(exact_domain_indices):
            domain = DOMAINS[domain_index]
            title = theme.title_template.format(domain_en=domain.en)
            primary_gain = _primary_gain(theme_index, domain_index)
            secondary_gain = _secondary_gain(theme_index, domain_index)
            cases.append(
                {
                    "case_id": f"retrieval-exact-{theme.slug}-{domain.slug}",
                    "query": (
                        f"找出标题为《{title}》的论文，"
                        f"它在 {domain.zh} 文献集合上报告了 {primary_gain} 个百分点的主要提升"
                        f"和 {secondary_gain} 个百分点的次要提升。"
                    ),
                    "top_k": 1,
                    "relevant_doc_ids": [f"{theme.slug}-{domain.slug}"],
                }
            )

        for case_offset, (left_index, right_index) in enumerate(pair_groups):
            left = DOMAINS[left_index]
            right = DOMAINS[right_index]
            signal_a = keywords[(theme_index + case_offset) % len(keywords)]
            signal_b = keywords[(theme_index + case_offset + 2) % len(keywords)]
            cases.append(
                {
                    "case_id": f"retrieval-semantic-pair-{theme.slug}-{left.slug}-{right.slug}",
                    "query": (
                        f"检索题名前缀接近 {title_prefix}，同时讨论{theme.survey_label}，强调{signal_a}与{signal_b}，"
                        f"并分别覆盖{left.zh}和{right.zh}场景的两篇代表论文。"
                    ),
                    "top_k": 2,
                    "relevant_doc_ids": [
                        f"{theme.slug}-{left.slug}",
                        f"{theme.slug}-{right.slug}",
                    ],
                }
            )

        for case_offset, (first_index, second_index, third_index) in enumerate(triple_groups):
            selected_domains = [
                DOMAINS[first_index],
                DOMAINS[second_index],
                DOMAINS[third_index],
            ]
            signal_a = keywords[(theme_index + case_offset + 1) % len(keywords)]
            signal_b = keywords[(theme_index + case_offset + 4) % len(keywords)]
            domain_phrase = "，".join(domain.zh for domain in selected_domains)
            cases.append(
                {
                    "case_id": (
                        f"retrieval-semantic-triple-{theme.slug}-"
                        f"{selected_domains[0].slug}-{selected_domains[1].slug}-{selected_domains[2].slug}"
                    ),
                    "query": (
                        f"找出题名前缀接近 {title_prefix}、围绕{theme.survey_label}、突出{signal_a}和{signal_b}，"
                        f"并覆盖{domain_phrase}三个场景的代表论文。"
                    ),
                    "top_k": 3,
                    "relevant_doc_ids": [
                        f"{theme.slug}-{domain.slug}"
                        for domain in selected_domains
                    ],
                }
            )
    return cases


def build_generation_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for theme_index, theme in enumerate(THEMES):
        for domain_index, domain in enumerate(DOMAINS):
            title = theme.title_template.format(domain_en=domain.en)
            cases.append(
                {
                    "case_id": f"generation-{theme.slug}-{domain.slug}",
                    "query": f"根据给定资料，总结《{title}》在 {domain.zh} 学术助手中的关键机制、实验发现和实践启示。",
                    "context_doc_ids": [f"{theme.slug}-{domain.slug}"],
                    "reference_points": _reference_points(
                        theme=theme,
                        domain=domain,
                        theme_index=theme_index,
                        domain_index=domain_index,
                    ),
                    "forbidden_keywords": list(theme.forbidden_keywords),
                }
            )
    return cases


def build_agent_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    documents = [
        {
            "theme": theme,
            "domain": domain,
            "title": theme.title_template.format(domain_en=domain.en),
        }
        for theme in THEMES
        for domain in DOMAINS
    ]

    explain_docs = documents[:25]
    compare_pairs = [(documents[index], documents[index + 1]) for index in range(0, 50, 2)]
    survey_specs = [THEMES[index % len(THEMES)] for index in range(25)]

    for item in explain_docs:
        cases.append(
            {
                "case_id": f"agent-explain-{item['theme'].slug}-{item['domain'].slug}",
                "query": f"只基于本地资料，解释《{item['title']}》的核心方法和主要应用场景",
                "expected_intent": "explain_concept",
                "expected_needs_input": False,
                "required_slots": ["topic", "rag_mode"],
                "expected_slot_values": {
                    "rag_mode": "local_only",
                },
                "required_trace_steps": ["memory_recall", "intent", "slots", "planning", "search", "write"],
                "optional_trace_steps": ["quality"],
                "success_keywords": [item["title"], item["theme"].survey_label],
                "artifact_expectations": {
                    "search_mode": "local_rag_only_by_instruction",
                    "min_local_hits": 1,
                },
            }
        )

    for left, right in compare_pairs:
        cases.append(
            {
                "case_id": f"agent-compare-{left['theme'].slug}-{left['domain'].slug}-{right['domain'].slug}",
                "query": f"只基于本地资料，比较《{left['title']}》和《{right['title']}》",
                "expected_intent": "compare_methods",
                "expected_needs_input": False,
                "required_slots": ["topic", "comparison_target", "rag_mode"],
                "expected_slot_values": {
                    "rag_mode": "local_only",
                },
                "required_trace_steps": ["memory_recall", "intent", "slots", "planning", "search", "analyze", "write"],
                "optional_trace_steps": ["quality"],
                "success_keywords": [left["theme"].survey_label, right["theme"].survey_label],
                "artifact_expectations": {
                    "search_mode": "local_rag_only_by_instruction",
                    "min_local_hits": 2,
                },
            }
        )

    citation_styles = ("apa", "ieee", "gb_t_7714", "chicago", "mla")
    style_labels = {
        "apa": "APA",
        "ieee": "IEEE",
        "gb_t_7714": "GB/T 7714",
        "chicago": "Chicago",
        "mla": "MLA",
    }
    organization_styles = ("topic", "timeline", "method", "application", "topic")
    organization_labels = {
        "topic": "按主题展开",
        "timeline": "按时间线展开",
        "method": "按方法线展开",
        "application": "按应用场景展开",
    }
    section_options = (
        ["摘要", "引言", "结论"],
        ["摘要", "相关工作", "结论"],
        ["摘要", "方法", "结论"],
        ["摘要", "挑战", "结论"],
        ["摘要", "未来工作", "结论"],
    )
    for index, theme in enumerate(survey_specs):
        citation_style = citation_styles[index % len(citation_styles)]
        organization_style = organization_styles[index % len(organization_styles)]
        required_sections = section_options[index % len(section_options)]
        min_references = 6 + (index % 5)
        cases.append(
            {
                "case_id": f"agent-survey-{theme.slug}-{index:02d}",
                "query": (
                    f"只基于本地资料，写一篇关于{theme.survey_label}的综述，要求用中文写，"
                    f"{organization_labels[organization_style]}，包含{'、'.join(required_sections)}，"
                    f"并使用 {style_labels[citation_style]} 格式，至少 {min_references} 篇参考文献，尽量详细"
                ),
                "expected_intent": "generate_survey",
                "expected_needs_input": False,
                "required_slots": [
                    "topic",
                    "rag_mode",
                    "language",
                    "organization_style",
                    "required_sections",
                    "citation_style",
                    "min_references",
                    "outline_depth",
                ],
                "expected_slot_values": {
                    "rag_mode": "local_only",
                    "language": "zh",
                    "organization_style": organization_style,
                    "required_sections": required_sections,
                    "citation_style": citation_style,
                    "min_references": min_references,
                    "outline_depth": "deep",
                },
                "required_trace_steps": ["memory_recall", "intent", "slots", "planning", "search", "analyze", "write"],
                "optional_trace_steps": ["quality"],
                "success_keywords": [theme.survey_label, "综述", required_sections[-1]],
                "artifact_expectations": {
                    "search_mode": "local_rag_only_by_instruction",
                    "min_local_hits": 2,
                },
            }
        )

    missing_reference_counts = [4, 5, 6, 7, 8]
    missing_org_styles = [
        ("timeline", "按时间线展开"),
        ("topic", "按主题展开"),
        ("method", "按方法线展开"),
        ("application", "按应用场景展开"),
    ]
    for index in range(25):
        org_style, org_label = missing_org_styles[index % len(missing_org_styles)]
        ref_count = missing_reference_counts[index % len(missing_reference_counts)]
        cases.append(
            {
                "case_id": f"agent-missing-topic-{index:02d}",
                "query": f"只基于本地资料，写一篇综述，要求用中文写，{org_label}，至少 {ref_count} 篇参考文献",
                "expected_intent": "generate_survey",
                "expected_needs_input": True,
                "required_slots": ["rag_mode", "min_references", "organization_style", "language"],
                "expected_slot_values": {
                    "rag_mode": "local_only",
                    "min_references": ref_count,
                    "organization_style": org_style,
                    "language": "zh",
                },
                "required_missing_slots": ["topic"],
                "required_trace_steps": ["memory_recall", "intent", "slots"],
                "forbidden_trace_steps": ["planning", "search", "write"],
                "success_keywords": ["请一次性补充", "topic"],
                "artifact_expectations": {},
            }
        )

    memory_theme = next(theme for theme in THEMES if theme.slug == "research_memory")
    memory_domains = DOMAINS[:10]
    for index, domain in enumerate(memory_domains):
        title = memory_theme.title_template.format(domain_en=domain.en)
        cases.append(
            {
                "case_id": f"agent-memory-short-long-{domain.slug}",
                "query": f"只基于本地资料，继续解释《{title}》的核心方法，并按我之前说的结构化偏好回答",
                "expected_intent": "explain_concept",
                "expected_needs_input": False,
                "required_slots": ["topic", "rag_mode"],
                "expected_slot_values": {
                    "rag_mode": "local_only",
                },
                "required_trace_steps": ["memory_recall", "intent", "slots", "planning", "search", "write"],
                "optional_trace_steps": ["quality"],
                "success_keywords": [title, memory_theme.survey_label],
                "artifact_expectations": {
                    "search_mode": "local_rag_only_by_instruction",
                    "min_local_hits": 1,
                },
                "memory_setup": {
                    "short_history": [
                        {
                            "role": "user",
                            "content": f"我偏好结构化回答，重点关注{domain.zh}领域的用户偏好、已读论文摘要和摘要层记忆。",
                        },
                        {
                            "role": "assistant",
                            "content": "好的，我会按研究问题、方法机制、实验发现和实践启示组织回答。",
                        },
                    ],
                    "long_term": [
                        {
                            "type": "preference",
                            "content": f"用户长期偏好：{domain.zh}研究记忆任务需要优先说明短期原文层、重点提炼层、摘要层，以及长期关键词召回。",
                            "metadata": {
                                "topic": "research_memory",
                                "domain": domain.slug,
                                "keywords": ["研究记忆", "短期记忆", "长期记忆", domain.zh],
                            },
                            "importance": 0.95,
                        }
                    ],
                    "other_user_long_term": [
                        {
                            "type": "preference",
                            "content": f"其他用户偏好：{domain.zh}研究记忆只关注无关的旧版检索历史字段。",
                            "metadata": {"topic": "noise", "domain": domain.slug},
                            "importance": 0.95,
                        }
                    ],
                },
                "memory_expectations": {
                    "min_short_raw": 3,
                    "min_short_highlights": 1,
                    "summary_required": True,
                    "min_long_count": 1,
                    "max_long_count": 3,
                },
            }
        )

    for index, domain in enumerate(memory_domains):
        title = memory_theme.title_template.format(domain_en=domain.en)
        cases.append(
            {
                "case_id": f"agent-memory-user-isolation-{domain.slug}",
                "query": f"只基于本地资料，解释《{title}》，不要使用其他用户的长期偏好",
                "expected_intent": "explain_concept",
                "expected_needs_input": False,
                "required_slots": ["topic", "rag_mode"],
                "expected_slot_values": {
                    "rag_mode": "local_only",
                },
                "required_trace_steps": ["memory_recall", "intent", "slots", "planning", "search", "write"],
                "optional_trace_steps": ["quality"],
                "success_keywords": [title, memory_theme.survey_label],
                "artifact_expectations": {
                    "search_mode": "local_rag_only_by_instruction",
                    "min_local_hits": 1,
                },
                "memory_setup": {
                    "short_history": [
                        {
                            "role": "user",
                            "content": f"我现在只验证{domain.zh}场景的用户专属长期记忆隔离。",
                        },
                        {
                            "role": "assistant",
                            "content": "明白，本轮只应使用当前用户的短期会话上下文。",
                        },
                    ],
                    "other_user_long_term": [
                        {
                            "type": "preference",
                            "content": f"其他用户长期偏好：{domain.zh}研究记忆要求召回关键词隔离测试和个性化排序。",
                            "metadata": {
                                "topic": "research_memory",
                                "domain": domain.slug,
                                "keywords": ["研究记忆", "关键词隔离", domain.zh],
                            },
                            "importance": 0.99,
                        }
                    ],
                },
                "memory_expectations": {
                    "min_short_raw": 3,
                    "min_short_highlights": 1,
                    "summary_required": True,
                    "long_count": 0,
                },
            }
        )

    return cases


def build_payloads() -> Dict[str, Dict[str, Any]]:
    agent_cases = build_agent_cases()
    corpus_documents = build_corpus_documents()
    retrieval_cases = build_retrieval_cases()
    generation_cases = build_generation_cases()
    corpus = {
        "metadata": {
            "name": "ScholarAgent 大规模论文知识库评测语料",
            "version": "3.0",
            "description": "围绕学术研究助手场景自建的评测语料库，包含 100+ 篇模拟论文条目，用于检索、生成与 agent 评测。",
            "document_count": len(corpus_documents),
            "theme_count": len(THEMES),
            "domain_count": len(DOMAINS),
        },
        "documents": corpus_documents,
    }
    retrieval = {
        "metadata": {
            "name": "ScholarAgent 检索评测集",
            "version": "3.0",
            "description": "面向论文知识库检索的评测集，重点衡量召回率、准确率和上下文相关性。",
            "case_count": len(retrieval_cases),
        },
        "cases": retrieval_cases,
    }
    generation = {
        "metadata": {
            "name": "ScholarAgent 生成评测集",
            "version": "3.0",
            "description": "面向论文知识库生成的评测集，使用金标准上下文单独衡量 faithfulness、答案真实性和答案相关性。",
            "case_count": len(generation_cases),
        },
        "cases": generation_cases,
    }
    agent = {
        "metadata": {
            "name": "ScholarAgent Agent 能力评测集",
            "version": "3.1",
            "description": "面向论文知识库任务流程的 agent 评测集，覆盖解释、比较、综述写作、缺槽追问、短期三层记忆和长期用户专属召回。",
            "case_count": len(agent_cases),
        },
        "cases": agent_cases,
    }
    legacy_rag = {
        "metadata": {
            "name": "ScholarAgent RAG 评测索引",
            "version": "3.0",
            "description": "兼容旧路径的索引文件，真实检索与生成评测请分别使用 retrieval_eval_dataset.json 和 generation_eval_dataset.json。",
            "retrieval_dataset": "retrieval_eval_dataset.json",
            "generation_dataset": "generation_eval_dataset.json",
        },
        "cases": [],
    }
    return {
        "rag_eval_corpus.json": corpus,
        "retrieval_eval_dataset.json": retrieval,
        "generation_eval_dataset.json": generation,
        "agent_eval_dataset.json": agent,
        "rag_eval_dataset.json": legacy_rag,
    }


def write_payloads(output_dir: Path | None = None) -> Dict[str, Path]:
    target_dir = output_dir or settings.evaluation_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    for filename, payload in build_payloads().items():
        path = target_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs[filename] = path
    return outputs


def main() -> None:
    outputs = write_payloads()
    print(
        json.dumps(
            {
                "output_dir": str(settings.evaluation_dir),
                "files": {name: str(path) for name, path in outputs.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
