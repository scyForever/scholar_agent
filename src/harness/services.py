from __future__ import annotations

from dataclasses import dataclass

from src.core.llm import LLMManager
from src.feedback.collector import FeedbackCollector
from src.memory.manager import MemoryManager
from src.planning.task_hierarchy import TaskHierarchyPlanner
from src.preprocessing.dialogue_manager import DialogueManager
from src.preprocessing.intent_classifier import IntentClassifier
from src.preprocessing.slot_filler import SlotFiller
from src.prompt_templates.manager import PromptTemplateManager
from src.quality.enhancer import QualityEnhancer
from src.rag.harness import RAGHarness
from src.rag.retriever import HybridRetriever
from src.reasoning.engine import ReasoningEngine
from src.skills import ResearchSkillset, ResearchSkillsHarness
from src.whitelist.manager import WhitelistManager
from src.whitebox.tracer import WhiteboxTracer


@dataclass(slots=True)
class ScholarAgentServices:
    llm: LLMManager
    tracer: WhiteboxTracer
    templates: PromptTemplateManager
    memory: MemoryManager
    feedback: FeedbackCollector
    whitelist: WhitelistManager
    dialogue: DialogueManager
    intent_classifier: IntentClassifier
    slot_filler: SlotFiller
    planner: TaskHierarchyPlanner
    retriever: HybridRetriever
    rag_harness: RAGHarness
    reasoning: ReasoningEngine
    quality: QualityEnhancer
    skills_harness: ResearchSkillsHarness
    research_skills: ResearchSkillset

    @classmethod
    def build_default(cls) -> "ScholarAgentServices":
        llm = LLMManager()
        tracer = WhiteboxTracer()
        templates = PromptTemplateManager()
        templates.ensure_default_templates()
        memory = MemoryManager()
        feedback = FeedbackCollector()
        whitelist = WhitelistManager()
        dialogue = DialogueManager()
        intent_classifier = IntentClassifier(llm, templates)
        slot_filler = SlotFiller()
        planner = TaskHierarchyPlanner()
        retriever = HybridRetriever(llm=llm)
        reasoning = ReasoningEngine(
            llm,
            tracer,
            retriever=retriever,
            whitelist=whitelist,
        )
        quality = QualityEnhancer(llm)
        research_skills = ResearchSkillset(memory)
        rag_harness = retriever.harness
        skills_harness = research_skills.harness
        return cls(
            llm=llm,
            tracer=tracer,
            templates=templates,
            memory=memory,
            feedback=feedback,
            whitelist=whitelist,
            dialogue=dialogue,
            intent_classifier=intent_classifier,
            slot_filler=slot_filler,
            planner=planner,
            retriever=retriever,
            rag_harness=rag_harness,
            reasoning=reasoning,
            quality=quality,
            skills_harness=skills_harness,
            research_skills=research_skills,
        )
