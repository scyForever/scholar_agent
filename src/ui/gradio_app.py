from __future__ import annotations

import html
import json
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List

import gradio as gr

from src.core.agent_v2 import AgentV2


WEB_SESSION_ID = "web-user"
BROWSER_HISTORY_KEY = "scholar_agent_history"

STEP_COLORS = {
    "memory_recall": "#0f766e",
    "intent": "#1d4ed8",
    "slots": "#7c3aed",
    "planning": "#c2410c",
    "search": "#0369a1",
    "llm": "#0f172a",
    "analyze": "#4f46e5",
    "debate": "#b45309",
    "write": "#15803d",
    "coder": "#15803d",
    "quality": "#be123c",
    "error": "#b91c1c",
}

STAGE_ORDER = ("analyze", "reasoning", "debate", "write")


def _seconds_from_start(started_at: str, timestamp: str) -> str:
    try:
        started = datetime.fromisoformat(started_at)
        current = datetime.fromisoformat(timestamp)
    except ValueError:
        return ""
    return f"+{(current - started).total_seconds():.1f}s"


def _summarize_step(step: Dict[str, Any]) -> str:
    step_type = str(step.get("type") or "")
    input_data = step.get("input") or {}
    output = step.get("output") or {}
    if step_type == "memory_recall":
        return f"召回记忆 {output.get('count', 0)} 条"
    if step_type == "intent":
        confidence = output.get("confidence")
        intent = output.get("intent", "")
        if confidence is None:
            return f"识别意图：{intent}"
        return f"识别意图：{intent}，置信度 {confidence}"
    if step_type == "slots":
        missing = len(output.get("missing") or [])
        return f"槽位填充完成，缺失 {missing} 项"
    if step_type == "planning":
        return f"任务等级：{output.get('task_level', '')}"
    if step_type == "search":
        paper_count = len(output.get("papers") or [])
        source_breakdown = output.get("source_breakdown") or {}
        local_hits = int(source_breakdown.get("local_rag") or 0)
        if local_hits:
            return f"检索到 {paper_count} 篇候选论文，本地 RAG 命中 {local_hits} 个片段"
        return f"检索到 {paper_count} 篇候选论文，来源 {source_breakdown}"
    if step_type == "llm":
        purpose = str(input_data.get("purpose") or "模型调用")
        provider = output.get("provider", "")
        model = output.get("model", "")
        status = output.get("status", "")
        if status == "running":
            return f"{purpose}：{provider} / {model}"
        if status == "success":
            latency = output.get("latency_ms")
            return f"{purpose}：{provider} / {model}，耗时 {latency} ms"
        if status == "error":
            return f"{purpose}失败：{provider} / {model}"
        return f"{purpose}：{provider} / {model}"
    if step_type == "analyze":
        return f"完成论文分析 {output.get('count', 0)} 篇"
    if step_type.startswith("reasoning:"):
        return f"推理模式：{step_type.split(':', 1)[1]}"
    if step_type == "debate":
        return f"完成多视角综合，支撑论文 {len(output.get('supporting_points') or [])} 篇"
    if step_type == "write":
        preview = str(output.get("answer_preview") or "")
        return f"完成写作，预览 {min(len(preview), 500)} 字符"
    if step_type == "coder":
        preview = str(output.get("answer_preview") or "")
        return f"完成代码生成，预览 {min(len(preview), 500)} 字符"
    if step_type == "quality":
        return f"质量校验结果：{output.get('verification', '')}"
    if step_type == "error":
        return str(output.get("error") or "执行失败")
    return "步骤执行完成"


def _step_title(step: Dict[str, Any]) -> str:
    step_type = str(step.get("type") or "unknown")
    if step_type == "llm":
        purpose = str((step.get("input") or {}).get("purpose") or "模型调用")
        return f"llm: {purpose}"
    if step_type == "error":
        return "error"
    return step_type


def _compact_error(error_text: str) -> str:
    lines = [line.strip() for line in str(error_text or "").splitlines() if line.strip()]
    if not lines:
        return "执行失败"
    return lines[-1]


def _display_steps(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    display_steps: List[Dict[str, Any]] = []
    llm_positions: Dict[int, int] = {}

    for raw_step in trace.get("steps") or []:
        step = dict(raw_step)
        if step.get("type") != "llm":
            display_steps.append(step)
            continue

        output = step.get("output") or {}
        call_id = output.get("call_id")
        if not isinstance(call_id, int):
            display_steps.append(step)
            continue

        existing_index = llm_positions.get(call_id)
        if existing_index is None:
            merged = {
                **step,
                "metadata": dict(step.get("metadata") or {}),
            }
            display_steps.append(merged)
            llm_positions[call_id] = len(display_steps) - 1
            continue

        existing = dict(display_steps[existing_index])
        merged_input = dict(existing.get("input") or {})
        merged_input.update(step.get("input") or {})
        merged_output = dict(existing.get("output") or {})
        merged_output.update(output)
        merged_metadata = dict(existing.get("metadata") or {})
        merged_metadata.update(step.get("metadata") or {})
        display_steps[existing_index] = {
            **existing,
            "input": merged_input,
            "output": merged_output,
            "metadata": merged_metadata,
            "timestamp": step.get("timestamp") or existing.get("timestamp"),
        }

    return display_steps


def _step_stage_key(step_type: str) -> str | None:
    if step_type == "analyze":
        return "analyze"
    if step_type.startswith("reasoning:"):
        return "reasoning"
    if step_type == "debate":
        return "debate"
    if step_type == "write":
        return "write"
    return None


def _collect_stage_models(trace: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    stage_models = {
        stage: {"triggered": False, "calls": []}
        for stage in STAGE_ORDER
    }
    for step in _display_steps(trace):
        step_type = str(step.get("type") or "")
        stage_key = _step_stage_key(step_type)
        if stage_key is not None:
            stage_models[stage_key]["triggered"] = True
        if step_type != "llm":
            continue
        input_data = step.get("input") or {}
        output = step.get("output") or {}
        stage = str(input_data.get("stage") or output.get("stage") or "").strip()
        if stage not in stage_models:
            continue
        stage_models[stage]["calls"].append(
            {
                "purpose": str(input_data.get("purpose") or ""),
                "provider": str(output.get("provider") or ""),
                "model": str(output.get("model") or ""),
                "status": str(output.get("status") or ""),
            }
        )
    return stage_models


def _stage_model_summary(stage_data: Dict[str, Any]) -> str:
    calls = list(stage_data.get("calls") or [])
    if not stage_data.get("triggered") and not calls:
        return "未触发"
    if not calls:
        return "未调用 LLM"

    grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for call in calls:
        provider = str(call.get("provider") or "unknown")
        model = str(call.get("model") or "unknown")
        key = (provider, model)
        entry = grouped.setdefault(
            key,
            {
                "count": 0,
                "purposes": [],
                "statuses": [],
            },
        )
        entry["count"] += 1
        purpose = str(call.get("purpose") or "")
        status = str(call.get("status") or "")
        if purpose and purpose not in entry["purposes"]:
            entry["purposes"].append(purpose)
        if status and status not in entry["statuses"]:
            entry["statuses"].append(status)

    parts: List[str] = []
    for (provider, model), entry in grouped.items():
        suffix = f" x{entry['count']}" if entry["count"] > 1 else ""
        if "error" in entry["statuses"]:
            suffix += "（含失败）"
        elif "running" in entry["statuses"]:
            suffix += "（进行中）"
        text = f"{provider} / {model}{suffix}"
        purposes = "、".join(entry["purposes"][:3])
        if purposes:
            text += f" · {purposes}"
        parts.append(text)
    return "；".join(parts)


def _search_chunk_details(step: Dict[str, Any]) -> str:
    output = step.get("output") or {}
    trace_payload = output.get("trace") or {}
    local_rag = trace_payload.get("local_rag") or {}
    results = local_rag.get("results") or []
    if not results:
        return ""

    cards: List[str] = []
    for index, item in enumerate(results[:5], start=1):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}
        title = str(metadata.get("title") or "未命名文档")
        pdf_path = str(metadata.get("pdf_path") or "")
        pdf_name = Path(pdf_path).name if pdf_path else "N/A"
        source_type = str(item.get("source_type") or "chunk")
        content = " ".join(str(item.get("content") or "").split())
        if len(content) > 280:
            content = content[:280] + "..."

        cards.append(
            (
                "<div style='margin-top:10px;border:1px solid #dbeafe;background:#f8fbff;"
                "border-radius:8px;padding:10px 12px;'>"
                f"<div style='font-weight:700;color:#0f172a;'>Chunk {index} · {html.escape(source_type)}</div>"
                f"<div style='margin-top:6px;font-size:13px;color:#1f2937;'><strong>标题：</strong>{html.escape(title)}</div>"
                f"<div style='margin-top:4px;font-size:13px;color:#1f2937;'><strong>PDF：</strong>{html.escape(pdf_name)}</div>"
                f"<div style='margin-top:4px;font-size:12px;color:#6b7280;word-break:break-all;'>{html.escape(pdf_path)}</div>"
                f"<div style='margin-top:8px;font-size:13px;color:#374151;line-height:1.6;'>{html.escape(content)}</div>"
                "</div>"
            )
        )

    return (
        "<div style='margin-top:10px;'>"
        "<div style='font-weight:700;color:#0f172a;'>本地 RAG 命中片段</div>"
        + "".join(cards)
        + "</div>"
    )


def _format_timeline(trace: Dict[str, Any]) -> str:
    steps = _display_steps(trace)
    stage_models = _collect_stage_models(trace)
    if not steps:
        return (
            "<div style='max-height:68vh;overflow-y:auto;scroll-behavior:smooth;padding-right:6px;'>"
            "<div style='color:#6b7280;padding:12px 0;'>当前还没有可展示的步骤。</div>"
            "</div>"
        )

    started_at = str(trace.get("started_at") or "")
    cards: List[str] = []
    for index, step in enumerate(steps, start=1):
        step_type = str(step.get("type") or "unknown")
        color = STEP_COLORS.get(step_type, "#475569")
        rel = _seconds_from_start(started_at, str(step.get("timestamp") or ""))
        summary = html.escape(_summarize_step(step))
        title = html.escape(_step_title(step))
        stage_key = _step_stage_key(step_type)
        stage_extra = ""
        if stage_key is not None:
            stage_summary = html.escape(_stage_model_summary(stage_models[stage_key]))
            stage_extra = (
                "<div style='margin-top:8px;font-size:13px;color:#111827;'>"
                f"<strong>模型：</strong>{stage_summary}"
                "</div>"
            )
        extra = stage_extra
        if step_type == "search":
            extra += _search_chunk_details(step)
        cards.append(
            (
                "<div style='border:1px solid #e5e7eb;border-left:6px solid "
                f"{color};border-radius:10px;padding:12px 14px;margin-bottom:10px;background:#ffffff;'>"
                f"<div style='display:flex;justify-content:space-between;gap:12px;align-items:center;'>"
                f"<div style='font-weight:700;color:#111827;'>{index}. {title}</div>"
                f"<div style='font-size:12px;color:#6b7280;'>{html.escape(rel)}</div>"
                "</div>"
                f"<div style='margin-top:8px;color:#374151;line-height:1.5;'>{summary}</div>"
                f"{extra}"
                "</div>"
            )
        )

    final_output = trace.get("final_output") or {}
    if isinstance(final_output, dict) and final_output.get("answer"):
        cards.append(
            (
                "<div style='border:1px solid #d1fae5;border-left:6px solid #15803d;"
                "border-radius:10px;padding:12px 14px;background:#f0fdf4;'>"
                "<div style='font-weight:700;color:#166534;'>最终输出</div>"
                f"<div style='margin-top:8px;color:#374151;line-height:1.5;'>"
                f"{html.escape(str(final_output.get('answer'))[:240])}"
                "</div>"
                "</div>"
            )
        )

    return (
        "<div style='max-height:68vh;overflow-y:auto;scroll-behavior:smooth;padding-right:6px;'>"
        + "".join(cards)
        + "</div>"
    )


def _step_choices(trace: Dict[str, Any]) -> List[str]:
    return [
        f"{index}. {_step_title(step)}"
        for index, step in enumerate(_display_steps(trace), start=1)
    ]


def _extract_intent(trace: Dict[str, Any]) -> str:
    for step in trace.get("steps") or []:
        if step.get("type") == "intent":
            return str((step.get("output") or {}).get("intent") or "")
    final_output = trace.get("final_output") or {}
    if isinstance(final_output, dict):
        return str(final_output.get("intent") or "")
    return ""


def _normalize_history(history: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "")
        if role not in {"user", "assistant"}:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _render_page_state(
    history: List[Dict[str, Any]],
    trace: Dict[str, Any],
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    str,
    str,
    str,
    str,
    Dict[str, Any],
    Any,
    str,
    str,
]:
    normalized_history = _normalize_history(history)
    whitebox = json.dumps(trace, ensure_ascii=False, indent=2) if trace else ""
    trace_id = str(trace.get("trace_id") or "")
    status = str(trace.get("status") or "")
    timeline = _format_timeline(trace)
    choices = _step_choices(trace)
    selected_step = choices[-1] if choices else None
    step_detail = _step_detail(trace, selected_step)
    return (
        normalized_history,
        normalized_history,
        _extract_intent(trace),
        trace_id,
        status,
        timeline,
        trace,
        gr.update(choices=choices, value=selected_step),
        step_detail,
        whitebox,
    )


def _render_outputs(
    history: List[Dict[str, Any]],
    trace: Dict[str, Any],
    assistant_content: str,
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    str,
    str,
    str,
    str,
    Dict[str, Any],
    Any,
    str,
    str,
    ]:
    rendered_history = history[:-1] + [{"role": "assistant", "content": assistant_content}]
    return _render_page_state(rendered_history, trace)


def _step_detail(trace: Dict[str, Any], selected_step: str | None) -> str:
    if not selected_step:
        return ""
    try:
        index = int(selected_step.split(".", 1)[0]) - 1
    except (TypeError, ValueError):
        return ""
    steps = _display_steps(trace)
    if index < 0 or index >= len(steps):
        return ""
    return json.dumps(steps[index], ensure_ascii=False, indent=2)


def create_app() -> gr.Blocks:
    agent = AgentV2()

    def _sync_dialogue_history(history: List[Dict[str, Any]]) -> None:
        state = agent.dialogue.get_state(WEB_SESSION_ID)
        state.history = _normalize_history(history)

    def handle_load(
        browser_history: List[Dict[str, Any]] | None,
    ) -> tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        str,
        str,
        str,
        str,
        Dict[str, Any],
        Any,
        str,
        str,
    ]:
        browser_items = _normalize_history(browser_history)
        state = agent.dialogue.get_state(WEB_SESSION_ID)
        backend_items = _normalize_history(state.history)
        if len(backend_items) >= len(browser_items):
            history = backend_items
        else:
            history = browser_items
            _sync_dialogue_history(history)
        trace = agent.tracer.get_trace(state.last_trace_id) if state.last_trace_id else {}
        return _render_page_state(history, trace)

    def handle_submit(
        message: str,
        history: List[Dict[str, Any]],
        pdf_file: str | None,
        fast_mode: bool,
        quality_mode: bool,
    ) -> Iterator[tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        str,
        str,
        str,
        str,
        Dict[str, Any],
        Any,
        str,
        str,
    ]]:
        history = _normalize_history(history)
        _sync_dialogue_history(history)
        if pdf_file:
            try:
                agent.index_pdf(pdf_file)
            except Exception:
                error_text = traceback.format_exc()
                assistant_content = f"PDF 建库失败：{_compact_error(error_text)}"
                error_history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": assistant_content},
                ]
                yield _render_outputs(error_history, {}, assistant_content)
                return

        agent.set_mode(fast_mode=fast_mode, enable_quality_enhance=quality_mode)
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "正在执行..."},
        ]
        shared: Dict[str, Any] = {"trace_id": "", "response": None, "error": None}

        def _run_chat() -> None:
            try:
                shared["response"] = agent.chat(
                    message,
                    session_id=WEB_SESSION_ID,
                    on_trace_start=lambda trace_id: shared.__setitem__("trace_id", trace_id),
                )
            except Exception:
                shared["error"] = traceback.format_exc()

        worker = threading.Thread(target=_run_chat, daemon=True)
        worker.start()

        last_signature = ""
        while worker.is_alive() or shared["trace_id"]:
            trace = agent.tracer.get_trace(shared["trace_id"]) if shared["trace_id"] else {}
            signature = json.dumps(trace, ensure_ascii=False, sort_keys=True) if trace else ""
            if signature != last_signature:
                last_signature = signature
                display_steps = _display_steps(trace)
                latest_step = display_steps[-1] if display_steps else {}
                latest_summary = _summarize_step(latest_step) if latest_step else "正在启动..."
                yield _render_outputs(history, trace, f"正在执行...\n\n{latest_summary}")
            if shared["response"] is not None or shared["error"] is not None:
                break
            time.sleep(0.3)

        worker.join()

        trace = agent.tracer.get_trace(shared["trace_id"]) if shared["trace_id"] else {}
        if shared["error"] is not None:
            error_message = f"执行失败：{_compact_error(shared['error'])}\n\n请查看步骤详情或原始 Trace。"
            yield _render_outputs(history, trace, error_message)
            return

        response = shared["response"]
        final_trace = response.whitebox or trace or {}
        yield _render_outputs(history, final_trace, response.answer)

    def handle_step_change(selected_step: str, trace: Dict[str, Any]) -> str:
        return _step_detail(trace or {}, selected_step)

    with gr.Blocks(title="ScholarAgent") as demo:
        gr.Markdown("# ScholarAgent\n多 Agent 学术研究助手")
        trace_state = gr.State({})
        browser_history = gr.BrowserState([], storage_key=BROWSER_HISTORY_KEY)
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="对话记录")
                message = gr.Textbox(label="输入问题", placeholder="例如：写一篇关于多智能体强化学习的综述")
                pdf_file = gr.File(label="上传 PDF 以建立本地知识库", file_types=[".pdf"], type="filepath")
                with gr.Row():
                    fast_mode = gr.Checkbox(label="快速模式", value=False)
                    quality_mode = gr.Checkbox(label="质量增强", value=False)
                submit = gr.Button("发送")
            with gr.Column(scale=2):
                intent_box = gr.Textbox(label="识别意图")
                trace_id_box = gr.Textbox(label="Trace ID")
                trace_status = gr.Textbox(label="执行状态")
                gr.Markdown("### 执行时间线")
                timeline = gr.HTML()
                step_selector = gr.Dropdown(label="查看步骤详情", choices=[], interactive=True)
                step_detail = gr.Code(label="步骤详情", language="json")
                with gr.Accordion("原始 Trace JSON", open=False):
                    whitebox = gr.Code(label="白盒追踪", language="json")

        submit.click(
            handle_submit,
            inputs=[message, browser_history, pdf_file, fast_mode, quality_mode],
            outputs=[
                chatbot,
                browser_history,
                intent_box,
                trace_id_box,
                trace_status,
                timeline,
                trace_state,
                step_selector,
                step_detail,
                whitebox,
            ],
            queue=True,
        )
        demo.load(
            handle_load,
            inputs=[browser_history],
            outputs=[
                chatbot,
                browser_history,
                intent_box,
                trace_id_box,
                trace_status,
                timeline,
                trace_state,
                step_selector,
                step_detail,
                whitebox,
            ],
            queue=False,
            show_progress="hidden",
        )
        step_selector.change(handle_step_change, inputs=[step_selector, trace_state], outputs=[step_detail])
    return demo.queue()
