from __future__ import annotations

import json
from typing import Any, List, Tuple

import gradio as gr

from src.core.agent_v2 import AgentV2


def create_app() -> gr.Blocks:
    agent = AgentV2()

    def handle_submit(
        message: str,
        history: List[Tuple[str, str]],
        pdf_file: str | None,
        fast_mode: bool,
        quality_mode: bool,
    ) -> tuple[List[Tuple[str, str]], str, str]:
        history = history or []
        if pdf_file:
            agent.index_pdf(pdf_file)

        agent.set_mode(fast_mode=fast_mode, enable_quality_enhance=quality_mode)
        response = agent.chat(message, session_id="web-user")
        history = history + [(message, response.answer)]
        whitebox = json.dumps(response.whitebox, ensure_ascii=False, indent=2)
        return history, whitebox, response.intent

    with gr.Blocks(title="ScholarAgent") as demo:
        gr.Markdown("# ScholarAgent\n多 Agent 学术研究助手")
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
                whitebox = gr.Code(label="白盒追踪", language="json")

        submit.click(
            handle_submit,
            inputs=[message, chatbot, pdf_file, fast_mode, quality_mode],
            outputs=[chatbot, whitebox, intent_box],
        )
    return demo
