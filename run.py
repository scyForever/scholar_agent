from __future__ import annotations

from src.core.agent_v2 import AgentV2
from src.ui.gradio_app import create_app


def run_cli() -> None:
    agent = AgentV2()
    print("ScholarAgent CLI 已启动，输入 exit 退出。")
    while True:
        query = input("\n你: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        response = agent.chat(query, session_id="cli-user")
        print(f"\nScholarAgent: {response.answer}")
        if response.trace_id:
            print(f"[trace_id] {response.trace_id}")


def run_web() -> None:
    app = create_app()
    app.launch()


def run_verify() -> None:
    import verify_features

    verify_features.main()


def main() -> None:
    print("请选择模式：")
    print("[1] Web界面")
    print("[2] 命令行")
    print("[3] 功能验证")
    choice = input("输入编号: ").strip()

    if choice == "1":
        run_web()
    elif choice == "2":
        run_cli()
    elif choice == "3":
        run_verify()
    else:
        print("无效选择。")


if __name__ == "__main__":
    main()
