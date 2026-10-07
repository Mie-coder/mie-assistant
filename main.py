"""加载配置并启动小咩；--plain 可使用原来的简单命令行。"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.types import Command
from app.agent import build_agent
from app.research_server import LocalResearchServer


def main() -> None:
    parser = argparse.ArgumentParser(description="MIE · 小咩助手")
    parser.add_argument("--plain", action="store_true", help="使用原来的纯文本聊天循环")
    args = parser.parse_args()
    # 根据文件位置定位配置，不依赖终端当前在哪个目录。
    app_dir = Path(__file__).resolve().parent
    # 独立项目只读取自己的配置；终端已设置的环境变量优先。
    load_dotenv(app_dir / ".env")
    os.environ.setdefault("LANGSMITH_PROJECT", "mie-assistant")

    # 服务只监听本机随机端口，不占用课程实验的 2024；退出时由本进程清理。
    print("正在启动本地研究服务…")
    with LocalResearchServer() as server:
        agent = build_agent(research_url=server.url)
        if not args.plain:
            from app.tui import MieApp

            MieApp(agent, model_name=os.environ["AGENTSEEK_MODEL"]).run()
        else:
            asyncio.run(plain_chat(agent))


async def plain_chat(agent):
    config = {"configurable":{"thread_id": str(uuid4())}}

    print("MIE · 小咩助手：已启动，可进行后台研究。输入 /exit 退出并停止研究服务。")

    while True:
        question = input("\n用户：").strip()
        if question == "/exit":
            print("MIE · 小咩助手：拜拜啦，下次聊～")
            break
        if not question:
            print("MIE · 小咩助手：请输入问题或指令。")
            continue
        print("MIE · 小咩助手：正在思考...")

        # 历史由Checkpointer管理，每轮只提交新的问题
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
        )

        # 一轮任务可能多次暂停，每次都要处理审批
        while result.get("__interrupt__"):
            approvals = {}

            for pause in result["__interrupt__"]:
                decisions = []

                for action in pause.value["action_requests"]:
                    print("\n待审批工具：",action["name"])
                    print(json.dumps(
                        action["args"],
                        ensure_ascii=False,
                        indent=2
                    )) 

                    answer = input("批准执行？输入 y 批准，其余输入拒绝：")
                    if answer.strip().lower() == "y":
                        decisions.append({"type":"approve"})
                    else:
                        decisions.append({
                            "type":"reject",
                            "message":"用户拒绝了此操作，请勿换一种方式重试。",
                        })
                approvals[pause.id] = {"decisions": decisions}
            result = await agent.ainvoke(
                Command(resume=approvals),
                config=config
            )
        print("\nMIE · 小咩助手：", result["messages"][-1].content)

if __name__ == "__main__":
    main()
