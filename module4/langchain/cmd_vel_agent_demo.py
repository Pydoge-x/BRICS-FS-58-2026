#!/usr/bin/env python3
"""LangChain 工具调用 → drive_turtle.sh → turtlesim (LangChain 1.x API)"""
import os
import subprocess
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

# 用脚本自身位置定位，避免相对路径/expanduser 依赖当前工作目录
DRIVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "drive_turtle.sh")


@tool
def move_turtle(command: str) -> str:
    """控制 turtlesim。command 取：前进、后退、左转、右转、停、巡检。"""
    r = subprocess.run([DRIVE, command], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return f"失败: {r.stderr or r.stdout}"
    return f"已执行: {command}"


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("请 export DEEPSEEK_API_KEY=...")
    llm = ChatOpenAI(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0,
    )
    # LangChain 1.x: create_agent 自带 ReAct 循环，无需 AgentExecutor / agent_scratchpad
    agent = create_agent(
        model=llm,
        tools=[move_turtle],
        system_prompt="你是海龟控制助手。用户要运动时必须调用 move_turtle 工具。",
    )
    q = os.environ.get("USER_QUERY", "请让海龟前进，然后停")
    result = agent.invoke({"messages": [{"role": "user", "content": q}]})
    for msg in result["messages"]:
        msg.pretty_print()


if __name__ == "__main__":
    main()
