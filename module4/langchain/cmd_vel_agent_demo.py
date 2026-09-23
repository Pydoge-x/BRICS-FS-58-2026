#!/usr/bin/env python3
"""LangChain 工具调用 → drive_turtle.sh → turtlesim"""
import os
import subprocess
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

DRIVE = os.path.expanduser("~/lab-m4/scripts/drive_turtle.sh")


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
    tools = [move_turtle]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是海龟控制助手。用户要运动时必须调用 move_turtle 工具。"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    ex = AgentExecutor(agent=agent, tools=tools, verbose=True)
    q = os.environ.get("USER_QUERY", "请让海龟前进，然后停")
    print(ex.invoke({"input": q}))


if __name__ == "__main__":
    main()
