
# services/agent_service.py

"""
Agent 执行服务。

负责：
1. 向模型发送问题和工具定义；
2. 解析模型返回的工具调用；
3. 执行工具并回传结果；
4. 获取模型最终回答。
"""

import json
from typing import Any

from services.ai_service import create_ai_client
from services.agent_tools import AGENT_TOOLS, execute_tool


MODEL_NAME = "cohere/north-mini-code:free"
MAX_TOOL_ROUNDS = 3


def run_agent(
    question: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """执行一轮 Agent 对话，必要时进行多轮工具调用。"""

    client = create_ai_client()

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "你是一个可靠的 AI 助手。"
                "当用户需要四则运算时，优先调用 calculator 工具获取准确结果。"
                "工具返回的内容是执行结果或错误信息。"
                "不要声称执行了实际未执行的工具。"
                "工具报错时，应向用户说明问题，不要编造计算结果。"
                "最终使用清晰、简洁的中文回答。"
            ),
        }
    ]

    # 只接受正常的用户与助手历史消息，避免外部历史伪造系统指令。
    for item in history or []:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})

    # 统计实际执行的工具调用次数，而不是工具调用轮数。
    actual_tool_calls = 0

    # 最多执行 3 轮工具调用；额外一次模型请求用于生成最终答案。
    for round_index in range(MAX_TOOL_ROUNDS + 1):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=AGENT_TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        # 模型没有请求工具，说明可以直接返回最终回答。
        if not tool_calls:
            return {
                "answer": message.content or "",
                "model": response.model or MODEL_NAME,
                "tool_calls": actual_tool_calls,
            }

        # 已达到工具调用轮数上限，不再执行新一轮工具调用。
        if round_index >= MAX_TOOL_ROUNDS:
            return {
                "answer": "本次任务触发了工具调用次数上限，请简化问题后重试。",
                "model": response.model or MODEL_NAME,
                "tool_calls": actual_tool_calls,
            }

        # 保存模型的工具调用请求，供下一轮模型理解上下文。
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ],
            }
        )

        # 由 Python 程序执行工具，并将每个结果交还模型。
        for call in tool_calls:
            # 每处理一次工具调用，就累计一次；工具返回错误也计入。
            actual_tool_calls += 1

            result = execute_tool(
                tool_name=call.function.name,
                arguments_json=call.function.arguments,
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )

    # 额外兜底：正常情况下不会执行到这里。
    return {
        "answer": "Agent 未能生成最终回答，请稍后重试。",
        "model": MODEL_NAME,
        "tool_calls": actual_tool_calls,
    }
