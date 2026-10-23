
from types import SimpleNamespace

from services import agent_service


def make_response(content=None, tool_calls=None):
    """构造模拟的模型响应。"""
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        model="mock-model",
    )


def make_calculator_call():
    """构造模型请求调用计算器的响应对象。"""
    return SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="calculator",
            arguments='{"a": 2, "b": 3, "operation": "add"}',
        ),
    )


def fake_client(responses):
    """按顺序返回预设响应，并记录模型请求。"""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )
    return client, calls


def test_agent_returns_direct_answer(monkeypatch):
    """不需要工具时，直接返回模型答案。"""
    client, calls = fake_client([
        make_response(content="Python 是一种编程语言。")
    ])
    monkeypatch.setattr(
        agent_service, "create_ai_client", lambda: client
    )

    result = agent_service.run_agent("Python 是什么？")

    assert result["answer"] == "Python 是一种编程语言。"
    assert result["tool_calls"] == 0
    assert len(calls) == 1


def test_agent_executes_tool_and_returns_final_answer(monkeypatch):
    """工具执行后，模型能基于工具结果生成最终答案。"""
    client, calls = fake_client([
        make_response(tool_calls=[make_calculator_call()]),
        make_response(content="计算结果是 5。"),
    ])
    monkeypatch.setattr(
        agent_service, "create_ai_client", lambda: client
    )

    result = agent_service.run_agent("计算 2 加 3")

    assert result["answer"] == "计算结果是 5。"
    assert result["tool_calls"] == 1
    assert len(calls) == 2

    # 第二次模型请求中应包含工具执行结果。
    messages = calls[1]["messages"]
    tool_messages = [
        message for message in messages
        if message.get("role") == "tool"
    ]

    assert len(tool_messages) == 1
    assert '"result": 5' in tool_messages[0]["content"]


def test_agent_stops_at_tool_round_limit(monkeypatch):
    """连续请求工具时，达到轮数限制后应停止。"""
    monkeypatch.setattr(agent_service, "MAX_TOOL_ROUNDS", 1)

    client, calls = fake_client([
        make_response(tool_calls=[make_calculator_call()]),
        make_response(tool_calls=[make_calculator_call()]),
    ])
    monkeypatch.setattr(
        agent_service, "create_ai_client", lambda: client
    )

    result = agent_service.run_agent("计算 2 加 3")

    assert "上限" in result["answer"]
    assert len(calls) == 2
