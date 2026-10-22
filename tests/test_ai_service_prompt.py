
from services.ai_service import build_messages


def test_rag_prompt_requires_sufficient_evidence():
    """检查 RAG 提示词是否包含证据充分性规则。"""
    context = "TCP 第三次握手发送 ACK，但没有给出具体确认号。"

    messages = build_messages(
        question="TCP 第三次握手的确认号是多少？",
        context=context,
    )

    system_prompt = messages[0]["content"]

    # 确保关键规则和本轮资料都被放进了模型请求。
    assert "相关性高不等于证据充分" in system_prompt
    assert "不得自行补全或猜测" in system_prompt
    assert context in system_prompt


def test_history_cannot_replace_knowledge_base_evidence():
    """检查历史记录保留正确，同时提示词要求优先依据当前资料。"""
    history = [
        {
            "role": "user",
            "content": "确认号是不是 12345？",
        },
        {
            "role": "assistant",
            "content": "确认号就是 12345。",
        },
    ]

    question = "请根据当前知识库资料确认具体确认号。"

    messages = build_messages(
        question=question,
        history=history,
        context="TCP 第三次握手发送 ACK，但没有给出具体确认号。",
    )

    system_prompt = messages[0]["content"]

    # 历史记录仍然保留，不能因为加强证据规则而丢失聊天上下文。
    assert messages[1] == history[0]
    assert messages[2] == history[1]

    # 当前问题仍然必须放在最后。
    assert messages[-1] == {
        "role": "user",
        "content": question,
    }

    # 明确要求历史对话不能替代当前知识库证据。
    assert "历史对话可以帮助理解上下文，但不能代替本轮知识库证据" in (
        system_prompt
    )


def test_normal_chat_still_works_without_rag_context():
    """检查普通聊天没有知识库资料时，仍能正常构造消息。"""
    question = "请用简单的话解释什么是 Python。"

    messages = build_messages(question=question)

    assert messages[0]["role"] == "system"
    assert messages[-1] == {
        "role": "user",
        "content": question,
    }

    # 没有知识库资料时，不应该误加入 RAG 专用规则。
    assert len(messages) == 2
    assert "【知识库问答规则】" not in messages[0]["content"]
