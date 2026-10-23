
# routers/agent.py

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.agent_service import run_agent


router = APIRouter(prefix="/agent", tags=["Agent"])


class AgentMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[AgentMessage] = Field(default_factory=list, max_length=20)


class AgentResponse(BaseModel):
    answer: str
    model: str
    tool_calls: int


@router.post("/chat", response_model=AgentResponse)
def agent_chat(request: AgentRequest):
    """独立的 Agent 对话接口，不修改现有聊天和 RAG 流程。"""

    try:
        return run_agent(
            question=request.question,
            history=[
                message.model_dump()
                for message in request.history
            ],
        )
    except Exception as exc:
        # 不把 API Key、请求头或底层异常详情返回给客户端。
        print(f"Agent 调用失败：{type(exc).__name__}")
        raise HTTPException(
            status_code=502,
            detail="Agent 调用失败，请检查模型服务配置后重试。",
        ) from exc
