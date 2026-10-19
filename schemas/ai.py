
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="用户当前提出的问题",
    )

    # 第一次聊天可以不传；继续已有聊天时传入会话 ID
    session_id: int | None = Field(
        default=None,
        gt=0,
        description="聊天会话 ID",
    )


class ChatResponse(BaseModel):
    session_id: int
    answer: str
    model: str
