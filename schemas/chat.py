
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ChatMessageResponse(BaseModel):
    # 允许 Pydantic 直接读取 SQLAlchemy 模型对象
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    role: str
    content: str
    created_at: datetime
