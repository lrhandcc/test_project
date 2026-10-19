
import sys
from pathlib import Path

# 将项目根目录加入 Python 模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def test_chat_history_returns_404_for_missing_session():
    """查询不存在的会话，应返回 404。"""
    response = client.get("/chat/999999999/messages")

    assert response.status_code == 404
    assert response.json()["detail"] == "聊天会话不存在"


def test_chat_returns_422_when_question_is_missing():
    """聊天请求缺少必填问题，应由 FastAPI 返回 422。"""
    response = client.post("/chat", json={})

    assert response.status_code == 422
