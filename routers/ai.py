
import json
import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAIError,
    RateLimitError,
)
from sqlalchemy.orm import Session

from database import get_db
from models.chat import ChatSession
from schemas.ai import ChatRequest, ChatResponse
from schemas.chat import ChatMessageResponse
from services.ai_service import ask_ai, stream_ai
from services.chat_service import (
    create_chat_session,
    get_chat_history,
    save_chat_exchange,
)
from rag.retriever import build_retriever

router = APIRouter()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_retriever():
    """缓存检索器，避免每次请求重复构建知识库索引。"""
    return build_retriever()


def get_or_create_session(db: Session, session_id: int | None):
    """获取已有聊天会话；没有会话 ID 时创建新会话。"""
    if session_id is None:
        return create_chat_session(db)

    chat_session = db.get(ChatSession, session_id)

    if chat_session is None:
        raise HTTPException(status_code=404, detail="聊天会话不存在")

    return chat_session


def load_history(db: Session, session_id: int):
    """读取历史消息，并转换成 AI 接口需要的格式。"""
    old_messages = get_chat_history(db, session_id)

    return [
        {"role": message.role, "content": message.content}
        for message in old_messages
    ]


def retrieve_context(question: str) -> str:
    """检索知识库，并整理成 AI 可以参考的上下文。"""
    retriever = get_retriever()
    results = retriever.search(question, top_k=2)

    # 分数用于观察检索效果，不代表答案正确的概率。
    context_parts = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            f"【参考片段 {index}】\n{result['chunk']}"
        )

    return "\n\n".join(context_parts)


def raise_ai_http_error(exc: Exception):
    """将 AI 服务异常转换成适当的 HTTP 错误。"""
    if isinstance(exc, RuntimeError):
        status_code = 500
        detail = "AI 服务配置异常或返回内容无效"
    elif isinstance(exc, APITimeoutError):
        status_code = 504
        detail = "AI 响应超时，请稍后重试"
    elif isinstance(exc, APIConnectionError):
        status_code = 502
        detail = "暂时无法连接 AI 服务"
    elif isinstance(exc, RateLimitError):
        status_code = 503
        detail = "AI 服务当前请求受限，请稍后重试"
    elif isinstance(exc, APIStatusError):
        status_code = 502
        detail = "AI 服务暂时无法处理请求"
    elif isinstance(exc, OpenAIError):
        status_code = 502
        detail = "AI 调用失败，请稍后重试"
    else:
        status_code = 500
        detail = "AI 服务发生异常"

    raise HTTPException(status_code=status_code, detail=detail)


@router.post("/chat", response_model=ChatResponse)
def chat_api(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """普通聊天：检索知识库、生成回答，并保存聊天记录。"""
    chat_session = get_or_create_session(db, request.session_id)
    session_id = chat_session.id
    history = load_history(db, session_id)

    try:
        context = retrieve_context(request.question)
        result = ask_ai(
            request.question,
            history,
            context=context,
        )
    except (
        RuntimeError,
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        APIStatusError,
        OpenAIError,
    ) as exc:
        logger.exception("普通 AI 聊天请求失败")
        raise_ai_http_error(exc)
    except Exception as exc:
        logger.exception("普通聊天的知识检索或处理失败")
        raise HTTPException(
            status_code=500,
            detail="知识检索或聊天处理失败，请稍后重试",
        ) from exc

    try:
        save_chat_exchange(
            db,
            session_id,
            request.question,
            result["answer"],
        )
    except Exception as exc:
        logger.exception("普通聊天记录保存失败")
        raise HTTPException(
            status_code=500,
            detail="聊天记录保存失败，请稍后重试",
        ) from exc

    return {
        "session_id": session_id,
        "answer": result["answer"],
        "model": result["model"],
    }


@router.post("/chat/stream")
def chat_stream_api(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """流式聊天：先检索知识库，再逐段生成并保存聊天记录。"""
    chat_session = get_or_create_session(db, request.session_id)
    session_id = chat_session.id
    history = load_history(db, session_id)

    # 在开始流式响应前检索，避免检索失败时已开始发送内容。
    try:
        context = retrieve_context(request.question)
    except Exception as exc:
        logger.exception("流式聊天的知识检索失败")
        raise HTTPException(
            status_code=500,
            detail="知识检索失败，请稍后重试",
        ) from exc

    def generate():
        answer_parts = []

        try:
            for content in stream_ai(
                request.question,
                history,
                context=context,
            ):
                answer_parts.append(content)

                event = {
                    "type": "content",
                    "content": content,
                }
                yield json.dumps(event, ensure_ascii=False) + "\n"

            answer = "".join(answer_parts)

            if not answer:
                raise RuntimeError("AI 模型没有返回有效回答")

            save_chat_exchange(
                db,
                session_id,
                request.question,
                answer,
            )

            event = {
                "type": "done",
                "session_id": session_id,
            }
            yield json.dumps(event, ensure_ascii=False) + "\n"

        except Exception:
            logger.exception("流式 AI 聊天请求失败")

            event = {
                "type": "error",
                "message": "AI 生成或聊天记录保存失败，请稍后重试",
            }
            yield json.dumps(event, ensure_ascii=False) + "\n"

        finally:
            db.close()

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/chat/{session_id}/messages",
    response_model=list[ChatMessageResponse],
)
def get_chat_messages_api(
    session_id: int,
    db: Session = Depends(get_db),
):
    """查询指定会话的聊天历史。"""
    chat_session = db.get(ChatSession, session_id)

    if chat_session is None:
        raise HTTPException(status_code=404, detail="聊天会话不存在")

    return get_chat_history(db, session_id)
