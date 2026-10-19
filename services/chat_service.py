
from sqlalchemy.orm import Session

from models.chat import ChatMessage, ChatSession


def create_chat_session(db: Session) -> ChatSession:
    session = ChatSession()
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def save_chat_message(
    db: Session,
    session_id: int,
    role: str,
    content: str,
) -> ChatMessage:
    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_chat_history(
    db: Session,
    session_id: int,
) -> list[ChatMessage]:
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id)
        .all()
    )


def save_chat_exchange(
    db: Session,
    session_id: int,
    question: str,
    answer: str,
) -> tuple[ChatMessage, ChatMessage]:
    """在同一个事务中保存用户问题和 AI 回答。"""
    user_message = ChatMessage(
        session_id=session_id,
        role="user",
        content=question,
    )

    assistant_message = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=answer,
    )

    try:
        db.add_all([user_message, assistant_message])
        db.commit()
        db.refresh(user_message)
        db.refresh(assistant_message)

        return user_message, assistant_message

    except Exception:
        db.rollback()
        raise