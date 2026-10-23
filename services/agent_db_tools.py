
# services/agent_db_tools.py

"""
Agent 的只读数据库工具。
只允许执行预先定义的 SQL，不接受模型传入任意 SQL。
"""

from typing import Any

from sqlalchemy import text

from database import SessionLocal


def query_users(
    operation: str,
    min_age: int | None = None,
) -> dict[str, Any]:
    """
    查询现有 users 表。

    operation:
        count       查询记录总数
        list        查询用户列表
        filter_age  查询年龄大于等于 min_age 的用户
    """

    db = SessionLocal()

    try:
        if operation == "count":
            sql = text("SELECT COUNT(*) AS total FROM users")
            rows = db.execute(sql).mappings().all()
            return {"total": rows[0]["total"]}

        if operation == "list":
            sql = text(
                "SELECT id, username, age "
                "FROM users ORDER BY id LIMIT 50"
            )
            rows = db.execute(sql).mappings().all()
            return {"users": [dict(row) for row in rows]}

        if operation == "filter_age":
            if min_age is None or isinstance(min_age, bool) or not isinstance(min_age, int):
                return {"error": "查询年龄条件必须是整数"}

            if not 0 <= min_age <= 150:
                return {"error": "年龄条件必须在 0 到 150 之间"}

            sql = text(
                "SELECT id, username, age "
                "FROM users WHERE age >= :min_age "
                "ORDER BY age, id LIMIT 50"
            )
            rows = db.execute(
                sql,
                {"min_age": min_age},
            ).mappings().all()

            return {"users": [dict(row) for row in rows]}

        return {"error": "不支持的数据库查询操作"}

    except Exception:
        # 不将数据库连接信息或 SQL 异常细节暴露给模型和用户。
        return {"error": "数据库查询失败"}

    finally:
        db.close()
