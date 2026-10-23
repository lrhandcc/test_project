
# services/agent_tools.py

"""Agent 工具定义与执行模块。"""

import json
from typing import Any

from services.agent_db_tools import query_users


# 向模型公开的工具清单。
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算两个数字的加、减、乘、除。",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个数字"},
                    "b": {"type": "number", "description": "第二个数字"},
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "运算类型",
                    },
                },
                "required": ["a", "b", "operation"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_users",
            "description": (
                "只读查询数据库 users 表。"
                "可查询用户总数、全部用户列表，"
                "或查询年龄大于等于指定值的用户。"
                "用户询问数据库中的用户或学生记录时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["count", "list", "filter_age"],
                        "description": (
                            "count 查询总数；list 查询列表；"
                            "filter_age 按最低年龄筛选"
                        ),
                    },
                    "min_age": {
                        "type": ["integer", "null"],
                        "description": (
                            "最低年龄，仅 filter_age 操作需要；"
                            "其他操作传 null"
                        ),
                    },
                },
                "required": ["operation", "min_age"],
                "additionalProperties": False,
            },
        },
    },
]


def calculator(a: float, b: float, operation: str) -> float:
    """执行基础四则运算。"""

    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ValueError("除数不能为 0")
        return a / b

    raise ValueError("不支持的运算类型")


def execute_tool(tool_name: str, arguments_json: str) -> str:
    """只分发到明确登记的工具，并检查输入参数。"""

    try:
        arguments: Any = json.loads(arguments_json)

        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象")

        if tool_name == "calculator":
            if set(arguments) != {"a", "b", "operation"}:
                raise ValueError("计算器参数不正确")

            a = arguments["a"]
            b = arguments["b"]
            operation = arguments["operation"]

            if (
                isinstance(a, bool)
                or not isinstance(a, (int, float))
                or isinstance(b, bool)
                or not isinstance(b, (int, float))
            ):
                raise ValueError("a 和 b 必须是数字")

            if operation not in {"add", "subtract", "multiply", "divide"}:
                raise ValueError("不支持的运算类型")

            result = calculator(a, b, operation)
            return json.dumps({"result": result}, ensure_ascii=False)

        if tool_name == "query_users":
            if set(arguments) != {"operation", "min_age"}:
                raise ValueError("数据库工具参数不正确")

            operation = arguments["operation"]
            min_age = arguments["min_age"]

            if operation not in {"count", "list", "filter_age"}:
                raise ValueError("不支持的数据库查询操作")

            if operation == "filter_age":
                if isinstance(min_age, bool) or not isinstance(min_age, int):
                    raise ValueError("筛选年龄必须是整数")
            elif min_age is not None:
                raise ValueError("count 和 list 操作的 min_age 必须为 null")

            result = query_users(
                operation=operation,
                min_age=min_age,
            )
            return json.dumps(result, ensure_ascii=False, default=str)

        return json.dumps(
            {"error": f"不允许调用工具：{tool_name}"},
            ensure_ascii=False,
        )

    except (json.JSONDecodeError, TypeError, ValueError, OverflowError) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    except Exception:
        return json.dumps({"error": "工具执行失败"}, ensure_ascii=False)
