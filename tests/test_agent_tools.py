
# tests/test_agent_tools.py

import json

import pytest

from services import agent_tools


def parse_result(result: str) -> dict:
    """将工具返回的 JSON 字符串转换为字典。"""
    return json.loads(result)


def test_calculator_add():
    """测试计算器正常执行加法。"""
    result = parse_result(
        agent_tools.execute_tool(
            "calculator",
            '{"a": 10, "b": 5, "operation": "add"}',
        )
    )

    assert result["result"] == 15


def test_calculator_divide_by_zero():
    """测试除数为零时返回错误，而不是崩溃。"""
    result = parse_result(
        agent_tools.execute_tool(
            "calculator",
            '{"a": 10, "b": 0, "operation": "divide"}',
        )
    )

    assert "error" in result
    assert "除数不能为 0" in result["error"]


def test_unknown_tool_is_rejected():
    """测试未登记的工具不能被执行。"""
    result = parse_result(
        agent_tools.execute_tool(
            "delete_database",
            "{}",
        )
    )

    assert "error" in result
    assert "不允许调用工具" in result["error"]


def test_invalid_database_operation_is_rejected(monkeypatch):
    """测试非法查询操作会被拒绝，不会访问数据库。"""

    def should_not_be_called(**kwargs):
        pytest.fail("非法操作不应该调用数据库工具")

    monkeypatch.setattr(
        agent_tools,
        "query_users",
        should_not_be_called,
    )

    result = parse_result(
        agent_tools.execute_tool(
            "query_users",
            '{"operation": "delete", "min_age": null}',
        )
    )

    assert "error" in result
    assert "不支持的数据库查询操作" in result["error"]


def test_database_exception_does_not_leak_details(monkeypatch):
    """测试数据库工具发生异常时，不向调用方泄露异常细节。"""

    def raise_database_error(**kwargs):
        raise RuntimeError("SECRET_DATABASE_CONNECTION_DETAILS")

    monkeypatch.setattr(
        agent_tools,
        "query_users",
        raise_database_error,
    )

    result = parse_result(
        agent_tools.execute_tool(
            "query_users",
            '{"operation": "count", "min_age": null}',
        )
    )

    assert result == {"error": "工具执行失败"}
    assert "SECRET_DATABASE_CONNECTION_DETAILS" not in str(result)


def test_malformed_json_is_rejected():
    """测试格式错误的 JSON 不会导致程序崩溃。"""
    result = parse_result(
        agent_tools.execute_tool(
            "calculator",
            '{"a": 10, "b": ',
        )
    )

    assert "error" in result


def test_extra_calculator_argument_is_rejected():
    """测试模型不能给计算器传入未定义的参数。"""
    result = parse_result(
        agent_tools.execute_tool(
            "calculator",
            '{"a": 10, "b": 5, "operation": "add", "sql": "DROP TABLE users"}',
        )
    )

    assert "error" in result
    assert "result" not in result


def test_age_out_of_range_is_rejected():
    """测试年龄超过允许范围时，不应执行数据库查询。"""
    result = parse_result(
        agent_tools.execute_tool(
            "query_users",
            '{"operation": "filter_age", "min_age": 151}',
        )
    )

    assert "error" in result
    assert "年龄" in result["error"]
