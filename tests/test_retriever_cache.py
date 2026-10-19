
import routers.ai as ai_router


def test_retriever_is_cached(monkeypatch):
    """连续获取检索器两次，底层构建函数只应执行一次。"""
    calls = []

    class FakeRetriever:
        pass

    fake_retriever = FakeRetriever()

    def fake_build_retriever():
        calls.append(1)
        return fake_retriever

    # 清理缓存，避免其他测试影响本测试。
    ai_router.get_retriever.cache_clear()

    monkeypatch.setattr(
        ai_router,
        "build_retriever",
        fake_build_retriever,
    )

    try:
        first = ai_router.get_retriever()
        second = ai_router.get_retriever()

        assert first is second
        assert len(calls) == 1
    finally:
        ai_router.get_retriever.cache_clear()
