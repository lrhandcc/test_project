
from rag.retriever import build_retriever


# 每个问题的 expected_terms 用于人工核对检索片段。
# 这是主题相关性评估，不代表片段必须逐字回答问题。
TEST_CASES = [
    {
        "question": "TCP 如何建立连接？",
        "expected_terms": ["TCP", "三次握手", "SYN"],
        "category": "in_scope",
    },
    {
        "question": "TCP 第一次握手发送什么？",
        "expected_terms": ["第一次握手", "SYN"],
        "category": "in_scope",
    },
    {
        "question": "TCP 第二次握手发送什么？",
        "expected_terms": ["第二次握手", "SYN-ACK"],
        "category": "in_scope",
    },
    {
        "question": "TCP 第三次握手的作用是什么？",
        "expected_terms": ["第三次握手", "ACK", "确认服务器的序列号"],
        "category": "in_scope",
    },
    {
        "question": "TCP 完成三次握手后会怎样？",
        "expected_terms": ["完成三次握手后", "传输数据"],
        "category": "in_scope",
    },
    {
        "question": "HTTP 是什么协议，属于哪一层？",
        "expected_terms": ["HTTP", "应用层"],
        "category": "in_scope",
    },
    {
        "question": "HTTP 请求包含哪些内容？",
        "expected_terms": ["请求方法", "路径", "请求头", "请求体"],
        "category": "in_scope",
    },
    {
        "question": "HTTP 通常用于哪些系统之间的通信？",
        "expected_terms": ["浏览器", "Web 服务器"],
        "category": "in_scope",
    },
    {
        "question": "量子计算机使用什么量子纠错算法？",
        "expected_terms": [],
        "category": "out_of_scope",
    },
    {
        "question": "如何使用神经网络训练图像分类模型？",
        "expected_terms": [],
        "category": "out_of_scope",
    },
]


def is_relevant(chunk: str, expected_terms: list[str]) -> bool:
    """根据人工指定的关键事实，判断片段是否相关。"""
    if not expected_terms:
        return False

    # 所有指定事实都出现在片段中，才算命中。
    return all(term in chunk for term in expected_terms)


def main():
    retriever = build_retriever()

    in_scope_cases = [
        case for case in TEST_CASES
        if case["category"] == "in_scope"
    ]

    hit_counts = {1: 0, 2: 0, 3: 0}

    print("=" * 70)
    print("RAG 检索效果评估（当前配置基线）")
    print("chunk_size=100, overlap=20, analyzer=char, ngram_range=(1, 2)")
    print("=" * 70)

    for case in TEST_CASES:
        question = case["question"]
        expected_terms = case["expected_terms"]
        category = case["category"]

        results = retriever.search(question, top_k=3)

        print(f"\n问题：{question}")
        print(f"类别：{category}")

        first_relevant_rank = None

        for rank, result in enumerate(results, start=1):
            chunk = result["chunk"]
            relevant = is_relevant(chunk, expected_terms)

            if relevant and first_relevant_rank is None:
                first_relevant_rank = rank

            print(
                f"\n排名：{rank} | "
                f"相似度：{result['score']:.4f} | "
                f"相关：{'是' if relevant else '否'}"
            )
            print(f"文本片段：\n{chunk}")

        if category == "in_scope":
            if first_relevant_rank is None:
                print("\n结果：前三条没有命中全部预期事实")
            else:
                print(f"\n结果：首次命中在第 {first_relevant_rank} 名")

            for k in hit_counts:
                if first_relevant_rank is not None and first_relevant_rank <= k:
                    hit_counts[k] += 1
        else:
            top_score = results[0]["score"] if results else 0.0
            print(f"\n知识库外问题最高分：{top_score:.4f}")
            print("请人工检查返回片段是否与问题相关。")

    total = len(in_scope_cases)

    print("\n" + "=" * 70)
    print("汇总（知识库内问题）")
    print("=" * 70)

    for k, count in hit_counts.items():
        rate = count / total if total else 0.0
        print(f"Hit@{k}: {count}/{total} = {rate:.1%}")

    print(
        "\n说明：当前指标基于人工指定的事实关键词组合，"
        "仍需核对打印出的文本片段。"
    )


if __name__ == "__main__":
    main()

# ==================== pytest 自动化测试 ====================

import pytest
from rag.retriever import TextRetriever


def test_build_retriever_has_chunks():
    """检查知识库能否成功加载。"""
    retriever = build_retriever()
    assert len(retriever.chunks) > 0


def test_search_returns_expected_fields():
    """检查检索结果的数据结构和分数范围。"""
    retriever = build_retriever()
    results = retriever.search("TCP 如何建立连接？", top_k=2)

    assert len(results) == 2

    for result in results:
        assert "chunk" in result
        assert "score" in result
        assert "overlap" in result
        assert 0.0 <= result["score"] <= 1.0
        assert 0.0 <= result["overlap"] <= 1.0


def test_search_rejects_empty_question():
    """检查空问题是否会被拒绝。"""
    retriever = TextRetriever(["TCP 是一种传输层协议。"])

    with pytest.raises(ValueError, match="问题不能为空"):
        retriever.search("   ")


def test_search_rejects_invalid_top_k():
    """检查非法 top_k 是否会被拒绝。"""
    retriever = TextRetriever(["TCP 是一种传输层协议。"])

    with pytest.raises(ValueError, match="top_k 必须大于 0"):
        retriever.search("TCP", top_k=0)


def test_retriever_rejects_empty_chunks():
    """检查空知识库是否会触发异常。"""
    with pytest.raises(ValueError, match="知识库不能为空"):
        TextRetriever([])