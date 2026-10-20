
from rag.retriever import DOCUMENT_PATH, TextRetriever
from rag.text_splitter import read_text_file, split_text


# 与基线实验使用相同的 10 个测试问题，确保不同配置可比较。
TEST_CASES = [
    {
        "question": "TCP 如何建立连接？",
        "expected": ["TCP", "三次握手", "SYN"],
    },
    {
        "question": "TCP 第一次握手发送什么？",
        "expected": ["第一次握手", "SYN"],
    },
    {
        "question": "TCP 第二次握手发送什么？",
        "expected": ["第二次握手", "SYN-ACK"],
    },
    {
        "question": "TCP 第三次握手的作用是什么？",
        "expected": ["第三次握手", "ACK", "确认服务器的序列号"],
    },
    {
        "question": "TCP 完成三次握手后会怎样？",
        "expected": ["完成三次握手后", "传输数据"],
    },
    {
        "question": "HTTP 是什么协议，属于哪一层？",
        "expected": ["HTTP", "应用层"],
    },
    {
        "question": "HTTP 请求包含哪些内容？",
        "expected": ["请求方法", "路径", "请求头", "请求体"],
    },
    {
        "question": "HTTP 通常用于哪些系统之间的通信？",
        "expected": ["浏览器", "Web 服务器"],
    },
    {
        "question": "量子计算机使用什么量子纠错算法？",
        "expected": [],
    },
    {
        "question": "如何使用神经网络训练图像分类模型？",
        "expected": [],
    },
]


def is_relevant(chunk: str, expected: list[str]) -> bool:
    """判断文本块是否包含问题要求的全部关键词。"""
    if not expected:
        return False

    return all(term in chunk for term in expected)


def evaluate_config(
    text: str,
    chunk_size: int,
    overlap: int,
) -> dict:
    """评估一组分块参数下的检索表现。"""
    chunks = split_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    retriever = TextRetriever(chunks)

    # 只统计前 8 个知识库内问题的 Hit@K。
    in_scope_cases = TEST_CASES[:8]
    hit_counts = {1: 0, 2: 0, 3: 0}
    first_hit_ranks = {}

    for case in in_scope_cases:
        question = case["question"]
        expected = case["expected"]

        # 注意：当前项目的方法名是 search，不是 retrieve。
        # 返回结果的正文键名是 chunk，相似度键名是 score。
        results = retriever.search(question, top_k=3)

        first_rank = None

        for rank, item in enumerate(results, start=1):
            if is_relevant(item["chunk"], expected):
                first_rank = rank
                break

        first_hit_ranks[question] = first_rank

        # Hit@K：前 K 个结果里至少有一个相关文本块，就算命中。
        for k in hit_counts:
            if first_rank is not None and first_rank <= k:
                hit_counts[k] += 1

    total_questions = len(in_scope_cases)

    # 另外观察两个知识库范围外问题的最高相似度。
    oos_scores = {}
    for case in TEST_CASES[8:]:
        results = retriever.search(case["question"], top_k=1)
        oos_scores[case["question"]] = (
            results[0]["score"] if results else 0.0
        )

    # 记录所有文本块的长度，便于检查分块变化。
    chunk_lengths = [len(chunk) for chunk in chunks]

    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunk_count": len(chunks),
        "chunk_lengths": chunk_lengths,
        "hit_at_1": hit_counts[1] / total_questions,
        "hit_at_2": hit_counts[2] / total_questions,
        "hit_at_3": hit_counts[3] / total_questions,
        "first_hit_ranks": first_hit_ranks,
        "oos_scores": oos_scores,
        "chunks": chunks,
    }


def print_result(result: dict) -> None:
    """打印单组实验结果。"""
    print("\n" + "=" * 65)
    print(
        f"chunk_size={result['chunk_size']}, "
        f"overlap={result['overlap']}"
    )
    print(f"文本块数量：{result['chunk_count']}")
    print(f"各块长度：{result['chunk_lengths']}")
    print(f"Hit@1：{result['hit_at_1']:.1%}")
    print(f"Hit@2：{result['hit_at_2']:.1%}")
    print(f"Hit@3：{result['hit_at_3']:.1%}")

    print("\n各问题第一次命中的排名：")
    for question, rank in result["first_hit_ranks"].items():
        print(f"  Rank={rank} | {question}")

    print("\n知识库范围外问题的最高相似度：")
    for question, score in result["oos_scores"].items():
        print(f"  {score:.4f} | {question}")


def print_summary(results: list[dict]) -> None:
    """汇总不同配置，方便横向比较。"""
    print("\n\n" + "#" * 75)
    print("实验结果汇总")
    print("#" * 75)
    print(
        f"{'chunk_size':>10} {'overlap':>8} {'块数':>6} "
        f"{'Hit@1':>9} {'Hit@2':>9} {'Hit@3':>9}"
    )

    for result in results:
        print(
            f"{result['chunk_size']:>10} "
            f"{result['overlap']:>8} "
            f"{result['chunk_count']:>6} "
            f"{result['hit_at_1']:>8.1%} "
            f"{result['hit_at_2']:>8.1%} "
            f"{result['hit_at_3']:>8.1%}"
        )


def find_actual_overlap(
    previous_chunk: str,
    next_chunk: str,
) -> str:
    """
    找出前一个文本块的结尾与下一个文本块开头
    实际共有的最长字符串。
    """
    max_length = min(len(previous_chunk), len(next_chunk))

    for length in range(max_length, 0, -1):
        if previous_chunk[-length:] == next_chunk[:length]:
            return next_chunk[:length]

    return ""


def print_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """打印每个文本块及相邻块之间的实际重叠内容。"""
    chunks = split_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    print("\n" + "-" * 65)
    print(
        f"分块详情：chunk_size={chunk_size}, "
        f"overlap={overlap}"
    )

    for index, chunk in enumerate(chunks):
        print(f"\n[Chunk {index + 1}] 长度={len(chunk)}")
        print(chunk)

        if index > 0:
            actual_overlap = find_actual_overlap(
                chunks[index - 1],
                chunk,
            )
            print(
                f"\n与上一个块的实际重叠长度："
                f"{len(actual_overlap)}"
            )
            print(f"实际重叠内容：{actual_overlap!r}")

    return chunks


def main() -> None:
    text = read_text_file(str(DOCUMENT_PATH))
    all_results = []

    # 实验一：改变 chunk_size，固定 overlap=20。
    print("\n第一组实验：比较不同 chunk_size")
    chunk_size_configs = [
        (80, 20),
        (100, 20),
        (150, 20),
        (200, 20),
    ]

    for chunk_size, overlap in chunk_size_configs:
        result = evaluate_config(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_results.append(result)
        print_result(result)

    print("\n\n第一组实验的分块详情：")
    print_chunks(text, chunk_size=150, overlap=20)

    # 实验二：固定 chunk_size=150，改变 overlap。
    print("\n\n第二组实验：比较不同 overlap")
    overlap_configs = [30, 40, 50, 60]

    for overlap in overlap_configs:
        result = evaluate_config(
            text,
            chunk_size=150,
            overlap=overlap,
        )
        all_results.append(result)
        print_result(result)
        print_chunks(
            text,
            chunk_size=150,
            overlap=overlap,
        )

    print_summary(all_results)


if __name__ == "__main__":
    main()
