
from rag.retriever import TextRetriever, DOCUMENT_PATH
from rag.text_splitter import read_text_file, split_text


# 与之前的评估保持一致：8 道知识库内问题，2 道范围外问题
TEST_CASES = [
    ("TCP 如何建立连接？", ["TCP", "三次握手", "SYN"]),
    ("TCP 第一次握手发送什么？", ["第一次握手", "SYN"]),
    ("TCP 第二次握手发送什么？", ["第二次握手", "SYN-ACK"]),
    ("TCP 第三次握手的作用是什么？",
     ["第三次握手", "ACK", "确认服务器的序列号"]),
    ("TCP 完成三次握手后会怎样？",
     ["完成三次握手后", "传输数据"]),
    ("HTTP 是什么协议，属于哪一层？", ["HTTP", "应用层"]),
    ("HTTP 请求包含哪些内容？",
     ["请求方法", "路径", "请求头", "请求体"]),
    ("HTTP 通常用于哪些系统之间的通信？",
     ["浏览器", "Web 服务器"]),
    ("量子计算机使用什么量子纠错算法？", []),
    ("如何使用神经网络训练图像分类模型？", []),
]


def is_relevant(chunk: str, expected_terms: list[str]) -> bool:
    """判断一个文本块是否包含该问题要求的全部关键词。"""
    if not expected_terms:
        return False
    return all(term in chunk for term in expected_terms)


def main():
    # 固定分块参数，保证这轮只研究 Top-K 的影响
    text = read_text_file(str(DOCUMENT_PATH))
    chunks = split_text(text, chunk_size=150, overlap=20)
    retriever = TextRetriever(chunks)

    print("=" * 65)
    print("Top-K 实验：固定 chunk_size=150，overlap=20")
    print(f"知识库文本块数量：{len(chunks)}")
    print("=" * 65)

    for top_k in [1, 2, 3]:
        print(f"\n{'-' * 65}")
        print(f"当前 top_k = {top_k}")
        print(f"{'-' * 65}")

        hit_count = 0
        in_scope_count = 0
        out_of_scope_scores = []

        for question, expected_terms in TEST_CASES:
            results = retriever.search(question, top_k=top_k)

            if expected_terms:
                in_scope_count += 1
                first_relevant_rank = None

                for rank, item in enumerate(results, start=1):
                    if is_relevant(item["chunk"], expected_terms):
                        first_relevant_rank = rank
                        break

                if first_relevant_rank is not None:
                    hit_count += 1

                print(
                    f"Rank={first_relevant_rank!s:4} | "
                    f"{question}"
                )
            else:
                # 范围外问题：记录排名第一的结果分数
                # 不把相似度直接当成可信度或拒答概率
                if results:
                    out_of_scope_scores.append(results[0]["score"])

        hit_rate = hit_count / in_scope_count * 100
        print(f"\nHit@{top_k}: {hit_rate:.1f}% "
              f"({hit_count}/{in_scope_count})")

        print("\n范围外问题的最高检索分数：")
        for (question, expected_terms), score in zip(
            [case for case in TEST_CASES if not case[1]],
            out_of_scope_scores,
        ):
            print(f"  {score:.4f} | {question}")

    print("\n实验完成。正式检索器配置未被修改。")


if __name__ == "__main__":
    main()
