
from rag.retriever import build_retriever


def main():
    retriever = build_retriever()

    questions = [
        "TCP 如何建立连接？",
        "HTTP 请求包含哪些内容？",
    ]

    for question in questions:
        print(f"\n问题：{question}")
        print("-" * 50)

        results = retriever.search(question, top_k=2)

        for rank, result in enumerate(results, start=1):
            print(f"\n排名：{rank}")
            print(f"相似度：{result['score']:.4f}")
            print(f"文本内容：\n{result['chunk']}")


if __name__ == "__main__":
    main()
