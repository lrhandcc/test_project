
from rag.retriever import TextRetriever
from rag.text_splitter import read_text_file, split_text
from rag.retriever import DOCUMENT_PATH
from rag.test_retriever import TEST_CASES, is_relevant


def main():
    # 独立实验：固定使用目前表现较好的切分方案。
    text = read_text_file(str(DOCUMENT_PATH))
    chunks = split_text(text, chunk_size=150, overlap=20)
    retriever = TextRetriever(chunks)

    # 只是实验候选值，不代表已经确定正式阈值。
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]

    scores_by_case = []

    print("=" * 65)
    print("RAG 相似度阈值实验")
    print(f"切分配置：chunk_size=150, overlap=20，片段数={len(chunks)}")
    print("=" * 65)

    for case in TEST_CASES:
        result = retriever.search(case["question"], top_k=1)[0]
        score = result["score"]

        # 知识库内问题：检查最高分片段是否包含所有预期事实。
        relevant = (
            case["category"] == "in_scope"
            and is_relevant(result["chunk"], case["expected_terms"])
        )

        scores_by_case.append({
            "question": case["question"],
            "category": case["category"],
            "score": score,
            "relevant": relevant,
        })

        print(
            f"\n[{case['category']}] {case['question']}\n"
            f"最高相似度：{score:.4f} | "
            f"Top-1 包含全部预期事实：{'是' if relevant else '否/不适用'}"
        )

    print("\n" + "=" * 65)
    print("不同阈值下的判断结果")
    print("=" * 65)

    for threshold in thresholds:
        accepted_in_scope = 0
        rejected_in_scope = 0
        rejected_out_scope = 0
        accepted_out_scope = 0

        for item in scores_by_case:
            accepted = item["score"] >= threshold

            if item["category"] == "in_scope":
                if accepted:
                    accepted_in_scope += 1
                else:
                    rejected_in_scope += 1
            else:
                if accepted:
                    accepted_out_scope += 1
                else:
                    rejected_out_scope += 1

        print(f"\n阈值 = {threshold:.2f}")
        print(
            f"知识库内：接收 {accepted_in_scope}/8，"
            f"拒绝 {rejected_in_scope}/8"
        )
        print(
            f"知识库外：拒绝 {rejected_out_scope}/2，"
            f"误接收 {accepted_out_scope}/2"
        )


if __name__ == "__main__":
    main()
