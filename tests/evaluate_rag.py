
"""
RAG 第一阶段评估：
复用正式接口中的检索器和过滤逻辑，不调用大模型。
运行位置：项目根目录
命令：.venv/bin/python tests/evaluate_rag.py
"""

import json
import sys
from pathlib import Path

# 保证从 tests 目录运行脚本时，也能导入项目中的模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routers.ai import get_retriever, retrieve_context


def main():
    cases_file = PROJECT_ROOT / "tests" / "rag_eval_cases.json"

    if not cases_file.exists():
        raise FileNotFoundError(f"找不到评估题库：{cases_file}")

    with cases_file.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    # 获取正式接口使用的检索器，而不是 qa.py 中的另一套配置
    retriever = get_retriever()

    answerable_total = 0
    answerable_passed = 0

    insufficient_total = 0
    insufficient_reached_model = 0

    out_of_domain_total = 0
    out_of_domain_passed = 0

    print("=" * 72)
    print("RAG 检索与证据过滤评估")
    print(f"评估题目数：{len(cases)}")
    print("=" * 72)

    for case in cases:
        question = case["question"]
        category = case["category"]

        # 查看正式检索器的前两个结果，用于分析检索质量
        hits = retriever.search(question, top_k=2)

        # 调用正式接口使用的过滤函数，判断问题是否会获得知识库上下文
        context = retrieve_context(question)
        has_context = bool(context)

        if category == "answerable":
            answerable_total += 1

            if has_context:
                answerable_passed += 1
                assessment = "符合预期：有上下文，可尝试回答"
            else:
                assessment = "需关注：知识库有答案，但检索未通过"

        elif category == "insufficient":
            insufficient_total += 1

            if has_context:
                insufficient_reached_model += 1
                assessment = "已进入模型：下一步检查是否识别证据不足"
            else:
                assessment = "检索层已拦截：安全，但尚未测试模型判断"

        elif category == "out_of_domain":
            out_of_domain_total += 1

            if not has_context:
                out_of_domain_passed += 1
                assessment = "符合预期：未提供知识库上下文"
            else:
                assessment = "需关注：无关问题通过了检索过滤"

        else:
            assessment = f"未知测试类别：{category}"

        print(f"\n[{case['id']}] {category}")
        print(f"问题：{question}")
        print(f"正式检索是否提供上下文：{'是' if has_context else '否'}")
        print(f"评估：{assessment}")

        if hits:
            print("原始检索结果（最多 2 条）：")
            for index, hit in enumerate(hits, start=1):
                score = float(hit.get("score") or 0)
                overlap = float(hit.get("overlap") or 0)
                chunk = str(hit.get("chunk") or "").replace("\n", " ")
                print(
                    f"  {index}. score={score:.3f}, "
                    f"overlap={overlap:.3f}"
                )
                print(f"     片段：{chunk[:160]}")
        else:
            print("原始检索结果：没有找到片段")

    print("\n" + "=" * 72)
    print("检索阶段汇总")
    print("=" * 72)
    print(
        f"明确可回答的问题：{answerable_passed}/{answerable_total} "
        "获得上下文"
    )
    print(
        f"证据不足的问题：{insufficient_reached_model}/"
        f"{insufficient_total} 进入模型判断"
    )
    print(
        f"知识库无关的问题：{out_of_domain_passed}/"
        f"{out_of_domain_total} 被检索层拦截"
    )

    print(
        "\n注意：这里只评估检索与过滤，不代表模型已经回答正确。"
        "证据不足的问题被检索层直接拦截，也不能证明模型的"
        "证据充分性判断能力。"
    )


if __name__ == "__main__":
    main()
