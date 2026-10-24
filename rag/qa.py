
from rag.retriever import TextRetriever, DOCUMENT_PATH
from rag.text_splitter import read_text_file, split_text


# 必须与阈值实验使用相同的切分配置。
CHUNK_SIZE = 150
OVERLAP = 20
SIMILARITY_THRESHOLD = 0.20

# 模块加载时构建一次检索器，后续请求重复使用。
text = read_text_file(str(DOCUMENT_PATH))
chunks = split_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
retriever = TextRetriever(chunks)


def retrieve_context(question: str) -> tuple[str | None, float]:
    """检索知识库；相关性分数不足时，不返回知识片段。"""
    results = retriever.search(question, top_k=1)

    if not results:
        return None, 0.0

    best_result = results[0]
    score = best_result["score"]

    if score < SIMILARITY_THRESHOLD:
        return None, score

    return best_result["chunk"], score
