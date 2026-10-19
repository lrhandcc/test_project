
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.text_splitter import read_text_file, split_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENT_PATH = PROJECT_ROOT / "rag" / "data" / "computer_network.txt"


class TextRetriever:
    """基于 TF-IDF 的轻量文本检索器。"""

    def __init__(self, chunks: list[str]):
        if not chunks:
            raise ValueError("知识库不能为空")

        self.chunks = chunks

        # 根据所有文本块建立词语特征空间。
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 2),
        )

        # 每一行对应一个文本块，每一列对应一个字符特征。
        self.chunk_vectors = self.vectorizer.fit_transform(chunks)

    def search(self, question: str, top_k: int = 2) -> list[dict]:
        """返回与问题最相关的文本块及其相似度。"""
        if not question.strip():
            raise ValueError("问题不能为空")

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        question_vector = self.vectorizer.transform([question])

        # 计算问题向量与每个文本块向量之间的余弦相似度。
        scores = cosine_similarity(
            question_vector,
            self.chunk_vectors,
        )[0]

        # 按相似度从高到低排列，取前 top_k 个。
        ranked_indices = scores.argsort()[::-1][:top_k]

        return [
            {
                "chunk": self.chunks[index],
                "score": float(scores[index]),
            }
            for index in ranked_indices
        ]


def build_retriever() -> TextRetriever:
    """读取知识文档、切分文本，并构建检索器。"""
    text = read_text_file(str(DOCUMENT_PATH))
    chunks = split_text(text, chunk_size=100, overlap=20)
    return TextRetriever(chunks)
