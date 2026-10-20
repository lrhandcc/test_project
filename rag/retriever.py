
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.text_splitter import read_text_file, split_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENT_PATH = PROJECT_ROOT / "rag" / "data" / "computer_network.txt"


def text_features(text: str) -> set[str]:
    """提取英文完整词和中文连续文本中的相邻双字词。"""
    features = set()
    parts = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text.lower())

    for part in parts:
        if re.fullmatch(r"[a-z0-9]+", part):
            features.add(f"en:{part}")
        elif len(part) == 1:
            features.add(f"zh:{part}")
        else:
            for i in range(len(part) - 1):
                features.add(f"zh:{part[i:i + 2]}")

    return features


def calculate_overlap(question: str, chunk: str) -> float:
    """计算问题与文本片段的词语特征重合比例。"""
    question_features = text_features(question)
    chunk_features = text_features(chunk)

    if not question_features:
        return 0.0

    return len(question_features & chunk_features) / len(question_features)


class TextRetriever:
    """基于 TF-IDF 的轻量文本检索器。"""

    def __init__(self, chunks: list[str]):
        if not chunks:
            raise ValueError("知识库不能为空")

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 2),
        )
        self.chunk_vectors = self.vectorizer.fit_transform(chunks)

    def search(self, question: str, top_k: int = 2) -> list[dict]:
        """返回相似度、词语重合比例及对应文本片段。"""
        if not question.strip():
            raise ValueError("问题不能为空")

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        question_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(
            question_vector,
            self.chunk_vectors,
        )[0]

        ranked_indices = scores.argsort()[::-1][:top_k]

        return [
            {
                "chunk": self.chunks[index],
                "score": float(scores[index]),
                "overlap": calculate_overlap(
                    question,
                    self.chunks[index],
                ),
            }
            for index in ranked_indices
        ]


def build_retriever() -> TextRetriever:
    """读取知识文档、切分文本，并构建检索器。"""
    text = read_text_file(str(DOCUMENT_PATH))
    chunks = split_text(text, chunk_size=100, overlap=20)
    return TextRetriever(chunks)
