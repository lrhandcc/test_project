
import re


def read_text_file(file_path: str) -> str:
    """读取 UTF-8 文本文件，正确处理中文。"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[str]:
    """
    优先按句子边界切分文本。

    chunk_size：每个文本块允许的最大字符数。
    overlap：相邻文本块尽量保留的重叠字符数。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0:
        raise ValueError("overlap 不能小于 0")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    text = text.strip()
    if not text:
        return []

    # 先按中英文句末标点或换行切分，尽量保留完整句子。
    sentences = re.split(r"(?<=[。！？.!?])\s*|\n+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    current_sentences: list[str] = []

    def current_text() -> str:
        return "\n".join(current_sentences)

    def emit_current() -> None:
        """保存当前文本块。"""
        if current_sentences:
            chunks.append(current_text())

    def split_long_sentence(sentence: str) -> list[str]:
        """单个句子太长时，才使用字符窗口切分。"""
        pieces = []
        start = 0
        step = chunk_size - overlap

        while start < len(sentence):
            pieces.append(sentence[start:start + chunk_size])
            start += step

        return pieces

    for sentence in sentences:
        # 单句超过块大小时，先保存当前块，再单独切分长句。
        if len(sentence) > chunk_size:
            emit_current()
            current_sentences = []

            long_chunks = split_long_sentence(sentence)
            chunks.extend(long_chunks)
            continue

        candidate = "\n".join(current_sentences + [sentence])

        if len(candidate) <= chunk_size:
            current_sentences.append(sentence)
            continue

        # 当前块装不下新句子，先保存当前块。
        emit_current()

        # 尽量从旧块末尾保留完整句子作为重叠内容。
        overlap_sentences = []
        overlap_length = 0

        for old_sentence in reversed(current_sentences):
            extra_length = len(old_sentence) + (
                1 if overlap_sentences else 0
            )
            if overlap_length + extra_length > overlap:
                break

            overlap_sentences.insert(0, old_sentence)
            overlap_length += extra_length

        current_sentences = overlap_sentences + [sentence]

        # 重叠的完整句子可能导致超长，此时从新句子开始。
        if len(current_text()) > chunk_size:
            current_sentences = [sentence]

    emit_current()
    return chunks
