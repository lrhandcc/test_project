
import sys
from pathlib import Path

# 将项目根目录加入 Python 模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from rag.text_splitter import read_text_file, split_text


def test_empty_text_returns_empty_list():
    assert split_text("   ") == []


def test_chunks_respect_size_limit():
    text = "第一句话。第二句话。第三句话。第四句话。"
    chunks = split_text(text, chunk_size=10, overlap=2)

    assert chunks
    assert all(len(chunk) <= 10 for chunk in chunks)


def test_invalid_parameters_raise_value_error():
    with pytest.raises(ValueError):
        split_text("测试", chunk_size=0)

    with pytest.raises(ValueError):
        split_text("测试", chunk_size=10, overlap=10)


def test_long_sentence_is_split():
    text = "A" * 25
    chunks = split_text(text, chunk_size=10, overlap=2)

    assert "".join(
        chunk if i == 0 else chunk[2:]
        for i, chunk in enumerate(chunks)
    ) == text


def test_read_chinese_text_file():
    file_path = PROJECT_ROOT / "rag" / "data" / "computer_network.txt"
    text = read_text_file(str(file_path))

    assert "TCP" in text
    assert "三次握手" in text
