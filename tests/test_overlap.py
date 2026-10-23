"""切块重叠（overlap）的单元测试。

背景：事实经常跨在块边界上。重叠能让边界两侧的块都包含完整语义，
但会让每块变长、相邻块重复——所以必须同时验证「有帮助」和「不越界」。
"""

from core import _overlap_tail, split_text


def _long_text(paragraphs: int = 6, sentences: int = 8) -> str:
    """构造多段落长文本，每段约 8 句 × 12 字 ≈ 96 字。"""
    blocks = []
    for p in range(paragraphs):
        blocks.append(
            "".join(f"第{p}段第{i}句用来测试切块行为。" for i in range(sentences))
        )
    return "\n\n".join(blocks)


class TestOverlapTail:
    def test_zero_returns_empty(self):
        assert _overlap_tail("随便一段文字。", 0) == ""

    def test_starts_at_sentence_boundary(self):
        text = "第一句话。第二句话。第三句话。"
        tail = _overlap_tail(text, 8)
        # 截出来应当从某个句号之后开始，而不是半句
        assert tail in text
        assert not tail.startswith("话。")

    def test_short_text_returns_all(self):
        assert _overlap_tail("短。", 100) == "短。"


class TestSplitWithOverlap:
    def test_default_keeps_backward_compatibility(self):
        """不传 overlap 时行为必须和以前完全一致。"""
        text = _long_text()
        assert split_text(text) == split_text(text, overlap=0)

    def test_overlap_zero_matches_explicit_zero(self):
        text = _long_text()
        assert split_text(text, 200, 400) == split_text(text, 200, 400, 0)

    def test_overlap_makes_chunks_share_text(self):
        text = _long_text()
        chunks = split_text(text, 200, 400, overlap=100)

        assert len(chunks) >= 2
        # 第二块开头应当出现第一块末尾的内容（共享片段）
        first_tail = chunks[0][-30:]
        assert first_tail in chunks[1]

    def test_chunks_stay_within_max_plus_overlap(self):
        text = _long_text()
        overlap = 120
        chunks = split_text(text, 200, 400, overlap=overlap)
        assert all(len(c) <= 400 + overlap for c in chunks), [len(c) for c in chunks]

    def test_content_is_not_lost(self):
        text = _long_text()
        joined = "".join(split_text(text, 200, 400, overlap=100))
        # 每段的第一句都应当还能找到（重叠不会丢内容）
        for p in range(6):
            assert f"第{p}段第0句" in joined

    def test_single_chunk_unchanged_by_overlap(self):
        text = "只有一小段文字。"
        assert split_text(text, 500, 800, overlap=100) == [text]

    def test_overlap_larger_than_chunk_does_not_crash(self):
        text = _long_text()
        chunks = split_text(text, 100, 200, overlap=500)
        assert chunks and all(isinstance(c, str) and c for c in chunks)
