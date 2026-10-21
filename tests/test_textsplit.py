"""文本切分与提示词拼接的单元测试。

这两个函数是检索质量的地基：切分错了，后面所有指标都无从谈起。
"""

from core import build_prompt, split_text


class TestSplitText:
    def test_empty_input_returns_empty(self):
        assert split_text("") == []
        assert split_text("   \n\n  ") == []

    def test_short_text_kept_as_single_chunk(self):
        text = "这是一段很短的文本。"
        assert split_text(text) == [text]

    def test_every_chunk_respects_max_size(self):
        text = "生成式人工智能正在改变内容生产的方式。" * 200  # 约 3800 字
        chunks = split_text(text, min_size=500, max_size=800)

        assert len(chunks) >= 4
        # 允许最后一个块不足 min_size，但不能超过 max_size
        assert all(len(c) <= 800 for c in chunks)

    def test_content_is_not_lost(self):
        sentences = [f"第{i}条：这是一句用来测试切分完整性的句子。" for i in range(80)]
        text = "".join(sentences)

        joined = "".join(split_text(text, min_size=200, max_size=400))
        for sentence in sentences:
            assert sentence in joined

    def test_long_single_sentence_is_hard_split(self):
        # 没有任何句末标点，只能硬切
        text = "字" * 2000
        chunks = split_text(text, min_size=500, max_size=800)

        assert all(len(c) <= 800 for c in chunks)
        assert "".join(chunks) == text

    def test_paragraph_breaks_are_preferred(self):
        # 每段约 840 字，两段合起来远超 max_size，必须切成多块
        text = ("第一段落内容。" * 60) + "\n\n" + ("第二段落内容。" * 60)
        chunks = split_text(text, min_size=300, max_size=600)
        assert len(chunks) >= 2

    def test_invalid_range_does_not_crash(self):
        # min_size > max_size 属于调用方错误，当前实现不校验，只保证不崩
        assert split_text("短文本", min_size=800, max_size=500)


class TestBuildPrompt:
    def test_contexts_are_numbered(self):
        prompt = build_prompt("试用期多久？", ["试用期 3 个月。", "薪资 80%。"])
        assert "[1] 试用期 3 个月。" in prompt
        assert "[2] 薪资 80%。" in prompt
        assert "【问题】\n试用期多久？" in prompt

    def test_empty_contexts_is_explicit(self):
        prompt = build_prompt("随便问问", [])
        assert "没有检索到任何相关资料" in prompt

    def test_contexts_are_truncated(self):
        contexts = [f"片段{i}" for i in range(10)]
        prompt = build_prompt("问题", contexts, max_chunks=3)
        assert "[3] 片段2" in prompt
        assert "片段3" not in prompt

    def test_long_context_is_kept_intact(self):
        # 不在 prompt 层截断单块内容（截断由检索与切块决定）
        long_text = "很长的上下文。" * 100
        assert long_text in build_prompt("问题", [long_text])
