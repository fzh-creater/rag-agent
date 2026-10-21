"""评测集解析测试：重点覆盖 Windows 下常见的编码坑与字段缺失。"""

import json

import pytest

from core import decode_text_file, load_eval_set


def _write(path, text, encoding):
    path.write_bytes(text.encode(encoding))
    return path


class TestDecode:
    def test_utf8_with_bom(self, tmp_dir):
        # 记事本“UTF-8 带 BOM”保存时，BOM 应被吃掉，不能污染首个字段
        raw = "\ufeff中文内容".encode()
        assert decode_text_file(raw) == "中文内容"

    def test_gbk(self, tmp_dir):
        # 记事本“ANSI”另存即为 GBK
        assert decode_text_file("中文内容".encode("gbk")) == "中文内容"


class TestLoadEvalSet:
    def test_reads_utf8(self, tmp_dir):
        path = tmp_dir / "eval_set.json"
        path.write_text(
            json.dumps([{"question": "问题一", "ground_truth": "答案一"}], ensure_ascii=False),
            encoding="utf-8",
        )
        items = load_eval_set(path)
        assert items == [
            {"question": "问题一", "ground_truth": "答案一", "expect_refusal": False}
        ]

    def test_reads_gbk(self, tmp_dir):
        path = tmp_dir / "eval_set.json"
        payload = json.dumps([{"question": "问题一", "ground_truth": "答案一"}], ensure_ascii=False)
        _write(path, payload, "gbk")
        assert load_eval_set(path)[0]["question"] == "问题一"

    def test_expect_refusal_flag_is_passed_through(self, tmp_dir):
        path = tmp_dir / "eval_set.json"
        path.write_text(
            json.dumps([{"question": "库里没有的问题", "expect_refusal": True}], ensure_ascii=False),
            encoding="utf-8",
        )
        assert load_eval_set(path)[0]["expect_refusal"] is True

    def test_plain_string_rows_are_allowed(self, tmp_dir):
        path = tmp_dir / "eval_set.json"
        path.write_text(json.dumps(["只有问题没有答案"], ensure_ascii=False), encoding="utf-8")
        assert load_eval_set(path) == [
            {"question": "只有问题没有答案", "ground_truth": "", "expect_refusal": False}
        ]

    def test_dict_wrapper_is_supported(self, tmp_dir):
        path = tmp_dir / "eval_set.json"
        path.write_text(
            json.dumps({"questions": [{"question": "Q"}]}, ensure_ascii=False), encoding="utf-8"
        )
        assert load_eval_set(path)[0]["question"] == "Q"

    def test_missing_question_raises(self, tmp_dir):
        path = tmp_dir / "eval_set.json"
        path.write_text(json.dumps([{"ground_truth": "只有答案"}]), encoding="utf-8")
        with pytest.raises(ValueError, match="缺少 question"):
            load_eval_set(path)

    def test_empty_list_raises(self, tmp_dir):
        path = tmp_dir / "eval_set.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError):
            load_eval_set(path)

    def test_invalid_json_raises_with_hint(self, tmp_dir):
        path = tmp_dir / "eval_set.json"
        path.write_text("{不是 json", encoding="utf-8")
        with pytest.raises(ValueError, match="无法解析"):
            load_eval_set(path)

    def test_missing_file_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError, match="eval_set"):
            load_eval_set(tmp_dir / "nope.json")

    def test_bundled_eval_set_is_valid(self):
        """仓库自带的 eval_set.json 必须能解析、且至少有一条带标准答案。"""
        from pathlib import Path

        bundled = Path(__file__).resolve().parents[1] / "eval_set.json"
        items = load_eval_set(bundled)
        assert len(items) >= 5
        assert sum(1 for item in items if item["ground_truth"]) >= 5
        # 至少要有一条“知识库里没有”的负样本，用来测幻觉
        assert any("未提及" in item["ground_truth"] for item in items)
