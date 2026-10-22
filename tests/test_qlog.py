"""问答日志与反馈的单元测试。

这套日志是「数据飞轮」的原料，所以它必须满足两个约束：
1. **写失败绝不抛异常**（埋点不能影响用户看答案）；
2. **坏行不能拖垮整个文件**（JSONL 的常见坑：进程被杀导致半行）。
"""

import json

import qlog


def test_log_query_returns_id_and_writes_line(tmp_dir, monkeypatch):
    monkeypatch.setenv("RAG_LOG_DIR", str(tmp_dir))
    qid = qlog.log_query(
        question="试用期多久？",
        answer="3 个月。",
        sources=[{"document_name": "手册.md", "chunk_id": 3, "score": 0.8}],
        retrieval_ms=12.3,
        generation_ms=456.7,
        model="deepseek-chat",
    )
    assert qid

    rows = qlog.load_queries()
    assert len(rows) == 1
    row = rows[0]
    assert row["query_id"] == qid
    assert row["question"] == "试用期多久？"
    assert row["hit_count"] == 1
    assert row["hits"][0]["document_name"] == "手册.md"
    # 只存块的 id 与分数，不把原文塞进日志
    assert "content" not in row["hits"][0]


def test_log_query_tolerates_missing_dir(tmp_dir, monkeypatch):
    """目录不存在也要能自动创建。"""
    target = tmp_dir / "deep" / "nested"
    monkeypatch.setenv("RAG_LOG_DIR", str(target))
    qlog.log_query(question="q", answer="a")
    assert (target / "queries.jsonl").is_file()


def test_log_query_never_raises_on_write_failure(tmp_dir, monkeypatch):
    """把日志目录指向一个文件（不可能创建目录），写入应静默失败。"""
    blocker = tmp_dir / "blocked"
    blocker.write_text("not a dir", encoding="utf-8")
    monkeypatch.setenv("RAG_LOG_DIR", str(blocker))
    qid = qlog.log_query(question="q", answer="a")
    assert qid  # 仍然返回 id，前端可以继续展示


def test_feedback_up_and_down(tmp_dir, monkeypatch):
    monkeypatch.setenv("RAG_LOG_DIR", str(tmp_dir))
    qlog.log_feedback("q1", "up")
    qlog.log_feedback("q2", "down", reason="有事实错误")

    rows = qlog.load_feedback()
    assert [r["rating"] for r in rows] == ["up", "down"]
    assert rows[1]["reason"] == "有事实错误"


def test_feedback_rejects_unknown_rating(tmp_dir, monkeypatch):
    monkeypatch.setenv("RAG_LOG_DIR", str(tmp_dir))
    try:
        qlog.log_feedback("q1", "maybe")
    except ValueError as exc:
        assert "rating" in str(exc)
    else:
        raise AssertionError("非法 rating 应该抛 ValueError")


def test_feedback_summary(tmp_dir, monkeypatch):
    monkeypatch.setenv("RAG_LOG_DIR", str(tmp_dir))
    qlog.log_feedback("q1", "up")
    qlog.log_feedback("q2", "up")
    qlog.log_feedback("q3", "down", reason="答非所问")
    qlog.log_feedback("q4", "down", reason="答非所问")

    summary = qlog.feedback_summary()
    assert summary["total"] == 4
    assert summary["up"] == 2
    assert summary["down"] == 2
    assert summary["up_rate"] == 0.5
    assert summary["reasons"] == {"答非所问": 2}


def test_feedback_summary_empty(tmp_dir, monkeypatch):
    monkeypatch.setenv("RAG_LOG_DIR", str(tmp_dir))
    assert qlog.feedback_summary()["total"] == 0
    assert qlog.feedback_summary()["up_rate"] is None


def test_bad_cases_join_feedback_with_question(tmp_dir, monkeypatch):
    monkeypatch.setenv("RAG_LOG_DIR", str(tmp_dir))
    qid = qlog.log_query(question="年假几天？", answer="不知道")
    qlog.log_feedback(qid, "down", reason="信息不完整")
    qlog.log_feedback(qlog.log_query(question="另一个问题", answer="ok"), "up")

    cases = qlog.bad_cases()
    assert len(cases) == 1
    assert cases[0]["question"] == "年假几天？"
    assert cases[0]["reason"] == "信息不完整"


def test_export_bad_cases_writes_draft(tmp_dir, monkeypatch):
    monkeypatch.setenv("RAG_LOG_DIR", str(tmp_dir))
    qid = qlog.log_query(question="试用期？", answer="不清楚")
    qlog.log_feedback(qid, "down", reason="答非所问")

    out = tmp_dir / "bad_cases.json"
    count = qlog.export_bad_cases(out)
    assert count == 1

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload[0]["question"] == "试用期？"
    assert payload[0]["ground_truth"] == ""  # 留给人工标注


def test_corrupted_line_is_skipped(tmp_dir, monkeypatch):
    monkeypatch.setenv("RAG_LOG_DIR", str(tmp_dir))
    path = tmp_dir / "queries.jsonl"
    path.write_text(
        '{"query_id": "ok1", "question": "q1"}\n{"broken": \n{"query_id": "ok2"}\n',
        encoding="utf-8",
    )
    rows = qlog.load_queries()
    assert [r["query_id"] for r in rows] == ["ok1", "ok2"]
