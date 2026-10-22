"""问答日志与用户反馈的落盘（纯标准库，无第三方依赖）。

为什么单独一个模块
------------------
一个 RAG 系统「能不能持续变好」，取决于有没有数据闭环：

    埋点 → 反馈 → 回流成评测集

这里只负责前两步的存储，第三步（把 👎 标注成 eval_set 新条目）是人工/脚本的事，
但有了这份日志才做得起来。

存储格式：JSONL（一行一条，追加写，坏一行不影响其它行）
    logs/queries.jsonl   每次问答一行
    logs/feedback.jsonl  每次 👍/👎 一行，用 query_id 关联

设计取舍
--------
- **不用数据库**：几十万行以内 JSONL 足够，且可以直接 grep / 用 pandas 读；
- **不存全文上下文**：只存命中块的 id 与分数，避免日志里堆积敏感原文；
- **坏数据不抛异常**：埋点失败绝不能让用户看不到回答，所有写盘错误都吞掉并返回 None。
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# 反馈原因固定几个选项，便于聚合统计（前端下拉框也用这份）
FEEDBACK_REASONS = (
    "答非所问",
    "有事实错误",
    "信息不完整",
    "引用不对",
    "其他",
)

_LOG_LOCK = threading.Lock()  # 多线程（Streamlit）下保证不写坏行


def log_dir() -> Path:
    """日志目录，可用环境变量 RAG_LOG_DIR 覆盖。"""
    return Path(os.getenv("RAG_LOG_DIR", "logs"))


def _append_jsonl(path: Path, record: Dict[str, Any]) -> bool:
    """追加一行 JSON。失败返回 False，绝不抛异常（埋点不能影响主流程）。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _LOG_LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return True
    except OSError:
        return False


def new_query_id() -> str:
    """生成一个短且可读的 query_id：时间戳 + 6 位随机。"""
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def log_query(
    question: str,
    answer: str,
    sources: Optional[Iterable[Dict[str, Any]]] = None,
    retrieval_ms: float = 0.0,
    generation_ms: float = 0.0,
    model: str = "",
    error: str = "",
    query_id: Optional[str] = None,
    session_id: str = "",
) -> Optional[str]:
    """记录一次问答，返回 query_id（写盘失败时仍返回 id，便于前端展示）。

    Args:
        sources: 检索命中的块，只抽取 id/文档名/分数，不存原文
    """
    qid = query_id or new_query_id()
    hits = [
        {
            "chunk_id": s.get("chunk_id"),
            "document_name": s.get("document_name"),
            "score": s.get("score"),
        }
        for s in (sources or [])
    ]
    record = {
        "query_id": qid,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "answer_chars": len(answer or ""),
        "hits": hits,
        "hit_count": len(hits),
        "retrieval_ms": round(retrieval_ms, 1),
        "generation_ms": round(generation_ms, 1),
        "model": model,
        "error": error,
    }
    _append_jsonl(log_dir() / "queries.jsonl", record)
    return qid


def log_feedback(
    query_id: str,
    rating: str,
    reason: str = "",
    comment: str = "",
) -> bool:
    """记录用户反馈。rating 取 "up" / "down"。"""
    if rating not in ("up", "down"):
        raise ValueError(f"rating 只能是 up 或 down，当前为 {rating!r}")
    record = {
        "query_id": query_id,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "rating": rating,
        "reason": reason,
        "comment": comment,
    }
    return _append_jsonl(log_dir() / "feedback.jsonl", record)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL，跳过坏行。"""
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # 写了一半的行，忽略
    except OSError:
        return []
    return rows


def load_queries(limit: int = 50) -> List[Dict[str, Any]]:
    """最近 N 条问答记录（按时间正序返回，方便直接展示）。"""
    rows = _read_jsonl(log_dir() / "queries.jsonl")
    return rows[-limit:] if limit else rows


def load_feedback(limit: int = 200) -> List[Dict[str, Any]]:
    """最近 N 条反馈。"""
    rows = _read_jsonl(log_dir() / "feedback.jsonl")
    return rows[-limit:] if limit else rows


def feedback_summary() -> Dict[str, Any]:
    """反馈聚合：总数、好评率、按原因分布。"""
    rows = load_feedback(limit=0)
    up = sum(1 for r in rows if r.get("rating") == "up")
    down = sum(1 for r in rows if r.get("rating") == "down")
    reasons: Dict[str, int] = {}
    for row in rows:
        if row.get("rating") == "down" and row.get("reason"):
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    total = up + down
    return {
        "total": total,
        "up": up,
        "down": down,
        "up_rate": round(up / total, 4) if total else None,
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def bad_cases(limit: int = 20) -> List[Dict[str, Any]]:
    """把 👎 的反馈和原始问题拼起来，用于人工标注后回流进评测集。"""
    by_id = {row.get("query_id"): row for row in load_queries(limit=0)}
    cases: List[Dict[str, Any]] = []
    for fb in load_feedback(limit=0):
        if fb.get("rating") != "down":
            continue
        q = by_id.get(fb.get("query_id"), {})
        cases.append(
            {
                "query_id": fb.get("query_id"),
                "ts": fb.get("ts"),
                "reason": fb.get("reason"),
                "question": q.get("question", ""),
                "answer": q.get("answer", ""),
                "hits": q.get("hits", []),
            }
        )
    return cases[-limit:]


def export_bad_cases(path: Path, limit: int = 200) -> int:
    """导出 👎 案例为「待标注」评测集草案（question 已填，ground_truth 留空）。

    人工补完 ground_truth 后，把条目合并进 eval_set.json 即完成一次回流。
    """
    cases = bad_cases(limit=limit)
    payload = [
        {
            "question": c["question"],
            "ground_truth": "",
            "_source": f"feedback:{c['reason']}",
            "_query_id": c["query_id"],
        }
        for c in cases
        if c["question"]
    ]
    if not payload:
        return 0
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(payload)


def _main() -> int:
    """命令行入口：

        python qlog.py            # 看当前反馈概况
        python qlog.py --export   # 导出 👎 案例到 logs/bad_cases.json（待标注）
    """
    import sys

    if "--export" in sys.argv:
        out = log_dir() / "bad_cases.json"
        count = export_bad_cases(out)
        if count:
            print(f"已导出 {count} 条待标注案例 -> {out}")
            print("补完 ground_truth 后合并进 eval_set.json，即完成一次数据回流。")
        else:
            print("还没有 👎 反馈，暂无待标注案例。")
        return 0

    summary = feedback_summary()
    print(f"日志目录: {log_dir().resolve()}")
    print(f"问答总数: {len(load_queries(limit=0))}")
    print(f"反馈数量: {summary['total']}（👍 {summary['up']} / 👎 {summary['down']}）")
    if summary["up_rate"] is not None:
        print(f"好评率  : {summary['up_rate']:.1%}")
    if summary["reasons"]:
        print("差评原因:")
        for reason, count in summary["reasons"].items():
            print(f"  - {reason}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
