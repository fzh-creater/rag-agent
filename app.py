"""RAG 本地知识库 · Streamlit 前端（问答 + 评测看板）。

页面结构（两个 Tab，一个进程）：
    1. 💬 智能问答：上传文档 → 建库 → 提问，回答下方可展开引用来源
    2. 📊 评测报告：读取 reports/ 下最新一次 RAGAS 评测结果，含指标卡、
       逐条得分、负样本拒答结论与优化建议；也可以在页面里直接触发一次评测

设计要点
--------
- 嵌入模型用 ``st.cache_resource`` 只加载一次（约 470MB），切 Tab / 刷新都不重载。
- 评测结果来自 ``run_eval.py`` 落盘的 JSON，页面只做“读 + 展示”，
  避免在 Web 进程里跑 ragas（其内部会 nest_asyncio，和 Streamlit 的事件循环打架）。
- 页面内触发评测走子进程，输出实时回显；跑完 ``st.rerun()`` 自动刷新看板。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import qlog  # noqa: E402
from rag_engine import LLMUnavailableError, RAGEngine, default_corpus_dir  # noqa: E402

st.set_page_config(page_title="RAG 知识库 · 问答与评测", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
        [data-testid="stToolbar"] {visibility: hidden;}
        footer {visibility: hidden;}
        .small-note {font-size: 0.82rem; color: #666;}
    </style>
    """,
    unsafe_allow_html=True,
)

METRIC_LABELS = {
    "faithfulness": "忠实度 Faithfulness",
    "answer_relevancy": "回答相关性 AnswerRelevancy",
    "context_precision": "上下文精确率 ContextPrecision",
    "context_recall": "上下文召回率 ContextRecall",
    "answer_correctness": "回答正确性 AnswerCorrectness",
}


@st.cache_resource(show_spinner="正在加载嵌入模型与向量索引...")
def get_engine() -> RAGEngine:
    """整个 Streamlit 会话只加载一次嵌入模型，切 Tab / 刷新页面都不会重载。"""
    return RAGEngine()


def load_reports(report_dir: Path) -> list[dict]:
    """读取 reports/ 下全部评测 JSON，按时间倒序返回。

    每项含 path / mtime / metrics / data / label。之所以全量读：用户可能先跑
    一次 `--retrieval-only`（快），再跑一次完整评测，两者都要能翻回来对比。
    """
    if not report_dir.is_dir():
        return []

    reports: list[dict] = []
    for path in sorted(report_dir.glob("eval_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = data.get("metrics") or {}
        top_k = (data.get("engine") or {}).get("top_k", "?")
        stamp = (data.get("generated_at") or "")[:16].replace("T", " ")
        kind = f"完整 · {len(metrics)} 指标" if metrics else "仅检索层"
        reports.append(
            {
                "path": path,
                "data": data,
                "metrics": metrics,
                "label": f"{stamp} ｜ {kind} ｜ top_k={top_k}",
            }
        )
    return reports


def fmt(value, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "N/A"


# ================== 全局：侧边栏 ==================
engine = get_engine()
stats = engine.stats()

# 匿名会话 id：只用于把同一轮对话的日志串起来，不含用户身份
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]

st.sidebar.title("🛠️ 知识库管理")
st.sidebar.caption(f"当前：{stats['documents']} 篇文档 / {stats['chunks']} 个向量块")
if stats["document_names"]:
    with st.sidebar.expander("已入库文档", expanded=False):
        for name in stats["document_names"]:
            st.write("· " + name)

uploaded_files = st.sidebar.file_uploader(
    "上传文档（PDF / Word / TXT / Markdown，可多选）",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

col_build, col_rebuild = st.sidebar.columns(2)
if uploaded_files and col_build.button("构建 / 更新", type="primary"):
    with st.sidebar.status("正在解析文档并构建向量索引...", expanded=True) as status:
        added = engine.build(uploaded_files, replace=False)
        status.update(label=f"完成：新增 {added} 个向量块", state="complete")
    st.rerun()

if col_rebuild.button("全量重建"):
    corpus = default_corpus_dir()
    with st.sidebar.status(f"正在按 {corpus}/ 全量重建索引...", expanded=True) as status:
        engine.reset_index()
        added = engine.ingest_directory(corpus)
        status.update(label=f"重建完成：{added} 个向量块", state="complete")
    st.rerun()

if stats["chunks"] == 0:
    st.sidebar.info(
        "知识库还是空的。可以直接点「全量重建」用仓库自带的 `data/samples/` 示例文档试跑，"
        "或上传自己的文档后点「构建 / 更新」。"
    )

st.sidebar.divider()
st.sidebar.caption(f"检索阈值 = {engine.score_threshold}")

_llm = engine.llm_diagnosis()
st.sidebar.caption(f"生成模型 = {_llm['model']}（{_llm['provider_label']}）")
if not (_llm["server_up"] and _llm["model_present"]):
    one_line = (
        "未安装 Ollama"
        if _llm["provider"] == "ollama" and not _llm["installed"]
        else "Ollama 服务未启动"
        if _llm["provider"] == "ollama" and not _llm["server_up"]
        else f"缺少模型 {_llm['model']}"
        if _llm["provider"] == "ollama"
        else "云端接口不可用"
    )
    st.sidebar.error(f"回答生成不可用：{one_line}")
    with st.sidebar.expander("怎么修", expanded=False):
        for hint in _llm["hints"]:
            st.markdown(hint)
        st.code("python run_eval.py --check-llm", language="powershell")

tab_chat, tab_eval = st.tabs(["💬 智能问答", "📊 评测报告"])


def render_sources(sources: list, expanded: bool = False) -> None:
    """引用列表：文件名 + 块号 + 相似度，可展开看原文片段。"""
    if not sources:
        return
    with st.expander(f"引用来源（{len(sources)} 条）", expanded=expanded):
        for rank, src in enumerate(sources, start=1):
            st.markdown(
                f"**[{rank}]** `{src['document_name']}` · chunk {src['chunk_id']} · "
                f"相似度 {src['score']}"
            )
            snippet = (src.get("snippet") or "").strip()
            if snippet:
                with st.expander("查看原文", expanded=False):
                    st.code(snippet, language="text")


def render_feedback(query_id: str) -> None:
    """👍/👎 反馈入口。点了立刻落盘，用于后续把坏例回流成评测集。"""
    if not query_id:
        return

    col_up, col_down, _ = st.columns([1, 1, 4])
    if col_up.button("👍 有帮助", key=f"up_{query_id}"):
        qlog.log_feedback(query_id, "up")
        st.toast("已记录，谢谢反馈")

    if col_down.button("👎 有问题", key=f"down_{query_id}"):
        st.session_state["feedback_open"] = query_id

    if st.session_state.get("feedback_open") == query_id:
        reason = st.selectbox(
            "哪里有问题？",
            qlog.FEEDBACK_REASONS,
            key=f"reason_{query_id}",
        )
        if st.button("提交反馈", key=f"submit_{query_id}"):
            qlog.log_feedback(query_id, "down", reason=reason)
            st.session_state["feedback_open"] = None
            st.toast("已记录，这条会进入待标注队列")


# ================== Tab 1：问答 ==================
with tab_chat:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown("### 🤖 RAG 知识库智能问答")
    st.caption("LangChain + FAISS + paraphrase-multilingual-MiniLM-L12-v2 + OpenAI 兼容生成接口")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    prompt = st.chat_input("请输入您的问题...")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            started = time.perf_counter()
            try:
                with st.spinner("正在检索知识库..."):
                    sources = engine.retrieve(prompt)
                retrieval_ms = (time.perf_counter() - started) * 1000
                contexts = [s["content"] for s in sources]

                if not sources:
                    st.warning("没有检索到超过阈值的内容，下面的回答可能缺少依据。")

                # 流式输出：首字延迟从「整段生成时间」降到 1 秒内
                gen_started = time.perf_counter()
                answer = st.write_stream(engine.generate_stream(prompt, contexts))
                generation_ms = (time.perf_counter() - gen_started) * 1000
                answer = answer or ""
            except LLMUnavailableError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:  # noqa: BLE001 兜底，避免整页崩掉
                st.error(f"生成失败：{type(exc).__name__}: {exc}")
                st.stop()

            if sources:
                render_sources(sources)
            else:
                st.info("本次没有检索到超过阈值的内容，回答可能缺少依据。")

            # 埋点：这条记录是后续「bad case → 评测集」的原料
            query_id = qlog.log_query(
                question=prompt,
                answer=answer,
                sources=sources,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                model=engine.chat_model_name(),
                session_id=st.session_state.get("session_id", ""),
            )
            st.session_state["last_query_id"] = query_id

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )

    # 反馈区必须放在 `if prompt` 之外：点 👍/👎 会触发 rerun，而 rerun 时
    # chat_input 返回 None，若把按钮渲染在里面，按钮会在被点后立刻消失。
    last_query_id = st.session_state.get("last_query_id")
    if last_query_id:
        st.caption(
            f"刚才这轮问答是否有帮助？你的反馈会进入待标注队列"
            f"（记录 id `{last_query_id}`）"
        )
        render_feedback(last_query_id)

# ================== Tab 2：评测看板 ==================
with tab_eval:
    report_dir = BASE_DIR / "reports"
    reports = load_reports(report_dir)

    st.markdown("### 📊 自动评测报告")
    if not reports:
        st.info(
            "还没有评测结果。先在终端执行一次：\n\n```\npython run_eval.py\n```\n\n"
            "跑完回到这里刷新即可看到指标。"
        )
        st.stop()

    # 默认选“最新一份带指标的完整报告”；只有检索层结果时才退回最新一份
    default_index = next((i for i, r in enumerate(reports) if r["metrics"]), 0)
    picked = st.selectbox(
        "选择报告",
        range(len(reports)),
        index=default_index,
        format_func=lambda i: reports[i]["label"],
        help="同一份问题集可以跑多次对比；完整评测才会包含 RAGAS 指标。",
    )
    report = reports[picked]["data"]

    meta = report.get("engine", {})
    generated_at = report.get("generated_at", "")
    st.caption(
        f"生成时间：{generated_at} ｜ 裁判模型：{meta.get('judge_model', 'N/A')} ｜ "
        f"知识库：{meta.get('index', {}).get('documents', '?')} 篇 / "
        f"{meta.get('index', {}).get('chunks', '?')} 块 ｜ "
        f"top_k={meta.get('top_k', '?')}，阈值={meta.get('score_threshold', '?')}"
    )

    metrics = report.get("metrics") or {}
    adjusted = report.get("metrics_excluding_negative") or {}
    samples = report.get("samples") or []

    if not metrics:
        st.warning("这份报告只包含检索层结果（生成模型当时不可用）。")
    else:
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items(), strict=False):
            delta = None
            if name in adjusted and isinstance(value, (int, float)):
                delta = f"排除负样本 {fmt(adjusted[name], 3)}"
            col.metric(METRIC_LABELS.get(name, name), fmt(value, 3), delta=delta, delta_color="off")

        with st.expander("指标含义与调优方向", expanded=False):
            st.markdown(
                "| 指标 | 衡量什么 | 偏低时先改什么 |\n"
                "|---|---|---|\n"
                "| Faithfulness | 回答里的陈述能否被上下文支持 | 加严提示词、降温度、减少上下文条数 |\n"
                "| Answer Relevancy | 回答是否正面回应问题 | 要求先给结论、把问题再强调一次 |\n"
                "| Context Precision | Top-K 里有多少块真正相关 | 加 rerank、缩小 chunk、换嵌入模型 |\n"
                "| Context Recall | 标准答案所需信息是否被召回 | 提 top_k、放大 chunk、换中文模型 |"
            )

    # ---- 多报告对比 ----
    full_reports = [r for r in reports if r["metrics"]]
    if len(full_reports) >= 2:
        with st.expander(f"多份报告对比（{len(full_reports)} 份）", expanded=False):
            rows = []
            for r in full_reports:
                row = {"报告": r["label"]}
                for name, value in r["metrics"].items():
                    if isinstance(value, (int, float)):
                        row[METRIC_LABELS.get(name, name).split()[0]] = round(value, 3)
                rows.append(row)
            st.dataframe(rows, use_container_width=True, hide_index=True)
            try:
                import pandas as pd

                st.bar_chart(pd.DataFrame(rows).set_index("报告"))
            except Exception:  # noqa: BLE001 画图失败不影响数据表
                pass
            st.caption("同一份问题集的不同配置（如 top_k 3 / 5）会在这里并排对比。")

    # ---- 检索层 ----
    if samples:
        st.markdown("#### 检索层")
        rows = []
        for i, s in enumerate(samples, start=1):
            r = s.get("retrieval", {})
            rows.append(
                {
                    "#": i,
                    "命中块": r.get("hits", 0),
                    "Top1 相似度": r.get("top1"),
                    "平均相似度": round(r.get("mean", 0) or 0, 4),
                    "问题": s["question"],
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

        hits = sum(1 for s in samples if s.get("retrieval", {}).get("hits", 0) > 0)
        top1 = [s["retrieval"]["top1"] for s in samples if s.get("retrieval", {}).get("top1")]
        c1, c2, c3 = st.columns(3)
        c1.metric("命中率", f"{hits}/{len(samples)}")
        c2.metric("平均 Top1 相似度", fmt(sum(top1) / len(top1) if top1 else None, 3))
        c3.metric("平均召回块数", fmt(sum(r.get("hits", 0) for r in [s.get("retrieval", {}) for s in samples]) / len(samples), 1))

    # ---- 逐条得分 ----
    ragas_rows = report.get("ragas_rows") or []
    if ragas_rows:
        st.markdown("#### 逐条得分")
        table = []
        for i, (row, s) in enumerate(zip(ragas_rows, samples, strict=False), start=1):
            item = {"#": i, "问题": s["question"]}
            for name in METRIC_LABELS:
                if name in row:
                    item[METRIC_LABELS[name].split()[0]] = round(row[name], 4) if isinstance(row[name], (int, float)) else row[name]
            table.append(item)
        st.dataframe(table, use_container_width=True, hide_index=True)

    # ---- 负样本 ----
    negatives = [(i + 1, s) for i, s in enumerate(samples) if s.get("expect_refusal")]
    has_answers = any((s.get("answer") or "").strip() for s in samples)
    if negatives and not has_answers:
        st.info("这份报告是「仅检索层」模式，没有生成回答，无法判断是否拒答。")
    elif negatives:
        st.markdown("#### 负样本（幻觉探针）")
        keywords = ("无法回答", "没有", "未提及", "未涉及", "不包含", "缺少", "不提供", "不存在")
        for idx, s in negatives:
            refused = any(k in s.get("answer", "") for k in keywords)
            if refused:
                st.success(f"#{idx} 正确拒答，未编造 — {s['question']}")
                st.caption(s.get("answer", "")[:160])
            else:
                st.error(f"#{idx} 未拒答，可能编造 — {s['question']}")
                st.caption(s.get("answer", "")[:160])
        st.caption(
            "注意：RAGAS 会把“拒答”视为 noncommittal，answer_relevancy 直接判 0，"
            "所以上面指标卡额外给出了「排除负样本」的均值。"
        )

    # ---- 建议 ----
    if report.get("advice"):
        st.markdown("#### 诊断与优化建议")
        for line in report["advice"]:
            st.markdown("- " + line)

    # ---- 重新运行 ----
    st.divider()
    st.markdown("#### 重新生成报告")
    st.caption("在页面内运行会调用真实模型（产生少量费用），大约 1~3 分钟。")
    if st.button("▶ 运行评测（retrieval-only 更快）", type="secondary"):
        with st.status("正在运行 run_eval.py ...", expanded=True) as status:
            log_box = st.empty()
            lines: list[str] = []
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            proc = subprocess.Popen(  # noqa: S603 本地脚本
                [sys.executable, "run_eval.py", "--quiet"],
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.append(line.rstrip())
                log_box.code("\n".join(lines[-18:]), language="text")
            proc.wait()
            status.update(
                label=f"运行结束（exit={proc.returncode}），正在刷新看板...",
                state="complete" if proc.returncode == 0 else "error",
            )
        st.rerun()

    with st.expander("命令行方式（推荐，输出更完整）", expanded=False):
        st.code(
            "python run_eval.py                # 完整评测\n"
            "python run_eval.py --retrieval-only  # 只测检索层\n"
            "python run_eval.py --top-k 3         # 换检索条数\n"
            "python run_eval.py --full            # 追加 answer_correctness",
            language="powershell",
        )
        st.caption(
            f"最近一次评测文件：{sorted(report_dir.glob('eval_report_*.json'))[-1].name}"
            if list(report_dir.glob("eval_report_*.json"))
            else ""
        )

st.sidebar.divider()
st.sidebar.caption(f"页面刷新时间：{datetime.now():%H:%M:%S}")
