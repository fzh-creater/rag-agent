"""RAG-Agent 自动评测脚本（适配 ragas 0.4.3）。

为什么之前的版本会报 TypeError
------------------------------
ragas 0.4.3 里同时存在两套互不兼容的指标 API：

  旧式（legacy）  ragas.metrics.Faithfulness / AnswerRelevancy ...
                  它们是 ragas.metrics.base.Metric 的子类，
                  只有它们能塞进 ragas.evaluate() 的 metrics 参数。
  新式（collections）ragas.metrics.collections.Faithfulness ...
                  继承自 collections.base.BaseMetric，只支持
                  metric.ascore(...) 或 @experiment 用法。

旧脚本从 collections 里 import 了 Faithfulness，又传给 evaluate()，于是
ragas/evaluation.py:132 的
    if isinstance(metrics, list) and any(not isinstance(m, Metric) for m in metrics)
判断失败，抛出
    TypeError: All metrics must be initialised metric objects...

本脚本的修法：用旧式指标类配合 evaluate()（ragas 0.4.3 里 evaluate() 仍然支持
llm_factory() 生成的 InstructorLLM，见 ragas/metrics/base.py:161 的说明）。

另外修掉了旧脚本的 3 个“分数好看但毫无意义”的问题：
1. 旧脚本把 answer 直接赋值为拼接后的 contexts —— 等于用资料回答资料，
   faithfulness 必然接近 1.0，指标彻底失效。本脚本改为调用 RAGEngine.generate()
   真的让大模型根据上下文生成回答。
2. 旧脚本的 4 个问题（试用期、华东智慧仓储……）和实际知识库（一篇生成式 AI 论文）
   完全无关，实测检索全部 0 命中。本脚本把问题集外置为 eval_set.json，
   并在报告里显式给出“空召回”告警。
3. 旧脚本用 qwen2.5:7b 当 embedding 模型（它是对话模型，Ollama 的 /v1/embeddings
   根本不存在该模型），本脚本直接复用 RAGEngine 已经加载的
   paraphrase-multilingual-MiniLM-L12-v2，零额外显存、和检索端语义一致。

用法
----
    python run_eval.py                  # 完整评测（检索 + ragas 生成指标）
    python run_eval.py --retrieval-only # 只测检索层（不需要生成模型）
    python run_eval.py --full           # 额外评测 answer_correctness
    python run_eval.py --rebuild        # 先按 uploads/ 重建索引再评测
    python run_eval.py --doctor         # 环境自检（版本/生成模型/索引/问题集）
    python run_eval.py --check-llm      # 只测生成模型连通性（最快定位 401/404）
    python run_eval.py --start-ollama   # 顺手后台拉起本地 Ollama

生成模型既可以走本地 Ollama，也可以走任意 OpenAI 兼容云端 API：
    $env:RAG_LLM_API_KEY="sk-..."
    $env:RAG_LLM_BASE_URL="https://api.deepseek.com/v1"
    $env:RAG_CHAT_MODEL="deepseek-chat"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from datasets import Dataset  # noqa: E402

# 这段补丁必须早于任何 `import ragas`：ragas 0.4.3 会 import
# langchain_community.chat_models.vertexai，而 langchain-community 0.4 已删除该模块。
import ragas_compat  # noqa: F401,E402  (import 即生效)
from core import (  # noqa: E402
    LLMUnavailableError,
    diagnose_llm,
    load_eval_set,
    resolve_llm_config,
    start_ollama_server,
)
from rag_engine import RAGEngine, default_corpus_dir  # noqa: E402

# 旧式指标类在 0.4.3 被标记为 deprecated（官方建议迁到 collections + @experiment），
# 但 evaluate() 至今只接受旧式 Metric 对象，所以这里必须用它，顺手把告警静音。
# 等 ragas 1.0 提供 evaluate() 的新式替代后再迁移，见 README 的“后续升级”一节。
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"Importing .* from 'ragas\.metrics' is deprecated.*",
)

# 指标中文名，用于报告排版
METRIC_LABELS = {
    "faithfulness": "忠实度 Faithfulness",
    "answer_relevancy": "回答相关性 AnswerRelevancy",
    "context_precision": "上下文精确率 ContextPrecision",
    "context_recall": "上下文召回率 ContextRecall",
    "answer_correctness": "回答正确性 AnswerCorrectness",
}

GRADE_THRESHOLDS = ((0.8, "优秀"), (0.6, "良好"), (0.4, "待改进"), (0.0, "严重问题"))

# 判定“模型是否老实说不知道”的关键词（用于负样本/幻觉探针）
REFUSAL_KEYWORDS = (
    "无法回答",
    "没有",
    "未提及",
    "未涉及",
    "不包含",
    "缺少",
    "不提供",
    "不存在",
)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def grade(score: float) -> str:
    for threshold, label in GRADE_THRESHOLDS:
        if score >= threshold:
            return label
    return "未知"


def fmt_score(value: Any) -> str:
    if value is None:
        return "  N/A "
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "  N/A "
    if math.isnan(number):
        return "  N/A "
    return f"{number:.4f}"


def safe_mean(values: List[Any]) -> Optional[float]:
    numbers = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return sum(numbers) / len(numbers) if numbers else None


# ---------------------------------------------------------------------------
# 第一步：跑 RAG 流水线，产出 question / answer / contexts
# ---------------------------------------------------------------------------

def collect_samples(
    engine: RAGEngine,
    items: List[Dict[str, str]],
    top_k: int,
    quiet: bool = False,
    with_generation: bool = True,
) -> List[Dict[str, Any]]:
    """对每个问题执行“检索（+ 生成）”，得到评测所需样本。

    with_generation=False 时跳过生成：既省时间，也避免在已知模型不可用的
    情况下逐条等待连接超时。
    """
    samples: List[Dict[str, Any]] = []
    total = len(items)
    for idx, item in enumerate(items, start=1):
        question = item["question"]
        if not quiet:
            stage = "检索并生成" if with_generation else "检索"
            print(f"  [{idx}/{total}] {stage}: {question[:40]}")

        raw_scores = engine.top_scores(question, k=top_k)  # 不过阈值，便于诊断
        results = engine.search(question, top_k=top_k)
        contexts = [r["content"] for r in results]

        started = datetime.now()
        answer, error = "", ""
        if with_generation:
            try:
                answer = engine.generate(question, contexts)
            except LLMUnavailableError as exc:
                answer, error = "", str(exc)
        elapsed = (datetime.now() - started).total_seconds()

        samples.append(
            {
                "question": question,
                "ground_truth": item["ground_truth"],
                "expect_refusal": bool(item.get("expect_refusal", False)),
                "answer": answer,
                "contexts": contexts,
                "error": error,
                "latency_s": round(elapsed, 2),
                "retrieval": {
                    "hits": len(contexts),
                    "top1": raw_scores[0] if raw_scores else None,
                    "mean": safe_mean(raw_scores),
                    "raw_scores": raw_scores,
                },
            }
        )
    return samples


# ---------------------------------------------------------------------------
# 第二步：构造 ragas 指标并执行 evaluate()
# ---------------------------------------------------------------------------

def build_llm_judge(cfg: Any):
    """构造 ragas 的裁判模型（InstructorLLM）。

    关键点：必须传 AsyncOpenAI。ragas 的 InstructorLLM 在 agenerate() 里会检查
    客户端是否异步（ragas/llms/base.py:1092），传同步 OpenAI 会在评分时报
    “Cannot use agenerate() with a synchronous client”。
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    client = AsyncOpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key or "not-needed",
        timeout=cfg.timeout,
        max_retries=2,
    )
    # temperature / max_tokens 通过 kwargs 传给 InstructorModelArgs
    return llm_factory(cfg.model, client=client, temperature=0.01, max_tokens=2048)


def build_embeddings(engine: RAGEngine):
    """复用 RAGEngine 已加载的 HuggingFace 嵌入模型，避免再加载一份 470MB 权重。"""
    from ragas.embeddings.base import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(engine.embeddings)


def build_metrics(llm: Any, embeddings: Any, with_reference: bool, full: bool) -> List[Any]:
    """按需装配指标。

    - faithfulness / answer_relevancy：不需要标准答案，任何数据集都能算；
    - context_precision / context_recall：必须提供 ground_truth；
    - answer_correctness：需要 ground_truth + embeddings，且要额外 1~2 次 LLM 调用。
    """
    from ragas.metrics import AnswerRelevancy, Faithfulness

    metrics: List[Any] = [Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=embeddings)]
    if with_reference:
        from ragas.metrics import ContextPrecision, ContextRecall

        metrics += [ContextPrecision(llm=llm), ContextRecall(llm=llm)]
        if full:
            from ragas.metrics import AnswerCorrectness

            metrics.append(AnswerCorrectness(llm=llm, embeddings=embeddings))
    return metrics


def run_ragas(
    samples: List[Dict[str, Any]],
    engine: RAGEngine,
    judge_cfg: Any,
    full: bool,
    quiet: bool,
):
    """调用 ragas.evaluate()，返回 (pandas.DataFrame, 指标名列表, 是否有标准答案)。"""
    from ragas import evaluate
    from ragas.run_config import RunConfig

    has_reference = all(s["ground_truth"] for s in samples)
    payload: Dict[str, List[Any]] = {
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
    }
    if has_reference:
        payload["ground_truth"] = [s["ground_truth"] for s in samples]

    dataset = Dataset.from_dict(payload)

    llm = build_llm_judge(judge_cfg)
    embeddings = build_embeddings(engine)
    metrics = build_metrics(llm, embeddings, with_reference=has_reference, full=full)
    metric_names = [m.name for m in metrics]

    # 本地 7B 模型单次调用较慢：放宽超时、控制并发，避免 429/超时批量失败
    run_config = RunConfig(timeout=300, max_workers=2, max_retries=2, max_wait=90)

    import ragas

    if not quiet:
        print(f"  ragas {ragas.__version__} | 指标: {', '.join(metric_names)}")
        print(f"  裁判模型: {judge_cfg.describe()}")
        print("  正在评分（本地模型通常需要数分钟）...")

    result = evaluate(
        dataset,
        metrics=metrics,
        run_config=run_config,
        show_progress=not quiet,
        raise_exceptions=False,  # 单条失败不中断整轮，失败样本会变成 NaN
    )
    return result.to_pandas(), metric_names, has_reference


# ---------------------------------------------------------------------------
# 第三步：生成报告
# ---------------------------------------------------------------------------

def print_retrieval_section(samples: List[Dict[str, Any]], engine: RAGEngine) -> None:
    print("\n[2] 检索层指标（阈值 %.2f）" % engine.score_threshold)
    header = f"  {'#':<3}{'命中块':<7}{'Top1相似度':<12}{'平均相似度':<12}{'生成(s)':<9}问题"
    print(header)
    print("  " + "-" * 96)
    for idx, s in enumerate(samples, start=1):
        r = s["retrieval"]
        print(
            f"  {idx:<3}{r['hits']:<7}{fmt_score(r['top1']):<12}"
            f"{fmt_score(r['mean']):<12}{s['latency_s']:<9}{s['question'][:40]}"
        )
    empty = [i for i, s in enumerate(samples, start=1) if s["retrieval"]["hits"] == 0]
    if empty:
        print(f"  ! 空召回样本: {empty} —— 检索阈值过高或知识库确实没有相关内容")
        print(f"    当前阈值 {engine.score_threshold}，Top1 分数普遍为 "
              f"{fmt_score(safe_mean([s['retrieval']['top1'] for s in samples]))}，建议调低 min_score")


def print_metric_section(df: Any, metric_names: List[str]) -> Dict[str, Optional[float]]:
    print("\n[3] RAGAS 指标汇总")
    print(f"  {'指标':<34}{'均值':<10}{'最小':<10}{'最大':<10}{'有效样本':<9}评级")
    print("  " + "-" * 96)
    means: Dict[str, Optional[float]] = {}
    for name in metric_names:
        if name not in df.columns:
            print(f"  {METRIC_LABELS.get(name, name):<34} 该指标未产生结果")
            means[name] = None
            continue
        series = df[name].dropna()
        mean_value = float(series.mean()) if len(series) else None
        means[name] = mean_value
        label = METRIC_LABELS.get(name, name)
        if mean_value is None:
            print(f"  {label:<34}{'N/A':<10}{'':<10}{'':<10}{0:<9}全部失败")
            continue
        print(
            f"  {label:<34}{mean_value:<10.4f}{float(series.min()):<10.4f}"
            f"{float(series.max()):<10.4f}{len(series):<9}{grade(mean_value)}"
        )
    return means


def print_sample_section(samples: List[Dict[str, Any]], df: Any, metric_names: List[str]) -> None:
    print("\n[4] 逐条得分")
    cols = [name for name in metric_names if name in df.columns]
    short = {
        "faithfulness": "Faith",
        "answer_relevancy": "Relev",
        "context_precision": "CtxP",
        "context_recall": "CtxR",
        "answer_correctness": "Corr",
    }
    header = "  " + f"{'#':<3}" + "".join(f"{short.get(c, c[:6]):<8}" for c in cols) + "问题"
    print(header)
    print("  " + "-" * 96)
    for i in range(len(samples)):
        cells = "".join(f"{fmt_score(df.iloc[i].get(c)):<8}" for c in cols)
        print("  " + f"{i + 1:<3}" + cells + samples[i]["question"][:36])


def print_refusal_section(
    samples: List[Dict[str, Any]],
    df: Any,
    metric_names: List[str],
) -> Dict[str, Optional[float]]:
    """负样本（幻觉探针）单独统计，并给出“排除负样本后”的均值。

    为什么必须单独看：RAGAS 把“拒答”视为 noncommittal，answer_relevancy 直接判 0；
    context_precision 也会因为“上下文里确实没有答案”而判 0。也就是说**答对了反而扣分**。
    混在总均值里会把整份报告拉低好几个点，且掩盖真实问题。
    """
    negatives = [i for i, s in enumerate(samples) if s.get("expect_refusal")]
    print("\n[3b] 负样本（幻觉探针）")
    if not negatives:
        print("  本次评测集未包含负样本；建议加 1~2 条“知识库里没有”的问题来测幻觉。")
        return {}

    for i in negatives:
        answer = samples[i].get("answer", "")
        refused = any(word in answer for word in REFUSAL_KEYWORDS)
        verdict = "✅ 正确拒答（未编造）" if refused else "❌ 未拒答（可能编造）"
        print(f"  #{i + 1} {verdict}：{samples[i]['question'][:40]}")
        print(f"      回答片段：{answer[:60].replace(chr(10), ' ') or '（空）'}")

    answerable = [i for i in range(len(samples)) if i not in negatives]
    adjusted: Dict[str, Optional[float]] = {}
    for name in metric_names:
        if name not in df.columns or not answerable:
            continue
        values = df.iloc[answerable][name].dropna()
        if len(values):
            adjusted[name] = float(values.mean())

    if adjusted:
        print("  排除负样本后的均值（更能反映真实问答质量）：")
        print(f"    {'指标':<34}{'全体均值':<12}{'排除负样本':<12}")
        for name, value in adjusted.items():
            overall = safe_mean([df.iloc[i].get(name) for i in range(len(samples))])
            print(f"    {METRIC_LABELS.get(name, name):<34}{fmt_score(overall):<12}{fmt_score(value):<12}")
    return adjusted


def print_diagnosis(
    means: Dict[str, Optional[float]],
    samples: List[Dict[str, Any]],
    engine: RAGEngine,
) -> List[str]:
    """根据指标给可执行的优化建议。返回建议列表（同时用于 JSON 报告）。"""
    advice: List[str] = []

    if any(s["retrieval"]["hits"] == 0 for s in samples):
        advice.append(
            "【检索层】存在空召回：先看 Top1 原始相似度，如果普遍 < 阈值，把 "
            "RAGEngine(score_threshold=...) 调到 Top1 分数之下（本项目实测 0.2~0.4）；"
            "如果知识库里确实没有该主题，就换问题，而不是调指标。"
        )

    faithfulness = means.get("faithfulness")
    if faithfulness is not None and faithfulness < 0.7:
        advice.append(
            "【忠实度偏低】回答里出现了资料未支持的内容。可：①在 SYSTEM_PROMPT 里再次强调"
            "“只依据资料作答”；②降低生成温度到 0.1 以下；③减少 MAX_CONTEXT_CHUNKS，"
            "避免模型在长上下文里“脑补”；④检查检索命中块是否真的包含答案。"
        )

    relevancy = means.get("answer_relevancy")
    if relevancy is not None and relevancy < 0.7:
        advice.append(
            "【相关性偏低】回答没有正面回应问题。可：①让提示词要求“先给结论再展开”；"
            "②把用户问题原样再强调一次；③检查是否被无关上下文带偏（结合 CtxP 一起看）。"
        )

    precision = means.get("context_precision")
    if precision is not None and precision < 0.7:
        advice.append(
            "【上下文精确率偏低】Top-K 里混入了不相关片段。可：①降低 top_k；②调高"
            " score_threshold 或加入 MMR 去冗余；③缩小 chunk（如 300~500 字）让块更聚焦；"
            "④加一层 rerank（bge-reranker）再截断。"
        )

    recall = means.get("context_recall")
    if recall is not None and recall < 0.7:
        advice.append(
            "【上下文召回率偏低】标准答案需要的信息没被检索到。可：①提高 top_k；"
            "②把 chunk 调大（800~1000 字）避免答案被切断；③优化切分策略（按标题层级切）；"
            "④换更强的嵌入模型（如 bge-m3 / text2vec-large-chinese）。"
        )

    correctness = means.get("answer_correctness")
    if correctness is not None and correctness < 0.6:
        advice.append(
            "【回答正确性偏低】与标准答案事实不一致。优先处理召回（CtxR）和忠实度（Faith），"
            "这两项修好后该项通常会自然回升。"
        )

    if not advice:
        advice.append("各项指标均达标，可继续扩充问题集做回归测试，或尝试更难的负样本问题。")

    print("\n[5] 诊断与优化建议")
    for line in advice:
        print("  - " + line)
    return advice


def save_reports(
    out_dir: Path,
    samples: List[Dict[str, Any]],
    df: Any,
    metric_names: List[str],
    means: Dict[str, Optional[float]],
    advice: List[str],
    engine: RAGEngine,
    judge_model: str,
    extra: Dict[str, Any],
) -> List[Path]:
    """把结果落盘为 JSON（完整）与 CSV（逐条，方便写进论文附录）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "engine": {
            "embedding_model": str(engine.model_name),
            "judge_model": judge_model,
            "score_threshold": engine.score_threshold,
            "top_k": engine.top_k,
            "index": engine.stats(),
        },
        "metrics": means,
        "advice": advice,
        "samples": samples,
        "ragas_rows": df.to_dict(orient="records") if df is not None else [],
        **extra,
    }
    json_path = out_dir / f"eval_report_{stamp}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    csv_path = out_dir / f"eval_scores_{stamp}.csv"
    if df is not None:
        # 只导出问题级得分表，避免把长上下文塞进 CSV
        table = df.copy()
        cols = [c for c in ["question", "answer", *metric_names] if c in table.columns]
        table[cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
    return [json_path, csv_path]


def show_reports(count: int, out_dir: Path) -> int:
    """只读查看最近 N 份历史报告：不加载嵌入模型、不调用任何接口、不产生费用。

    对应命令 ``python run_eval.py --show``（或 ``--show 3`` 看最近三份对比）。
    输出与完整评测的 [1][3][3b][5] 段保持一致，方便直接贴到文档里。

    选报告的策略：只取 1 份时优先给“带 RAGAS 指标”的完整报告——
    否则刚跑过一次 ``--retrieval-only`` 就会看到一份没有分数的报告。
    """
    loaded: list[tuple[Path, dict]] = []
    for path in out_dir.glob("eval_report_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        loaded.append((path, data))

    if not loaded:
        print(f"[提示] {out_dir} 下还没有任何报告。先跑一次：python run_eval.py")
        return 1

    loaded.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
    if count <= 1:
        with_metrics = [item for item in loaded if item[1].get("metrics")]
        selected = with_metrics[:1] or loaded[:1]
    else:
        selected = loaded[:count]

    for idx, (path, data) in enumerate(selected, start=1):
        engine_info = data.get("engine") or {}
        samples = data.get("samples") or []
        metrics = data.get("metrics") or {}
        adjusted = data.get("metrics_excluding_negative") or {}
        index_info = engine_info.get("index") or {}

        print("=" * 92)
        print(f"[{idx}] {path.name}")
        print(f"  生成时间 : {data.get('generated_at', '?')}")
        print(f"  模式     : {data.get('mode', '?')}    裁判模型: {engine_info.get('judge_model', '?')}")
        print(
            f"  知识库   : {index_info.get('documents', '?')} 篇 / {index_info.get('chunks', '?')} 块"
            f"    top_k={engine_info.get('top_k', '?')}    阈值={engine_info.get('score_threshold', '?')}"
        )

        if metrics:
            print(f"\n  {'指标':<28}{'全体均值':<12}{'排除负样本':<12}")
            print("  " + "-" * 62)
            for name, value in metrics.items():
                label = METRIC_LABELS.get(name, name)
                print(f"  {label:<28}{fmt_score(value):<12}{fmt_score(adjusted.get(name)):<12}")
        else:
            print("\n  （仅检索层：当时生成模型不可用，没有 RAGAS 指标）")

        if samples:
            hits = sum(1 for s in samples if (s.get("retrieval") or {}).get("hits", 0) > 0)
            top1_values = [
                (s.get("retrieval") or {}).get("top1") for s in samples
            ]
            top1_values = [v for v in top1_values if isinstance(v, (int, float))]
            avg_top1 = sum(top1_values) / len(top1_values) if top1_values else None
            print(
                f"\n  检索层   : 命中 {hits}/{len(samples)}，平均 Top1 相似度 {fmt_score(avg_top1)}"
            )

        for i, sample in enumerate(samples):
            if not sample.get("expect_refusal"):
                continue
            answer = (sample.get("answer") or "").strip()
            if not answer:
                print(f"  负样本   : #{i + 1} 无回答（仅检索层）")
            else:
                refused = any(k in answer for k in REFUSAL_KEYWORDS)
                print(f"  负样本   : #{i + 1} " + ("✅ 正确拒答" if refused else "❌ 未拒答（可能编造）"))

        advice = data.get("advice") or []
        if advice:
            print("\n  建议：")
            for line in advice:
                print("    - " + line)

        print(f"\n  文件：{path}")
    print("=" * 92)
    print(f"共 {len(selected)} 份报告（按时间倒序）。要重新评测：python run_eval.py")
    return 0


def render_report(
    samples: List[Dict[str, Any]],
    engine: RAGEngine,
    df: Any,
    metric_names: List[str],
    judge_model: str,
    has_reference: bool,
) -> tuple[Dict[str, Optional[float]], List[str], Dict[str, Optional[float]]]:
    print("\n" + "=" * 100)
    print(" RAG-Agent 自动评测报告")
    print(" 生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 100)

    stats = engine.stats()
    print("[1] 运行环境")
    print(f"  嵌入模型  : {engine.model_name}")
    print(f"  裁判/生成 : {judge_model}")
    print(f"  知识库    : {stats['documents']} 篇文档 / {stats['chunks']} 个向量块")
    if stats["document_names"]:
        print(f"  文档清单  : {', '.join(stats['document_names'][:5])}"
              + (" ..." if len(stats["document_names"]) > 5 else ""))
    print(f"  评测样本  : {len(samples)} 条" + ("（含标准答案）" if has_reference else "（无标准答案）"))

    print_retrieval_section(samples, engine)
    means = print_metric_section(df, metric_names)
    adjusted = print_refusal_section(samples, df, metric_names)
    print_sample_section(samples, df, metric_names)
    advice = print_diagnosis(means, samples, engine)
    print("\n" + "=" * 100)
    return means, advice, adjusted


# ---------------------------------------------------------------------------
# 环境自检
# ---------------------------------------------------------------------------

def print_llm_diagnosis(cfg: Any) -> dict:
    """打印生成端点的分层诊断，并给出“缺什么补什么”的命令。"""
    info = diagnose_llm(cfg)
    print(f"\n[生成模型] {info['provider_label']}")
    print(f"  接口地址 : {info['base_url']}")
    print(f"  目标模型 : {info['model']}")

    if info["provider"] == "ollama":
        if info["installed"]:
            print(f"  Ollama   : 已安装 -> {info['exe']}")
        else:
            print("  Ollama   : 未安装（PATH 与常见安装目录都没找到）")
        print(f"  服务状态 : {'在线' if info['server_up'] else '未连接'}")
        if info["models"]:
            print(f"  已有模型 : {', '.join(info['models'][:8])}")
        print(f"  目标模型 : {'已就绪' if info['model_present'] else '不在列表里'}")
    else:
        print(f"  连接状态 : {'正常' if info['server_up'] else '失败'}")
        if info["models"]:
            print(f"  可用模型 : {', '.join(info['models'][:8])}")
        if info["error"]:
            print(f"  错误信息 : {info['error']}")

    if info["hints"]:
        print("\n  需要处理：")
        for hint in info["hints"]:
            for line in hint.splitlines():
                print("    " + line)
    else:
        print("\n  状态：可用")
    return info


def run_doctor(engine: RAGEngine, questions_path: Path) -> int:
    print("=" * 60)
    print(" 环境自检")
    print("=" * 60)

    import platform

    print(f"Python      : {platform.python_version()} ({sys.executable})")

    for pkg in ("ragas", "datasets", "openai", "langchain-community", "faiss-cpu", "sentence-transformers"):
        try:
            from importlib.metadata import version

            print(f"{pkg:<12}: {version(pkg)}")
        except Exception:  # noqa: BLE001
            print(f"{pkg:<12}: 未安装  <-- 需要 pip install -r requirements.txt")

    try:
        import ragas

        print(f"ragas 导入  : OK (v{ragas.__version__})")
    except Exception as exc:  # noqa: BLE001
        print(f"ragas 导入  : 失败 -> {type(exc).__name__}: {exc}")

    info = print_llm_diagnosis(engine.llm_config)

    stats = engine.stats()
    print("\n[知识库]")
    print(f"  FAISS 索引 : {stats['documents']} 篇文档 / {stats['chunks']} 个块")
    if stats["chunks"] == 0:
        print("             索引为空 -> 先 python run_eval.py --rebuild，或用 app.py 上传文档")

    try:
        items = load_eval_set(questions_path)
        print(f"  问题集     : {questions_path.name}，共 {len(items)} 条")
    except Exception as exc:  # noqa: BLE001
        print(f"  问题集     : 读取失败 -> {exc}")

    # 结论：接口连得上 + 有索引就算能跑；模型名是否出现在 /models 里只作提示，
    # 因为部分网关不返回完整列表（实测按名字调用仍成功）。
    can_generate = info["server_up"]
    print("\n" + "-" * 60)
    if can_generate and stats["chunks"] > 0:
        print("结论：可以跑完整评测 -> python run_eval.py")
        print("     建议先跑一次：python run_eval.py --check-llm")
    elif stats["chunks"] == 0:
        print("结论：索引为空，先建库 -> python run_eval.py --rebuild")
    else:
        print("结论：检索可用，但生成端点连不上。")
        print("     现在可以直接跑：python run_eval.py --retrieval-only")
        print("     想拿到忠实度/相关性分数，请按上面「需要处理」的提示补齐配置。")
    print("-" * 60)
    return 0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG-Agent 自动评测（RAGAS）")
    parser.add_argument("--questions", default="eval_set.json", help="评测问题集路径（默认 eval_set.json）")
    parser.add_argument("--top-k", type=int, default=5, help="检索返回的块数（默认 5）")
    parser.add_argument("--min-score", type=float, default=None, help="检索相似度阈值（默认用引擎配置）")
    parser.add_argument("--judge-model", default=None, help="RAGAS 裁判模型（默认用生成模型）")
    parser.add_argument("--retrieval-only", action="store_true", help="只评测检索层，不调用 ragas")
    parser.add_argument("--full", action="store_true", help="额外评测 answer_correctness")
    parser.add_argument("--rebuild", action="store_true", help="先按 uploads/ 重建索引再评测")
    parser.add_argument("--doctor", action="store_true", help="只做环境自检")
    parser.add_argument("--check-llm", action="store_true", help="发一条真实请求，测试生成模型连通性")
    parser.add_argument("--start-ollama", action="store_true", help="后台启动本地 Ollama 后再继续")
    parser.add_argument(
        "--show",
        nargs="?",
        const=1,
        type=int,
        default=None,
        metavar="N",
        help="只读查看最近 N 份历史报告（默认 1），不重跑评测、不产生费用",
    )
    parser.add_argument("--quiet", action="store_true", help="减少进度输出")
    parser.add_argument("--out-dir", default="reports", help="报告输出目录（默认 reports/）")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    questions_path = Path(args.questions)
    if not questions_path.is_absolute():
        questions_path = HERE / questions_path

    out_dir = HERE / args.out_dir

    # --show 走只读分支：不加载嵌入模型（省 10 秒），也不调用任何接口
    if args.show is not None:
        return show_reports(args.show, out_dir)

    # 先把生成端点定下来：--judge-model 只覆盖模型名，其余走环境变量/默认值
    llm_cfg = resolve_llm_config(model=args.judge_model)

    engine_kwargs: Dict[str, Any] = {
        "top_k": args.top_k,
        "llm_provider": llm_cfg.provider,
        "llm_base_url": llm_cfg.base_url,
        "llm_api_key": llm_cfg.api_key,
        "chat_model": llm_cfg.model,
    }
    if args.min_score is not None:
        engine_kwargs["score_threshold"] = args.min_score

    try:
        engine = RAGEngine(**engine_kwargs)
    except Exception:  # noqa: BLE001
        print("\n[致命错误] RAG 引擎初始化失败（通常是嵌入模型加载不了）：")
        traceback.print_exc()
        return 1

    # --start-ollama：顺手把本地服务拉起来，省掉“另开一个终端”的步骤
    if args.start_ollama:
        print("正在尝试后台启动 Ollama ...")
        if start_ollama_server(engine.llm_config):
            print("Ollama 已就绪。")
        else:
            print("启动失败或超时，请看下面的诊断。")

    if args.doctor:
        return run_doctor(engine, questions_path)

    if args.check_llm:
        print(f"\n[连通性测试] {engine.llm_config.describe()}")
        ok, message, elapsed = engine.check_llm()
        if ok:
            print(f"  成功：{elapsed:.2f}s，模型回复 {message!r}")
            return 0
        print(f"  失败：{elapsed:.2f}s，{message}")
        print_llm_diagnosis(engine.llm_config)
        return 1

    print("Starting RAGAS Automated Evaluation...")

    if args.rebuild:
        corpus = default_corpus_dir()
        print(f"重建索引：扫描 {corpus}/ ...")
        engine.reset_index()
        engine.ingest_directory(corpus)

    try:
        items = load_eval_set(questions_path)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[致命错误] {exc}")
        return 1

    stats = engine.stats()
    if stats["chunks"] == 0:
        print("\n[致命错误] 向量索引为空，无法评测。请先执行：python run_eval.py --rebuild")
        return 1

    print(f"知识库: {stats['documents']} 篇文档 / {stats['chunks']} 个块")
    print(f"生成端点: {engine.llm_config.describe()}")

    # 是否降级只由“真实调用”决定，不看 /models 列表：
    # 不少网关（含 deepseek 中转）的 /models 不返回完整列表，但按名字调用是通的。
    # 误判成不可用会让用户白跑一遍只拿到检索层结果。
    if not diagnose_llm(engine.llm_config)["server_up"]:
        print("\n[提示] 生成端点连不上，将只做检索层评测。")
        print_llm_diagnosis(engine.llm_config)
        args.retrieval_only = True
    else:
        ok, message, elapsed = engine.check_llm()
        if not ok:
            print(f"\n[提示] 生成模型连通性测试失败（{elapsed:.1f}s）：{message}")
            print_llm_diagnosis(engine.llm_config)
            args.retrieval_only = True
        else:
            print(f"连通性: OK（{elapsed:.1f}s）")

    print("Generating data from RAG engine, please wait...")
    samples = collect_samples(
        engine,
        items,
        top_k=args.top_k,
        quiet=args.quiet,
        with_generation=not args.retrieval_only,
    )

    judge_model = engine.llm_config.model
    generation_failed = [s for s in samples if s["error"]]
    if generation_failed:
        print("\n[提示] 有 %d 条样本生成失败，示例：" % len(generation_failed))
        print(generation_failed[0]["error"])
        args.retrieval_only = True

    if args.retrieval_only:
        print("\n[1] 运行环境")
        print(f"  嵌入模型  : {engine.model_name}")
        print(f"  知识库    : {stats['documents']} 篇文档 / {stats['chunks']} 个块")
        print_retrieval_section(samples, engine)
        print("\n[说明] 当前只完成了检索层评测。要拿到 RAGAS 的忠实度/相关性分数，"
              "请先让生成模型可用（见上方诊断）后运行：python run_eval.py")
        print("\n[3] 检索层汇总")
        print(f"  平均召回块数 : {fmt_score(safe_mean([s['retrieval']['hits'] for s in samples]))}")
        print(f"  平均 Top1 相似度 : {fmt_score(safe_mean([s['retrieval']['top1'] for s in samples]))}")
        save_reports(
            HERE / args.out_dir,
            samples,
            None,
            [],
            {},
            ["仅完成检索层评测（生成模型不可用）"],
            engine,
            judge_model,
            {"mode": "retrieval-only"},
        )
        return 2

    try:
        df, metric_names, has_reference = run_ragas(
            samples,
            engine,
            judge_cfg=engine.llm_config,
            full=args.full,
            quiet=args.quiet,
        )
    except Exception:  # noqa: BLE001
        print("\n[致命错误] RAGAS 评分阶段失败：")
        traceback.print_exc()
        print("\n常见原因与修法：")
        print("  1) ModuleNotFoundError: langchain_community.chat_models.vertexai")
        print("     -> 必须在本文件最前面 import ragas_compat（本脚本已做）")
        print("  2) TypeError: All metrics must be initialised metric objects")
        print("     -> 不要从 ragas.metrics.collections 导入指标，用 ragas.metrics 的旧式类")
        print("  3) 连接被拒绝 / 404 / 401")
        print("     -> 先跑 python run_eval.py --check-llm，按输出逐层排查")
        print("  4) 单条样本全部 NaN")
        print("     -> 多半是模型没按 JSON 结构化输出，换更强的裁判模型：")
        print("        python run_eval.py --judge-model qwen2.5:14b")
        return 1

    means, advice, adjusted = render_report(
        samples, engine, df, metric_names, judge_model, has_reference
    )

    paths = save_reports(
        HERE / args.out_dir,
        samples,
        df,
        metric_names,
        means,
        advice,
        engine,
        judge_model,
        {
            "mode": "full" if not args.retrieval_only else "retrieval-only",
            "metrics_excluding_negative": adjusted,
        },
    )
    print("\n报告已保存:")
    for path in paths:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断。")
        sys.exit(130)
