"""RAG 引擎：文档解析 → 切分 → 向量化 → FAISS 检索 → 大模型生成。

对外只有三个动作：

    engine = RAGEngine()                 # 加载本地嵌入模型
    engine.build(uploaded_files)         # 入库（Streamlit 上传对象或路径）
    engine.query_with_sources(question)  # 检索 + 生成，返回答案与引用

设计取舍
--------
- **嵌入与生成解耦**：检索用 sentence-transformers（本地、免费、可离线），
  生成用任意 OpenAI 兼容接口（Ollama 或云端）。换生成模型不用重建索引。
- **索引与模型同源**：从磁盘加载索引时显式传回 ``distance_strategy``，
  否则内积索引会被按欧氏距离解释，重启前后分数口径不一致。
- **阈值可调**：MiniLM 多语种对中文的余弦相似度普遍落在 0.2~0.6，
  默认阈值取 0.25；``top_scores()`` 可看未过滤的原始分布用于调参。

依赖说明：torch / faiss / langchain 只在实例化 RAGEngine 时才真正加载，
因此 ``import rag_engine`` 本身很轻，方便测试与工具复用。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

# 屏蔽 langchain-community 的“停止维护”提示：只是 deprecation 告警，
# 混在评测输出里容易被误认为报错。
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*langchain-community.*sunset.*",
)

# 国内网络下载 HF 模型更稳
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 关闭 huggingface_hub 的 Xet 传输（其 CDN 在国内常不可达，会卡死下载）
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from core import (  # noqa: E402  纯逻辑，无重依赖
    MAX_CONTEXT_CHUNKS,
    SYSTEM_PROMPT,
    LLMUnavailableError,  # 兼容旧导入路径
    build_prompt,
    diagnose_llm,
    resolve_llm_config,
    split_text,
)

# 默认嵌入模型：多语言轻量模型，中英文检索均衡
# 可用 RAG_EMBEDDING_MODEL 覆盖（支持模型名或本地绝对路径）
DEFAULT_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL") or "paraphrase-multilingual-MiniLM-L12-v2"

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def default_corpus_dir() -> Path:
    """建库默认扫描哪个目录。

    优先用 ``uploads/``（用户自己上传的文档）；空的话退回仓库自带的
    ``data/samples/``，让 clone 下来的人不用准备文档就能跑通演示。
    """
    for candidate in (Path("uploads"), Path("data/samples"), Path("documents")):
        if not candidate.is_dir():
            continue
        if any(p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS for p in candidate.iterdir()):
            return candidate
    return Path("uploads")


def parse_document(file_path: Union[str, Path]) -> str:
    """解析单个文档为纯文本（PDF / docx / txt / md）。

    解析器按需导入：只有真正处理该类型文件时才加载 pypdf / python-docx。
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages: List[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 单页损坏直接跳过
                pages.append("")
        return "\n".join(pages)

    if suffix == ".docx":
        from docx import Document as DocxDocument

        doc = DocxDocument(str(path))
        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if row_text:
                    parts.append(row_text)
        return "\n\n".join(parts)

    if suffix in {".txt", ".md"}:
        raw = path.read_bytes()
        for encoding in ("utf-8", "gbk", "utf-16"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")

    raise ValueError(f"不支持的文件类型: {path}（仅支持 PDF/Word/TXT/MD）")


class RAGEngine:
    """本地 RAG 引擎：解析 → 切分 → 向量化 → FAISS → 检索 → 生成。"""

    def __init__(
        self,
        index_dir: str = "faiss_index",
        model_name: str = DEFAULT_MODEL_NAME,
        chunk_min_chars: int = 500,
        chunk_max_chars: int = 800,
        top_k: int = 5,
        score_threshold: float = 0.25,
        upload_dir: str = "uploads",
        chat_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        request_timeout: Optional[float] = None,
        llm_provider: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
    ) -> None:
        """初始化引擎并加载嵌入模型（这一步会占用约 470MB 内存）。

        Args:
            index_dir: FAISS 索引目录
            model_name: 嵌入模型名或本地路径
            chunk_min_chars / chunk_max_chars: 切分块长度区间
            top_k: 检索默认返回条数
            score_threshold: 余弦相似度阈值；本模型对中文普遍 0.2~0.6，
                设 0.6 会导致“永远检索不到内容”
            upload_dir: 上传文件落盘目录
            chat_model: 生成模型名；None 表示按环境变量/自动探测
            ollama_base_url: 旧参数名，等价于把 llm_base_url 指向本地 Ollama
            request_timeout: 单次生成超时（秒）
            llm_provider / llm_base_url / llm_api_key: 生成端点显式配置
        """
        if chunk_min_chars > chunk_max_chars:
            raise ValueError("chunk_min_chars 不能大于 chunk_max_chars")

        # 仓库里不带模型权重：优先用本地 models/，否则交给 HF 自动下载（走镜像）
        local_model_dir = Path("models") / model_name
        if (local_model_dir / "modules.json").exists():
            model_name = str(local_model_dir)

        from langchain_community.vectorstores import FAISS
        from langchain_community.vectorstores.utils import DistanceStrategy

        try:  # langchain 1.x 把 HuggingFaceEmbeddings 拆到了独立包
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        self._FAISS = FAISS
        self._DistanceStrategy = DistanceStrategy

        self.index_dir = Path(index_dir)
        self.upload_dir = Path(upload_dir)
        self.model_name = model_name
        self.chunk_min_chars = chunk_min_chars
        self.chunk_max_chars = chunk_max_chars
        self.top_k = top_k
        self.score_threshold = score_threshold

        self.llm_config = resolve_llm_config(
            provider=llm_provider,
            base_url=llm_base_url or ollama_base_url,
            api_key=llm_api_key,
            model=chat_model,
            timeout=request_timeout,
        )
        # 兼容旧属性名
        self.chat_model = self.llm_config.model
        self.ollama_base_url = self.llm_config.base_url
        self.request_timeout = self.llm_config.timeout

        # normalize_embeddings=True：归一化后内积即余弦相似度
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT
        self.vectorstore: Any = None
        self._next_chunk_id = 0
        self._llm_client: Any = None

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------

    def ingest(self, file_paths: Iterable[Union[str, Path]]) -> int:
        """解析并入库一批文档，返回新增块数量；已有索引时增量追加。"""
        all_chunks: List[Any] = []
        from langchain_core.documents import Document

        for file_path in file_paths:
            path = Path(file_path)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                print(f"[跳过] 不支持的文件类型: {path}")
                continue

            text = parse_document(path)
            chunks = split_text(text, self.chunk_min_chars, self.chunk_max_chars)
            if not chunks:
                print(f"[跳过] 未提取到有效文本: {path}")
                continue

            document_name = path.name
            for chunk in chunks:
                all_chunks.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "document_name": document_name,
                            "chunk_id": self._next_chunk_id,
                            "source": str(path),
                        },
                    )
                )
                self._next_chunk_id += 1
            print(f"[解析] {document_name}: 切分为 {len(chunks)} 个块")

        if not all_chunks:
            return 0

        if self.vectorstore is None:
            self.load_index()
        if self.vectorstore is None:
            self.vectorstore = self._FAISS.from_documents(
                all_chunks,
                self.embeddings,
                distance_strategy=self.distance_strategy,
            )
        else:
            self.vectorstore.add_documents(all_chunks)

        self.save()
        return len(all_chunks)

    def ingest_directory(self, dir_path: Union[str, Path]) -> int:
        """扫描目录下所有受支持文档并入库。"""
        directory = Path(dir_path)
        if not directory.exists():
            return 0
        files = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            print(f"[提示] {directory}/ 中没有可解析的 PDF/Word/TXT/MD 文件")
            return 0
        return self.ingest(files)

    def build(self, uploaded_files: Iterable[Any], replace: bool = False) -> int:
        """接收 Streamlit 的 UploadedFile：落盘到 uploads/ 后入库。

        Args:
            uploaded_files: UploadedFile 列表（也兼容路径字符串）
            replace: True 时先清空旧索引，按传入文件全量重建
        """
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        saved: List[Path] = []
        for uploaded in uploaded_files:
            if isinstance(uploaded, (str, Path)):
                saved.append(Path(uploaded))
                continue
            name = Path(getattr(uploaded, "name", "upload.bin")).name
            target = self.upload_dir / name
            if hasattr(uploaded, "getbuffer"):
                target.write_bytes(uploaded.getbuffer())
            elif hasattr(uploaded, "read"):
                target.write_bytes(uploaded.read())
            else:
                raise TypeError(f"无法识别的上传对象类型: {type(uploaded)!r}")
            saved.append(target)

        if replace:
            self.reset_index()
        return self.ingest(saved)

    def reset_index(self) -> None:
        """删除内存与磁盘上的索引。"""
        self.vectorstore = None
        self._next_chunk_id = 0
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
            print(f"[重建] 已删除旧索引目录: {self.index_dir}")

    def save(self) -> None:
        """把 FAISS 索引与 chunk_id 计数器持久化。"""
        if self.vectorstore is None:
            return
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.index_dir))
        (self.index_dir / "meta.json").write_text(
            json.dumps({"next_chunk_id": self._next_chunk_id}, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_index(self) -> None:
        """加载本地索引；不存在时保持为空。

        ``distance_strategy`` 必须与建库时一致，否则分数语义会被改写。
        """
        if self.vectorstore is not None:
            return
        if not (self.index_dir / "index.faiss").exists():
            return
        # pickle 反序列化：仅加载本进程自己生成的索引
        self.vectorstore = self._FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
            distance_strategy=self.distance_strategy,
        )
        meta_path = self.index_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._next_chunk_id = int(meta.get("next_chunk_id", 0))

    def stats(self) -> dict:
        """索引概况：文档数、块数、文档名列表。"""
        self.load_index()
        if self.vectorstore is None:
            return {"documents": 0, "chunks": 0, "document_names": []}
        docstore = getattr(self.vectorstore, "docstore", None)
        names: List[str] = []
        for doc_id in getattr(self.vectorstore, "index_to_docstore_id", {}).values():
            try:
                doc = docstore.search(doc_id)
            except Exception:  # noqa: BLE001
                continue
            name = doc.metadata.get("document_name", "")
            if name and name not in names:
                names.append(name)
        chunks = len(getattr(self.vectorstore, "index_to_docstore_id", {}))
        return {"documents": len(names), "chunks": chunks, "document_names": names}

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[dict]:
        """检索与问题最相似的文档块（按余弦相似度降序，低于阈值丢弃）。

        Returns:
            每项含 content / document_name / chunk_id / score
        """
        self.load_index()
        if self.vectorstore is None:
            return []

        k = top_k if top_k is not None else self.top_k
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        # 多取 4 倍候选，阈值过滤后仍能凑够 top_k
        candidates = self.vectorstore.similarity_search_with_score(query, k=k * 4)
        results: List[dict] = []
        for doc, score in candidates:
            if score < threshold:
                continue
            results.append(
                {
                    "content": doc.page_content,
                    "document_name": doc.metadata.get("document_name", ""),
                    "chunk_id": doc.metadata.get("chunk_id", -1),
                    "score": round(float(score), 4),
                }
            )
            if len(results) >= k:
                break
        return results

    def top_scores(self, query: str, k: int = 5) -> List[float]:
        """不做阈值过滤的原始 Top-K 相似度（调阈值与评测诊断用）。"""
        self.load_index()
        if self.vectorstore is None:
            return []
        pairs = self.vectorstore.similarity_search_with_score(query, k=k)
        return [round(float(score), 4) for _, score in pairs]

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------

    def chat_model_name(self) -> str:
        return self.llm_config.model

    def _client(self):
        """惰性创建 OpenAI 兼容客户端。"""
        if self._llm_client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise LLMUnavailableError("缺少 openai 依赖，请执行: pip install openai") from exc
            cfg = self.llm_config
            self._llm_client = OpenAI(
                base_url=cfg.base_url,
                api_key=cfg.api_key or "not-needed",
                timeout=cfg.timeout,
                # 服务没起来时立刻报错，避免评测脚本每条样本白等几秒
                max_retries=0,
            )
        return self._llm_client

    def llm_available(self) -> bool:
        return bool(diagnose_llm(self.llm_config)["server_up"])

    def llm_diagnosis(self) -> dict:
        """分层诊断结果（界面与 doctor 共用）。"""
        return diagnose_llm(self.llm_config)

    def check_llm(self, timeout: float = 60.0) -> tuple:
        """发一条极短请求，验证“能连上 + 模型名对 + 能返回”。

        Returns:
            (是否成功, 说明或模型回复, 耗时秒)
        """
        from datetime import datetime

        started = datetime.now()
        try:
            resp = self._client().chat.completions.create(
                model=self.chat_model_name(),
                messages=[{"role": "user", "content": "回答两个字：收到"}],
                temperature=0,
                max_tokens=16,
            )
            elapsed = (datetime.now() - started).total_seconds()
            reply = (resp.choices[0].message.content or "").strip().replace("\n", " ")
            return True, reply or "(空回复)", elapsed
        except Exception as exc:  # noqa: BLE001
            elapsed = (datetime.now() - started).total_seconds()
            return False, f"{type(exc).__name__}: {exc}", elapsed

    def build_prompt(self, question: str, contexts: List[str]) -> str:
        return build_prompt(question, contexts, MAX_CONTEXT_CHUNKS)

    def generate(self, question: str, contexts: List[str]) -> str:
        """调用大模型，根据上下文生成回答（失败抛 LLMUnavailableError）。"""
        prompt = self.build_prompt(question, contexts)
        try:
            resp = self._client().chat.completions.create(
                model=self.chat_model_name(),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=800,
            )
        except Exception as exc:  # noqa: BLE001 统一转成可读错误
            hints = diagnose_llm(self.llm_config)["hints"] or ["检查接口地址、模型名与网络后重试。"]
            raise LLMUnavailableError(
                "无法调用生成模型：\n"
                f"  端点: {self.llm_config.base_url}\n"
                f"  模型: {self.chat_model_name()}\n"
                f"  原始错误: {type(exc).__name__}: {exc}\n"
                "建议按下面顺序排查：\n  - " + "\n  - ".join(hints)
            ) from exc
        return (resp.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------
    # 端到端问答
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: Optional[int] = None, min_score: Optional[float] = None) -> str:
        """检索 + 生成，只返回答案字符串。"""
        return self.query_with_sources(question, top_k=top_k, min_score=min_score)["answer"]

    def query_with_sources(
        self,
        question: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> dict:
        """检索 + 生成，同时返回引用来源。

        Returns:
            {"answer", "sources": [{document_name, chunk_id, score}], "contexts": [str]}
        """
        results = self.search(question, top_k=top_k, score_threshold=min_score)
        contexts = [item["content"] for item in results]
        answer = self.generate(question, contexts)
        sources = [
            {
                "document_name": item["document_name"],
                "chunk_id": item["chunk_id"],
                "score": item["score"],
            }
            for item in results
        ]
        return {"answer": answer, "sources": sources, "contexts": contexts}


def main() -> None:
    """命令行演示。

    用法：
        python rag_engine.py                      # 用 data/samples 或 documents/ 建库并示例检索
        python rag_engine.py "你的问题"            # 检索 + 生成
        python rag_engine.py --rebuild             # 删旧索引重建
        python rag_engine.py --retrieval-only      # 只检索，不调用大模型
    """
    engine = RAGEngine()

    if "--rebuild" in sys.argv and engine.index_dir.exists():
        engine.reset_index()

    # 建库目录：优先 documents/，否则用仓库自带的 data/samples/
    candidates = [Path("documents"), Path("data/samples")]
    if not (engine.index_dir / "index.faiss").exists():
        source = next((p for p in candidates if p.exists()), None)
        if source is None:
            print("未找到索引，也没有 documents/ 或 data/samples/ 目录。")
            print("请把 PDF/Word/TXT/MD 放进 documents/ 后重试。")
            return
        print(f"开始构建索引，扫描 {source}/ ...")
        engine.ingest_directory(source)

    engine.load_index()

    query_parts = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    query = " ".join(query_parts) if query_parts else "公司的试用期是多久？"

    print(f"\n查询: {query}\n")
    results = engine.search(query)
    if not results:
        print(f"没有满足阈值（{engine.score_threshold}）的结果。")
        print(f"Top-5 原始相似度: {engine.top_scores(query)}")
        return

    for rank, item in enumerate(results, start=1):
        preview = item["content"][:100].replace("\n", " ")
        print(f"Top{rank} score={item['score']} | {item['document_name']} (chunk_id={item['chunk_id']})")
        print(f"    {preview}...")

    if "--retrieval-only" in sys.argv:
        return

    try:
        answer = engine.generate(query, [item["content"] for item in results])
    except LLMUnavailableError as exc:
        print(f"\n[生成跳过] {exc}")
        return
    print("\n回答:\n" + answer)


if __name__ == "__main__":
    main()
