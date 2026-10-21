"""纯逻辑核心：不依赖 torch / faiss / langchain，可在无模型环境下导入与测试。

这里的函数都是“输入 → 输出”的纯计算或轻量系统探测，因此 CI 不需要安装
sentence-transformers（约 2GB）就能覆盖大部分容易写错的地方：

- ``split_text``       中文友好的定长切分（边界、超长句、残余块）
- ``build_prompt``     检索上下文拼提示词（编号、截断、空上下文）
- ``load_eval_set``    评测集解析（UTF-8/GBK 兼容、缺字段报错）
- ``resolve_llm_config`` / ``diagnose_llm``  生成端点的多后端配置与分层诊断

重活（向量化、FAISS、ragas）都在 rag_engine.py 与 run_eval.py 里。
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_dotenv(path: Optional[Path] = None) -> int:
    """把项目根目录的 ``.env`` 读进环境变量（已存在的不覆盖）。

    只支持 ``KEY=VALUE`` 与 ``#`` 注释，不引入第三方依赖。
    这样配一次 key 就能长期生效，不必在每个新终端里重新 ``export``。

    Returns:
        实际写入的变量个数。
    """
    env_path = Path(path) if path is not None else Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return 0

    applied = 0
    try:
        content = env_path.read_text(encoding="utf-8-sig")
    except (UnicodeDecodeError, OSError):
        return 0

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # 显式设置过的环境变量优先级更高，不被 .env 覆盖
        if key and key not in os.environ:
            os.environ[key] = value
            applied += 1
    return applied


# import 即生效：项目根目录有 .env 时自动加载（该文件已被 .gitignore 排除）
load_dotenv()

# ---------------------------------------------------------------------------
# 生成模型：多后端配置（本地 Ollama / 任意 OpenAI 兼容云服务）
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_BASE_URL = (
    os.getenv("RAG_OLLAMA_BASE_URL", "").strip() or "http://localhost:11434/v1"
)
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

PROVIDER_LABELS = {
    "ollama": "本地 Ollama",
    "openai": "OpenAI 兼容云端 API",
}

# 交给生成模型的上下文条数上限（太多会超出小模型的上下文窗口）
MAX_CONTEXT_CHUNKS = int(os.getenv("RAG_MAX_CONTEXT_CHUNKS", "5"))

SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。你只能依据用户提供的【参考资料】回答问题，"
    "不得使用参考资料之外的知识，也不要编造事实。"
    "如果参考资料不足以回答问题，请直接说明“根据现有知识库无法回答该问题”，"
    "并指出缺少哪方面的信息。"
    "回答使用简体中文，条理清晰；引用某条资料时用 [编号] 标注来源。"
)

# Ollama 常见安装位置：装完但没重开终端时 PATH 里没有，这里兜底找一下
OLLAMA_EXE_CANDIDATES = (
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Ollama" / "ollama.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Ollama" / "ollama.exe",
)


class LLMUnavailableError(RuntimeError):
    """生成模型不可用（未安装 / 未启动 / 缺模型 / Key 无效）。"""


@dataclass
class LLMConfig:
    """一次生成/评测要用的模型端点配置。"""

    provider: str = "ollama"
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    api_key: str = "ollama"
    model: str = "qwen2.5:7b"
    timeout: float = 180.0

    @property
    def is_ollama(self) -> bool:
        return self.provider == "ollama"

    @property
    def provider_label(self) -> str:
        return PROVIDER_LABELS.get(self.provider, self.provider)

    def describe(self) -> str:
        return f"{self.provider_label} · {self.model} · {self.base_url}"


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _http_json(url: str, timeout: float = 3.0) -> Any:
    """GET 一个 JSON 接口（探测 Ollama 是否在线、列出本地模型）。"""
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 本地地址
        return json.loads(resp.read().decode("utf-8"))


def ollama_base_url_to_root(base_url: str) -> str:
    """把 OpenAI 兼容地址 http://host:11434/v1 还原成 http://host:11434。"""
    return base_url.rstrip("/").removesuffix("/v1")


def list_ollama_models(base_url: str = DEFAULT_OLLAMA_BASE_URL, timeout: float = 3.0) -> List[str]:
    """返回本地 Ollama 已安装的模型名列表；不可用时返回空列表。"""
    try:
        data = _http_json(f"{ollama_base_url_to_root(base_url)}/api/tags", timeout=timeout)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return []
    models = data.get("models", []) if isinstance(data, dict) else []
    names = []
    for item in models:
        name = item.get("name") or item.get("model")
        if name:
            names.append(name)
    return names


def resolve_chat_model(base_url: str = DEFAULT_OLLAMA_BASE_URL) -> str:
    """确定本地模型名：环境变量优先，其次取已安装的第一个，最后兜底 qwen2.5:7b。"""
    explicit = os.getenv("RAG_CHAT_MODEL", "").strip()
    if explicit:
        return explicit
    installed = list_ollama_models(base_url)
    if installed:
        # 优先挑通用对话模型（排除 embedding 类）
        for name in installed:
            lowered = name.lower()
            if any(key in lowered for key in ("qwen", "llama", "deepseek", "glm", "mistral")):
                return name
        return installed[0]
    return "qwen2.5:7b"


def resolve_llm_config(
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> LLMConfig:
    """把“显式参数 → 环境变量 → 默认值”合成一份配置。

    优先级示例：``RAG_LLM_API_KEY`` 存在时默认走云端；否则指向 localhost 走 Ollama。
    """
    api_key = (api_key or _first_env("RAG_LLM_API_KEY", "OPENAI_API_KEY")).strip()
    base_url = (base_url or _first_env("RAG_LLM_BASE_URL", "OPENAI_BASE_URL")).strip()
    model = (model or _first_env("RAG_CHAT_MODEL", "RAG_JUDGE_MODEL")).strip()
    provider = (provider or _first_env("RAG_LLM_PROVIDER")).strip().lower()

    if provider not in ("ollama", "openai"):
        if provider:
            raise ValueError(f"RAG_LLM_PROVIDER 只能是 ollama 或 openai，当前为 {provider!r}")
        if api_key:
            provider = "openai"
        elif base_url and "localhost" not in base_url and "127.0.0.1" not in base_url:
            provider = "openai"
        else:
            provider = "ollama"

    if not base_url:
        base_url = DEFAULT_OPENAI_BASE_URL if provider == "openai" else DEFAULT_OLLAMA_BASE_URL

    if not api_key:
        api_key = "ollama" if provider == "ollama" else ""

    if not model:
        model = resolve_chat_model(base_url) if provider == "ollama" else "gpt-4o-mini"

    if timeout is None:
        try:
            timeout = float(os.getenv("RAG_LLM_TIMEOUT", "180"))
        except ValueError:
            timeout = 180.0

    return LLMConfig(
        provider=provider,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        timeout=timeout,
    )


def find_ollama_executable() -> Optional[Path]:
    """在 PATH 和常见安装目录里找 ollama(.exe)。"""
    found = shutil.which("ollama")
    if found:
        return Path(found)
    for candidate in OLLAMA_EXE_CANDIDATES:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def diagnose_llm(cfg: LLMConfig) -> Dict[str, Any]:
    """分层诊断：装没装 / 起没起 / 模型在不在 / Key 对不对。

    返回结构同时给 doctor、Streamlit 侧边栏和报错文案复用，
    避免每处各写一套 if，结论不一致。
    """
    info: Dict[str, Any] = {
        "provider": cfg.provider,
        "provider_label": cfg.provider_label,
        "base_url": cfg.base_url,
        "model": cfg.model,
        "installed": None,
        "exe": None,
        "server_up": False,
        "models": [],
        "model_present": False,
        "error": "",
        "hints": [],
    }

    if cfg.is_ollama:
        exe = find_ollama_executable()
        info["installed"] = bool(exe)
        info["exe"] = str(exe) if exe else None

        models = list_ollama_models(cfg.base_url)
        info["models"] = models
        info["server_up"] = bool(models)
        target = cfg.model.split(":")[0].lower()
        info["model_present"] = any(
            name.lower() == cfg.model.lower() or name.split(":")[0].lower() == target
            for name in models
        )
    else:
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=cfg.base_url,
                api_key=cfg.api_key or "sk-missing",
                timeout=10,
                max_retries=0,
            )
            listed = [m.id for m in client.models.list().data]
            info["server_up"] = True
            info["models"] = listed[:20]
            # 有的服务不实现 /models，只要请求没报错就认为可用
            info["model_present"] = (
                any(cfg.model in name or name in cfg.model for name in listed)
                if listed
                else True
            )
        except Exception as exc:  # noqa: BLE001 网络/鉴权/地址错误统一记录
            info["error"] = f"{type(exc).__name__}: {exc}"

    info["hints"] = llm_hints(info, cfg)
    return info


def llm_hints(info: Dict[str, Any], cfg: LLMConfig) -> List[str]:
    """按“缺什么补什么”的顺序给出可执行命令（纯函数，便于单测）。"""
    if not cfg.is_ollama:
        if info["server_up"] and info["model_present"]:
            return []
        hints = []
        if not cfg.api_key:
            hints.append(
                "没有读到 API Key。设置后再跑：\n"
                '    $env:RAG_LLM_API_KEY="sk-..."\n'
                '    $env:RAG_LLM_BASE_URL="https://api.deepseek.com/v1"\n'
                '    $env:RAG_CHAT_MODEL="deepseek-chat"'
            )
        if info["error"]:
            hints.append(f"接口调用失败：{info['error']}")
        if info["server_up"] and not info["model_present"]:
            # 仅提醒，不据此判定不可用：不少网关的 /models 不返回完整列表，
            # 但按名字调用依然成功（实测 deepseek 中转就是这种情况）。
            hints.append(
                f"提示：/models 未列出 {cfg.model}"
                f"（返回的是：{', '.join(info['models'][:6]) or '空'}）。"
                "若调用报 404 再换模型名，否则可以忽略。"
            )
        return hints

    # ---- 本地 Ollama ----
    if not info["installed"]:
        return [
            "没有检测到 Ollama —— 不是“没启动”，是根本没安装。二选一：",
            "  A. 命令行安装（推荐，装完重开一个终端）：\n"
            "       winget install Ollama.Ollama",
            "  B. 手动下载安装包：https://ollama.com/download/OllamaSetup.exe",
            "  装好后执行：ollama pull qwen2.5:7b（约 4.7GB）",
            "不想下载 5GB 也可以走云端 API（OpenAI 兼容）：\n"
            '       $env:RAG_LLM_API_KEY="你的key"\n'
            '       $env:RAG_LLM_BASE_URL="https://api.deepseek.com/v1"\n'
            '       $env:RAG_CHAT_MODEL="deepseek-chat"',
        ]

    if not info["server_up"]:
        exe = info["exe"] or "ollama"
        return [
            "已安装 Ollama，但服务没在跑。任选其一：",
            f'  1. 新开一个终端：& "{exe}" serve',
            "  2. 直接执行：python run_eval.py --start-ollama",
            "  3. 从开始菜单启动 “Ollama”（托盘出现羊驼图标即可）",
        ]

    if not info["model_present"]:
        return [
            f"服务在跑，但没有模型 {cfg.model}。拉取：",
            f"    ollama pull {cfg.model}",
            f"  当前已有：{', '.join(info['models']) or '（空）'}",
            "  也可以直接指定已有模型：python run_eval.py --judge-model <模型名>",
        ]

    return []


def start_ollama_server(cfg: Optional[LLMConfig] = None, wait_seconds: int = 90) -> bool:
    """后台拉起 ``ollama serve`` 并等待端口可用（给 --start-ollama 用）。"""
    import subprocess
    import time

    cfg = cfg or resolve_llm_config()
    if not cfg.is_ollama:
        return False
    if list_ollama_models(cfg.base_url):
        return True

    exe = find_ollama_executable()
    if exe is None:
        return False

    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
        subprocess, "DETACHED_PROCESS", 0
    )
    try:
        subprocess.Popen(  # noqa: S603 本地已知可执行文件
            [str(exe), "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )
    except OSError:
        return False

    for _ in range(wait_seconds):
        time.sleep(1)
        if list_ollama_models(cfg.base_url):
            return True
    return False


# ---------------------------------------------------------------------------
# 文本切分（中文友好，不依赖任何第三方库）
# ---------------------------------------------------------------------------

def split_text(text: str, min_size: int = 500, max_size: int = 800) -> List[str]:
    """把文本切分为长度尽量落在 [min_size, max_size] 区间的块。

    1. 细分：先按空行切段落；超长段落按句末标点切句；单句仍超长则硬切；
    2. 贪心打包：不断累进片段，直到再加一个就超过 max_size；
       - 当前块已达 min_size：保存，用新片段开新块；
       - 当前块未达 min_size：从新片段截取剩余空间凑满后保存；
    3. 末尾不足 min_size 的残余作为最后一块保留。
    """
    import re

    if not text or not text.strip():
        return []

    segments: List[str] = []
    for paragraph in re.split(r"\n{2,}", text):  # 按空行切分段落
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_size:
            segments.append(paragraph)
            continue
        for sentence in re.split(r"(?<=[。！？!?])", paragraph):  # 句末标点切句
            sentence = sentence.strip()
            if not sentence:
                continue
            while len(sentence) > max_size:  # 单句仍超长时硬切
                segments.append(sentence[:max_size])
                sentence = sentence[max_size:]
            if sentence:
                segments.append(sentence)

    chunks: List[str] = []
    buffer = ""
    for segment in segments:
        candidate = f"{buffer}\n{segment}" if buffer else segment
        if len(candidate) <= max_size:
            buffer = candidate
            continue

        if len(buffer) >= min_size:
            chunks.append(buffer)
            buffer = segment  # 片段本身不超过 max_size
        else:
            take = max_size - len(buffer) - 1  # 减去 1 个换行符
            chunks.append(f"{buffer}\n{segment[:take]}")
            buffer = segment[take:].lstrip()

    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


def build_prompt(question: str, contexts: List[str], max_chunks: Optional[int] = None) -> str:
    """把检索到的上下文拼成带编号的提示词。"""
    limit = max_chunks if max_chunks is not None else MAX_CONTEXT_CHUNKS
    blocks = [f"[{idx}] {text.strip()}" for idx, text in enumerate(contexts[:limit], start=1)]
    joined = "\n\n".join(blocks) if blocks else "（没有检索到任何相关资料）"
    return (
        f"【参考资料】\n{joined}\n\n"
        f"【问题】\n{question.strip()}\n\n"
        "请严格依据上面的参考资料作答；资料里没有的内容不要编造。"
    )


# ---------------------------------------------------------------------------
# 评测集读取（兼容记事本存成 ANSI/GBK 的情况）
# ---------------------------------------------------------------------------

def decode_text_file(raw: bytes) -> str:
    """按 UTF-8(含 BOM) → GBK → UTF-16 依次尝试解码。"""
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "gbk", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "unknown", raw, 0, len(raw), f"已尝试 UTF-8 / GBK / UTF-16 均失败：{last_error}"
    )


def load_eval_set(path: Path) -> List[Dict[str, str]]:
    """读取评测问题集：JSON 数组，元素形如 ``{"question": ..., "ground_truth": ...}``。"""
    if not path.exists():
        raise FileNotFoundError(
            f"找不到评测问题集: {path}\n"
            "请先创建 eval_set.json（格式见 README 或项目自带示例）。"
        )

    text = decode_text_file(path.read_bytes())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"无法解析 {path.name}：{exc}\n请确认它是合法 JSON，且用 UTF-8 保存。"
        ) from exc

    if isinstance(data, dict):  # 兼容 {"questions": [...]} 写法
        data = data.get("questions", [])

    items: List[Dict[str, str]] = []
    for idx, row in enumerate(data, start=1):
        if isinstance(row, str):
            row = {"question": row}
        question = (row.get("question") or "").strip()
        if not question:
            raise ValueError(f"{path.name} 第 {idx} 条缺少 question 字段")
        items.append(
            {
                "question": question,
                "ground_truth": (row.get("ground_truth") or "").strip(),
                # 负样本：知识库里本来就没有答案，用来检验模型会不会编
                "expect_refusal": bool(row.get("expect_refusal", False)),
            }
        )
    if not items:
        raise ValueError(f"{path.name} 是空的")
    return items
