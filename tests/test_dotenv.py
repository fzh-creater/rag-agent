"""`.env` 自动加载的单元测试。

真实场景：用户在 .env 里配一次 key，之后每个新终端都能直接跑评测，
不必每次 `$env:RAG_LLM_API_KEY=...`。配置错了也不该静默生效——所以
「已存在的环境变量优先」这条必须锁住。
"""

import os

import core
from core import load_dotenv


def test_missing_file_is_noop(tmp_dir):
    assert load_dotenv(tmp_dir / "nope.env") == 0


def test_parses_simple_pairs(tmp_dir, clean_env):
    path = tmp_dir / ".env"
    path.write_text(
        "# 注释行\n\nRAG_LLM_API_KEY=sk-abc\nRAG_CHAT_MODEL=deepseek-chat\n",
        encoding="utf-8",
    )
    assert load_dotenv(path) == 2
    assert os.environ.get("RAG_LLM_API_KEY") == "sk-abc"
    assert os.environ.get("RAG_CHAT_MODEL") == "deepseek-chat"


def test_quotes_are_stripped(tmp_dir, clean_env):
    path = tmp_dir / ".env"
    path.write_text('RAG_LLM_BASE_URL="https://api.deepseek.com/v1"\n', encoding="utf-8")
    load_dotenv(path)
    assert os.environ.get("RAG_LLM_BASE_URL") == "https://api.deepseek.com/v1"


def test_existing_env_wins(tmp_dir, clean_env):
    clean_env.setenv("RAG_CHAT_MODEL", "from-shell")
    path = tmp_dir / ".env"
    path.write_text("RAG_CHAT_MODEL=from-dotenv\n", encoding="utf-8")
    load_dotenv(path)
    assert os.environ.get("RAG_CHAT_MODEL") == "from-shell"


def test_bom_and_crlf_are_tolerated(tmp_dir, clean_env):
    # Windows 记事本默认带 BOM + CRLF
    path = tmp_dir / ".env"
    path.write_bytes("\ufeffRAG_CHAT_MODEL=qwen2.5:7b\r\n".encode("utf-8"))
    load_dotenv(path)
    assert os.environ.get("RAG_CHAT_MODEL") == "qwen2.5:7b"


def test_broken_lines_are_skipped(tmp_dir, clean_env):
    path = tmp_dir / ".env"
    path.write_text("这不是配置\n=没有键\nRAG_CHAT_MODEL=ok\n", encoding="utf-8")
    load_dotenv(path)
    assert os.environ.get("RAG_CHAT_MODEL") == "ok"


def test_module_import_loads_project_dotenv(clean_env):
    """core 被 import 时应当尝试加载仓库根目录的 .env（存在才生效）。"""
    assert callable(core.load_dotenv)
