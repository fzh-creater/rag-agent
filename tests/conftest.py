import shutil
import sys
import uuid
from pathlib import Path

import pytest

# 让测试可以直接 import 仓库根目录的 core / rag_engine
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TMP_ROOT = ROOT / "tests" / ".tmp"

# 会干扰配置解析的环境变量，逐个清掉
_ENV_KEYS = (
    "RAG_LLM_PROVIDER",
    "RAG_LLM_BASE_URL",
    "RAG_LLM_API_KEY",
    "RAG_CHAT_MODEL",
    "RAG_JUDGE_MODEL",
    "RAG_OLLAMA_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "RAG_LLM_TIMEOUT",
)


@pytest.fixture()
def clean_env(monkeypatch):
    """隔离外部环境变量，保证配置解析测试可复现。"""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


@pytest.fixture()
def tmp_dir():
    """临时目录。

    刻意不用 pytest 的 ``tmp_path``：它底层走 ``tempfile.mkdtemp``，
    会把目录权限设为 0o700（Windows 上体现为一条限制性 ACL），
    在受限环境里连自己都读不了。这里用普通 ``mkdir``，权限走继承。
    """
    path = _TMP_ROOT / uuid.uuid4().hex[:8]
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
