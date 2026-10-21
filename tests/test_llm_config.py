"""生成端点配置与分层诊断的单元测试（不联网，全部用假数据结构驱动）。"""

import pytest

from core import LLMConfig, llm_hints, resolve_llm_config


class TestResolveConfig:
    def test_explicit_provider_wins(self, clean_env):
        cfg = resolve_llm_config(
            provider="openai",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-1",
            model="deepseek-chat",
        )
        assert cfg.provider == "openai"
        assert cfg.model == "deepseek-chat"
        assert cfg.api_key == "sk-1"
        assert cfg.base_url == "https://api.deepseek.com/v1"

    def test_api_key_implies_cloud(self, clean_env):
        clean_env.setenv("RAG_LLM_API_KEY", "sk-abc")
        clean_env.setenv("RAG_CHAT_MODEL", "gpt-4o-mini")
        cfg = resolve_llm_config()
        assert cfg.provider == "openai"
        assert cfg.api_key == "sk-abc"

    def test_remote_base_url_without_key_is_cloud(self, clean_env):
        clean_env.setenv("RAG_LLM_BASE_URL", "https://api.moonshot.cn/v1")
        clean_env.setenv("RAG_CHAT_MODEL", "moonshot-v1-8k")
        cfg = resolve_llm_config()
        assert cfg.provider == "openai"

    def test_localhost_defaults_to_ollama(self, clean_env):
        clean_env.setenv("RAG_OLLAMA_BASE_URL", "http://localhost:11434/v1")
        clean_env.setenv("RAG_CHAT_MODEL", "qwen2.5:7b")
        cfg = resolve_llm_config()
        assert cfg.provider == "ollama"
        assert cfg.api_key == "ollama"  # 本地服务不校验 Key，用占位值

    def test_trailing_slash_is_normalized(self, clean_env):
        cfg = resolve_llm_config(
            provider="openai",
            base_url="https://api.deepseek.com/v1/",
            api_key="sk-1",
            model="deepseek-chat",
        )
        assert not cfg.base_url.endswith("/")

    def test_invalid_provider_raises(self, clean_env):
        with pytest.raises(ValueError, match="RAG_LLM_PROVIDER"):
            resolve_llm_config(provider="anthropic")

    def test_timeout_env_is_applied(self, clean_env):
        clean_env.setenv("RAG_LLM_TIMEOUT", "42")
        cfg = resolve_llm_config(provider="ollama", model="qwen2.5:7b")
        assert cfg.timeout == 42.0

    def test_broken_timeout_falls_back(self, clean_env):
        clean_env.setenv("RAG_LLM_TIMEOUT", "not-a-number")
        cfg = resolve_llm_config(provider="ollama", model="qwen2.5:7b")
        assert cfg.timeout == 180.0

    def test_describe_contains_three_parts(self):
        cfg = LLMConfig(provider="ollama", model="qwen2.5:7b", base_url="http://x/v1")
        described = cfg.describe()
        assert "本地 Ollama" in described
        assert "qwen2.5:7b" in described
        assert "http://x/v1" in described


def _info(**overrides):
    """构造一份 diagnose_llm 的输出结构，便于单独测提示文案。"""
    base = {
        "provider": "ollama",
        "provider_label": "本地 Ollama",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:7b",
        "installed": True,
        "exe": r"C:\Ollama\ollama.exe",
        "server_up": True,
        "models": ["qwen2.5:7b"],
        "model_present": True,
        "error": "",
    }
    base.update(overrides)
    return base


class TestHints:
    def test_ready_has_no_hints(self):
        cfg = LLMConfig(provider="ollama", model="qwen2.5:7b")
        assert llm_hints(_info(), cfg) == []

    def test_not_installed_says_install_not_start(self):
        cfg = LLMConfig(provider="ollama", model="qwen2.5:7b")
        hints = "\n".join(llm_hints(_info(installed=False, server_up=False, model_present=False), cfg))
        assert "没安装" in hints
        assert "winget install Ollama.Ollama" in hints
        # 关键回归点：以前这里只会叫人 `ollama serve`，对没装的机器毫无用处
        assert "ollama serve" not in hints

    def test_installed_but_down_suggests_serve(self):
        cfg = LLMConfig(provider="ollama", model="qwen2.5:7b")
        hints = "\n".join(llm_hints(_info(server_up=False, model_present=False), cfg))
        assert "serve" in hints
        assert "--start-ollama" in hints

    def test_model_missing_suggests_pull(self):
        cfg = LLMConfig(provider="ollama", model="qwen2.5:7b")
        hints = "\n".join(llm_hints(_info(models=["llama3:8b"], model_present=False), cfg))
        assert "ollama pull qwen2.5:7b" in hints
        assert "llama3:8b" in hints  # 提示里要列出已有模型

    def test_cloud_without_key_mentions_env_vars(self):
        cfg = LLMConfig(provider="openai", model="deepseek-chat", api_key="")
        hints = "\n".join(
            llm_hints(
                _info(provider="openai", server_up=False, model_present=False, error="401"),
                cfg,
            )
        )
        assert "RAG_LLM_API_KEY" in hints
        assert "401" in hints

    def test_cloud_ready_has_no_hints(self):
        cfg = LLMConfig(provider="openai", model="deepseek-chat", api_key="sk-1")
        info = _info(provider="openai", models=["deepseek-chat"], model_present=True)
        assert llm_hints(info, cfg) == []

    def test_model_not_listed_is_only_a_hint(self):
        """回归：网关的 /models 不列目标模型时，不能判定为不可用。

        实测某个 deepseek 中转返回的模型列表是 deepseek-flash / deepseek-v4-pro，
        但按 deepseek-chat 调用照样成功。若这里报“不可用”，整轮评测会被误降级。
        """
        cfg = LLMConfig(provider="openai", model="deepseek-chat", api_key="sk-1")
        info = _info(provider="openai", models=["deepseek-flash"], model_present=False)
        hints = "\n".join(llm_hints(info, cfg))
        assert "提示" in hints
        assert "deepseek-chat" in hints
        assert "不可用" not in hints
        assert "RAG_LLM_API_KEY" not in hints  # key 明明有，别再让人去配
