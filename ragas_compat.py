"""ragas 0.4.3 与 langchain-community >= 0.4 的导入兼容补丁。

问题现象
--------
只要机器上装的是 langchain-community >= 0.4（本项目实测 0.4.2），执行
`import ragas` 就会直接崩溃：

    ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'

原因
----
ragas/llms/base.py 第 12 行在模块顶层无条件执行
    from langchain_community.chat_models.vertexai import ChatVertexAI
而 langchain-community 0.4 已经把 vertexai 这个“顺带打包”的模块删掉了
（迁移到独立包 langchain-google-vertexai），于是 ragas 导入即失败。

补救
----
在 import ragas 之前，把一个占位模块塞进 sys.modules，让这行 import 成功。
占位类只在 ragas 的类型判断分支里被引用，本项目不使用 VertexAI，因此安全。

注意：必须在任何 `import ragas` 之前导入本模块（本文件 import 时自动生效）。
"""

from __future__ import annotations

import sys
import types

_MODULE_NAME = "langchain_community.chat_models.vertexai"
_APPLIED = False


def apply() -> bool:
    """安装占位模块；返回 True 表示本次真的做了替换。"""
    global _APPLIED
    if _APPLIED or _MODULE_NAME in sys.modules:
        _APPLIED = True
        return False

    placeholder = types.ModuleType(_MODULE_NAME)
    placeholder.__doc__ = "占位模块，用于修复 ragas 0.4.x 对已删除模块的导入。"

    class ChatVertexAI:  # noqa: D401 仅作为类型占位
        """占位类：只为满足 ragas 里的 isinstance / 类型注解。"""

    placeholder.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
    sys.modules[_MODULE_NAME] = placeholder
    _APPLIED = True
    return True


apply()

__all__ = ["apply"]
