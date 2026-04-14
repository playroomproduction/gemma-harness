"""Tool package — auto-registers all tools on import."""

# Import all tool modules to trigger @register_tool decorators.
from tools import time_tool  # noqa: F401
from tools import web  # noqa: F401
from tools import filesystem  # noqa: F401
from tools import code_executor  # noqa: F401
from tools import git_tools  # noqa: F401
from tools import memory  # noqa: F401
