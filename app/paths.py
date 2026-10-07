"""固定应用与工作资料位置，功能模块移动时不改变用户数据路径。"""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = APP_DIR.parent / "workspace"
