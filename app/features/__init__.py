"""可选功能开关；不在这里导入功能实现。"""
import os


def game2048_enabled():
    """默认按需可用；设为 0 关闭，只有 1 表示启用。"""
    return os.environ.get("MIE_ENABLE_2048", "1").strip() == "1"
