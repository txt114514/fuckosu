"""构造无需文件 I/O 的训练默认设置，供兼容调用方复用。"""

from traning.conf.settings import Settings


DEFAULT_SETTINGS = Settings()

__all__ = ["DEFAULT_SETTINGS"]
