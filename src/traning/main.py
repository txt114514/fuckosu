"""训练包的公开命令入口。"""

from traning.core.app.cli import app


def main() -> None:
    """运行 canonical 训练命令行。"""

    app()


__all__ = ["app", "main"]
