"""定义可视化持久化、渲染和安全停止请求的异常类型。"""

class VisualizationError(RuntimeError):
    """仪表盘状态无法持久化或渲染时抛出。"""


class TrainingStopRequested(RuntimeError):
    """适配器要求训练在安全边界停止时抛出。"""
