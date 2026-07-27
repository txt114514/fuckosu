"""终端仪表盘渲染器的公开入口。"""

from visualization.core.renderers.plain_renderer import PlainDashboardRenderer
from visualization.core.renderers.rich_renderer import RichDashboardRenderer

__all__ = ["PlainDashboardRenderer", "RichDashboardRenderer"]
