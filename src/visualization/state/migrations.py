"""将历史仪表盘状态提升到当前可读取的结构版本。"""

CURRENT_DASHBOARD_STATE_VERSION = "dashboard-state-v1"


def migrate_dashboard_state(raw: dict) -> dict:
    """返回当前版本状态；迁移时复制输入并留下可追溯日志。"""

    if raw.get("schema_version") == CURRENT_DASHBOARD_STATE_VERSION:
        return raw
    # 不原地修改调用方持有的历史快照，避免迁移失败时破坏唯一可恢复数据。
    migrated = dict(raw)
    migrated["schema_version"] = CURRENT_DASHBOARD_STATE_VERSION
    migrated.setdefault("migration_log", []).append("initialized dashboard-state-v1")
    return migrated
