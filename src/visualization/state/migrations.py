CURRENT_DASHBOARD_STATE_VERSION = "dashboard-state-v1"


def migrate_dashboard_state(raw: dict) -> dict:
    if raw.get("schema_version") == CURRENT_DASHBOARD_STATE_VERSION:
        return raw
    migrated = dict(raw)
    migrated["schema_version"] = CURRENT_DASHBOARD_STATE_VERSION
    migrated.setdefault("migration_log", []).append("initialized dashboard-state-v1")
    return migrated
