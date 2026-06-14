"""Prefect task wrappers for the training workflow."""


def require_success(stage: str, success: bool) -> bool:
    if not success:
        raise RuntimeError(f"{stage} 存在失败项，详情见处理状态数据库")
    return True


__all__ = [
    "av",
    "clip",
    "difficulty",
    "importer",
    "match",
    "require_success",
    "segment",
    "verify",
]
