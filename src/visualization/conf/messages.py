from __future__ import annotations

from typing import Mapping


STATUS_NAMES: dict[str, str] = {
    "pending": "等待中",
    "scanning": "正在扫描",
    "converting": "正在转换",
    "checking": "正在检查",
    "running": "正在运行",
    "passed": "已通过",
    "warning": "存在警告",
    "failed": "已失败",
    "skipped": "已跳过",
    "interrupted": "已中断",
    "completed": "已完成",
    "training": "训练中",
    "evaluating": "评估中",
    "promoted": "已晋升",
    "promotable": "满足晋升条件",
    "pruned": "已淘汰",
    "continue": "继续训练",
    "stopped": "已停止",
}

PIPELINE_PHASE_NAMES: dict[str, str] = {
    "startup": "启动",
    "data_preparation": "数据准备",
    "pretrain_check": "训练前检查",
    "progressive_preparation": "渐进训练准备",
    "training": "正式训练",
    "completed": "已完成",
    "failed": "已失败",
}

SEVERITY_NAMES: dict[str, str] = {
    "info": "信息",
    "success": "成功",
    "warning": "警告",
    "error": "错误",
    "critical": "严重错误",
}

GRADE_NAMES: dict[str, str] = {
    "unrated": "未评级",
    "observing": "观察中",
    "reached": "达到等级",
    "recheck": "等待复评",
    "promotable": "满足晋升条件",
    "promoted": "已晋升",
    "continue": "继续训练",
    "near_prune": "接近淘汰",
    "pruned": "已淘汰",
    "stopped": "因异常停止",
}

MESSAGE_TEMPLATES: dict[str, str] = {
    "dashboard_started": "训练控制台已启动：{run_id}",
    "dashboard_stopped": "训练控制台已停止：{status}",
    "stage_started": "{stage}开始",
    "stage_finished": "{stage}{status}",
    "raw_data_unchanged": "原始数据未发生变化，无需重新转换",
    "score_updated": "当前评分更新为 {score}",
    "best_score_updated": "新全局最高分：{score}",
    "checkpoint_saved": "已保存检查点：{path}",
    "gallery_saved": "Gallery 已生成：{path}",
    "artifact_validated": "Artifact 校验完成：{path}",
    "resource_warning": "警告：GPU 显存使用率达到 {ratio}",
    "resource_critical": "严重警告：GPU 显存接近上限，当前 {reserved}/{total} GB",
    "inheritance_saved": "已生成继承状态：{path}",
    "inheritance_loaded": "下一次训练继承成功：{path}",
    "inheritance_downgraded": "训练继承已降级：{reason}",
    "dataset_exhausted": "训练集已用完，当前训练无法继续",
    "user_interrupted": "用户中断，正在保存可继承状态",
    "fatal_error": "不可恢复错误：{error}",
}


def render_message(message_key: str, args: Mapping[str, object] | None = None) -> str:
    template = MESSAGE_TEMPLATES.get(message_key, message_key)
    try:
        return template.format(**dict(args or {}))
    except Exception:
        return template
