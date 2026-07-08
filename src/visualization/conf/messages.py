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
    "off": "已关闭",
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

VIEW_KIND_NAMES: dict[str, str] = {
    "startup": "启动准备",
    "training": "正式训练",
}

PAGE_NAMES: dict[str, str] = {
    "overview": "概览",
    "parameters": "参数",
    "tests": "测试",
    "scores": "评分",
    "resources": "资源",
    "events": "事件",
}

DISPLAY_TEXT_NAMES: dict[str, str] = {
    "stage": "阶段",
    "status": "状态",
    "score": "评分",
    "loss": "损失",
    "trial": "试验",
    "trial_id": "试验",
    "level": "等级",
    "phase": "流程阶段",
    "progress": "进度",
    "warning": "警告",
    "warnings": "警告",
    "device": "设备",
    "split": "数据划分",
    "path": "路径",
    "checkpoint_path": "检查点",
    "N/A": "不适用",
    "GPU bridge": "GPU 桥接检查",
    "NVML": "NVML 监控",
    "nvidia-smi": "nvidia-smi 监控",
    "host-exec nvidia-smi": "host-exec 桥接监控",
    "训练 readiness": "训练就绪检查",
    "最终 readiness": "最终就绪检查",
    "target_readiness": "目标就绪检查",
    "full_training": "完整训练",
    "target": "目标阶段",
    "level_a": "等级 A",
    "level_b": "等级 B",
    "level_c": "等级 C",
    "single_point": "单点",
    "multi_point": "多点",
    "slider": "滑条",
    "point_slider": "点滑条",
    "spinner": "转盘",
    "long_sequence": "长序列",
    "single_click": "单点检测",
    "multi_slider": "多滑条",
    "split 构建或校验": "数据划分构建或校验",
    "schema 与数据质量检查": "数据结构与质量检查",
    "真实 checkpoint 恢复": "真实检查点恢复",
    "inheritance 继承包": "继承状态打包",
    "checkpoint": "检查点",
    "passed": "通过",
    "failed": "失败",
}

DISPLAY_TEXT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("Level ", "等级 "),
    ("level_", "等级 "),
)

DISPLAY_TEXT_FRAGMENTS: tuple[tuple[str, str], ...] = (
    ("GPU bridge", "GPU 桥接检查"),
    ("host-exec", "host-exec 桥接"),
    ("nvidia-smi", "nvidia-smi 监控"),
    ("NVML", "NVML 监控"),
    ("Failed to initialize NVML", "NVML 初始化失败"),
    ("NVIDIA-SMI has failed", "nvidia-smi 启动失败"),
    ("readiness", "就绪检查"),
    ("checkpoint", "检查点"),
    ("inheritance", "继承状态"),
    ("split", "数据划分"),
    ("schema", "数据结构"),
    ("session", "会话"),
    ("gate", "门禁"),
    ("data-check failed", "数据检查失败"),
    ("RampGateError", "渐进训练门禁错误"),
    ("RuntimeError", "运行时错误"),
    ("ValueError", "取值错误"),
    ("IndexError", "索引错误"),
    ("TypeError", "类型错误"),
    ("OSError", "系统错误"),
    ("FileNotFoundError", "文件未找到错误"),
    ("PermissionError", "权限错误"),
    ("CalledProcessError", "子进程错误"),
    ("AssertionError", "断言错误"),
    ("quality below ASHA prune floor", "质量分低于 ASHA 淘汰下限"),
    ("quality score", "质量分"),
    ("below pass threshold", "低于通过阈值"),
    ("spatial steps did not reach requested level", "空间训练步数未达到要求等级"),
    ("temporal steps did not reach requested level", "时序训练步数未达到要求等级"),
    ("gallery is empty or not saved", "图集为空或未保存"),
    ("score report missing", "评分报告缺失"),
    ("next job missing", "下一次训练任务缺失"),
    ("run-job --dry-run failed", "训练任务 dry-run 失败"),
    ("artifact validation failed", "模型产物校验失败"),
    ("artifact smoke produced non-finite outputs", "模型产物 smoke 测试输出非有限值"),
    (
        "CUDA is not visible; run ramp-to-full through host-exec",
        "CUDA 不可见，请通过 host-exec 运行 ramp-to-full",
    ),
    (
        "less than 10 GiB free space for ramp outputs",
        "渐进训练输出目录剩余空间不足 10 GiB",
    ),
    ("full_checks failed; see", "完整检查失败，日志："),
)

MESSAGE_TEMPLATES: dict[str, str] = {
    "dashboard_started": "训练控制台已启动：{run_id}",
    "dashboard_stopped": "训练控制台已停止：{status}",
    "stage_started": "{stage}开始",
    "stage_finished": "{stage}{status}",
    "stage_lifecycle_started": "阶段开始：{stage}",
    "stage_lifecycle_passed": "阶段完成：{stage}（{status}）",
    "stage_lifecycle_warning": "阶段警告：{stage}（{status}）",
    "stage_lifecycle_skipped": "阶段跳过：{stage}",
    "stage_lifecycle_failed": "阶段失败：{stage}（{status}）",
    "raw_data_unchanged": "原始数据未发生变化，无需重新转换",
    "score_updated": "当前评分更新为 {score}",
    "best_score_updated": "新全局最高分：{score}",
    "checkpoint_saved": "已保存检查点：{path}",
    "gallery_saved": "评估图集已生成：{path}",
    "artifact_validated": "模型产物校验完成：{path}",
    "resource_warning": "警告：GPU 显存使用率达到 {ratio}",
    "resource_critical": "严重警告：GPU 显存接近上限，当前 {reserved}/{total} GB",
    "resource_monitor_warning": "GPU 监控暂不可用：{error}",
    "resource_monitor_restored": "GPU 监控已恢复：{source}",
    "inheritance_saved": "已生成继承状态：{path}",
    "inheritance_loaded": "下一次训练继承成功：{path}",
    "inheritance_downgraded": "训练继承已降级：{reason}",
    "dataset_exhausted": "训练集已用完，当前训练无法继续",
    "user_interrupted": "用户中断，正在保存可继承状态",
    "fatal_error": "不可恢复错误：{error}",
}


def display_status(value: object) -> str:
    text = _text(value)
    return STATUS_NAMES.get(text, text or "无")


def display_pipeline_phase(value: object) -> str:
    text = _text(value)
    return PIPELINE_PHASE_NAMES.get(text, text or "无")


def display_grade(value: object) -> str:
    text = _text(value)
    return GRADE_NAMES.get(text, STATUS_NAMES.get(text, text or "未评级"))


def display_view_kind(value: object) -> str:
    text = _text(value)
    return VIEW_KIND_NAMES.get(text, text or "未知视图")


def display_page(value: object) -> str:
    text = _text(value)
    return PAGE_NAMES.get(text, display_text(text))


def display_text(value: object) -> str:
    text = _text(value)
    if not text:
        return "无"
    if text in DISPLAY_TEXT_NAMES:
        return DISPLAY_TEXT_NAMES[text]
    if text in STATUS_NAMES:
        return STATUS_NAMES[text]
    if text in PIPELINE_PHASE_NAMES:
        return PIPELINE_PHASE_NAMES[text]
    if text in GRADE_NAMES:
        return GRADE_NAMES[text]
    if _looks_like_ramp_level(text):
        return f"等级 {text.upper()}"
    for prefix, replacement in DISPLAY_TEXT_PREFIXES:
        if text.startswith(prefix):
            return f"{replacement}{text.removeprefix(prefix).upper()}"
    translated = text
    for fragment, replacement in DISPLAY_TEXT_FRAGMENTS:
        translated = translated.replace(fragment, replacement)
    return translated


def render_message(message_key: str, args: Mapping[str, object] | None = None) -> str:
    template = MESSAGE_TEMPLATES.get(message_key, message_key)
    try:
        return template.format(**_message_args(args or {}))
    except Exception:
        return template


def _message_args(args: Mapping[str, object]) -> dict[str, object]:
    translated: dict[str, object] = {}
    for key, value in args.items():
        if key in {
            "status",
            "stage",
            "phase",
            "grade",
            "level",
            "reason",
            "error",
            "source",
        }:
            translated[key] = display_text(value)
        else:
            translated[key] = value
    return translated


def _text(value: object) -> str:
    return str(value) if value is not None else ""


def _looks_like_ramp_level(text: str) -> bool:
    if text in {"a", "b"}:
        return True
    return len(text) <= 4 and text.startswith("c") and text[1:].isdigit()
