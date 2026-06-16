#!/usr/bin/env python3
"""Build compact, deterministic indexes for project Python packages."""

from __future__ import annotations

import argparse
import ast
import builtins
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


WORKSPACE = Path(__file__).resolve().parents[1]
BEFORE_SOURCE_ROOT = WORKSPACE / "src" / "before_traning"
TRAINING_SOURCE_ROOT = WORKSPACE / "src" / "traning"
BEFORE_CODEX_INDEX = BEFORE_SOURCE_ROOT / "docs" / "CODEX_INDEX.md"
TRAINING_CODEX_INDEX = TRAINING_SOURCE_ROOT / "docs" / "CODEX_INDEX.md"

GENERATED_NOTICE = (
    "> 自动生成文件，请勿手工修改。运行 "
    "`python project_index/build_index.py` 重建。"
)

ROLE_OVERRIDES = {
    "main.py": "Typer CLI 入口；合并命令行覆盖项，选择 direct/Prefect runner，并渲染阶段结果。",
    "conf/settings.py": "Pydantic 配置模型与 YAML/JSON 加载；解析相对路径、切片抖动和去音频配置，并兼容旧配置层级。",
    "conf/defaults.py": "创建全局默认 Settings 实例，供兼容 core 构造器使用。",
    "conf/artifacts.py": "保存固定训练前产物文件名契约。",
    "conf/legacy_config.py": "旧 builder API 的兼容层；把 Settings 展平、覆盖并按构造函数签名转发。",
    "conf/field_groups.py": "集中声明处理器字段组，负责批量赋值和处理器之间的参数转发。",
    "conf/runtime.py": "把 Prefect home 固定到仓库内可写目录。",
    "state/manifest_schema.py": "SQLModel 训练包 manifest 与谱面解析缓存表。",
    "state/segment_schema.py": "SQLModel 视频片段数据集索引表。",
    "state/status_schema.py": "独立的 SQLModel 状态表、处理步骤规范化和 detail JSON 编解码。",
    "state/process_status.py": "按谱面文件夹读写 SQLite 处理状态，并迁移旧 process_status.json。",
    "core/beatmap/beatmap.py": "谱面阶段统一公开入口；集中导出三个处理器、阶段函数和 pipeline API。",
    "core/beatmap/importer.py": "完整谱面导入实施；扫描 .osz、更新 manifest、写文件和导入状态。",
    "core/beatmap/verify.py": "完整 verify 实施；调用 ProcessingGuard、标准谱面缓存并导出 verify.txt。",
    "core/beatmap/difficulty.py": "完整难度实施；调用 ProcessingGuard、读取难度并更新 SQLite manifest。",
    "core/beatmap/pipeline.py": "声明七阶段注册表与统一 Pipeline API，并用分组表选择谱面/视频阶段。",
    "core/video/match.py": "视频匹配业务入口；处理“全部已有视频”的正常跳过情况。",
    "core/video/av.py": "AV 对齐业务入口；把音频算法和状态参数传给 VideoAVProcessor。",
    "core/video/clip.py": "固定区域裁剪业务入口。",
    "core/video/segment.py": "最终谱面视频切分处理器；映射设置、调度分类、生成产物并更新状态。",
    "core/video/pipeline.py": "顺序组合视频匹配、AV 对齐、裁剪和谱面切分。",
    "core/audio/match.py": "重导出音频匹配处理器，作为 core 层稳定入口。",
    "core/video/matching/builders.py": "视频顺序匹配器的兼容配置 builder。",
    "core/video/matching/matching.py": "视频匹配策略入口；在音频匹配与时间顺序重命名之间切换。",
    "core/video/matching/renamer.py": "按录像时间与 manifest 顺序移动视频，并支持异常回滚。",
    "core/audio/matching/preflight.py": "同步视频匹配状态，收集待匹配文件夹和候选视频。",
    "core/audio/matching/steps.py": "调用 AV 信号 API 计算视频/歌曲配对得分并做一对一选择。",
    "core/audio/matching/wrapup.py": "应用音频匹配结果，移动视频、回写状态并支持回滚。",
    "core/audio/matching/matching.py": "组合音频匹配处理器，并注入 AV 对齐算法能力。",
    "core/video/av_processing/preflight.py": "校验 AV 参数/状态步骤并定位阶段输入输出。",
    "core/video/av_processing/steps.py": "执行单文件夹 AV 对齐阶段和状态推进。",
    "core/video/av_processing/wrapup.py": "记录 AV 阶段进度、完成细节和失败状态。",
    "core/video/av_processing/av_processing.py": "组合 AV 处理器并初始化配置、存储和状态依赖。",
    "core/video/clipping/preflight.py": "校验裁剪状态步骤和单文件夹前置条件。",
    "core/video/clipping/steps.py": "调用通用 crop_video API 执行单文件夹原地裁剪。",
    "core/video/clipping/wrapup.py": "记录裁剪进度、实际坐标和失败状态。",
    "core/video/clipping/clipping.py": "组合固定区域裁剪处理器并校验阶段配置。",
    "Lib/common/pathspec.py": "统一后缀到 gitwildmatch PathSpec 的转换与文件过滤。",
    "Lib/common/sequence.py": "统一生成带定宽数字的稳定序列名称。",
    "Lib/common/batch.py": "配置规格辅助函数与文件夹批处理模板。",
    "Lib/common/failures.py": "统一提取异常类型、报错函数和模块，并生成状态 detail 与控制台文本。",
    "Lib/common/processing.py": "通用目录/文件检查、前置步骤检查、完成态对齐和失败状态回写 API。",
    "Lib/tasks/tasks.py": "通用 task 规格、注册器和循环 Prefect task 生成 API。",
    "Lib/tasks/flows.py": "通用 direct/Prefect 循环执行 Pipeline API 与构建函数。",
    "Lib/tools/ffmpeg.py": "提供 ffmpeg/ffprobe 参数构造与音频提取、裁切、默认去音频分段和裁剪高层 API。",
    "Lib/beatmap/manifest.py": "SQLite manifest 仓储；管理内部目录、谱面缓存和可读对照表。",
    "Lib/beatmap/order.py": "旧 OrderFolderWalker 的兼容导出；业务代码使用 ManifestFolderWalker。",
    "Lib/beatmap/package.py": "通过 SQLite manifest 创建和同步允许使用的内部谱面目录。",
    "Lib/beatmap/folder_store.py": "受 manifest 约束的源文件读写、输出目录创建和原子目录替换。",
    "Lib/beatmap/osu_metadata.py": "从 .osu 指定 section 读取 AudioFilename 和 OverallDifficulty。",
    "Lib/beatmap/osu_parser.py": "解析 .osu sections、timing points 和 HitObjects，并生成结构化对象。",
    "Lib/beatmap/osz.py": "解压单个 .osz 并读取目标 .osu 与音频字节。",
    "Lib/beatmap/standard.py": "解析或从 manifest SQLite 缓存读取完整 osu!standard 谱面。",
    "Lib/beatmap/hit_objects.py": "Circle、Slider、Spinner 的轻量数据模型。",
    "Lib/beatmap/timing_points.py": "osu 原始 timing point 数据模型。",
    "Lib/video/av_processing/steps.py": "可复用 AV 信号算法：采样、粗细相关、hit 校正和裁切窗口计算。",
    "Lib/video/clipping/geometry.py": "按参考分辨率缩放裁剪矩形，并校验边界和编码偶数尺寸。",
    "Lib/video/segment_dataset.py": "用 SQLite 管理视频片段索引、导出 CSV 并校验数据集文件完整性。",
    "Lib/video/segmentation/planner.py": "构建对象恰好归属一次的原子片段，支持稳定前置时间抖动，并将完整原子片段组合为长序列维度。",
    "Lib/video/segmentation/segmentation.py": "根据显式参数调用 planner，返回原子与长序列计划集合。",
}

TRAINING_ROLE_OVERRIDES = {
    "main.py": "Typer CLI；执行数据检查、样本预览和训练阶段注册表。",
    "cli.py": "python -m 入口；复用 main.py 中的同一个 Typer app。",
    "conf/settings.py": "训练配置模型与 YAML 加载；解析数据集路径、item 划分、点击频率上限并校验采样和分块参数。",
    "conf/defaults.py": "创建默认训练 Settings。",
    "core/pipeline.py": "声明训练阶段注册表；当前登记 data_input，后续扩展空间、时序和导出阶段。",
    "core/env_check.py": "收集 Python、PyTorch/CUDA、GPU、FFmpeg 和关键依赖状态，供 CLI 环境检查使用。",
    "core/data_input/data_input.py": "数据输入模块公开门面；提供检查、Dataset 和 DataLoader。",
    "core/data_input/preflight.py": "扫描训练片段并生成数量、类别、维度和问题报告。",
    "core/data_input/loader.py": "把配置映射为 SegmentFrameDataset 与 PyTorch DataLoader。",
    "Lib/data/annotation.py": "beatmap.json 的 Pydantic 契约和按帧可见 HitObject 筛选。",
    "Lib/data/discovery.py": "发现 video.mp4 与 beatmap.json 配对并构建稳定片段记录。",
    "Lib/data/dataset.py": "按片段帧索引解码原分辨率 RGB Tensor 和可变长度标签。",
    "Lib/data/video_reader.py": "带有限打开文件缓存的 OpenCV 视频帧读取器。",
    "Lib/data/tiling.py": "构建覆盖完整画面的重叠 patch 窗口并返回 Tensor 视图。",
    "Lib/data/sampling.py": "根据片段时长、FPS 和步长建立帧引用表。",
    "Lib/data/collate.py": "组装图像批次并保留可变长度样本元数据。",
    "Lib/metrics/scoring.py": "实现点与 slider 的空间、时间、1.5x 膨胀路径覆盖和组合评分。",
    "Lib/metrics/sequence.py": "按点击时间模拟目标一次性命中、重叠目标递进、最小点击间隔限制和错误归因。",
    "Lib/visualization/render.py": "把帧 Tensor、osu 标签和共享坐标变换渲染为标注图片。",
    "Lib/visualization/selection.py": "根据 HitObject 起始时间反推最接近的采样帧。",
    "Lib/visualization/display.py": "通过独立 ffplay 子进程把标注图片显示到主机 X11。",
    "Lib/visualization/gallery.py": "选择批次最高分 trial，并按通过状态和六个子项目随机保存标注帧图集。",
    "Lib/visualization/output_identity.py": "为 traning_example 输出分配进程安全的递增次数和 UTC 时间标识。",
    "core/visualization/service.py": "可选可视化故障隔离、一次性告警和训练步频率控制。",
    "core/visualization/preview.py": "组装 Dataset、单帧点击标注和批次最佳参数图集。",
    "state/run_state.py": "保存 trial、课程阶段、rung、预算和全局步数的运行状态。",
    "state/checkpoint_schema.py": "检查点 lineage 与模型/优化器/scheduler/AMP 恢复契约。",
    "state/experiment_schema.py": "三层参数、TPE/随机来源、ASHA trial、课程和独立评估契约。",
    "state/gallery_schema.py": "批次 trial 分数、稳定帧引用、错误归因和最佳参数图集输入契约。",
}

SUMMARY_OVERRIDES = {
    "__init__": "初始化实例依赖、配置和运行状态。",
    "__post_init__": "完成 dataclass 初始化后的派生字段设置。",
    "main": "独立脚本入口，构建处理器并执行。",
    "run": "执行该处理器的完整工作流。",
    "process_one": "处理 manifest 中的单个内部谱面文件夹。",
    "progress_message": "生成当前批处理进度文本。",
    "handle_failure": "处理单文件夹失败并同步失败状态。",
    "from_settings": "从 Settings 创建处理器实例。",
}

PREFIX_SUMMARIES = (
    ("build_", "构建并返回"),
    ("_build_", "构建"),
    ("load_", "加载"),
    ("_load_", "加载"),
    ("read_", "读取"),
    ("_read_", "读取"),
    ("write_", "写入"),
    ("_write_", "写入"),
    ("parse_", "解析"),
    ("_parse_", "解析"),
    ("normalize_", "规范化"),
    ("_normalize_", "规范化"),
    ("validate_", "校验"),
    ("_validate_", "校验"),
    ("ensure_", "确保"),
    ("_ensure_", "确保"),
    ("resolve_", "解析并定位"),
    ("_resolve_", "解析并定位"),
    ("sync_", "同步"),
    ("_sync_", "同步"),
    ("mark_", "更新状态为"),
    ("_mark_", "更新状态为"),
    ("estimate_", "估算"),
    ("_estimate_", "估算"),
    ("extract_", "提取"),
    ("_extract_", "提取"),
    ("list_", "列出"),
    ("_list_", "列出"),
    ("filter_", "筛选"),
    ("_filter_", "筛选"),
    ("apply_", "应用"),
    ("_apply_", "应用"),
    ("select_", "选择"),
    ("_select_", "选择"),
    ("import_", "导入"),
    ("export_", "导出"),
    ("match_", "匹配"),
    ("crop_", "裁剪"),
    ("prepare_", "顺序准备"),
    ("get_", "获取"),
    ("_get_", "获取"),
    ("is_", "判断是否"),
    ("_is_", "判断是否"),
)

READ_CALLS = {
    "json.load",
    "read",
    "read_bytes",
    "read_text",
    "wavfile.read",
}
WRITE_CALLS = {
    "mkdir",
    "open",
    "rename",
    "replace",
    "touch",
    "unlink",
    "write",
    "write_bytes",
    "write_text",
}
PROCESS_CALLS = {
    "_run_command",
    "run_ffmpeg",
    "subprocess.run",
}
DATABASE_CALLS = {
    "Session",
    "create_engine",
    "session.add",
    "session.commit",
    "session.exec",
}
BUILTIN_NAMES = set(dir(builtins))


@dataclass(frozen=True)
class Symbol:
    path: Path
    qualname: str
    short_name: str
    kind: str
    line: int
    end_line: int
    signature: str
    decorators: tuple[str, ...]
    bases: tuple[str, ...]
    doc: str
    calls: tuple[str, ...]
    tags: tuple[str, ...]


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return dotted_name(node.func)
    return ""


def first_doc_line(node: ast.AST) -> str:
    doc = ast.get_docstring(node, clean=True) or ""
    return doc.splitlines()[0].strip() if doc else ""


def format_arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    rendered = ast.unparse(node.args)
    return_annotation = (
        f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    )
    async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return f"{async_prefix}({rendered}){return_annotation}"


class CallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_name(node.func)
        if name:
            self.calls.add(name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


def collect_calls(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    collector = CallCollector()
    for statement in node.body:
        collector.visit(statement)
    return tuple(sorted(collector.calls))


def tags_for(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    decorators: Iterable[str],
    calls: Iterable[str],
) -> tuple[str, ...]:
    tags: set[str] = set()
    decorator_text = " ".join(decorators)
    call_set = set(calls)
    call_leaves = {name.rsplit(".", 1)[-1] for name in call_set}

    if "app.command" in decorator_text or "app.callback" in decorator_text:
        tags.add("CLI")
    if "flow" in decorator_text:
        tags.add("PREFECT-FLOW")
    if "task" in decorator_text:
        tags.add("PREFECT-TASK")
    if "validator" in decorator_text:
        tags.add("VALIDATOR")
    if "property" in decorator_text:
        tags.add("PROPERTY")
    if call_set & READ_CALLS or call_leaves & READ_CALLS:
        tags.add("IO-R")
    if call_set & WRITE_CALLS or call_leaves & WRITE_CALLS:
        tags.add("IO-W")
    if call_set & PROCESS_CALLS or call_leaves & PROCESS_CALLS:
        tags.add("PROCESS")
    if call_set & DATABASE_CALLS or call_leaves & DATABASE_CALLS:
        tags.add("DB")
    if isinstance(node, ast.ClassDef):
        tags.add("CLASS")
    return tuple(sorted(tags))


class SymbolVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stack: list[tuple[str, str]] = []
        self.symbols: list[Symbol] = []

    def _qualname(self, name: str) -> str:
        return ".".join([item[0] for item in self.stack] + [name])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        decorators = tuple(dotted_name(item) or ast.unparse(item) for item in node.decorator_list)
        bases = tuple(ast.unparse(base) for base in node.bases)
        self.symbols.append(
            Symbol(
                path=self.path,
                qualname=self._qualname(node.name),
                short_name=node.name,
                kind="class",
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                signature="",
                decorators=decorators,
                bases=bases,
                doc=first_doc_line(node),
                calls=(),
                tags=tags_for(node, decorators, ()),
            )
        )
        self.stack.append((node.name, "class"))
        self.generic_visit(node)
        self.stack.pop()

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        decorators = tuple(dotted_name(item) or ast.unparse(item) for item in node.decorator_list)
        calls = collect_calls(node)
        parent_kinds = {item[1] for item in self.stack}
        if "function" in parent_kinds:
            kind = "nested"
        elif "class" in parent_kinds:
            kind = "method"
        else:
            kind = "function"
        self.symbols.append(
            Symbol(
                path=self.path,
                qualname=self._qualname(node.name),
                short_name=node.name,
                kind=kind,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                signature=format_arguments(node),
                decorators=decorators,
                bases=(),
                doc=first_doc_line(node),
                calls=calls,
                tags=tags_for(node, decorators, calls),
            )
        )
        self.stack.append((node.name, "function"))
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)


def source_files(source_root: Path) -> list[Path]:
    return sorted(
        path
        for path in source_root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def parse_file(path: Path) -> tuple[ast.Module, list[Symbol]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = SymbolVisitor(path.relative_to(WORKSPACE))
    visitor.visit(tree)
    return tree, visitor.symbols


def local_dependencies(
    tree: ast.Module,
    prefixes: tuple[str, ...],
) -> tuple[str, ...]:
    dependencies: set[str] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith(prefixes)
        ):
            dependencies.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(prefixes):
                    dependencies.add(alias.name)
    return tuple(sorted(dependencies))


def module_role(
    relative_path: str,
    role_overrides: dict[str, str],
) -> str:
    if relative_path in role_overrides:
        return role_overrides[relative_path]
    if relative_path.endswith("/__init__.py") or relative_path == "__init__.py":
        return "包导出边界；集中暴露该目录的稳定名称。"
    return "Python 模块；具体职责见下方符号及调用。"


def human_name(name: str) -> str:
    return name.strip("_").replace("_", " ") or name


def summary_for(symbol: Symbol) -> str:
    if symbol.doc:
        return symbol.doc.rstrip(".。") + "。"
    if symbol.short_name in SUMMARY_OVERRIDES:
        return SUMMARY_OVERRIDES[symbol.short_name]
    for prefix, action in PREFIX_SUMMARIES:
        if symbol.short_name.startswith(prefix):
            subject = human_name(symbol.short_name[len(prefix) :])
            return f"{action} `{subject}` 对应的数据或结果。"
    if symbol.kind == "class":
        return f"封装 `{human_name(symbol.short_name)}` 相关数据或行为。"
    return f"执行 `{human_name(symbol.short_name)}` 对应逻辑。"


def relevant_calls(symbol: Symbol, known_names: set[str]) -> tuple[str, ...]:
    selected: list[str] = []
    for call in symbol.calls:
        leaf = call.rsplit(".", 1)[-1]
        if leaf in BUILTIN_NAMES:
            continue
        if leaf in known_names or call.startswith(("self.", "cls.")):
            selected.append(call)
    return tuple(selected[:6])


def kind_label(kind: str) -> str:
    return {
        "class": "C",
        "function": "F",
        "method": "M",
        "nested": "N",
    }[kind]


def render_symbol_index(
    lines: list[str],
    parsed: list[tuple[Path, ast.Module, list[Symbol]]],
    *,
    source_root: Path,
    role_overrides: dict[str, str],
    dependency_prefixes: tuple[str, ...],
) -> str:
    all_symbols = [symbol for _, _, symbols in parsed for symbol in symbols]
    known_names = {symbol.short_name for symbol in all_symbols}
    functions = sum(symbol.kind != "class" for symbol in all_symbols)
    classes = len(all_symbols) - functions
    lines.extend(
        [
        "## 符号索引",
        "",
        f"覆盖 `{len(parsed)}` 个 Python 文件、`{functions}` 个命名函数/方法、"
        f"`{classes}` 个类。匿名 lambda 不单独列出。",
        "",
        "图例：`F` 模块函数，`M` 方法，`N` 嵌套函数，`C` 类；"
        "`IO-R/IO-W` 文件读写，`DB` 数据库，`PROCESS` 外部进程。",
        "",
        ]
    )

    for path, tree, symbols in parsed:
        if not symbols:
            continue
        relative = path.relative_to(source_root).as_posix()
        workspace_path = path.relative_to(WORKSPACE).as_posix()
        lines.extend(
            [
                f"## `{workspace_path}`",
                "",
                f"职责：{module_role(relative, role_overrides)}",
            ]
        )
        dependencies = local_dependencies(tree, dependency_prefixes)
        if dependencies:
            rendered_dependencies = ", ".join(f"`{item}`" for item in dependencies)
            lines.append(f"工程依赖：{rendered_dependencies}")

        lines.append("")
        for symbol in symbols:
            location = (
                f"L{symbol.line}"
                if symbol.line == symbol.end_line
                else f"L{symbol.line}-L{symbol.end_line}"
            )
            if symbol.kind == "class":
                bases = f"({', '.join(symbol.bases)})" if symbol.bases else ""
                display = f"{symbol.qualname}{bases}"
            else:
                display = f"{symbol.qualname}{symbol.signature}"
            tags = f" [{' '.join(symbol.tags)}]" if symbol.tags else ""
            calls = relevant_calls(symbol, known_names)
            call_text = (
                " 调用：" + ", ".join(f"`{call}`" for call in calls) + "。"
                if calls
                else ""
            )
            lines.append(
                f"- `{kind_label(symbol.kind)} {location}` `{display}`{tags}："
                f"{summary_for(symbol)}{call_text}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_before_index(
    parsed: list[tuple[Path, ast.Module, list[Symbol]]],
) -> str:
    lines = [
        "# before_traning Codex Index",
        "",
        GENERATED_NOTICE,
        "",
        "面向 Codex 的低 token 工程导航；先按阶段定位，再读取命中的源码。",
        "",
        "## 调用分层",
        "",
        "```text",
        "main.py -> core/pipeline.py:TRAINING_PIPELINE -> core stages",
        "        -> Lib reusable APIs -> state / filesystem / SQLite / ffmpeg",
        "```",
        "",
        "## 七阶段入口",
        "",
        "| key | Core 入口 | 完成状态 |",
        "|---|---|---|",
        "| `import_beatmaps` | `core/beatmap/importer.py` | `osu_imported`, `audio_imported` |",
        "| `verify_export` | `core/beatmap/verify.py` | `verify_exported` |",
        "| `difficulty_export` | `core/beatmap/difficulty.py` | `difficulty_exported` |",
        "| `video_match` | `core/video/match.py` | `video_matched` |",
        "| `av_correspondence` | `core/video/av.py` | `av_corresponded` |",
        "| `clip` | `core/video/clip.py` | `video_processed` |",
        "| `video_segment` | `core/video/segment.py` | `video_segmented` |",
        "",
        "快速查询：`python project_index/build_index.py --lookup 符号名`。",
        "",
    ]
    return render_symbol_index(
        lines,
        parsed,
        source_root=BEFORE_SOURCE_ROOT,
        role_overrides=ROLE_OVERRIDES,
        dependency_prefixes=("before_traning", "package"),
    )


def render_training_index(
    parsed: list[tuple[Path, ast.Module, list[Symbol]]],
) -> str:
    lines = [
        "# traning Codex Index",
        "",
        GENERATED_NOTICE,
        "",
        "面向 Codex 的低 token 工程导航；先按阶段定位，再读取命中的源码。",
        "",
        "## 调用分层",
        "",
        "```text",
        "main.py -> core/pipeline.py:TRAINING_STAGES",
        "        -> core/data_input (配置映射、preflight、组装)",
        "        -> Lib/data (发现、标签、采样、解码、分块、collate)",
        "        -> state (run / experiment / checkpoint metadata)",
        "```",
        "",
        "## 六阶段入口",
        "",
        "| key | Core 入口 | 当前状态 |",
        "|---|---|---|",
        "| `data_input` | `core/data_input/data_input.py` | 已实现并登记 |",
        "| `spatial` | `core/spatial` | 目录边界已建立 |",
        "| `candidate_cache` | `core/candidate_cache` | 目录边界已建立 |",
        "| `temporal` | `core/temporal` | 目录边界已建立 |",
        "| `evaluation` | `core/evaluation` | 目录边界已建立 |",
        "| `export` | `core/export` | 目录边界已建立 |",
        "",
        "快速查询：`python project_index/build_index.py --lookup 符号名`。",
        "",
    ]
    return render_symbol_index(
        lines,
        parsed,
        source_root=TRAINING_SOURCE_ROOT,
        role_overrides=TRAINING_ROLE_OVERRIDES,
        dependency_prefixes=("traning", "package"),
    )


def parse_project(
    source_root: Path,
) -> list[tuple[Path, ast.Module, list[Symbol]]]:
    parsed: list[tuple[Path, ast.Module, list[Symbol]]] = []
    for path in source_files(source_root):
        tree, symbols = parse_file(path)
        parsed.append((path, tree, symbols))
    return parsed


def build_outputs(
    before_parsed: list[tuple[Path, ast.Module, list[Symbol]]],
    training_parsed: list[tuple[Path, ast.Module, list[Symbol]]],
) -> dict[Path, str]:
    return {
        BEFORE_CODEX_INDEX: render_before_index(before_parsed),
        TRAINING_CODEX_INDEX: render_training_index(training_parsed),
    }


def write_outputs(outputs: dict[Path, str]) -> None:
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(WORKSPACE)}")


def check_outputs(outputs: dict[Path, str]) -> int:
    stale: list[Path] = []
    for path, expected in outputs.items():
        actual = path.read_text(encoding="utf-8") if path.exists() else None
        if actual != expected:
            stale.append(path)
    if stale:
        for path in stale:
            print(f"stale {path.relative_to(WORKSPACE)}")
        print("run: python project_index/build_index.py")
        return 1
    print("project indexes are current")
    return 0


def lookup_symbols(
    parsed: list[tuple[Path, ast.Module, list[Symbol]]],
    query: str,
) -> int:
    normalized_query = query.casefold()
    all_symbols = [symbol for _, _, symbols in parsed for symbol in symbols]
    known_names = {symbol.short_name for symbol in all_symbols}
    matches = [
        symbol
        for symbol in all_symbols
        if normalized_query in symbol.qualname.casefold()
        or normalized_query in symbol.path.as_posix().casefold()
    ]
    if not matches:
        print(f"no symbol or module matched: {query}")
        return 1

    for symbol in matches:
        location = (
            f"{symbol.line}"
            if symbol.line == symbol.end_line
            else f"{symbol.line}-{symbol.end_line}"
        )
        if symbol.kind == "class":
            bases = f"({', '.join(symbol.bases)})" if symbol.bases else ""
            display = f"{symbol.qualname}{bases}"
        else:
            display = f"{symbol.qualname}{symbol.signature}"
        tags = f" [{' '.join(symbol.tags)}]" if symbol.tags else ""
        print(
            f"{symbol.path.as_posix()}:{location} "
            f"{kind_label(symbol.kind)} {display}{tags}"
        )
        print(f"  {summary_for(symbol)}")
        calls = relevant_calls(symbol, known_names)
        if calls:
            print("  calls: " + ", ".join(calls))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if generated indexes differ from the current Python source",
    )
    parser.add_argument(
        "--lookup",
        metavar="TEXT",
        help="print only symbols/modules whose qualified name or path contains TEXT",
    )
    args = parser.parse_args()
    before_parsed = parse_project(BEFORE_SOURCE_ROOT)
    training_parsed = parse_project(TRAINING_SOURCE_ROOT)
    all_parsed = before_parsed + training_parsed
    if args.lookup:
        return lookup_symbols(all_parsed, args.lookup)
    outputs = build_outputs(before_parsed, training_parsed)
    return check_outputs(outputs) if args.check else (write_outputs(outputs) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
