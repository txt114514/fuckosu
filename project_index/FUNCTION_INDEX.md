# Function Index

> 自动生成文件，请勿手工修改。运行 `python project_index/build_index.py` 重建。

覆盖 `89` 个 Python 文件、`343` 个命名函数/方法、`70` 个类。匿名 lambda 不单独列出。

图例：`F` 模块函数，`M` 方法，`N` 嵌套函数，`C` 类；`IO-R/IO-W` 文件读写，`DB` 数据库，`PROCESS` 外部进程。

使用顺序：先读 `PROJECT_MAP.md`，再在 `FUNCTION_LOCATIONS.md` 定位，最后只读取本文件对应模块块和源码行。

## `src/Traning/Lib/artifacts.py`

职责：Python 模块；具体职责见下方符号及调用。

- 无命名函数、方法或类。

## `src/Traning/Lib/audio/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。

- 无命名函数、方法或类。

## `src/Traning/Lib/audio/matching/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`Traning.Lib.audio.matching.matching`

- 无命名函数、方法或类。

## `src/Traning/Lib/audio/matching/matching.py`

职责：组合音频匹配处理器，并复用 AV 对齐器的信号处理能力。
工程依赖：`Traning.Lib.audio.matching.preflight`, `Traning.Lib.audio.matching.steps`, `Traning.Lib.audio.matching.wrapup`, `Traning.Lib.beatmap.folder_store`, `Traning.Lib.beatmap.manifest`, `Traning.Lib.common.batch`, `Traning.Lib.common.pathspec`, `Traning.Lib.defaults`, `Traning.Lib.video.av_processing`, `Traning.conf`, `Traning.conf.legacy_config`, `Traning.state.process_status`

- `F L30-L31` `_load_audio_match_experiment_config(config: ConfigReader) -> dict[str, object]`：加载 `audio match experiment config` 对应的数据或结果。
  关键调用：`read_config_values`。
- `F L34-L42` `build_audio_match_experiment_from_config_or_default(config_path: Path | None=None) -> 'AudioMatchExperiment'`：构建并返回 `audio match experiment from config or default` 对应的数据或结果。
  关键调用：`build_from_config_or_default`。
- `C L45-L85` `AudioMatchExperiment(AudioMatchWrapUpMixin, AudioMatchStepsMixin, AudioMatchPreflightMixin)` [CLASS]：封装 `AudioMatchExperiment` 相关数据或行为。
- `M L50-L85` `AudioMatchExperiment.__init__(self, settings: Settings=DEFAULTS, status_manager: ProcessStatusManager | None=None, **overrides: object)`：初始化实例依赖、配置和运行状态。
  关键调用：`BeatmapFolderStore`, `ManifestFolderWalker`, `ProcessStatusManager`, `VideoAVProcessor`, `assign_group`, `forward_kwargs`。
- `C L88-L89` `AudioMatchProcessor(AudioMatchExperiment)` [CLASS]：Task-aligned name for the audio-based video matching processor。
- `F L92-L94` `main()`：独立脚本入口，构建处理器并执行。
  关键调用：`build_audio_match_experiment_from_config_or_default`, `experiment.run`。

## `src/Traning/Lib/audio/matching/preflight.py`

职责：同步已存在视频的状态，收集待匹配文件夹和候选视频。
工程依赖：`Traning.Lib.common.failures`, `Traning.Lib.common.pathspec`

- `C L9-L97` `AudioMatchPreflightMixin` [CLASS]：封装 `AudioMatchPreflightMixin` 相关数据或行为。
- `M L10-L15` `AudioMatchPreflightMixin._folder_has_video(self, folder_name: str) -> bool`：执行 `folder has video` 对应逻辑。
  关键调用：`matches_name`, `self.store.get_folder_path`。
- `M L17-L45` `AudioMatchPreflightMixin._sync_video_matched_status(self, folder_name: str)`：同步 `video matched status` 对应的数据或结果。
  关键调用：`failure_detail`, `self._folder_has_video`, `self.status_manager.ensure_status_file`, `self.status_manager.is_step_done`, `self.status_manager.mark_step_done`, `self.status_manager.mark_step_pending`。
- `M L47-L62` `AudioMatchPreflightMixin._pending_folder_names(self) -> list[str]`：执行 `pending folder names` 对应逻辑。
  关键调用：`self._folder_has_video`, `self._sync_video_matched_status`, `self.store.file_exists`, `self.store.folder_exists`, `self.store.get_folder_path`, `self.walker.read_folder_names`。
- `M L64-L74` `AudioMatchPreflightMixin._candidate_folder_names(self, *, include_existing_video: bool) -> list[str]`：执行 `candidate folder names` 对应逻辑。
  关键调用：`self._pending_folder_names`, `self.store.file_exists`, `self.store.folder_exists`, `self.walker.read_folder_names`。
- `M L76-L97` `AudioMatchPreflightMixin._candidate_videos(self, *, allow_fallback: bool) -> list[Path]`：执行 `candidate videos` 对应逻辑。
  关键调用：`filter_files`, `self.video_root.exists`, `self.video_root.iterdir`, `self.walker.read_folder_names`。

## `src/Traning/Lib/audio/matching/steps.py`

职责：提取音频特征、计算视频/歌曲配对得分并做贪心一对一选择。
工程依赖：`Traning.Lib.common.failures`

- `C L12-L194` `AudioMatchStepsMixin` [CLASS]：封装 `AudioMatchStepsMixin` 相关数据或行为。
- `M L13-L17` `AudioMatchStepsMixin._extract_samples(self, source_path: Path, *, from_video: bool) -> np.ndarray`：提取 `samples` 对应的数据或结果。
  关键调用：`self.aligner._extract_audio_to_wav`, `self.aligner._load_wav_samples`。
- `M L19-L32` `AudioMatchStepsMixin._build_alignment_features(self, samples: np.ndarray) -> dict[str, np.ndarray]`：构建 `alignment features` 对应的数据或结果。
  关键调用：`self.aligner._build_feature_series`, `self.aligner._build_music_refine_series`。
- `M L34-L69` `AudioMatchStepsMixin._estimate_offset_from_features(self, video_features: dict[str, np.ndarray], song_features: dict[str, np.ndarray]) -> tuple[float, float, float]`：估算 `offset from features` 对应的数据或结果。
  关键调用：`self.aligner._estimate_best_start_frame`。
- `M L71-L79` `AudioMatchStepsMixin._result_sort_key(self, item: dict[str, Any]) -> tuple[float, float, float, float]`：执行 `result sort key` 对应逻辑。
  关键调用：`item.get`。
- `M L81-L173` `AudioMatchStepsMixin._score_pairs(self, videos: list[Path], folder_names: list[str]) -> list[dict[str, Any]]`：执行 `score pairs` 对应逻辑。
  关键调用：`exception_detail`, `format_failure`, `self._build_alignment_features`, `self._estimate_offset_from_features`, `self._extract_samples`, `self.aligner._estimate_verify_adjustment_seconds`。
- `M L175-L194` `AudioMatchStepsMixin._select_greedy_matches(self, pair_results: list[dict[str, Any]]) -> list[dict[str, Any]]`：选择 `greedy matches` 对应的数据或结果。

## `src/Traning/Lib/audio/matching/wrapup.py`

职责：展示和应用音频匹配结果，移动视频、回写状态并支持回滚。
工程依赖：`Traning.Lib.common.failures`

- `C L9-L160` `AudioMatchWrapUpMixin` [CLASS]：封装 `AudioMatchWrapUpMixin` 相关数据或行为。
- `M L10-L26` `AudioMatchWrapUpMixin._print_greedy_matches(self, matches: list[dict[str, Any]])`：执行 `print greedy matches` 对应逻辑。
- `M L28-L133` `AudioMatchWrapUpMixin._apply_matches(self, matches: list[dict[str, Any]], pending_folder_names: list[str], candidate_videos: list[Path])` [IO-W]：应用 `matches` 对应的数据或结果。
  关键调用：`exception_detail`, `failure_detail`, `self.status_manager.mark_step_done`, `self.status_manager.mark_step_pending`, `self.store.get_file_path`。
- `M L135-L160` `AudioMatchWrapUpMixin.run(self, *, apply_matches: bool=False, allow_fallback_videos: bool | None=None)`：执行该处理器的完整工作流。
  关键调用：`self._apply_matches`, `self._candidate_folder_names`, `self._candidate_videos`, `self._print_greedy_matches`, `self._score_pairs`, `self._select_greedy_matches`。

## `src/Traning/Lib/beatmap/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。

- 无命名函数、方法或类。

## `src/Traning/Lib/beatmap/difficulty.py`

职责：单谱面难度读取、manifest 存储与按区间查询。
工程依赖：`Traning.Lib.beatmap.difficulty_batch`, `Traning.Lib.beatmap.folder_store`, `Traning.Lib.beatmap.manifest`, `Traning.Lib.beatmap.osu_metadata`, `Traning.Lib.defaults`, `Traning.conf`, `Traning.state.process_status`

- `C L18-L20` `DifficultyEntry` [CLASS]：封装 `DifficultyEntry` 相关数据或行为。
- `C L23-L134` `DifficultyFileManager(DifficultyBatchMixin)` [CLASS]：封装 `DifficultyFileManager` 相关数据或行为。
- `M L24-L42` `DifficultyFileManager.__init__(self, store: BeatmapFolderStore, walker: ManifestFolderWalker | None=None, settings: Settings=DEFAULTS, status_manager: ProcessStatusManager | None=None, **_overrides: object)`：初始化实例依赖、配置和运行状态。
  关键调用：`ProcessStatusManager`。
- `M L44-L49` `DifficultyFileManager.write_difficulty(self, folder_name: str, difficulty_value: float) -> None`：写入 `difficulty` 对应的数据或结果。
  关键调用：`self.manifest.set_difficulty`。
- `M L51-L55` `DifficultyFileManager.read_difficulty(self, folder_name: str) -> float`：读取 `difficulty` 对应的数据或结果。
  关键调用：`self.manifest.difficulty_for`。
- `M L57-L103` `DifficultyFileManager.export_one(self, folder_name: str, overwrite: bool=False) -> str`：导出 `one` 对应的数据或结果。
  关键调用：`read_overall_difficulty`, `self.manifest.difficulty_for`, `self.status_manager.ensure_status_file`, `self.status_manager.load_status`, `self.status_manager.mark_step_done`, `self.store.find_osu_files`。
- `M L105-L134` `DifficultyFileManager.list_difficulties(self, min_difficulty: float | None=None, max_difficulty: float | None=None) -> List[DifficultyEntry]`：列出 `difficulties` 对应的数据或结果。
  关键调用：`DifficultyEntry`, `self.read_difficulty`, `self.walker.read_folder_names`。
- `C L137-L138` `BeatmapDifficultyProcessor(DifficultyFileManager)` [CLASS]：Task-aligned name for the beatmap difficulty export processor。

## `src/Traning/Lib/beatmap/difficulty_batch.py`

职责：难度导出的批处理循环、计数、状态回写和整体结果返回。
工程依赖：`Traning.Lib.common.failures`

- `C L6-L32` `DifficultyBatchMixin` [CLASS]：封装 `DifficultyBatchMixin` 相关数据或行为。
- `M L7-L32` `DifficultyBatchMixin.run(self, overwrite: bool=False) -> bool`：执行该处理器的完整工作流。
  关键调用：`exception_detail`, `format_exception`, `self.export_one`, `self.status_manager.ensure_status_file`, `self.status_manager.mark_step_pending`, `self.store.folder_exists`。

## `src/Traning/Lib/beatmap/folder_store.py`

职责：受 SQLite manifest 约束的谱面目录文件读写。
工程依赖：`Traning.Lib.beatmap.manifest`, `Traning.Lib.common.pathspec`, `Traning.state.manifest_schema`

- `C L14-L170` `BeatmapFolderStore` [CLASS]：严格规则：。
- `M L22-L36` `BeatmapFolderStore.__init__(self, target_root: str, manifest_filename: str=MANIFEST_DB_FILENAME)`：初始化实例依赖、配置和运行状态。
  关键调用：`ManifestFolderWalker`, `self.target_root.exists`。
- `M L38-L46` `BeatmapFolderStore._normalize_folder_name(self, folder_name: str) -> str`：规范化 `folder name` 对应的数据或结果。
- `M L48-L49` `BeatmapFolderStore._registered_names(self) -> set[str]`：执行 `registered names` 对应逻辑。
  关键调用：`self.walker.read_folder_names`。
- `M L51-L53` `BeatmapFolderStore.is_registered(self, folder_name: str) -> bool`：判断是否 `registered` 对应的数据或结果。
  关键调用：`self._normalize_folder_name`, `self._registered_names`。
- `M L55-L59` `BeatmapFolderStore._assert_registered(self, folder_name: str)`：执行 `assert registered` 对应逻辑。
  关键调用：`self.is_registered`。
- `M L61-L64` `BeatmapFolderStore.get_folder_path(self, folder_name: str) -> Path`：获取 `folder path` 对应的数据或结果。
  关键调用：`self._assert_registered`, `self._normalize_folder_name`。
- `M L66-L68` `BeatmapFolderStore.folder_exists(self, folder_name: str) -> bool`：执行 `folder exists` 对应逻辑。
  关键调用：`self.get_folder_path`。
- `M L70-L76` `BeatmapFolderStore._require_existing_folder(self, folder_name: str) -> Path`：执行 `require existing folder` 对应逻辑。
  关键调用：`self.get_folder_path`。
- `M L78-L81` `BeatmapFolderStore.find_files(self, folder_name: str, pattern: str='*') -> List[Path]`：执行 `find files` 对应逻辑。
  关键调用：`filter_files`, `gitwildmatch_spec`, `self._require_existing_folder`。
- `M L83-L84` `BeatmapFolderStore.find_osu_files(self, folder_name: str) -> List[Path]`：执行 `find osu files` 对应逻辑。
  关键调用：`self.find_files`。
- `M L86-L91` `BeatmapFolderStore.get_file_path(self, folder_name: str, filename: str) -> Path`：获取 `file path` 对应的数据或结果。
  关键调用：`self._require_existing_folder`。
- `M L93-L94` `BeatmapFolderStore.file_exists(self, folder_name: str, filename: str) -> bool`：执行 `file exists` 对应逻辑。
  关键调用：`self.get_file_path`, `self.get_file_path.exists`。
- `M L96-L132` `BeatmapFolderStore.write_text(self, folder_name: str, filename: str, content: str, mode: WriteMode='overwrite') -> str` [IO-W]：通用文本写入接口。
  关键调用：`file_path.write_text`, `self.get_file_path`。
- `M L134-L151` `BeatmapFolderStore.write_lines(self, folder_name: str, filename: str, lines: Iterable[str], mode: WriteMode='overwrite', add_trailing_newline: bool=True) -> str` [IO-W]：写入 `lines` 对应的数据或结果。
  关键调用：`self.write_text`。
- `M L153-L164` `BeatmapFolderStore.append_line(self, folder_name: str, filename: str, line: str) -> str` [IO-W]：执行 `append line` 对应逻辑。
  关键调用：`self.write_text`。
- `M L166-L170` `BeatmapFolderStore.read_text(self, folder_name: str, filename: str) -> str` [IO-R]：读取 `text` 对应的数据或结果。
  关键调用：`file_path.read_text`, `self.get_file_path`。

## `src/Traning/Lib/beatmap/hit_objects.py`

职责：Circle、Slider、Spinner 的轻量数据模型。

- `C L8-L11` `HitObject` [CLASS]：封装 `HitObject` 相关数据或行为。
- `C L13-L17` `Circle(HitObject)` [CLASS]：封装 `Circle` 相关数据或行为。
- `M L16-L17` `Circle.__post_init__(self)`：完成 dataclass 初始化后的派生字段设置。
- `C L19-L25` `Slider(HitObject)` [CLASS]：封装 `Slider` 相关数据或行为。
- `M L24-L25` `Slider.__post_init__(self)`：完成 dataclass 初始化后的派生字段设置。
- `C L27-L29` `Spinner(HitObject)` [CLASS]：封装 `Spinner` 相关数据或行为。
- `M L28-L29` `Spinner.__post_init__(self)`：完成 dataclass 初始化后的派生字段设置。

## `src/Traning/Lib/beatmap/importing/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`Traning.Lib.beatmap.importing.entry`, `Traning.Lib.beatmap.importing.importing`

- 无命名函数、方法或类。

## `src/Traning/Lib/beatmap/importing/entry.py`

职责：单个 .osz 扫描结果的数据模型。

- `C L8-L16` `OsuEntry` [CLASS]：封装 `OsuEntry` 相关数据或行为。

## `src/Traning/Lib/beatmap/importing/importing.py`

职责：组合谱面导入 mixin，初始化路径规格、包管理器和状态管理器。
工程依赖：`Traning.Lib.beatmap.importing.scanner`, `Traning.Lib.beatmap.importing.wrapup`, `Traning.Lib.beatmap.importing.writer`, `Traning.Lib.beatmap.package`, `Traning.Lib.common.batch`, `Traning.Lib.common.pathspec`, `Traning.Lib.defaults`, `Traning.conf`, `Traning.conf.legacy_config`, `Traning.state.process_status`

- `F L27-L28` `_load_osu_osz_processor_config(config: ConfigReader) -> dict[str, object]`：加载 `osu osz processor config` 对应的数据或结果。
  关键调用：`read_config_values`。
- `F L31-L39` `build_osu_osz_processor_from_config_or_default(config_path: Path | None=None) -> 'OsuOszProcessor'`：构建并返回 `osu osz processor from config or default` 对应的数据或结果。
  关键调用：`build_from_config_or_default`。
- `F L42-L50` `build_beatmap_import_processor_from_config_or_default(config_path: Path | None=None) -> 'BeatmapImportProcessor'`：构建并返回 `beatmap import processor from config or default` 对应的数据或结果。
  关键调用：`build_from_config_or_default`。
- `C L53-L82` `OsuOszProcessor(OszImportWrapUpMixin, OszImportWriterMixin, OszScannerMixin)` [CLASS]：封装 `OsuOszProcessor` 相关数据或行为。
- `M L54-L82` `OsuOszProcessor.__init__(self, settings: Settings=DEFAULTS, **overrides: object)`：初始化实例依赖、配置和运行状态。
  关键调用：`PackageUpdater`, `ProcessStatusManager`, `assign_group`, `settings_namespace`, `suffix_spec`。
- `C L85-L86` `BeatmapImportProcessor(OsuOszProcessor)` [CLASS]：Task-aligned name for the beatmap import core processor。
- `F L89-L91` `main()`：独立脚本入口，构建处理器并执行。
  关键调用：`build_beatmap_import_processor_from_config_or_default`, `processor.run`。

## `src/Traning/Lib/beatmap/importing/scanner.py`

职责：解压 .osz，筛选目标 .osu/.mp3，并按导出时间建立导入项。
工程依赖：`Traning.Lib.beatmap.importing.entry`, `Traning.Lib.beatmap.osu_metadata`, `Traning.Lib.common.failures`, `Traning.Lib.common.pathspec`

- `C L13-L113` `OszScannerMixin` [CLASS]：封装 `OszScannerMixin` 相关数据或行为。
- `M L14-L19` `OszScannerMixin._is_target_osu(self, path: Path) -> bool`：判断是否 `target osu` 对应的数据或结果。
  关键调用：`matches_name`。
- `M L21-L34` `OszScannerMixin._read_audio_bytes(self, osu_path: Path) -> tuple[str, bytes]` [IO-R]：读取 `audio bytes` 对应的数据或结果。
  关键调用：`matches_name`, `read_audio_filename`。
- `M L36-L62` `OszScannerMixin._scan_single_osz(self, osz_path: Path) -> OsuEntry | None` [IO-R]：执行 `scan single osz` 对应逻辑。
  关键调用：`OsuEntry`, `self._is_target_osu`, `self._read_audio_bytes`。
- `M L64-L113` `OszScannerMixin._scan_all_entries_in_time_order(self) -> list[OsuEntry]`：执行 `scan all entries in time order` 对应逻辑。
  关键调用：`filter_files`, `format_exception`, `self._scan_single_osz`, `self.export_dir.exists`, `self.export_dir.iterdir`。

## `src/Traning/Lib/beatmap/importing/wrapup.py`

职责：串联谱面扫描、写入、额外目录告警和汇总输出。

- `C L4-L28` `OszImportWrapUpMixin` [CLASS]：封装 `OszImportWrapUpMixin` 相关数据或行为。
- `M L5-L28` `OszImportWrapUpMixin.run(self) -> bool`：执行该处理器的完整工作流。
  关键调用：`self._rebuild_manifest`, `self._scan_all_entries_in_time_order`, `self._sync_folders_and_copy_files`, `self.updater.find_unregistered_existing_folders`。

## `src/Traning/Lib/beatmap/importing/writer.py`

职责：更新 manifest、分配内部编号并写入谱面/音频及状态。
工程依赖：`Traning.Lib.beatmap.importing.entry`, `Traning.Lib.beatmap.manifest`

- `C L7-L67` `OszImportWriterMixin` [CLASS]：封装 `OszImportWriterMixin` 相关数据或行为。
- `M L8-L22` `OszImportWriterMixin._rebuild_manifest(self, entries: list[OsuEntry])`：执行 `rebuild manifest` 对应逻辑。
  关键调用：`ManifestEntry`, `self.updater.replace_manifest`。
- `M L24-L67` `OszImportWriterMixin._sync_folders_and_copy_files(self, entries: list[OsuEntry])` [IO-W]：同步 `folders and copy files` 对应的数据或结果。
  关键调用：`self.status_manager.ensure_status_file`, `self.status_manager.mark_step_done`, `self.updater.create_folder_if_registered`, `self.updater.load_registered_names`, `self.updater.sync_folders_from_manifest`。

## `src/Traning/Lib/beatmap/manifest.py`

职责：SQLite manifest 仓储；分配稳定内部目录 ID、迁移旧 order.txt 并导出对照表。
工程依赖：`Traning.state.manifest_schema`

- `C L21-L25` `ManifestEntry` [CLASS]：封装 `ManifestEntry` 相关数据或行为。
- `C L28-L322` `PackageManifest` [CLASS]：Small SQLite manifest for stable internal folder IDs and processing order。
- `M L31-L48` `PackageManifest.__init__(self, target_root: str, manifest_filename: str=MANIFEST_DB_FILENAME, legacy_order_filename: str='order.txt', table_filename: str=MANIFEST_TABLE_FILENAME)` [DB IO-W]：初始化实例依赖、配置和运行状态。
  关键调用：`self._ensure_schema`, `self._migrate_legacy_difficulty_files`, `self._migrate_legacy_order`, `self.export_table`, `self.target_root.mkdir`。
- `M L50-L60` `PackageManifest._ensure_schema(self) -> None`：确保 `schema` 对应的数据或结果。
- `M L62-L65` `PackageManifest._all_items(self) -> list[PackageManifestItem]` [DB]：执行 `all items` 对应逻辑。
- `M L67-L73` `PackageManifest._normalize_source_name(self, source_name: str) -> str`：规范化 `source name` 对应的数据或结果。
- `M L75-L83` `PackageManifest._next_folder_number(self, items: list[PackageManifestItem]) -> int`：执行 `next folder number` 对应逻辑。
- `M L85-L86` `PackageManifest._folder_name(self, number: int) -> str`：执行 `folder name` 对应逻辑。
- `M L88-L93` `PackageManifest._legacy_osu_filename(self, source_name: str) -> str | None`：执行 `legacy osu filename` 对应逻辑。
- `M L95-L144` `PackageManifest._rename_legacy_folders(self, mappings: list[tuple[str, str]]) -> None` [IO-W]：执行 `rename legacy folders` 对应逻辑。
  关键调用：`self._restore_legacy_folders`。
- `M L146-L159` `PackageManifest._restore_legacy_folders(self, mappings: list[tuple[str, str]]) -> None` [IO-W]：执行 `restore legacy folders` 对应逻辑。
- `M L161-L203` `PackageManifest._migrate_legacy_order(self) -> None` [DB IO-R IO-W]：执行 `migrate legacy order` 对应逻辑。
  关键调用：`PackageManifestItem`, `self._all_items`, `self._folder_name`, `self._legacy_osu_filename`, `self._normalize_source_name`, `self._rename_legacy_folders`。
- `M L205-L226` `PackageManifest._migrate_legacy_difficulty_files(self) -> None` [DB IO-R IO-W]：执行 `migrate legacy difficulty files` 对应逻辑。
  关键调用：`difficulty_path.read_text`。
- `M L228-L238` `PackageManifest.export_table(self) -> Path` [IO-W]：导出 `table` 对应的数据或结果。
  关键调用：`self._all_items`, `self.table_path.with_name`, `temp_path.replace`。
- `M L240-L275` `PackageManifest.replace(self, entries: list[ManifestEntry]) -> dict[str, str]` [DB]：执行 `replace` 对应逻辑。
  关键调用：`PackageManifestItem`, `by_source.get`, `self._folder_name`, `self._next_folder_number`, `self._normalize_source_name`, `self.export_table`。
- `M L277-L284` `PackageManifest.read_folder_names(self) -> list[str]` [DB]：读取 `folder names` 对应的数据或结果。
- `M L286-L287` `PackageManifest.read_all_folder_names(self) -> list[str]`：读取 `all folder names` 对应的数据或结果。
  关键调用：`self._all_items`。
- `M L289-L295` `PackageManifest.source_name_for(self, folder_name: str) -> str | None` [DB]：执行 `source name for` 对应逻辑。
- `M L297-L310` `PackageManifest.set_difficulty(self, folder_name: str, difficulty_value: float) -> None` [DB IO-W]：执行 `set difficulty` 对应逻辑。
- `M L312-L319` `PackageManifest.difficulty_for(self, folder_name: str) -> float | None` [DB]：执行 `difficulty for` 对应逻辑。
- `M L321-L322` `PackageManifest.is_active(self, folder_name: str) -> bool`：判断是否 `active` 对应的数据或结果。
  关键调用：`self.read_folder_names`。
- `C L325-L343` `ManifestFolderWalker` [CLASS]：封装 `ManifestFolderWalker` 相关数据或行为。
- `M L326-L337` `ManifestFolderWalker.__init__(self, target_root: str, manifest_filename: str=MANIFEST_DB_FILENAME)`：初始化实例依赖、配置和运行状态。
  关键调用：`PackageManifest`, `self.target_root.exists`。
- `M L339-L340` `ManifestFolderWalker.read_folder_names(self) -> list[str]`：读取 `folder names` 对应的数据或结果。
  关键调用：`self.manifest.read_folder_names`。
- `M L342-L343` `ManifestFolderWalker.source_name_for(self, folder_name: str) -> str | None`：执行 `source name for` 对应逻辑。
  关键调用：`self.manifest.source_name_for`。

## `src/Traning/Lib/beatmap/order.py`

职责：旧 OrderFolderWalker 的兼容导出；业务代码使用 ManifestFolderWalker。
工程依赖：`Traning.Lib.beatmap.manifest`

- 无命名函数、方法或类。

## `src/Traning/Lib/beatmap/osu_metadata.py`

职责：从 .osu 指定 section 读取 AudioFilename 和 OverallDifficulty。

- `F L6-L31` `read_section_key(osu_path: Path, section_name: str, key_name: str) -> str` [IO-W]：读取 `section key` 对应的数据或结果。
- `F L34-L35` `read_audio_filename(osu_path: Path) -> str`：读取 `audio filename` 对应的数据或结果。
  关键调用：`read_section_key`。
- `F L38-L39` `read_overall_difficulty(osu_path: Path) -> float`：读取 `overall difficulty` 对应的数据或结果。
  关键调用：`read_section_key`。

## `src/Traning/Lib/beatmap/package.py`

职责：通过 SQLite manifest 创建和同步允许使用的内部谱面目录。
工程依赖：`Traning.Lib.beatmap.manifest`, `Traning.Lib.defaults`, `Traning.conf`, `Traning.conf.legacy_config`

- `C L14-L101` `PackageUpdater` [CLASS]：规则：。
- `M L22-L42` `PackageUpdater.__init__(self, target_root: str | Settings, **overrides: object)` [IO-W]：初始化实例依赖、配置和运行状态。
  关键调用：`PackageManifest`, `self.target_root.mkdir`, `settings_namespace`。
- `M L44-L45` `PackageUpdater.load_manifest_folder_names(self) -> List[str]`：加载 `manifest folder names` 对应的数据或结果。
  关键调用：`self.manifest.read_folder_names`。
- `M L47-L48` `PackageUpdater.load_registered_names(self) -> set[str]`：加载 `registered names` 对应的数据或结果。
  关键调用：`self.manifest.read_all_folder_names`。
- `M L50-L51` `PackageUpdater.replace_manifest(self, entries: list[ManifestEntry]) -> dict[str, str]` [IO-W]：执行 `replace manifest` 对应逻辑。
  关键调用：`self.manifest.replace`。
- `M L53-L54` `PackageUpdater.is_registered(self, folder_name: str) -> bool`：判断是否 `registered` 对应的数据或结果。
  关键调用：`self.manifest.is_active`。
- `M L56-L71` `PackageUpdater.create_folder_if_registered(self, folder_name: str) -> Path` [IO-W]：只有在 manifest 中启用的内部 ID 才允许创建/使用对应文件夹。
  关键调用：`self.is_registered`。
- `M L73-L83` `PackageUpdater.sync_folders_from_manifest(self) -> List[Path]` [IO-W]：Create active manifest folders in processing order。
  关键调用：`self.load_manifest_folder_names`。
- `M L85-L101` `PackageUpdater.find_unregistered_existing_folders(self) -> List[Path]`：返回 target_root 下存在，但不在 manifest 中登记的目录。
  关键调用：`self.ignore_spec.match_file`, `self.load_registered_names`, `self.target_root.iterdir`。

## `src/Traning/Lib/beatmap/timing_points.py`

职责：osu 原始 timing point 数据模型。

- `C L7-L15` `OsuOriginalTimingPoint` [CLASS]：封装 `OsuOriginalTimingPoint` 相关数据或行为。

## `src/Traning/Lib/beatmap/verification/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`Traning.Lib.beatmap.verification.verification`

- 无命名函数、方法或类。

## `src/Traning/Lib/beatmap/verification/parser.py`

职责：解析 .osu sections、timing points 和 hit objects，并统一生成文本或结构化对象数据。
工程依赖：`Traning.Lib.beatmap.hit_objects`, `Traning.Lib.beatmap.timing_points`

- `C L10-L212` `VerifyOsuParser` [CLASS]：封装 `VerifyOsuParser` 相关数据或行为。
- `M L11-L35` `VerifyOsuParser.parse_sections(self, osu_path: Path) -> tuple[str | None, dict[str, list[str]]]` [IO-W]：解析 `sections` 对应的数据或结果。
- `M L37-L44` `VerifyOsuParser.parse_key_value_section(self, lines: List[str]) -> dict[str, str]`：解析 `key value section` 对应的数据或结果。
- `M L46-L68` `VerifyOsuParser.parse_timing_points(self, lines: List[str]) -> List[OsuOriginalTimingPoint]`：解析 `timing points` 对应的数据或结果。
  关键调用：`OsuOriginalTimingPoint`。
- `M L70-L97` `VerifyOsuParser.get_effective_timing(self, t: int, timing_points: List[OsuOriginalTimingPoint]) -> tuple[OsuOriginalTimingPoint, float]`：获取 `effective timing` 对应的数据或结果。
- `M L99-L170` `VerifyOsuParser.parse_hitobjects(self, hitobject_lines: List[str], timing_points: List[OsuOriginalTimingPoint], slider_multiplier: float) -> List[object]`：解析 `hitobjects` 对应的数据或结果。
  关键调用：`Circle`, `Slider`, `Spinner`, `self.get_effective_timing`。
- `M L172-L188` `VerifyOsuParser.objects_to_lines(self, objects: List[object]) -> List[str]`：执行 `objects to lines` 对应逻辑。
- `M L190-L212` `VerifyOsuParser.hit_object_to_dict(self, hit_object: HitObject, *, time_offset_ms: int=0) -> dict[str, Any]`：执行 `hit object to dict` 对应逻辑。

## `src/Traning/Lib/beatmap/verification/steps.py`

职责：单文件夹 verify.txt 导出和 verify_exported 状态更新。
工程依赖：`Traning.Lib.common.batch`

- `C L6-L73` `VerifyStepsMixin` [CLASS]：封装 `VerifyStepsMixin` 相关数据或行为。
- `M L7-L73` `VerifyStepsMixin.process_one(self, folder_name: str, overwrite: bool=False) -> BatchProcessResult`：处理 manifest 中的单个内部谱面文件夹。
  关键调用：`general.get`, `sections.get`, `self.parser.objects_to_lines`, `self.parser.parse_hitobjects`, `self.parser.parse_key_value_section`, `self.parser.parse_sections`。

## `src/Traning/Lib/beatmap/verification/verification.py`

职责：组合 verify 导出处理器及兼容 builder/脚本入口。
工程依赖：`Traning.Lib.artifacts`, `Traning.Lib.beatmap.folder_store`, `Traning.Lib.beatmap.manifest`, `Traning.Lib.beatmap.verification.parser`, `Traning.Lib.beatmap.verification.steps`, `Traning.Lib.beatmap.verification.wrapup`, `Traning.Lib.common.batch`, `Traning.Lib.defaults`, `Traning.conf`, `Traning.conf.legacy_config`, `Traning.state.process_status`

- `F L27-L28` `_load_verify_exporter_config(config: ConfigReader) -> dict[str, object]`：加载 `verify exporter config` 对应的数据或结果。
  关键调用：`read_config_values`。
- `C L31-L48` `VerifyExporter(VerifyWrapUpMixin, VerifyStepsMixin, FolderBatchProcessor)` [CLASS]：封装 `VerifyExporter` 相关数据或行为。
- `M L32-L48` `VerifyExporter.__init__(self, walker: ManifestFolderWalker, store: BeatmapFolderStore, settings: Settings=DEFAULTS, status_manager: ProcessStatusManager | None=None, **overrides: object)`：初始化实例依赖、配置和运行状态。
  关键调用：`ProcessStatusManager`, `VerifyOsuParser`, `super.__init__`。
- `C L51-L52` `BeatmapVerifyExporter(VerifyExporter)` [CLASS]：Task-aligned name for the beatmap verify export processor。
- `F L55-L71` `_build_verify_exporter_from_config(settings: Settings=DEFAULTS, **overrides: object) -> BeatmapVerifyExporter`：构建 `verify exporter from config` 对应的数据或结果。
  关键调用：`BeatmapFolderStore`, `BeatmapVerifyExporter`, `ManifestFolderWalker`, `settings_namespace`。
- `F L74-L82` `build_verify_exporter_from_config_or_default(config_path: Path | None=None) -> BeatmapVerifyExporter`：构建并返回 `verify exporter from config or default` 对应的数据或结果。
  关键调用：`build_from_config_or_default`。
- `F L85-L88` `build_beatmap_verify_exporter_from_config_or_default(config_path: Path | None=None) -> BeatmapVerifyExporter`：构建并返回 `beatmap verify exporter from config or default` 对应的数据或结果。
  关键调用：`build_verify_exporter_from_config_or_default`。
- `F L91-L93` `main()`：独立脚本入口，构建处理器并执行。
  关键调用：`build_verify_exporter_from_config_or_default`, `exporter.run`。

## `src/Traning/Lib/beatmap/verification/wrapup.py`

职责：verify 导出失败时回写 pending 状态。
工程依赖：`Traning.Lib.common.failures`

- `C L6-L14` `VerifyWrapUpMixin` [CLASS]：封装 `VerifyWrapUpMixin` 相关数据或行为。
- `M L7-L14` `VerifyWrapUpMixin.handle_failure(self, folder_name: str, error: Exception)`：处理单文件夹失败并同步失败状态。
  关键调用：`exception_detail`, `self.status_manager.ensure_status_file`, `self.status_manager.mark_step_pending`, `self.store.folder_exists`。

## `src/Traning/Lib/common/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`Traning.Lib.common.batch`, `Traning.Lib.common.failures`, `Traning.Lib.common.pathspec`

- 无命名函数、方法或类。

## `src/Traning/Lib/common/batch.py`

职责：配置规格辅助函数与文件夹批处理模板。
工程依赖：`Traning.Lib.common.failures`

- `C L36-L39` `ConfigValueSpec` [CLASS]：封装 `ConfigValueSpec` 相关数据或行为。
- `F L46-L54` `_normalize_config_path(path: ConfigPathInput) -> tuple[str, ...]`：规范化 `config path` 对应的数据或结果。
- `F L57-L70` `_normalize_config_path_group(paths: ConfigPathGroupInput) -> tuple[tuple[str, ...], ...]`：规范化 `config path group` 对应的数据或结果。
  关键调用：`_normalize_config_path`。
- `F L73-L81` `_config_values(reader_name: str, **entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config values` 对应逻辑。
  关键调用：`ConfigValueSpec`, `_normalize_config_path_group`。
- `F L84-L85` `config_resolved_paths(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config resolved paths` 对应逻辑。
  关键调用：`_config_values`。
- `F L88-L89` `config_filenames(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config filenames` 对应逻辑。
  关键调用：`_config_values`。
- `F L92-L93` `config_nonempty_strs(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config nonempty strs` 对应逻辑。
  关键调用：`_config_values`。
- `F L96-L97` `config_bools(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config bools` 对应逻辑。
  关键调用：`_config_values`。
- `F L100-L101` `config_string_tuples(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config string tuples` 对应逻辑。
  关键调用：`_config_values`。
- `F L104-L105` `config_suffix_tuples(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config suffix tuples` 对应逻辑。
  关键调用：`_config_values`。
- `F L108-L109` `config_positive_ints(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config positive ints` 对应逻辑。
  关键调用：`_config_values`。
- `F L112-L113` `config_nonnegative_ints(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config nonnegative ints` 对应逻辑。
  关键调用：`_config_values`。
- `F L116-L117` `config_positive_floats(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config positive floats` 对应逻辑。
  关键调用：`_config_values`。
- `F L120-L121` `config_floats(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config floats` 对应逻辑。
  关键调用：`_config_values`。
- `F L124-L133` `merge_config_specs(*spec_groups: ConfigValueSpec | Iterable[ConfigValueSpec]) -> tuple[ConfigValueSpec, ...]`：执行 `merge config specs` 对应逻辑。
- `F L136-L147` `prefix_config_keys(specs: Iterable[ConfigValueSpec], prefix: str) -> tuple[ConfigValueSpec, ...]`：执行 `prefix config keys` 对应逻辑。
  关键调用：`ConfigValueSpec`。
- `F L150-L165` `read_config_values(config_reader: Any, *spec_groups: ConfigValueSpec | Iterable[ConfigValueSpec]) -> dict[str, Any]` [IO-R]：读取 `config values` 对应的数据或结果。
  关键调用：`config_reader.read`。
- `C L168-L170` `FolderWalkerLike(Protocol)` [CLASS]：封装 `FolderWalkerLike` 相关数据或行为。
- `M L169-L170` `FolderWalkerLike.read_folder_names(self) -> list[str]`：读取 `folder names` 对应的数据或结果。
- `C L173-L244` `FolderBatchProcessor(ABC)` [CLASS]：Shared shell for folder-based batch processors。
- `M L178-L181` `FolderBatchProcessor.__init__(self)`：初始化实例依赖、配置和运行状态。
- `M L183-L189` `FolderBatchProcessor.progress_message(self, index: int, total: int, folder_name: str) -> str | None`：生成当前批处理进度文本。
- `M L191-L192` `FolderBatchProcessor.iter_folder_names(self) -> list[str]`：执行 `iter folder names` 对应逻辑。
  关键调用：`self.walker.read_folder_names`。
- `M L194-L195` `FolderBatchProcessor.handle_failure(self, folder_name: str, error: Exception)`：处理单文件夹失败并同步失败状态。
- `M L198-L203` `FolderBatchProcessor.process_one(self, folder_name: str, overwrite: bool=False) -> BatchProcessResult`：处理 manifest 中的单个内部谱面文件夹。
- `M L205-L216` `FolderBatchProcessor._record_result(self, folder_name: str, result: BatchProcessResult)`：执行 `record result` 对应逻辑。
- `M L218-L222` `FolderBatchProcessor._print_summary(self)`：执行 `print summary` 对应逻辑。
- `M L224-L244` `FolderBatchProcessor.run(self, overwrite: bool=False) -> bool`：执行该处理器的完整工作流。
  关键调用：`format_exception`, `self._print_summary`, `self._record_result`, `self.handle_failure`, `self.iter_folder_names`, `self.process_one`。

## `src/Traning/Lib/common/failures.py`

职责：统一提取异常类型、报错函数和模块，并生成状态 detail 与控制台文本。

- `F L11-L21` `_error_traceback(error: BaseException) -> TracebackType | None`：执行 `error traceback` 对应逻辑。
  关键调用：`traceback.tb_frame.f_globals.get`。
- `F L24-L29` `callable_location(function: Callable[..., object]) -> tuple[str, str]`：执行 `callable location` 对应逻辑。
- `F L32-L42` `exception_location(error: BaseException) -> tuple[str, str]`：执行 `exception location` 对应逻辑。
  关键调用：`_error_traceback`, `frame.f_globals.get`。
- `F L45-L59` `failure_detail(message: str, function: Callable[..., object], *, error_type: str='ProcessingStateError', **context: Any) -> dict[str, Any]`：执行 `failure detail` 对应逻辑。
  关键调用：`callable_location`。
- `F L62-L70` `exception_detail(error: BaseException, **context: Any) -> dict[str, Any]`：执行 `exception detail` 对应逻辑。
  关键调用：`exception_location`。
- `F L73-L77` `format_failure(detail: dict[str, Any]) -> str`：执行 `format failure` 对应逻辑。
  关键调用：`detail.get`。
- `F L80-L81` `format_exception(error: BaseException) -> str`：执行 `format exception` 对应逻辑。
  关键调用：`exception_detail`, `format_failure`。

## `src/Traning/Lib/common/pathspec.py`

职责：统一后缀到 gitwildmatch PathSpec 的转换与文件过滤。

- `F L9-L15` `suffix_pattern(suffix: str) -> str`：执行 `suffix pattern` 对应逻辑。
- `F L18-L19` `suffix_patterns(suffixes: Iterable[str]) -> tuple[str, ...]`：执行 `suffix patterns` 对应逻辑。
  关键调用：`suffix_pattern`。
- `F L22-L23` `gitwildmatch_spec(patterns: Iterable[str]) -> pathspec.PathSpec`：执行 `gitwildmatch spec` 对应逻辑。
- `F L26-L27` `suffix_spec(suffixes: Iterable[str]) -> pathspec.PathSpec`：执行 `suffix spec` 对应逻辑。
  关键调用：`gitwildmatch_spec`, `suffix_patterns`。
- `F L30-L31` `matches_name(spec: pathspec.PathSpec, path: Path | str) -> bool`：执行 `matches name` 对应逻辑。
- `F L34-L35` `filter_files(paths: Iterable[Path], spec: pathspec.PathSpec) -> list[Path]`：筛选 `files` 对应的数据或结果。
  关键调用：`matches_name`。

## `src/Traning/Lib/defaults.py`

职责：创建全局默认 Settings 实例，供兼容构造器使用。
工程依赖：`Traning.conf`

- 无命名函数、方法或类。

## `src/Traning/Lib/tools/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`Traning.Lib.tools.ffmpeg`

- 无命名函数、方法或类。

## `src/Traning/Lib/tools/ffmpeg.py`

职责：构造并执行 ffmpeg/ffprobe 命令，读取流起点、媒体时长和视频尺寸。

- `F L38-L42` `_command_error_text(result: subprocess.CompletedProcess[str], unknown_error: str) -> str`：执行 `command error text` 对应逻辑。
- `F L45-L51` `_run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]` [PROCESS]：执行 `run command` 对应逻辑。
  关键调用：`subprocess.run`。
- `F L54-L57` `run_ffmpeg(args: Sequence[str])` [PROCESS]：执行 `run ffmpeg` 对应逻辑。
  关键调用：`_command_error_text`, `_run_command`。
- `F L60-L82` `build_extract_wav_args(source_path: Path, output_path: Path, *, sample_rate: int, from_video: bool) -> tuple[str, ...]`：构建并返回 `extract wav args` 对应的数据或结果。
- `F L85-L104` `build_trim_video_args(source_video_path: Path, output_video_path: Path, *, trim_start_seconds: float, trim_duration_seconds: float) -> tuple[str, ...]`：构建并返回 `trim video args` 对应的数据或结果。
- `F L107-L126` `build_segment_video_args(source_video_path: Path, output_video_path: Path, *, trim_start_seconds: float, trim_duration_seconds: float) -> tuple[str, ...]`：构建并返回 `segment video args` 对应的数据或结果。
- `F L129-L148` `build_crop_video_args(source_video_path: Path, output_video_path: Path, *, crop_left: int, crop_top: int, crop_width: int, crop_height: int) -> tuple[str, ...]`：构建并返回 `crop video args` 对应的数据或结果。
- `F L151-L168` `run_ffprobe_json(args: Sequence[str], *, error_prefix: str) -> dict[str, Any]` [PROCESS]：执行 `run ffprobe json` 对应逻辑。
  关键调用：`_command_error_text`, `_run_command`。
- `F L171-L193` `get_audio_stream_start_time(source_path: Path) -> float`：获取 `audio stream start time` 对应的数据或结果。
  关键调用：`get`, `payload.get`, `run_ffprobe_json`。
- `F L196-L216` `get_media_duration_seconds(source_path: Path) -> float`：获取 `media duration seconds` 对应的数据或结果。
  关键调用：`payload.get`, `payload.get.get`, `run_ffprobe_json`。
- `F L219-L240` `get_video_size(video_path: Path) -> tuple[int, int]`：获取 `video size` 对应的数据或结果。
  关键调用：`payload.get`, `run_ffprobe_json`。

## `src/Traning/Lib/video/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。

- 无命名函数、方法或类。

## `src/Traning/Lib/video/av_processing/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`Traning.Lib.video.av_processing.av_processing`

- 无命名函数、方法或类。

## `src/Traning/Lib/video/av_processing/av_processing.py`

职责：组合 AV 处理器，初始化文件存储、算法参数和状态管理器。
工程依赖：`Traning.Lib.beatmap.folder_store`, `Traning.Lib.common.batch`, `Traning.Lib.common.pathspec`, `Traning.Lib.defaults`, `Traning.Lib.video.av_processing.preflight`, `Traning.Lib.video.av_processing.steps`, `Traning.Lib.video.av_processing.wrapup`, `Traning.conf`, `Traning.conf.legacy_config`, `Traning.state.process_status`

- `F L27-L28` `_load_av_correspondence_processor_config(config: ConfigReader) -> dict[str, object]`：加载 `av correspondence processor config` 对应的数据或结果。
  关键调用：`read_config_values`。
- `F L31-L39` `build_av_correspondence_processor_from_config_or_default(config_path: Path | None=None) -> 'AVCorrespondenceProcessor'`：构建并返回 `av correspondence processor from config or default` 对应的数据或结果。
  关键调用：`build_from_config_or_default`。
- `C L42-L84` `AVCorrespondenceProcessor(AVWrapUpMixin, AVCoreStepsMixin, AVPreflightMixin, FolderBatchProcessor)` [CLASS]：封装 `AVCorrespondenceProcessor` 相关数据或行为。
- `M L48-L84` `AVCorrespondenceProcessor.__init__(self, settings: Settings=DEFAULTS, status_manager: ProcessStatusManager | None=None, **overrides: object)`：初始化实例依赖、配置和运行状态。
  关键调用：`BeatmapFolderStore`, `ProcessStatusManager`, `assign_group`, `self._ensure_status_steps_registered`, `self._validate_config`, `self.status_step.strip`。
- `C L87-L88` `VideoAVProcessor(AVCorrespondenceProcessor)` [CLASS]：Task-aligned name for the video AV correspondence processor。
- `F L91-L93` `main()`：独立脚本入口，构建处理器并执行。
  关键调用：`build_av_correspondence_processor_from_config_or_default`, `processor.run`。

## `src/Traning/Lib/video/av_processing/preflight.py`

职责：校验 AV 参数/状态步骤，定位源视频、音频、verify 和输出文件。
工程依赖：`Traning.Lib.common.failures`, `Traning.Lib.common.pathspec`

- `C L11-L109` `AVPreflightMixin` [CLASS]：封装 `AVPreflightMixin` 相关数据或行为。
- `M L12-L28` `AVPreflightMixin._validate_config(self, config)`：校验 `config` 对应的数据或结果。
- `M L30-L39` `AVPreflightMixin._ensure_status_steps_registered(self)`：确保 `status steps registered` 对应的数据或结果。
- `M L41-L59` `AVPreflightMixin._resolve_source_video_path(self, folder_name: str) -> Path`：解析并定位 `source video path` 对应的数据或结果。
  关键调用：`filter_files`, `self.store.get_folder_path`。
- `M L61-L65` `AVPreflightMixin._resolve_song_audio_path(self, folder_name: str) -> Path`：解析并定位 `song audio path` 对应的数据或结果。
  关键调用：`self.store.get_file_path`。
- `M L67-L68` `AVPreflightMixin._resolve_verify_path(self, folder_name: str) -> Path`：解析并定位 `verify path` 对应的数据或结果。
  关键调用：`self.store.get_file_path`。
- `M L70-L97` `AVPreflightMixin._sync_output_status(self, folder_name: str) -> tuple[bool, bool]`：同步 `output status` 对应的数据或结果。
  关键调用：`failure_detail`, `self.status_manager.is_step_done`, `self.status_manager.mark_step_done`, `self.status_manager.mark_step_pending`, `self.store.file_exists`, `self.store.get_file_path`。
- `M L99-L109` `AVPreflightMixin._ensure_required_steps_done(self, folder_name: str)`：确保 `required steps done` 对应的数据或结果。
  关键调用：`self.status_manager.is_step_done`。

## `src/Traning/Lib/video/av_processing/steps.py`

职责：AV 核心算法：提取采样、粗细相关、hit 校正、窗口校验和视频裁切。
工程依赖：`Traning.Lib.common.batch`, `Traning.Lib.tools.ffmpeg`

- `C L15-L412` `AVCoreStepsMixin` [CLASS]：封装 `AVCoreStepsMixin` 相关数据或行为。
- `M L16-L24` `AVCoreStepsMixin._extract_audio_to_wav(self, source_path: Path, output_path: Path, from_video: bool)` [PROCESS]：提取 `audio to wav` 对应的数据或结果。
  关键调用：`build_extract_wav_args`, `run_ffmpeg`。
- `M L26-L46` `AVCoreStepsMixin._load_wav_samples(self, wav_path: Path) -> np.ndarray` [IO-R]：加载 `wav samples` 对应的数据或结果。
  关键调用：`wavfile.read`。
- `M L48-L54` `AVCoreStepsMixin._normalize_series(self, values: np.ndarray) -> np.ndarray`：规范化 `series` 对应的数据或结果。
- `M L56-L81` `AVCoreStepsMixin._build_feature_series(self, samples: np.ndarray, target_hz: int, mode: str='energy') -> np.ndarray`：构建 `feature series` 对应的数据或结果。
  关键调用：`self._normalize_series`。
- `M L83-L95` `AVCoreStepsMixin._lowpass_samples(self, samples: np.ndarray) -> np.ndarray`：执行 `lowpass samples` 对应逻辑。
- `M L97-L103` `AVCoreStepsMixin._build_music_refine_series(self, samples: np.ndarray) -> np.ndarray`：构建 `music refine series` 对应的数据或结果。
  关键调用：`self._build_feature_series`, `self._lowpass_samples`。
- `M L105-L132` `AVCoreStepsMixin._estimate_best_start_frame(self, long_series: np.ndarray, short_series: np.ndarray) -> tuple[float, float]`：估算 `best start frame` 对应的数据或结果。
- `M L134-L180` `AVCoreStepsMixin._estimate_offset_seconds(self, video_audio_samples: np.ndarray, song_audio_samples: np.ndarray) -> tuple[float, float, float]`：估算 `offset seconds` 对应的数据或结果。
  关键调用：`self._build_feature_series`, `self._build_music_refine_series`, `self._estimate_best_start_frame`。
- `M L182-L195` `AVCoreStepsMixin._parse_verify_hit_times_ms(self, verify_path: Path) -> list[int]` [IO-R]：解析 `verify hit times ms` 对应的数据或结果。
  关键调用：`verify_path.read_text`。
- `M L197-L210` `AVCoreStepsMixin._build_verify_click_train(self, hit_times_ms: list[int], length_frames: int) -> np.ndarray`：构建 `verify click train` 对应的数据或结果。
  关键调用：`self._normalize_series`。
- `M L212-L261` `AVCoreStepsMixin._estimate_verify_adjustment_seconds(self, transient_series: np.ndarray, verify_path: Path, base_offset_seconds: float) -> tuple[float, dict[str, float]] | None`：估算 `verify adjustment seconds` 对应的数据或结果。
  关键调用：`self._build_verify_click_train`, `self._normalize_series`, `self._parse_verify_hit_times_ms`。
- `M L263-L285` `AVCoreStepsMixin._validate_trim_window(self, offset_seconds: float, song_duration_seconds: float, video_duration_seconds: float) -> float`：校验 `trim window` 对应的数据或结果。
- `M L287-L301` `AVCoreStepsMixin._trim_video(self, source_video_path: Path, output_video_path: Path, trim_start_seconds: float, trim_duration_seconds: float)` [PROCESS]：执行 `trim video` 对应逻辑。
  关键调用：`build_trim_video_args`, `run_ffmpeg`。
- `M L303-L412` `AVCoreStepsMixin.process_one(self, folder_name: str, overwrite: bool=False) -> BatchProcessResult`：处理 manifest 中的单个内部谱面文件夹。
  关键调用：`self._build_feature_series`, `self._ensure_required_steps_done`, `self._estimate_offset_seconds`, `self._estimate_verify_adjustment_seconds`, `self._extract_audio_to_wav`, `self._load_wav_samples`。

## `src/Traning/Lib/video/av_processing/wrapup.py`

职责：记录 AV 阶段进度、完成细节和失败状态。
工程依赖：`Traning.Lib.common.failures`

- `C L6-L74` `AVWrapUpMixin` [CLASS]：封装 `AVWrapUpMixin` 相关数据或行为。
- `M L7-L11` `AVWrapUpMixin._update_progress(self, folder_name: str, stage: str, detail: dict | None=None)`：执行 `update progress` 对应逻辑。
  关键调用：`self.status_manager.mark_step_pending`。
- `M L13-L14` `AVWrapUpMixin.progress_message(self, index: int, total: int, folder_name: str) -> str | None`：生成当前批处理进度文本。
- `M L16-L58` `AVWrapUpMixin._build_done_detail(self, *, source_video_path, output_video_path, song_audio_path, verify_path, raw_offset_seconds: float, verify_adjustment_seconds: float, global_offset_seconds: float, offset_seconds: float, trim_start_seconds: float, song_duration_seconds: float, score: float, coarse_score: float, verify_detail: dict[str, float] | None) -> dict`：构建 `done detail` 对应的数据或结果。
- `M L60-L65` `AVWrapUpMixin._mark_done(self, folder_name: str, **detail_kwargs)`：更新状态为 `done` 对应的数据或结果。
  关键调用：`self._build_done_detail`, `self.status_manager.mark_step_done`。
- `M L67-L74` `AVWrapUpMixin.handle_failure(self, folder_name: str, error: Exception)`：处理单文件夹失败并同步失败状态。
  关键调用：`exception_detail`, `self.status_manager.ensure_status_file`, `self.status_manager.mark_step_pending`, `self.store.folder_exists`。

## `src/Traning/Lib/video/clipping/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`Traning.Lib.video.clipping.clipping`

- 无命名函数、方法或类。

## `src/Traning/Lib/video/clipping/clipping.py`

职责：组合固定区域裁剪处理器并校验配置。
工程依赖：`Traning.Lib.beatmap.folder_store`, `Traning.Lib.common.batch`, `Traning.Lib.common.failures`, `Traning.Lib.defaults`, `Traning.Lib.video.clipping.geometry`, `Traning.Lib.video.clipping.preflight`, `Traning.Lib.video.clipping.steps`, `Traning.Lib.video.clipping.wrapup`, `Traning.conf`, `Traning.conf.legacy_config`, `Traning.state.process_status`

- `F L24-L36` `build_fixed_region_video_crop_processor_from_config_or_default(config_path: Path | None=None) -> 'FixedRegionVideoCropProcessor'`：构建并返回 `fixed region video crop processor from config or default` 对应的数据或结果。
  关键调用：`FixedRegionVideoCropProcessor`, `format_exception`, `load_settings`。
- `C L39-L112` `FixedRegionVideoCropProcessor(ClipWrapUpMixin, ClipStepsMixin, ClipGeometryMixin, ClipPreflightMixin, FolderBatchProcessor)` [CLASS]：封装 `FixedRegionVideoCropProcessor` 相关数据或行为。
- `M L47-L52` `FixedRegionVideoCropProcessor.from_settings(cls, settings: Settings, status_manager: ProcessStatusManager | None=None) -> 'FixedRegionVideoCropProcessor'`：从 Settings 创建处理器实例。
- `M L54-L112` `FixedRegionVideoCropProcessor.__init__(self, settings: Settings=DEFAULTS, status_manager: ProcessStatusManager | None=None, **overrides: object)`：初始化实例依赖、配置和运行状态。
  关键调用：`BeatmapFolderStore`, `ProcessStatusManager`, `assign_group`, `group_values`, `self._ensure_status_steps_registered`, `settings_namespace`。
- `C L115-L116` `VideoClipProcessor(FixedRegionVideoCropProcessor)` [CLASS]：Task-aligned name for the fixed-region video clip processor。
- `F L119-L121` `main()`：独立脚本入口，构建处理器并执行。
  关键调用：`build_fixed_region_video_crop_processor_from_config_or_default`, `processor.run`。

## `src/Traning/Lib/video/clipping/geometry.py`

职责：按参考分辨率缩放裁剪矩形，并校验边界和编码偶数尺寸。
工程依赖：`Traning.Lib.tools.ffmpeg`

- `C L8-L95` `ClipGeometryMixin` [CLASS]：封装 `ClipGeometryMixin` 相关数据或行为。
- `M L9-L10` `ClipGeometryMixin.get_video_size(self, video_path: Path) -> tuple[int, int]`：获取 `video size` 对应的数据或结果。
- `M L12-L18` `ClipGeometryMixin._scale_crop_coordinate(self, value: int, reference_size: int, video_size: int) -> int`：执行 `scale crop coordinate` 对应逻辑。
- `M L20-L75` `ClipGeometryMixin._resolve_scaled_crop(self, video_width: int, video_height: int) -> dict[str, int]`：解析并定位 `scaled crop` 对应的数据或结果。
  关键调用：`self._scale_crop_coordinate`。
- `M L77-L92` `ClipGeometryMixin._validate_crop_bounds(self, video_path: Path) -> tuple[int, int, dict[str, int]]`：校验 `crop bounds` 对应的数据或结果。
  关键调用：`self._resolve_scaled_crop`, `self.get_video_size`。
- `M L94-L95` `ClipGeometryMixin.describe_crop_for_video(self, video_path: Path) -> tuple[int, int, dict[str, int]]`：执行 `describe crop for video` 对应逻辑。
  关键调用：`self._validate_crop_bounds`。

## `src/Traning/Lib/video/clipping/preflight.py`

职责：校验裁剪所需状态步骤和单文件夹前置条件。

- `C L4-L32` `ClipPreflightMixin` [CLASS]：封装 `ClipPreflightMixin` 相关数据或行为。
- `M L5-L14` `ClipPreflightMixin._ensure_status_steps_registered(self)`：确保 `status steps registered` 对应的数据或结果。
- `M L16-L32` `ClipPreflightMixin._ensure_folder_ready(self, folder_name: str, overwrite: bool) -> bool`：确保 `folder ready` 对应的数据或结果。
  关键调用：`self.status_manager.ensure_status_file`, `self.status_manager.is_step_done`, `self.store.folder_exists`。

## `src/Traning/Lib/video/clipping/steps.py`

职责：单文件夹原地裁剪，使用临时文件保证替换完整。
工程依赖：`Traning.Lib.common.batch`, `Traning.Lib.tools.ffmpeg`

- `C L9-L47` `ClipStepsMixin` [CLASS]：封装 `ClipStepsMixin` 相关数据或行为。
- `M L10-L29` `ClipStepsMixin._crop_video_in_place(self, video_path: Path, crop_info: dict[str, int])` [IO-W PROCESS]：执行 `crop video in place` 对应逻辑。
  关键调用：`build_crop_video_args`, `run_ffmpeg`, `temp_output_path.replace`。
- `M L31-L47` `ClipStepsMixin.process_one(self, folder_name: str, overwrite: bool=False) -> BatchProcessResult`：处理 manifest 中的单个内部谱面文件夹。
  关键调用：`self._crop_video_in_place`, `self._ensure_folder_ready`, `self._mark_cropping`, `self._mark_done`, `self._validate_crop_bounds`, `self.store.get_file_path`。

## `src/Traning/Lib/video/clipping/wrapup.py`

职责：记录裁剪进度、参考坐标、实际坐标和失败状态。
工程依赖：`Traning.Lib.common.failures`

- `C L8-L71` `ClipWrapUpMixin` [CLASS]：封装 `ClipWrapUpMixin` 相关数据或行为。
- `M L9-L10` `ClipWrapUpMixin.progress_message(self, index: int, total: int, folder_name: str) -> str | None`：生成当前批处理进度文本。
- `M L12-L20` `ClipWrapUpMixin._reference_detail(self) -> dict[str, int]`：执行 `reference detail` 对应逻辑。
- `M L22-L37` `ClipWrapUpMixin._mark_cropping(self, folder_name: str, video_path: Path, crop_info: dict[str, int])`：更新状态为 `cropping` 对应的数据或结果。
  关键调用：`self._reference_detail`, `self.status_manager.mark_step_pending`。
- `M L39-L58` `ClipWrapUpMixin._mark_done(self, folder_name: str, video_path: Path, video_width: int, video_height: int, crop_info: dict[str, int])`：更新状态为 `done` 对应的数据或结果。
  关键调用：`self._reference_detail`, `self.status_manager.mark_step_done`。
- `M L60-L71` `ClipWrapUpMixin.handle_failure(self, folder_name: str, error: Exception)`：处理单文件夹失败并同步失败状态。
  关键调用：`exception_detail`, `self._reference_detail`, `self.status_manager.ensure_status_file`, `self.status_manager.mark_step_pending`, `self.store.folder_exists`。

## `src/Traning/Lib/video/cut.py`

职责：预留的空模块，目前没有实现。

- 无命名函数、方法或类。

## `src/Traning/Lib/video/matching/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`Traning.Lib.video.matching.builders`, `Traning.Lib.video.matching.matching`, `Traning.Lib.video.matching.renamer`

- 无命名函数、方法或类。

## `src/Traning/Lib/video/matching/builders.py`

职责：视频顺序匹配器的兼容配置 builder。
工程依赖：`Traning.Lib.common.batch`, `Traning.Lib.video.matching.renamer`, `Traning.conf.legacy_config`

- `F L14-L15` `_load_video_package_renamer_config(config: ConfigReader) -> dict[str, object]`：加载 `video package renamer config` 对应的数据或结果。
  关键调用：`read_config_values`。
- `F L18-L26` `build_video_package_renamer_from_config_or_default(config_path: Path | None=None) -> VideoPackageRenamer`：构建并返回 `video package renamer from config or default` 对应的数据或结果。
  关键调用：`build_from_config_or_default`。

## `src/Traning/Lib/video/matching/matching.py`

职责：视频匹配策略入口；在音频匹配与时间顺序重命名之间切换。
工程依赖：`Traning.Lib.defaults`, `Traning.Lib.video.matching.renamer`, `Traning.conf`, `Traning.conf.legacy_config`

- `C L9-L39` `VideoMatchProcessor` [CLASS]：Video matching entry point used by the video/match task。
- `M L12-L23` `VideoMatchProcessor.__init__(self, settings: Settings=DEFAULTS, **overrides: object)`：初始化实例依赖、配置和运行状态。
  关键调用：`assign_group`, `settings_namespace`。
- `M L25-L39` `VideoMatchProcessor.run(self) -> None`：执行该处理器的完整工作流。
  关键调用：`AudioMatchProcessor`, `AudioMatchProcessor.run`, `VideoMatchRenamer`, `VideoMatchRenamer.run`, `forward_kwargs`。

## `src/Traning/Lib/video/matching/renamer.py`

职责：按录像时间与 manifest 顺序一一对应移动视频，并支持异常回滚。
工程依赖：`Traning.Lib.beatmap.manifest`, `Traning.Lib.common.failures`, `Traning.Lib.common.pathspec`, `Traning.Lib.defaults`, `Traning.conf`, `Traning.conf.legacy_config`, `Traning.state.process_status`

- `C L22-L187` `VideoPackageRenamer` [CLASS]：封装 `VideoPackageRenamer` 相关数据或行为。
- `M L23-L46` `VideoPackageRenamer.__init__(self, settings: Settings=DEFAULTS, status_manager: ProcessStatusManager | None=None, **overrides: object)`：初始化实例依赖、配置和运行状态。
  关键调用：`ManifestFolderWalker`, `ProcessStatusManager`, `assign_group`, `settings_namespace`, `suffix_spec`。
- `M L48-L52` `VideoPackageRenamer._folder_has_video(self, folder_path: Path) -> bool`：执行 `folder has video` 对应逻辑。
  关键调用：`matches_name`。
- `M L54-L69` `VideoPackageRenamer._parse_video_time(self, path: Path) -> datetime`：解析 `video time` 对应的数据或结果。
- `M L71-L82` `VideoPackageRenamer._list_videos_in_time_order(self) -> list[Path]`：列出 `videos in time order` 对应的数据或结果。
  关键调用：`filter_files`, `self._parse_video_time`, `self.video_root.exists`, `self.video_root.iterdir`。
- `M L84-L115` `VideoPackageRenamer._pending_folder_names(self) -> list[str]`：执行 `pending folder names` 对应逻辑。
  关键调用：`failure_detail`, `self._folder_has_video`, `self.status_manager.ensure_status_file`, `self.status_manager.is_step_done`, `self.status_manager.mark_step_done`, `self.status_manager.mark_step_pending`。
- `M L117-L142` `VideoPackageRenamer._build_rename_plan(self) -> list[tuple[str, Path, Path]]`：构建 `rename plan` 对应的数据或结果。
  关键调用：`self._list_videos_in_time_order`, `self._pending_folder_names`。
- `M L144-L187` `VideoPackageRenamer.run(self)` [IO-W]：执行该处理器的完整工作流。
  关键调用：`exception_detail`, `self._build_rename_plan`, `self.status_manager.mark_step_done`, `self.status_manager.mark_step_pending`。
- `C L190-L191` `VideoMatchRenamer(VideoPackageRenamer)` [CLASS]：Task-aligned name for sequence-based video matching。

## `src/Traning/Lib/video/segmentation/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`Traning.Lib.video.segmentation.segmentation`

- 无命名函数、方法或类。

## `src/Traning/Lib/video/segmentation/planner.py`

职责：复用 osu!standard 解析器和 slider 曲线 API，按 200ms 高优先级窗口或半缩圈时间+路径重合率聚组，并保证每个对象恰好归属一个片段。
工程依赖：`Traning.Lib.beatmap.hit_objects`, `Traning.Lib.beatmap.verification.parser`

- `C L35-L47` `ParsedStandardBeatmap` [CLASS]：封装 `ParsedStandardBeatmap` 相关数据或行为。
- `M L46-L47` `ParsedStandardBeatmap.approach_preempt_ms(self) -> float` [PROPERTY]：执行 `approach preempt ms` 对应逻辑。
  关键调用：`approach_preempt_ms`。
- `C L51-L83` `SegmentPlan` [CLASS]：封装 `SegmentPlan` 相关数据或行为。
- `M L66-L67` `SegmentPlan.duration_seconds(self) -> float` [PROPERTY]：执行 `duration seconds` 对应逻辑。
- `M L70-L71` `SegmentPlan.pre_context_seconds(self) -> float` [PROPERTY]：执行 `pre context seconds` 对应逻辑。
- `M L74-L75` `SegmentPlan.post_context_seconds(self) -> float` [PROPERTY]：执行 `post context seconds` 对应逻辑。
- `M L78-L79` `SegmentPlan.clip_start_ms(self) -> int` [PROPERTY]：执行 `clip start ms` 对应逻辑。
- `M L82-L83` `SegmentPlan.clip_end_ms(self) -> int` [PROPERTY]：执行 `clip end ms` 对应逻辑。
- `F L86-L123` `parse_standard_beatmap(osu_path: Path) -> ParsedStandardBeatmap`：解析 `standard beatmap` 对应的数据或结果。
  关键调用：`ParsedStandardBeatmap`, `VerifyOsuParser`, `difficulty.get`, `general.get`, `parser.parse_hitobjects`, `parser.parse_key_value_section`。
- `F L126-L127` `parse_standard_hit_objects(osu_path: Path) -> list[HitObject]`：解析 `standard hit objects` 对应的数据或结果。
  关键调用：`parse_standard_beatmap`。
- `F L130-L137` `approach_preempt_ms(approach_rate: float) -> float`：执行 `approach preempt ms` 对应逻辑。
- `F L140-L152` `circle_radius_from_size(circle_size: float) -> float`：执行 `circle radius from size` 对应逻辑。
- `F L155-L166` `circle_overlap_ratio(distance: float, radius: float) -> float`：执行 `circle overlap ratio` 对应逻辑。
- `F L169-L188` `_slider_polyline(slider: Slider) -> tuple[tuple[float, float], ...]`：执行 `slider polyline` 对应逻辑。
- `F L191-L198` `_object_polyline(hit_object: HitObject) -> tuple[tuple[float, float], ...]`：执行 `object polyline` 对应逻辑。
  关键调用：`_slider_polyline`。
- `F L201-L219` `_point_to_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float`：执行 `point to segment distance` 对应逻辑。
- `F L222-L230` `_orientation(first: tuple[float, float], second: tuple[float, float], third: tuple[float, float]) -> float`：执行 `orientation` 对应逻辑。
- `F L233-L255` `_segments_intersect(first_start: tuple[float, float], first_end: tuple[float, float], second_start: tuple[float, float], second_end: tuple[float, float]) -> bool`：执行 `segments intersect` 对应逻辑。
  关键调用：`_orientation`, `_point_to_segment_distance`。
- `F L258-L310` `_polyline_distance(first: tuple[tuple[float, float], ...], second: tuple[tuple[float, float], ...]) -> float`：执行 `polyline distance` 对应逻辑。
  关键调用：`_point_to_segment_distance`, `_segments_intersect`。
- `F L313-L323` `hit_objects_overlap_ratio(first: HitObject, second: HitObject, *, circle_radius: float) -> float`：执行 `hit objects overlap ratio` 对应逻辑。
  关键调用：`_object_polyline`, `_polyline_distance`, `circle_overlap_ratio`。
- `F L326-L395` `group_hit_objects(hit_objects: list[HitObject], overlap_merge_window_ms: int, *, circle_size: float=5.0, min_circle_overlap_ratio: float=0.5, priority_merge_window_ms: int=0, use_priority_merge: bool=True) -> list[list[HitObject]]`：执行 `group hit objects` 对应逻辑。
  关键调用：`_object_polyline`, `_polyline_distance`, `circle_overlap_ratio`, `circle_radius_from_size`。
- `F L398-L418` `classify_hit_group(hit_group: list[HitObject]) -> SegmentCategory`：Classify by contained object types; mixed groups may contain many sliders。
- `F L421-L510` `build_segment_plans(hit_objects: list[HitObject], *, approach_preempt_ratio: float, circle_size: float, min_circle_overlap_ratio: float, priority_merge_window_ms: int, use_priority_merge: bool, approach_preempt_seconds: float, post_context_seconds: float, video_duration_seconds: float) -> list[SegmentPlan]`：构建并返回 `segment plans` 对应的数据或结果。
  关键调用：`SegmentPlan`, `circle_radius_from_size`, `classify_hit_group`, `group_hit_objects`。

## `src/Traning/Lib/video/segmentation/segmentation.py`

职责：为每个训练样本目录生成 video.mp4、相对时间 beatmap.json、CSV 索引和状态。
工程依赖：`Traning.Lib.beatmap.folder_store`, `Traning.Lib.beatmap.hit_objects`, `Traning.Lib.beatmap.verification.parser`, `Traning.Lib.common.batch`, `Traning.Lib.common.failures`, `Traning.Lib.defaults`, `Traning.Lib.tools.ffmpeg`, `Traning.Lib.video.segmentation.planner`, `Traning.conf`, `Traning.state.process_status`

- `C L76-L579` `VideoSegmentationProcessor(FolderBatchProcessor)` [CLASS]：封装 `VideoSegmentationProcessor` 相关数据或行为。
- `M L77-L118` `VideoSegmentationProcessor.__init__(self, settings: Settings=DEFAULTS, status_manager: ProcessStatusManager | None=None)`：初始化实例依赖、配置和运行状态。
  关键调用：`BeatmapFolderStore`, `ProcessStatusManager`, `VerifyOsuParser`, `self._ensure_status_steps_registered`, `self._recover_interrupted_outputs`, `self._sync_manifest_table`。
- `M L120-L132` `VideoSegmentationProcessor._recover_interrupted_outputs(self) -> None` [IO-W]：执行 `recover interrupted outputs` 对应逻辑。
  关键调用：`self._remove_output_path`, `self.segment_root.glob`, `self.segment_root.mkdir`。
- `M L134-L142` `VideoSegmentationProcessor._sync_manifest_table(self) -> None` [IO-W]：同步 `manifest table` 对应的数据或结果。
  关键调用：`self.segment_root.mkdir`, `temporary_table.replace`。
- `M L144-L154` `VideoSegmentationProcessor._ensure_status_steps_registered(self) -> None`：确保 `status steps registered` 对应的数据或结果。
- `M L156-L157` `VideoSegmentationProcessor.progress_message(self, index: int, total: int, folder_name: str) -> str`：生成当前批处理进度文本。
- `M L159-L160` `VideoSegmentationProcessor._output_directory(self, folder_name: str) -> Path`：执行 `output directory` 对应逻辑。
- `M L162-L185` `VideoSegmentationProcessor._output_complete(self, folder_name: str) -> bool` [IO-W]：执行 `output complete` 对应逻辑。
  关键调用：`self._output_directory`。
- `M L187-L194` `VideoSegmentationProcessor._ensure_required_steps_done(self, folder_name: str) -> None`：确保 `required steps done` 对应的数据或结果。
  关键调用：`self.status_manager.is_step_done`。
- `M L196-L200` `VideoSegmentationProcessor._segment_directory_name(self, index: int, plan: SegmentPlan) -> str`：执行 `segment directory name` 对应逻辑。
- `M L202-L208` `VideoSegmentationProcessor._overlap_merge_window_ms(self, beatmap: ParsedStandardBeatmap) -> int`：执行 `overlap merge window ms` 对应逻辑。
- `M L210-L223` `VideoSegmentationProcessor._write_segment(self, source_video_path: Path, output_path: Path, plan: SegmentPlan) -> None` [PROCESS]：写入 `segment` 对应的数据或结果。
  关键调用：`build_segment_video_args`, `run_ffmpeg`。
- `M L225-L240` `VideoSegmentationProcessor._serialize_hit_object(self, hit_object: HitObject, source_index: int, clip_start_ms: int) -> dict[str, object]`：执行 `serialize hit object` 对应逻辑。
  关键调用：`self.parser.hit_object_to_dict`。
- `M L242-L300` `VideoSegmentationProcessor._write_beatmap_data(self, output_path: Path, *, folder_name: str, source_osu_path: Path, segment_id: str, beatmap: ParsedStandardBeatmap, plan: SegmentPlan) -> None` [IO-W]：写入 `beatmap data` 对应的数据或结果。
  关键调用：`output_path.write_text`, `self._overlap_merge_window_ms`, `self._serialize_hit_object`。
- `M L302-L317` `VideoSegmentationProcessor._write_segment_table(self, output_directory: Path, rows: list[dict[str, object]]) -> None` [IO-W]：写入 `segment table` 对应的数据或结果。
- `M L319-L323` `VideoSegmentationProcessor._remove_output_path(self, path: Path) -> None` [IO-W]：执行 `remove output path` 对应逻辑。
- `M L325-L465` `VideoSegmentationProcessor._build_output(self, folder_name: str, source_video_path: Path, source_osu_path: Path, beatmap: ParsedStandardBeatmap, plans: list[SegmentPlan]) -> tuple[Path, dict[str, int]]` [IO-W]：构建 `output` 对应的数据或结果。
  关键调用：`self._output_directory`, `self._overlap_merge_window_ms`, `self._remove_output_path`, `self._segment_directory_name`, `self._write_beatmap_data`, `self._write_segment`。
- `M L467-L570` `VideoSegmentationProcessor.process_one(self, folder_name: str, overwrite: bool=False) -> BatchProcessResult`：处理 manifest 中的单个内部谱面文件夹。
  关键调用：`build_segment_plans`, `failure_detail`, `get_media_duration_seconds`, `parse_standard_beatmap`, `self._build_output`, `self._ensure_required_steps_done`。
- `M L572-L579` `VideoSegmentationProcessor.handle_failure(self, folder_name: str, error: Exception) -> None`：处理单文件夹失败并同步失败状态。
  关键调用：`exception_detail`, `self.status_manager.ensure_status_file`, `self.status_manager.mark_step_pending`, `self.store.folder_exists`。

## `src/Traning/conf/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`Traning.conf.runtime`, `Traning.conf.settings`

- 无命名函数、方法或类。

## `src/Traning/conf/field_groups.py`

职责：集中声明处理器字段组，负责批量赋值和处理器之间的参数转发。

- `F L72-L73` `group_values(config: SimpleNamespace, group: str) -> tuple[Any, ...]`：执行 `group values` 对应逻辑。
- `F L76-L78` `assign_group(target: Any, config: SimpleNamespace, group: str) -> None`：执行 `assign group` 对应逻辑。
- `F L81-L82` `forward_kwargs(source: Any, group: str) -> dict[str, Any]`：执行 `forward kwargs` 对应逻辑。

## `src/Traning/conf/legacy_config.py`

职责：旧 builder API 的兼容层；把 Settings 展平、覆盖并按构造函数签名转发。
工程依赖：`Traning.Lib.artifacts`, `Traning.Lib.common.failures`, `Traning.conf`, `Traning.conf.field_groups`

- `C L20-L21` `CheckDataConfigError(Exception)` [CLASS]：封装 `CheckDataConfigError` 相关数据或行为。
- `C L24-L37` `ConfigReader` [CLASS]：Small compatibility reader for legacy build_* helpers。
- `M L27-L31` `ConfigReader.__init__(self, path: Path | None=None)`：初始化实例依赖、配置和运行状态。
  关键调用：`load_settings`, `self.settings.model_dump`。
- `M L33-L34` `ConfigReader.get(self, *_paths: tuple[str, ...]) -> Any`：执行 `get` 对应逻辑。
- `M L36-L37` `ConfigReader.read(self, *_args: Any, **_kwargs: Any) -> Any`：执行 `read` 对应逻辑。
- `F L52-L53` `load_config(config_path: Path | None=None) -> ConfigReader`：加载 `config` 对应的数据或结果。
  关键调用：`ConfigReader`。
- `F L56-L57` `load_process_steps_config(config_path: Path | None=None) -> tuple[str, ...]`：加载 `process steps config` 对应的数据或结果。
  关键调用：`load_settings`。
- `F L60-L69` `load_process_steps_config_or_default(config_path: Path | None=None, default_steps: Iterable[str] | None=None) -> tuple[str, ...]`：加载 `process steps config or default` 对应的数据或结果。
  关键调用：`load_process_steps_config`。
- `F L72-L119` `settings_kwargs(settings: Settings, processor: str | None=None) -> dict[str, Any]`：执行 `settings kwargs` 对应逻辑。
- `F L122-L139` `_coerce_like(reference: Any, value: Any) -> Any`：执行 `coerce like` 对应逻辑。
- `F L142-L153` `settings_namespace(settings: Settings, processor: str | None=None, overrides: Mapping[str, Any] | None=None) -> SimpleNamespace`：执行 `settings namespace` 对应逻辑。
  关键调用：`_coerce_like`, `settings_kwargs`。
- `F L156-L157` `_settings_kwargs(settings: Settings) -> dict[str, Any]`：执行 `settings kwargs` 对应逻辑。
  关键调用：`settings_kwargs`。
- `F L160-L174` `_filter_builder_kwargs(builder: Callable[..., T], config: Mapping[str, Any]) -> dict[str, Any]`：筛选 `builder kwargs` 对应的数据或结果。
- `F L177-L183` `build_from_config(builder: Callable[..., T], _loaders: Iterable[Callable[[ConfigReader], Mapping[str, Any]]], config_path: Path | None=None) -> T`：构建并返回 `from config` 对应的数据或结果。
  关键调用：`_filter_builder_kwargs`, `_settings_kwargs`, `load_settings`。
- `F L186-L200` `build_from_config_or_default(builder: Callable[..., T], loaders: Iterable[Callable[[ConfigReader], Mapping[str, Any]]], config_path: Path | None=None, default_builder: Callable[[], T] | None=None) -> T`：构建并返回 `from config or default` 对应的数据或结果。
  关键调用：`build_from_config`, `format_exception`。

## `src/Traning/conf/runtime.py`

职责：把 Prefect home 固定到仓库内可写目录。
工程依赖：`Traning.conf.settings`

- `F L9-L12` `ensure_prefect_home(repo_root: Path=REPO_ROOT) -> Path`：确保 `prefect home` 对应的数据或结果。

## `src/Traning/conf/settings.py`

职责：Pydantic 配置模型与 YAML/JSON 加载；解析相对路径并兼容旧配置层级。

- `C L16-L17` `SettingsError(Exception)` [CLASS]：封装 `SettingsError` 相关数据或行为。
- `C L20-L22` `RuntimeSettings(BaseModel)` [CLASS]：封装 `RuntimeSettings` 相关数据或行为。
- `C L25-L28` `CheckDataSettings(BaseModel)` [CLASS]：封装 `CheckDataSettings` 相关数据或行为。
- `C L31-L44` `VideoClipSettings(BaseModel)` [CLASS]：封装 `VideoClipSettings` 相关数据或行为。
- `M L41-L44` `VideoClipSettings._finite_offset(cls, value: float) -> float` [VALIDATOR]：执行 `finite offset` 对应逻辑。
- `C L47-L54` `FileManagementSettings(BaseModel)` [CLASS]：封装 `FileManagementSettings` 相关数据或行为。
- `C L57-L59` `FileFormatSettings(BaseModel)` [CLASS]：封装 `FileFormatSettings` 相关数据或行为。
- `C L62-L68` `AVSettings(BaseModel)` [CLASS]：封装 `AVSettings` 相关数据或行为。
- `C L71-L73` `AudioMatchSettings(BaseModel)` [CLASS]：封装 `AudioMatchSettings` 相关数据或行为。
- `C L76-L84` `PackageSettings(BaseModel)` [CLASS]：封装 `PackageSettings` 相关数据或行为。
- `C L87-L95` `ClipSettings(BaseModel)` [CLASS]：封装 `ClipSettings` 相关数据或行为。
- `C L98-L135` `SegmentSettings(BaseModel)` [CLASS]：封装 `SegmentSettings` 相关数据或行为。
- `M L109-L112` `SegmentSettings._nonnegative_interval(cls, value: int) -> int` [VALIDATOR]：执行 `nonnegative interval` 对应逻辑。
- `M L116-L119` `SegmentSettings._approach_ratio(cls, value: float) -> float` [VALIDATOR]：执行 `approach ratio` 对应逻辑。
- `M L123-L126` `SegmentSettings._overlap_ratio(cls, value: float) -> float` [VALIDATOR]：执行 `overlap ratio` 对应逻辑。
- `M L130-L135` `SegmentSettings._nonnegative_context(cls, value: float) -> float` [VALIDATOR]：执行 `nonnegative context` 对应逻辑。
- `C L138-L150` `ProgressSettings(BaseModel)` [CLASS]：封装 `ProgressSettings` 相关数据或行为。
- `C L153-L186` `Settings(BaseSettings)` [CLASS]：封装 `Settings` 相关数据或行为。
- `M L173-L174` `Settings.target_root(self) -> Path` [PROPERTY]：执行 `target root` 对应逻辑。
- `M L177-L178` `Settings.overwrite(self) -> bool` [PROPERTY]：执行 `overwrite` 对应逻辑。
- `M L181-L182` `Settings.continue_on_error(self) -> bool` [PROPERTY]：执行 `continue on error` 对应逻辑。
- `M L185-L186` `Settings.global_offset_ms(self) -> float` [PROPERTY]：执行 `global offset ms` 对应逻辑。
- `F L189-L199` `_resolve_paths(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]`：解析并定位 `paths` 对应的数据或结果。
  关键调用：`file_management.get`, `raw.get`。
- `F L202-L237` `_extract_nested(raw: dict[str, Any]) -> dict[str, Any]`：提取 `nested` 对应的数据或结果。
  关键调用：`ProgressSettings`, `parameters.get`, `progress.get`, `raw.get`, `required_steps.get`, `status_steps.get`。
- `F L240-L257` `_read_config(config_path: Path) -> dict[str, Any]` [IO-R IO-W]：读取 `config` 对应的数据或结果。
  关键调用：`SettingsError`。
- `F L260-L266` `load_settings(config_path: Path | None=None) -> Settings`：加载 `settings` 对应的数据或结果。
  关键调用：`Settings`, `SettingsError`, `_extract_nested`, `_read_config`, `_resolve_paths`。

## `src/Traning/core/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。

- 无命名函数、方法或类。

## `src/Traning/core/audio/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`Traning.core.audio.match`

- 无命名函数、方法或类。

## `src/Traning/core/audio/match.py`

职责：重导出音频匹配处理器，作为 core 层稳定入口。
工程依赖：`Traning.Lib.audio.matching`

- 无命名函数、方法或类。

## `src/Traning/core/beatmap/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`Traning.core.beatmap.difficulty`, `Traning.core.beatmap.importer`, `Traning.core.beatmap.pipeline`, `Traning.core.beatmap.verify`

- 无命名函数、方法或类。

## `src/Traning/core/beatmap/difficulty.py`

职责：读取谱面难度并写入 SQLite manifest 的业务入口。
工程依赖：`Traning.Lib.beatmap.difficulty`, `Traning.conf`

- `F L13-L23` `export_difficulty(settings: Settings) -> bool`：导出 `difficulty` 对应的数据或结果。
  关键调用：`BeatmapDifficultyProcessor`, `BeatmapDifficultyProcessor.run`, `build_store`。

## `src/Traning/core/beatmap/importer.py`

职责：谱面导入业务入口；把 Settings 映射到 BeatmapImportProcessor。
工程依赖：`Traning.Lib.beatmap.importing`, `Traning.conf`

- `F L11-L22` `import_beatmaps(settings: Settings) -> bool`：导入 `beatmaps` 对应的数据或结果。
  关键调用：`BeatmapImportProcessor`, `BeatmapImportProcessor.run`。

## `src/Traning/core/beatmap/pipeline.py`

职责：顺序组合谱面导入、校验导出和难度导出。
工程依赖：`Traning.conf`, `Traning.core.beatmap.difficulty`, `Traning.core.beatmap.importer`, `Traning.core.beatmap.verify`

- `F L9-L14` `prepare_beatmaps(settings: Settings) -> dict[str, bool]`：顺序准备 `beatmaps` 对应的数据或结果。
  关键调用：`export_difficulty`, `export_verify`, `import_beatmaps`。

## `src/Traning/core/beatmap/verify.py`

职责：verify.txt 导出业务入口，并构建受 manifest 约束的文件存储。
工程依赖：`Traning.Lib.beatmap.folder_store`, `Traning.Lib.beatmap.verification`, `Traning.conf`

- `F L12-L16` `build_store(settings: Settings) -> BeatmapFolderStore`：构建并返回 `store` 对应的数据或结果。
  关键调用：`BeatmapFolderStore`。
- `F L19-L28` `export_verify(settings: Settings) -> bool`：导出 `verify` 对应的数据或结果。
  关键调用：`BeatmapVerifyExporter`, `BeatmapVerifyExporter.run`, `build_store`。

## `src/Traning/core/flows/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。

- 无命名函数、方法或类。

## `src/Traning/core/flows/pipeline.py`

职责：用统一阶段表驱动七阶段 Prefect/direct 编排，并处理继续执行策略。
工程依赖：`Traning.Lib.common.failures`, `Traning.conf`, `Traning.core.beatmap`, `Traning.core.tasks.av`, `Traning.core.tasks.clip`, `Traning.core.tasks.difficulty`, `Traning.core.tasks.importer`, `Traning.core.tasks.match`, `Traning.core.tasks.segment`, `Traning.core.tasks.verify`, `Traning.core.video`

- `C L39-L45` `PipelineStage` [CLASS]：封装 `PipelineStage` 相关数据或行为。
- `F L108-L116` `_call_stage(label: str, stage_func: StageCall, continue_on_error: bool) -> bool`：执行 `call stage` 对应逻辑。
  关键调用：`format_exception`。
- `F L119-L120` `_enabled(override: bool | None, default: bool) -> bool`：执行 `enabled` 对应逻辑。
- `F L123-L144` `_run_stages(settings: Settings, overrides: dict[str, bool | None], *, use_prefect: bool) -> dict[str, bool]`：执行 `run stages` 对应逻辑。
  关键调用：`_call_stage`, `_enabled`。
- `F L148-L171` `train_pipeline(settings: Settings | None=None, run_get_files: bool | None=None, run_verify_export: bool | None=None, run_difficulty_export: bool | None=None, run_video_match: bool | None=None, run_av_correspondence: bool | None=None, run_clip_stage: bool | None=None, run_segment_stage: bool | None=None) -> dict[str, bool]` [PREFECT-FLOW]：执行 `train pipeline` 对应逻辑。
  关键调用：`_run_stages`, `load_settings`。
- `F L174-L197` `train_pipeline_direct(settings: Settings | None=None, run_get_files: bool | None=None, run_verify_export: bool | None=None, run_difficulty_export: bool | None=None, run_video_match: bool | None=None, run_av_correspondence: bool | None=None, run_clip_stage: bool | None=None, run_segment_stage: bool | None=None) -> dict[str, bool]`：执行 `train pipeline direct` 对应逻辑。
  关键调用：`_run_stages`, `load_settings`。
- `C L200-L245` `TemporaryTrainingRunner` [CLASS]：封装 `TemporaryTrainingRunner` 相关数据或行为。
- `M L201-L202` `TemporaryTrainingRunner.__init__(self, config_path: Path | None=None)`：初始化实例依赖、配置和运行状态。
  关键调用：`load_settings`。
- `M L204-L245` `TemporaryTrainingRunner.run(self, overwrite: bool=False, run_check_data: bool=True, run_get_files: bool=True, run_verify_export: bool=True, run_difficulty_export: bool=True, run_video_clip: bool=True, run_video_match: bool=True, run_av_correspondence: bool=True, run_clip_stage: bool=True, run_segment_stage: bool=True, use_audio_match_experiment: bool=True, global_offset_ms: float | None=None, continue_on_error: bool=False) -> dict[str, bool]`：执行该处理器的完整工作流。
  关键调用：`self.settings.model_copy`, `self.settings.runtime.model_copy`, `self.settings.video_clip.model_copy`, `train_pipeline_direct`。

## `src/Traning/core/tasks/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。

- `F L4-L7` `require_success(stage: str, success: bool) -> bool`：执行 `require success` 对应逻辑。

## `src/Traning/core/tasks/av.py`

职责：Prefect task 薄包装；调用同名 core 业务函数。
工程依赖：`Traning.conf`, `Traning.core.tasks`, `Traning.core.video.av`

- `F L15-L16` `av_correspondence_task(settings: Settings) -> bool` [PREFECT-TASK]：执行 `av correspondence task` 对应逻辑。
  关键调用：`av_correspondence`, `require_success`。

## `src/Traning/core/tasks/clip.py`

职责：Prefect task 薄包装；调用同名 core 业务函数。
工程依赖：`Traning.conf`, `Traning.core.tasks`, `Traning.core.video.clip`

- `F L15-L16` `crop_video_task(settings: Settings) -> bool` [PREFECT-TASK]：裁剪 `video task` 对应的数据或结果。
  关键调用：`crop_video`, `require_success`。

## `src/Traning/core/tasks/difficulty.py`

职责：Prefect task 薄包装；调用同名 core 业务函数。
工程依赖：`Traning.conf`, `Traning.core.beatmap.difficulty`, `Traning.core.tasks`

- `F L15-L16` `export_difficulty_task(settings: Settings) -> bool` [PREFECT-TASK]：导出 `difficulty task` 对应的数据或结果。
  关键调用：`export_difficulty`, `require_success`。

## `src/Traning/core/tasks/importer.py`

职责：Prefect task 薄包装；调用同名 core 业务函数。
工程依赖：`Traning.conf`, `Traning.core.beatmap.importer`, `Traning.core.tasks`

- `F L15-L16` `import_beatmaps_task(settings: Settings) -> bool` [PREFECT-TASK]：导入 `beatmaps task` 对应的数据或结果。
  关键调用：`import_beatmaps`, `require_success`。

## `src/Traning/core/tasks/match.py`

职责：Prefect task 薄包装；调用同名 core 业务函数。
工程依赖：`Traning.conf`, `Traning.core.tasks`, `Traning.core.video.match`

- `F L15-L16` `match_videos_task(settings: Settings) -> bool` [PREFECT-TASK]：匹配 `videos task` 对应的数据或结果。
  关键调用：`match_videos`, `require_success`。

## `src/Traning/core/tasks/segment.py`

职责：Prefect task 薄包装；调用同名 core 业务函数。
工程依赖：`Traning.conf`, `Traning.core.tasks`, `Traning.core.video.segment`

- `F L15-L16` `segment_videos_task(settings: Settings) -> bool` [PREFECT-TASK]：执行 `segment videos task` 对应逻辑。
  关键调用：`require_success`, `segment_videos`。

## `src/Traning/core/tasks/verify.py`

职责：Prefect task 薄包装；调用同名 core 业务函数。
工程依赖：`Traning.conf`, `Traning.core.beatmap.verify`, `Traning.core.tasks`

- `F L15-L16` `export_verify_task(settings: Settings) -> bool` [PREFECT-TASK]：导出 `verify task` 对应的数据或结果。
  关键调用：`export_verify`, `require_success`。

## `src/Traning/core/video/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`Traning.core.video.av`, `Traning.core.video.clip`, `Traning.core.video.match`, `Traning.core.video.pipeline`, `Traning.core.video.segment`

- 无命名函数、方法或类。

## `src/Traning/core/video/av.py`

职责：AV 对齐业务入口；把音频算法和状态参数传给 VideoAVProcessor。
工程依赖：`Traning.Lib.video.av_processing`, `Traning.conf`

- `F L13-L32` `av_correspondence(settings: Settings) -> bool`：执行 `av correspondence` 对应逻辑。
  关键调用：`VideoAVProcessor`, `VideoAVProcessor.run`。

## `src/Traning/core/video/clip.py`

职责：固定区域裁剪业务入口。
工程依赖：`Traning.Lib.video.clipping`, `Traning.conf`

- `F L11-L16` `crop_video(settings: Settings) -> bool`：裁剪 `video` 对应的数据或结果。
  关键调用：`VideoClipProcessor.from_settings`, `VideoClipProcessor.from_settings.run`。

## `src/Traning/core/video/match.py`

职责：视频匹配业务入口；处理“全部已有视频”的正常跳过情况。
工程依赖：`Traning.Lib.video.matching`, `Traning.conf`

- `F L11-L33` `match_videos(settings: Settings) -> bool`：匹配 `videos` 对应的数据或结果。
  关键调用：`VideoMatchProcessor`, `VideoMatchProcessor.run`。

## `src/Traning/core/video/pipeline.py`

职责：顺序组合视频匹配、AV 对齐、裁剪和谱面切分。
工程依赖：`Traning.conf`, `Traning.core.video.av`, `Traning.core.video.clip`, `Traning.core.video.match`, `Traning.core.video.segment`

- `F L10-L16` `prepare_videos(settings: Settings) -> dict[str, bool]`：顺序准备 `videos` 对应的数据或结果。
  关键调用：`av_correspondence`, `crop_video`, `match_videos`, `segment_videos`。

## `src/Traning/core/video/segment.py`

职责：最终谱面视频切分业务入口。
工程依赖：`Traning.Lib.video.segmentation`, `Traning.conf`

- `F L11-L16` `segment_videos(settings: Settings) -> bool`：执行 `segment videos` 对应逻辑。
  关键调用：`VideoSegmentationProcessor`, `VideoSegmentationProcessor.run`。

## `src/Traning/main.py`

职责：Typer CLI 入口；合并命令行覆盖项，选择 direct/Prefect runner，并渲染阶段结果。
工程依赖：`Traning.conf`, `Traning.core.flows.pipeline`

- `F L25-L26` `_resolve(default: bool, override: bool | None) -> bool`：执行 `resolve` 对应逻辑。
- `F L29-L30` `_skip(default: bool, skip_flag: bool) -> bool`：执行 `skip` 对应逻辑。
- `F L33-L60` `_settings(config: Path | None, overwrite: bool | None=None, continue_on_error: bool | None=None, use_audio_match_experiment: bool | None=None, global_offset_ms: float | None=None) -> Settings`：执行 `settings` 对应逻辑。
  关键调用：`_resolve`, `load_settings`。
- `F L63-L72` `_render(results: dict[str, bool], elapsed: float)`：执行 `render` 对应逻辑。
- `F L75-L84` `_run(settings: Settings, **stages: bool | None) -> int`：执行 `run` 对应逻辑。
  关键调用：`_render`, `os.environ.get`。
- `F L88-L132` `run_command(config: Path | None=typer.Option(None, '--config', help='config.yaml/json path.'), overwrite: bool | None=typer.Option(None, '--overwrite/--no-overwrite'), continue_on_error: bool | None=typer.Option(None, '--continue-on-error/--stop-on-error'), skip_get_files: bool=typer.Option(False, '--skip-get-files'), skip_verify_export: bool=typer.Option(False, '--skip-verify-export'), skip_difficulty_export: bool=typer.Option(False, '--skip-difficulty-export'), skip_video_match: bool=typer.Option(False, '--skip-video-match'), skip_av_correspondence: bool=typer.Option(False, '--skip-av-correspondence'), skip_clip: bool=typer.Option(False, '--skip-clip'), skip_segment: bool=typer.Option(False, '--skip-segment'), use_audio_match_experiment: bool | None=typer.Option(None, '--use-audio-match-experiment/--disable-audio-match-experiment'), global_offset_ms: float | None=typer.Option(None, '--global-offset-ms'))` [CLI]：执行 `run command` 对应逻辑。
  关键调用：`_run`, `_settings`, `_skip`。
- `F L136-L152` `verify_command(config: Path | None=typer.Option(None, '--config', help='config.yaml/json path.'), overwrite: bool | None=typer.Option(None, '--overwrite/--no-overwrite'), continue_on_error: bool | None=typer.Option(None, '--continue-on-error/--stop-on-error'))` [CLI]：执行 `verify command` 对应逻辑。
  关键调用：`_run`, `_settings`。
- `F L156-L176` `match_command(config: Path | None=typer.Option(None, '--config', help='config.yaml/json path.'), overwrite: bool | None=typer.Option(None, '--overwrite/--no-overwrite'), continue_on_error: bool | None=typer.Option(None, '--continue-on-error/--stop-on-error'), use_audio_match_experiment: bool | None=typer.Option(None, '--use-audio-match-experiment/--disable-audio-match-experiment'))` [CLI]：匹配 `command` 对应的数据或结果。
  关键调用：`_run`, `_settings`。
- `F L180-L197` `clip_command(config: Path | None=typer.Option(None, '--config', help='config.yaml/json path.'), overwrite: bool | None=typer.Option(None, '--overwrite/--no-overwrite'), continue_on_error: bool | None=typer.Option(None, '--continue-on-error/--stop-on-error'), global_offset_ms: float | None=typer.Option(None, '--global-offset-ms'))` [CLI]：执行 `clip command` 对应逻辑。
  关键调用：`_run`, `_settings`。
- `F L201-L217` `segment_command(config: Path | None=typer.Option(None, '--config', help='config.yaml/json path.'), overwrite: bool | None=typer.Option(None, '--overwrite/--no-overwrite'), continue_on_error: bool | None=typer.Option(None, '--continue-on-error/--stop-on-error'))` [CLI]：执行 `segment command` 对应逻辑。
  关键调用：`_run`, `_settings`。
- `F L221-L223` `default_command(ctx: typer.Context)` [CLI]：执行 `default command` 对应逻辑。
  关键调用：`_run`, `load_settings`。
- `F L226-L234` `main(argv: list[str] | None=None) -> int`：独立脚本入口，构建处理器并执行。

## `src/Traning/state/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。

- `F L7-L12` `__getattr__(name: str)`：执行 `getattr` 对应逻辑。

## `src/Traning/state/manifest_schema.py`

职责：SQLModel 训练包 manifest 表；保存内部目录 ID、原谱面名和处理顺序。

- `C L9-L24` `PackageManifestItem(SQLModel)` [CLASS]：封装 `PackageManifestItem` 相关数据或行为。

## `src/Traning/state/process_status.py`

职责：按谱面文件夹读写 SQLite 处理状态，并迁移旧 process_status.json。
工程依赖：`Traning.Lib.beatmap.manifest`, `Traning.conf`, `Traning.state.manifest_schema`, `Traning.state.status_schema`

- `C L25-L208` `ProcessStatusManager` [CLASS]：封装 `ProcessStatusManager` 相关数据或行为。
- `M L26-L48` `ProcessStatusManager.__init__(self, target_root: str, manifest_filename: str=MANIFEST_DB_FILENAME, status_filename: str='process_status.json', process_steps: Iterable[str] | None=None, db_filename: str=STATUS_DB_FILENAME)` [DB]：初始化实例依赖、配置和运行状态。
  关键调用：`ManifestFolderWalker`, `load_settings`, `normalize_process_steps`。
- `M L50-L56` `ProcessStatusManager._normalize_folder_name(self, folder_name: str) -> str`：规范化 `folder name` 对应的数据或结果。
- `M L58-L59` `ProcessStatusManager._registered_names(self) -> set[str]`：执行 `registered names` 对应逻辑。
  关键调用：`self.walker.read_folder_names`。
- `M L61-L66` `ProcessStatusManager._assert_registered(self, folder_name: str)`：执行 `assert registered` 对应逻辑。
  关键调用：`self._normalize_folder_name`, `self._registered_names`。
- `M L68-L74` `ProcessStatusManager._require_existing_folder(self, folder_name: str) -> Path`：执行 `require existing folder` 对应逻辑。
  关键调用：`self._assert_registered`, `self._normalize_folder_name`。
- `M L76-L77` `ProcessStatusManager._default_status(self) -> dict[str, Any]`：执行 `default status` 对应逻辑。
  关键调用：`default_status`。
- `M L79-L81` `ProcessStatusManager._validate_step(self, step: str)`：校验 `step` 对应的数据或结果。
- `M L83-L84` `ProcessStatusManager._normalize_status(self, raw_status: dict[str, Any] | None) -> dict[str, Any]`：规范化 `status` 对应的数据或结果。
  关键调用：`normalize_status`。
- `M L86-L97` `ProcessStatusManager._select_record(self, session: Session, folder_name: str, step: str) -> ProcessStepStatus | None` [DB]：选择 `record` 对应的数据或结果。
- `M L99-L105` `ProcessStatusManager._has_records(self, folder_name: str) -> bool` [DB]：执行 `has records` 对应逻辑。
- `M L107-L121` `ProcessStatusManager._migrate_legacy_status_key(self, folder_name: str) -> None` [DB]：执行 `migrate legacy status key` 对应逻辑。
  关键调用：`self.walker.source_name_for`。
- `M L123-L128` `ProcessStatusManager._load_legacy_json(self, folder_name: str) -> dict[str, Any] | None` [IO-R IO-W]：加载 `legacy json` 对应的数据或结果。
  关键调用：`self.get_status_path`。
- `M L130-L132` `ProcessStatusManager.get_status_path(self, folder_name: str) -> Path`：获取 `status path` 对应的数据或结果。
  关键调用：`self._require_existing_folder`。
- `M L134-L158` `ProcessStatusManager.load_status(self, folder_name: str) -> dict[str, Any]` [DB IO-W]：加载 `status` 对应的数据或结果。
  关键调用：`decode_detail`, `self._default_status`, `self._has_records`, `self._load_legacy_json`, `self._migrate_legacy_status_key`, `self._normalize_folder_name`。
- `M L160-L178` `ProcessStatusManager.save_status(self, folder_name: str, status: dict[str, Any])` [DB]：执行 `save status` 对应逻辑。
  关键调用：`ProcessStepStatus`, `encode_detail`, `self._normalize_folder_name`, `self._normalize_status`, `self._require_existing_folder`, `self._select_record`。
- `M L180-L183` `ProcessStatusManager.ensure_status_file(self, folder_name: str) -> dict[str, Any]`：确保 `status file` 对应的数据或结果。
  关键调用：`self.load_status`, `self.save_status`。
- `M L185-L188` `ProcessStatusManager.is_step_done(self, folder_name: str, step: str) -> bool`：判断是否 `step done` 对应的数据或结果。
  关键调用：`self._validate_step`, `self.load_status`。
- `M L190-L196` `ProcessStatusManager.mark_step_done(self, folder_name: str, step: str, detail: Any=None)`：更新状态为 `step done` 对应的数据或结果。
  关键调用：`self._validate_step`, `self.load_status`, `self.save_status`。
- `M L198-L204` `ProcessStatusManager.mark_step_pending(self, folder_name: str, step: str, detail: Any=None)`：更新状态为 `step pending` 对应的数据或结果。
  关键调用：`self._validate_step`, `self.load_status`, `self.save_status`。
- `M L206-L208` `ProcessStatusManager.get_steps_summary(self, folder_name: str) -> dict[str, bool]`：获取 `steps summary` 对应的数据或结果。
  关键调用：`self.load_status`。

## `src/Traning/state/status_schema.py`

职责：独立的 SQLModel 状态表、处理步骤规范化和 detail JSON 编解码。

- `C L22-L34` `ProcessStepStatus(SQLModel)` [CLASS]：封装 `ProcessStepStatus` 相关数据或行为。
- `F L37-L56` `normalize_process_steps(process_steps: Iterable[str]) -> tuple[str, ...]`：规范化 `process steps` 对应的数据或结果。
- `F L59-L69` `default_status(process_steps: Iterable[str]) -> dict[str, Any]`：执行 `default status` 对应逻辑。
- `F L72-L94` `normalize_status(raw_status: dict[str, Any] | None, process_steps: Iterable[str]) -> dict[str, Any]`：规范化 `status` 对应的数据或结果。
  关键调用：`default_status`, `raw_status.get`, `raw_step.get`, `raw_steps.get`。
- `F L97-L100` `encode_detail(detail: Any) -> str | None`：执行 `encode detail` 对应逻辑。
- `F L103-L109` `decode_detail(detail_json: str | None) -> Any`：执行 `decode detail` 对应逻辑。
