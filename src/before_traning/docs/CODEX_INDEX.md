# before_traning Codex Index

> 自动生成文件，请勿手工修改。运行 `python project_index/build_index.py` 重建。

面向 Codex 的低 token 工程导航；先按阶段定位，再读取命中的源码。

## 调用分层

```text
main.py -> core/pipeline.py:TRAINING_PIPELINE -> core stages
        -> Lib reusable APIs -> state / filesystem / SQLite / ffmpeg
tests/startup_checks/runner.py -> settings/pipeline/raw-data startup checks
tests/full_checks/runner.py -> full pytest checks
```

## 七阶段入口

| key | Core 入口 | 完成状态 |
|---|---|---|
| `import_beatmaps` | `core/beatmap/importer.py` | `osu_imported`, `audio_imported` |
| `verify_export` | `core/beatmap/verify.py` | `verify_exported` |
| `difficulty_export` | `core/beatmap/difficulty.py` | `difficulty_exported` |
| `video_match` | `core/video/match.py` | `video_matched` |
| `av_correspondence` | `core/video/av.py` | `av_corresponded` |
| `clip` | `core/video/clip.py` | `video_processed` |
| `video_segment` | `core/video/segment.py` | `video_segmented` |

快速查询：`python project_index/build_index.py --lookup 符号名`。

## 符号索引

覆盖 `88` 个 Python 文件、`459` 个命名函数/方法、`85` 个类。匿名 lambda 不单独列出。

图例：`F` 模块函数，`M` 方法，`N` 嵌套函数，`C` 类；`IO-R/IO-W` 文件读写，`DB` 数据库，`PROCESS` 外部进程。

## `src/before_traning/Lib/beatmap/folder_store.py`

职责：受 manifest 约束的源文件读写、输出目录创建和原子目录替换。
工程依赖：`before_traning.Lib.beatmap.manifest`, `before_traning.Lib.common.pathspec`, `before_traning.state.manifest_schema`

- `C L18-L242` `BeatmapFolderStore` [CLASS]：严格规则：。
- `M L26-L40` `BeatmapFolderStore.__init__(self, target_root: str, manifest_filename: str=MANIFEST_DB_FILENAME)`：初始化实例依赖、配置和运行状态。 调用：`ManifestFolderWalker`, `self.target_root.exists`。
- `M L42-L50` `BeatmapFolderStore._normalize_folder_name(self, folder_name: str) -> str`：规范化 `folder name` 对应的数据或结果。
- `M L52-L53` `BeatmapFolderStore._registered_names(self) -> set[str]`：执行 `registered names` 对应逻辑。 调用：`self.walker.read_folder_names`。
- `M L55-L57` `BeatmapFolderStore.is_registered(self, folder_name: str) -> bool`：判断是否 `registered` 对应的数据或结果。 调用：`self._normalize_folder_name`, `self._registered_names`。
- `M L59-L63` `BeatmapFolderStore._assert_registered(self, folder_name: str)`：执行 `assert registered` 对应逻辑。 调用：`self.is_registered`。
- `M L65-L68` `BeatmapFolderStore.get_folder_path(self, folder_name: str) -> Path`：获取 `folder path` 对应的数据或结果。 调用：`self._assert_registered`, `self._normalize_folder_name`。
- `M L70-L72` `BeatmapFolderStore.folder_exists(self, folder_name: str) -> bool`：执行 `folder exists` 对应逻辑。 调用：`self.get_folder_path`。
- `M L74-L80` `BeatmapFolderStore._require_existing_folder(self, folder_name: str) -> Path`：执行 `require existing folder` 对应逻辑。 调用：`self.get_folder_path`。
- `M L82-L85` `BeatmapFolderStore.find_files(self, folder_name: str, pattern: str='*') -> List[Path]`：执行 `find files` 对应逻辑。 调用：`filter_files`, `gitwildmatch_spec`, `self._require_existing_folder`。
- `M L87-L88` `BeatmapFolderStore.find_osu_files(self, folder_name: str) -> List[Path]`：执行 `find osu files` 对应逻辑。 调用：`self.find_files`。
- `M L90-L95` `BeatmapFolderStore.get_file_path(self, folder_name: str, filename: str) -> Path`：获取 `file path` 对应的数据或结果。 调用：`self._require_existing_folder`。
- `M L97-L98` `BeatmapFolderStore.file_exists(self, folder_name: str, filename: str) -> bool`：执行 `file exists` 对应逻辑。 调用：`self.get_file_path`, `self.get_file_path.exists`。
- `M L100-L136` `BeatmapFolderStore.write_text(self, folder_name: str, filename: str, content: str, mode: WriteMode='overwrite') -> str` [IO-W]：通用文本写入接口。 调用：`file_path.write_text`, `self.get_file_path`。
- `M L138-L155` `BeatmapFolderStore.write_lines(self, folder_name: str, filename: str, lines: Iterable[str], mode: WriteMode='overwrite', add_trailing_newline: bool=True) -> str` [IO-W]：写入 `lines` 对应的数据或结果。 调用：`self.write_text`。
- `M L157-L168` `BeatmapFolderStore.append_line(self, folder_name: str, filename: str, line: str) -> str` [IO-W]：执行 `append line` 对应逻辑。 调用：`self.write_text`。
- `M L170-L174` `BeatmapFolderStore.read_text(self, folder_name: str, filename: str) -> str` [IO-R]：读取 `text` 对应的数据或结果。 调用：`file_path.read_text`, `self.get_file_path`。
- `M L176-L187` `BeatmapFolderStore.create_output_directory(self, base_directory: Path, *relative_parts: str) -> Path` [IO-W]：执行 `create output directory` 对应逻辑。
- `M L189-L207` `BeatmapFolderStore.recover_atomic_outputs(self, output_root: Path, *, namespace: str) -> None` [IO-W]：执行 `recover atomic outputs` 对应逻辑。
- `M L210-L242` `BeatmapFolderStore.atomic_output_folder(self, output_root: Path, folder_name: str, *, namespace: str) -> Iterator[Path]` [IO-W]：执行 `atomic output folder` 对应逻辑。 调用：`self._assert_registered`, `self.recover_atomic_outputs`。

## `src/before_traning/Lib/beatmap/hit_objects.py`

职责：Circle、Slider、Spinner 的轻量数据模型。

- `C L8-L11` `HitObject` [CLASS]：封装 `HitObject` 相关数据或行为。
- `C L13-L17` `Circle(HitObject)` [CLASS]：封装 `Circle` 相关数据或行为。
- `M L16-L17` `Circle.__post_init__(self)`：完成 dataclass 初始化后的派生字段设置。
- `C L19-L25` `Slider(HitObject)` [CLASS]：封装 `Slider` 相关数据或行为。
- `M L24-L25` `Slider.__post_init__(self)`：完成 dataclass 初始化后的派生字段设置。
- `C L27-L29` `Spinner(HitObject)` [CLASS]：封装 `Spinner` 相关数据或行为。
- `M L28-L29` `Spinner.__post_init__(self)`：完成 dataclass 初始化后的派生字段设置。

## `src/before_traning/Lib/beatmap/manifest.py`

职责：SQLite manifest 仓储；管理内部目录、谱面缓存和可读对照表。
工程依赖：`before_traning.Lib.common.sequence`, `before_traning.state.manifest_schema`

- `C L28-L32` `ManifestEntry` [CLASS]：封装 `ManifestEntry` 相关数据或行为。
- `C L35-L393` `PackageManifest` [CLASS]：Small SQLite manifest for stable internal folder IDs and processing order。
- `M L38-L56` `PackageManifest.__init__(self, target_root: str, manifest_filename: str=MANIFEST_DB_FILENAME, legacy_order_filename: str='order.txt', table_filename: str=MANIFEST_TABLE_FILENAME)` [DB IO-W]：初始化实例依赖、配置和运行状态。 调用：`self._ensure_schema`, `self._migrate_legacy_difficulty_files`, `self._migrate_legacy_order`, `self.export_table`, `self.target_root.mkdir`。
- `M L58-L68` `PackageManifest._ensure_schema(self) -> None`：确保 `schema` 对应的数据或结果。
- `M L70-L73` `PackageManifest._all_items(self) -> list[PackageManifestItem]` [DB]：执行 `all items` 对应逻辑。 调用：`select`。
- `M L75-L81` `PackageManifest._normalize_source_name(self, source_name: str) -> str`：规范化 `source name` 对应的数据或结果。
- `M L83-L91` `PackageManifest._next_folder_number(self, items: list[PackageManifestItem]) -> int`：执行 `next folder number` 对应逻辑。
- `M L93-L94` `PackageManifest._folder_name(self, number: int) -> str`：执行 `folder name` 对应逻辑。 调用：`format_sequence_name`。
- `M L96-L101` `PackageManifest._legacy_osu_filename(self, source_name: str) -> str | None`：执行 `legacy osu filename` 对应逻辑。
- `M L103-L152` `PackageManifest._rename_legacy_folders(self, mappings: list[tuple[str, str]]) -> None` [IO-W]：执行 `rename legacy folders` 对应逻辑。 调用：`self._restore_legacy_folders`。
- `M L154-L167` `PackageManifest._restore_legacy_folders(self, mappings: list[tuple[str, str]]) -> None` [IO-W]：执行 `restore legacy folders` 对应逻辑。
- `M L169-L211` `PackageManifest._migrate_legacy_order(self) -> None` [DB IO-R IO-W]：执行 `migrate legacy order` 对应逻辑。 调用：`PackageManifestItem`, `self._all_items`, `self._folder_name`, `self._legacy_osu_filename`, `self._normalize_source_name`, `self._rename_legacy_folders`。
- `M L213-L234` `PackageManifest._migrate_legacy_difficulty_files(self) -> None` [DB IO-R IO-W]：执行 `migrate legacy difficulty files` 对应逻辑。 调用：`difficulty_path.read_text`, `select`。
- `M L236-L248` `PackageManifest.export_table(self, destination: Path | None=None) -> Path` [IO-W]：导出 `table` 对应的数据或结果。 调用：`self._all_items`, `temp_path.replace`。
- `M L250-L285` `PackageManifest.replace(self, entries: list[ManifestEntry]) -> dict[str, str]` [DB]：执行 `replace` 对应逻辑。 调用：`PackageManifestItem`, `by_source.get`, `select`, `self._folder_name`, `self._next_folder_number`, `self._normalize_source_name`。
- `M L287-L294` `PackageManifest.read_folder_names(self) -> list[str]` [DB]：读取 `folder names` 对应的数据或结果。 调用：`select`。
- `M L296-L297` `PackageManifest.read_all_folder_names(self) -> list[str]`：读取 `all folder names` 对应的数据或结果。 调用：`self._all_items`。
- `M L299-L305` `PackageManifest.source_name_for(self, folder_name: str) -> str | None` [DB]：执行 `source name for` 对应逻辑。 调用：`select`。
- `M L307-L320` `PackageManifest.set_difficulty(self, folder_name: str, difficulty_value: float) -> None` [DB IO-W]：执行 `set difficulty` 对应逻辑。 调用：`select`。
- `M L322-L329` `PackageManifest.difficulty_for(self, folder_name: str) -> float | None` [DB]：执行 `difficulty for` 对应逻辑。 调用：`select`。
- `M L331-L367` `PackageManifest.save_beatmap_data(self, folder_name: str, *, osu_filename: str, source_mtime_ns: int, schema_version: int, payload: dict[str, object]) -> None` [DB]：执行 `save beatmap data` 对应逻辑。 调用：`BeatmapDataRecord`, `select`, `self.is_active`。
- `M L369-L390` `PackageManifest.beatmap_data_for(self, folder_name: str) -> tuple[str, int, int, dict[str, object]] | None` [DB]：执行 `beatmap data for` 对应逻辑。 调用：`select`。
- `M L392-L393` `PackageManifest.is_active(self, folder_name: str) -> bool`：判断是否 `active` 对应的数据或结果。 调用：`self.read_folder_names`。
- `C L396-L414` `ManifestFolderWalker` [CLASS]：封装 `ManifestFolderWalker` 相关数据或行为。
- `M L397-L408` `ManifestFolderWalker.__init__(self, target_root: str, manifest_filename: str=MANIFEST_DB_FILENAME)`：初始化实例依赖、配置和运行状态。 调用：`PackageManifest`, `self.target_root.exists`。
- `M L410-L411` `ManifestFolderWalker.read_folder_names(self) -> list[str]`：读取 `folder names` 对应的数据或结果。 调用：`self.manifest.read_folder_names`。
- `M L413-L414` `ManifestFolderWalker.source_name_for(self, folder_name: str) -> str | None`：执行 `source name for` 对应逻辑。 调用：`self.manifest.source_name_for`。

## `src/before_traning/Lib/beatmap/osu_metadata.py`

职责：从 .osu 指定 section 读取 AudioFilename 和 OverallDifficulty。

- `F L6-L31` `read_section_key(osu_path: Path, section_name: str, key_name: str) -> str` [IO-W]：读取 `section key` 对应的数据或结果。
- `F L34-L35` `read_audio_filename(osu_path: Path) -> str`：读取 `audio filename` 对应的数据或结果。 调用：`read_section_key`。
- `F L38-L39` `read_overall_difficulty(osu_path: Path) -> float`：读取 `overall difficulty` 对应的数据或结果。 调用：`read_section_key`。

## `src/before_traning/Lib/beatmap/osu_parser.py`

职责：解析 .osu sections、timing points 和 HitObjects，并生成结构化对象。
工程依赖：`before_traning.Lib.beatmap.hit_objects`, `before_traning.Lib.beatmap.timing_points`

- `C L10-L212` `VerifyOsuParser` [CLASS]：封装 `VerifyOsuParser` 相关数据或行为。
- `M L11-L35` `VerifyOsuParser.parse_sections(self, osu_path: Path) -> tuple[str | None, dict[str, list[str]]]` [IO-W]：解析 `sections` 对应的数据或结果。
- `M L37-L44` `VerifyOsuParser.parse_key_value_section(self, lines: List[str]) -> dict[str, str]`：解析 `key value section` 对应的数据或结果。
- `M L46-L68` `VerifyOsuParser.parse_timing_points(self, lines: List[str]) -> List[OsuOriginalTimingPoint]`：解析 `timing points` 对应的数据或结果。 调用：`OsuOriginalTimingPoint`。
- `M L70-L97` `VerifyOsuParser.get_effective_timing(self, t: int, timing_points: List[OsuOriginalTimingPoint]) -> tuple[OsuOriginalTimingPoint, float]`：获取 `effective timing` 对应的数据或结果。
- `M L99-L170` `VerifyOsuParser.parse_hitobjects(self, hitobject_lines: List[str], timing_points: List[OsuOriginalTimingPoint], slider_multiplier: float) -> List[object]`：解析 `hitobjects` 对应的数据或结果。 调用：`Circle`, `Slider`, `Spinner`, `self.get_effective_timing`。
- `M L172-L188` `VerifyOsuParser.objects_to_lines(self, objects: List[object]) -> List[str]`：执行 `objects to lines` 对应逻辑。
- `M L190-L212` `VerifyOsuParser.hit_object_to_dict(self, hit_object: HitObject, *, time_offset_ms: int=0) -> dict[str, Any]`：执行 `hit object to dict` 对应逻辑。

## `src/before_traning/Lib/beatmap/osz.py`

职责：解压单个 .osz 并读取目标 .osu 与音频字节。
工程依赖：`before_traning.Lib.beatmap.osu_metadata`

- `C L12-L20` `OsuEntry` [CLASS]：封装 `OsuEntry` 相关数据或行为。
- `F L23-L66` `read_osz_entry(osz_path: Path, *, keyword: str, audio_output_filename: str) -> OsuEntry | None` [IO-R]：读取 `osz entry` 对应的数据或结果。 调用：`OsuEntry`, `read_audio_filename`。

## `src/before_traning/Lib/beatmap/package.py`

职责：通过 SQLite manifest 创建和同步允许使用的内部谱面目录。
工程依赖：`before_traning.Lib.beatmap.manifest`, `before_traning.state.manifest_schema`

- `C L12-L98` `PackageUpdater` [CLASS]：规则：。
- `M L20-L39` `PackageUpdater.__init__(self, target_root: str | Path, manifest_filename: str=MANIFEST_DB_FILENAME, ignore_patterns: Iterable[str]=())` [IO-W]：初始化实例依赖、配置和运行状态。 调用：`PackageManifest`, `self.target_root.mkdir`。
- `M L41-L42` `PackageUpdater.load_manifest_folder_names(self) -> List[str]`：加载 `manifest folder names` 对应的数据或结果。 调用：`self.manifest.read_folder_names`。
- `M L44-L45` `PackageUpdater.load_registered_names(self) -> set[str]`：加载 `registered names` 对应的数据或结果。 调用：`self.manifest.read_all_folder_names`。
- `M L47-L48` `PackageUpdater.replace_manifest(self, entries: list[ManifestEntry]) -> dict[str, str]` [IO-W]：执行 `replace manifest` 对应逻辑。 调用：`self.manifest.replace`。
- `M L50-L51` `PackageUpdater.is_registered(self, folder_name: str) -> bool`：判断是否 `registered` 对应的数据或结果。 调用：`self.manifest.is_active`。
- `M L53-L68` `PackageUpdater.create_folder_if_registered(self, folder_name: str) -> Path` [IO-W]：只有在 manifest 中启用的内部 ID 才允许创建/使用对应文件夹。 调用：`self.is_registered`。
- `M L70-L80` `PackageUpdater.sync_folders_from_manifest(self) -> List[Path]` [IO-W]：Create active manifest folders in processing order。 调用：`self.load_manifest_folder_names`。
- `M L82-L98` `PackageUpdater.find_unregistered_existing_folders(self) -> List[Path]`：返回 target_root 下存在，但不在 manifest 中登记的目录。 调用：`self.ignore_spec.match_file`, `self.load_registered_names`, `self.target_root.iterdir`。

## `src/before_traning/Lib/beatmap/standard.py`

职责：解析或从 manifest SQLite 缓存读取完整 osu!standard 谱面。
工程依赖：`before_traning.Lib.beatmap.folder_store`, `before_traning.Lib.beatmap.hit_objects`, `before_traning.Lib.beatmap.osu_parser`

- `C L21-L33` `ParsedStandardBeatmap` [CLASS]：封装 `ParsedStandardBeatmap` 相关数据或行为。
- `M L32-L33` `ParsedStandardBeatmap.approach_preempt_ms(self) -> float` [PROPERTY]：执行 `approach preempt ms` 对应逻辑。 调用：`approach_preempt_ms`。
- `F L36-L43` `approach_preempt_ms(approach_rate: float) -> float`：执行 `approach preempt ms` 对应逻辑。
- `F L46-L90` `parse_standard_beatmap(osu_path: Path, parser: VerifyOsuParser | None=None) -> ParsedStandardBeatmap`：解析 `standard beatmap` 对应的数据或结果。 调用：`ParsedStandardBeatmap`, `VerifyOsuParser`, `difficulty.get`, `general.get`, `parser.parse_hitobjects`, `parser.parse_key_value_section`。
- `F L93-L97` `parse_standard_hit_objects(osu_path: Path, parser: VerifyOsuParser | None=None) -> list[HitObject]`：解析 `standard hit objects` 对应的数据或结果。 调用：`parse_standard_beatmap`。
- `F L100-L116` `_beatmap_to_payload(beatmap: ParsedStandardBeatmap, parser: VerifyOsuParser) -> dict[str, object]`：执行 `beatmap to payload` 对应逻辑。 调用：`parser.hit_object_to_dict`。
- `F L119-L151` `_hit_object_from_payload(payload: dict[str, object]) -> HitObject`：执行 `hit object from payload` 对应逻辑。 调用：`Circle`, `Slider`, `Spinner`, `payload.get`。
- `F L154-L173` `_beatmap_from_payload(payload: dict[str, object]) -> ParsedStandardBeatmap`：执行 `beatmap from payload` 对应逻辑。 调用：`ParsedStandardBeatmap`, `_hit_object_from_payload`, `payload.get`。
- `F L176-L207` `load_standard_beatmap(store: BeatmapFolderStore, folder_name: str, *, refresh: bool=False, parser: VerifyOsuParser | None=None) -> tuple[Path, ParsedStandardBeatmap]`：加载 `standard beatmap` 对应的数据或结果。 调用：`VerifyOsuParser`, `_beatmap_from_payload`, `_beatmap_to_payload`, `parse_standard_beatmap`, `store.find_osu_files`, `store.walker.manifest.beatmap_data_for`。

## `src/before_traning/Lib/beatmap/timing_points.py`

职责：osu 原始 timing point 数据模型。

- `C L7-L15` `OsuOriginalTimingPoint` [CLASS]：封装 `OsuOriginalTimingPoint` 相关数据或行为。

## `src/before_traning/Lib/common/batch.py`

职责：配置规格辅助函数与文件夹批处理模板。
工程依赖：`before_traning.Lib.common.failures`

- `C L36-L39` `ConfigValueSpec` [CLASS]：封装 `ConfigValueSpec` 相关数据或行为。
- `F L46-L54` `_normalize_config_path(path: ConfigPathInput) -> tuple[str, ...]`：规范化 `config path` 对应的数据或结果。
- `F L57-L70` `_normalize_config_path_group(paths: ConfigPathGroupInput) -> tuple[tuple[str, ...], ...]`：规范化 `config path group` 对应的数据或结果。 调用：`_normalize_config_path`。
- `F L73-L81` `_config_values(reader_name: str, **entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config values` 对应逻辑。 调用：`ConfigValueSpec`, `_normalize_config_path_group`。
- `F L84-L85` `config_resolved_paths(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config resolved paths` 对应逻辑。 调用：`_config_values`。
- `F L88-L89` `config_filenames(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config filenames` 对应逻辑。 调用：`_config_values`。
- `F L92-L93` `config_nonempty_strs(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config nonempty strs` 对应逻辑。 调用：`_config_values`。
- `F L96-L97` `config_bools(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config bools` 对应逻辑。 调用：`_config_values`。
- `F L100-L101` `config_string_tuples(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config string tuples` 对应逻辑。 调用：`_config_values`。
- `F L104-L105` `config_suffix_tuples(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config suffix tuples` 对应逻辑。 调用：`_config_values`。
- `F L108-L109` `config_positive_ints(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config positive ints` 对应逻辑。 调用：`_config_values`。
- `F L112-L113` `config_nonnegative_ints(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config nonnegative ints` 对应逻辑。 调用：`_config_values`。
- `F L116-L117` `config_positive_floats(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config positive floats` 对应逻辑。 调用：`_config_values`。
- `F L120-L121` `config_floats(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]`：执行 `config floats` 对应逻辑。 调用：`_config_values`。
- `F L124-L133` `merge_config_specs(*spec_groups: ConfigValueSpec | Iterable[ConfigValueSpec]) -> tuple[ConfigValueSpec, ...]`：执行 `merge config specs` 对应逻辑。
- `F L136-L147` `prefix_config_keys(specs: Iterable[ConfigValueSpec], prefix: str) -> tuple[ConfigValueSpec, ...]`：执行 `prefix config keys` 对应逻辑。 调用：`ConfigValueSpec`。
- `F L150-L165` `read_config_values(config_reader: Any, *spec_groups: ConfigValueSpec | Iterable[ConfigValueSpec]) -> dict[str, Any]` [IO-R]：读取 `config values` 对应的数据或结果。 调用：`config_reader.read`。
- `C L168-L170` `FolderWalkerLike(Protocol)` [CLASS]：封装 `FolderWalkerLike` 相关数据或行为。
- `M L169-L170` `FolderWalkerLike.read_folder_names(self) -> list[str]`：读取 `folder names` 对应的数据或结果。
- `C L173-L244` `FolderBatchProcessor(ABC)` [CLASS]：Shared shell for folder-based batch processors。
- `M L178-L181` `FolderBatchProcessor.__init__(self)`：初始化实例依赖、配置和运行状态。
- `M L183-L189` `FolderBatchProcessor.progress_message(self, index: int, total: int, folder_name: str) -> str | None`：生成当前批处理进度文本。
- `M L191-L192` `FolderBatchProcessor.iter_folder_names(self) -> list[str]`：执行 `iter folder names` 对应逻辑。 调用：`self.walker.read_folder_names`。
- `M L194-L195` `FolderBatchProcessor.handle_failure(self, folder_name: str, error: Exception)`：处理单文件夹失败并同步失败状态。
- `M L198-L203` `FolderBatchProcessor.process_one(self, folder_name: str, overwrite: bool=False) -> BatchProcessResult`：处理 manifest 中的单个内部谱面文件夹。
- `M L205-L216` `FolderBatchProcessor._record_result(self, folder_name: str, result: BatchProcessResult)`：执行 `record result` 对应逻辑。
- `M L218-L222` `FolderBatchProcessor._print_summary(self)`：执行 `print summary` 对应逻辑。
- `M L224-L244` `FolderBatchProcessor.run(self, overwrite: bool=False) -> bool`：执行该处理器的完整工作流。 调用：`format_exception`, `self._print_summary`, `self._record_result`, `self.handle_failure`, `self.iter_folder_names`, `self.process_one`。

## `src/before_traning/Lib/common/failures.py`

职责：统一提取异常类型、报错函数和模块，并生成状态 detail 与控制台文本。

- `F L11-L21` `_error_traceback(error: BaseException) -> TracebackType | None`：执行 `error traceback` 对应逻辑。 调用：`traceback.tb_frame.f_globals.get`。
- `F L24-L29` `callable_location(function: Callable[..., object]) -> tuple[str, str]`：执行 `callable location` 对应逻辑。
- `F L32-L42` `exception_location(error: BaseException) -> tuple[str, str]`：执行 `exception location` 对应逻辑。 调用：`_error_traceback`, `frame.f_globals.get`。
- `F L45-L59` `failure_detail(message: str, function: Callable[..., object], *, error_type: str='ProcessingStateError', **context: Any) -> dict[str, Any]`：执行 `failure detail` 对应逻辑。 调用：`callable_location`。
- `F L62-L70` `exception_detail(error: BaseException, **context: Any) -> dict[str, Any]`：执行 `exception detail` 对应逻辑。 调用：`exception_location`。
- `F L73-L77` `format_failure(detail: dict[str, Any]) -> str`：执行 `format failure` 对应逻辑。 调用：`detail.get`。
- `F L80-L81` `format_exception(error: BaseException) -> str`：执行 `format exception` 对应逻辑。 调用：`exception_detail`, `format_failure`。

## `src/before_traning/Lib/common/pathspec.py`

职责：统一后缀到 gitwildmatch PathSpec 的转换与文件过滤。

- `F L9-L15` `suffix_pattern(suffix: str) -> str`：执行 `suffix pattern` 对应逻辑。
- `F L18-L19` `suffix_patterns(suffixes: Iterable[str]) -> tuple[str, ...]`：执行 `suffix patterns` 对应逻辑。 调用：`suffix_pattern`。
- `F L22-L23` `gitwildmatch_spec(patterns: Iterable[str]) -> pathspec.PathSpec`：执行 `gitwildmatch spec` 对应逻辑。
- `F L26-L27` `suffix_spec(suffixes: Iterable[str]) -> pathspec.PathSpec`：执行 `suffix spec` 对应逻辑。 调用：`gitwildmatch_spec`, `suffix_patterns`。
- `F L30-L31` `matches_name(spec: pathspec.PathSpec, path: Path | str) -> bool`：执行 `matches name` 对应逻辑。
- `F L34-L35` `filter_files(paths: Iterable[Path], spec: pathspec.PathSpec) -> list[Path]`：筛选 `files` 对应的数据或结果。 调用：`matches_name`。

## `src/before_traning/Lib/common/processing.py`

职责：通用目录/文件检查、前置步骤检查、完成态对齐和失败状态回写 API。
工程依赖：`before_traning.Lib.common.failures`, `before_traning.Lib.common.pathspec`

- `C L16-L24` `FolderStoreLike(Protocol)` [CLASS]：封装 `FolderStoreLike` 相关数据或行为。
- `M L17-L18` `FolderStoreLike.folder_exists(self, folder_name: str) -> bool`：执行 `folder exists` 对应逻辑。
- `M L20-L21` `FolderStoreLike.file_exists(self, folder_name: str, filename: str) -> bool`：执行 `file exists` 对应逻辑。
- `M L23-L24` `FolderStoreLike.find_files(self, folder_name: str, pattern: str='*') -> list[Path]`：执行 `find files` 对应逻辑。
- `C L27-L50` `StatusManagerLike(Protocol)` [CLASS]：封装 `StatusManagerLike` 相关数据或行为。
- `M L30-L31` `StatusManagerLike.ensure_status_file(self, folder_name: str) -> dict[str, Any]`：确保 `status file` 对应的数据或结果。
- `M L33-L34` `StatusManagerLike.is_step_done(self, folder_name: str, step: str) -> bool`：判断是否 `step done` 对应的数据或结果。
- `M L36-L42` `StatusManagerLike.mark_step_done(self, folder_name: str, step: str, detail: Any=None) -> None`：更新状态为 `step done` 对应的数据或结果。
- `M L44-L50` `StatusManagerLike.mark_step_pending(self, folder_name: str, step: str, detail: Any=None) -> None`：更新状态为 `step pending` 对应的数据或结果。
- `F L53-L62` `matching_files(directory: Path, spec: pathspec.PathSpec, *, sort_key: SortKey | None=None) -> list[Path]`：执行 `matching files` 对应逻辑。 调用：`filter_files`。
- `C L66-L174` `ProcessingGuard` [CLASS]：封装 `ProcessingGuard` 相关数据或行为。
- `M L72-L79` `ProcessingGuard.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L81-L98` `ProcessingGuard.prepare_folder(self, folder_name: str, *, required_patterns: Iterable[str]=()) -> dict[str, tuple[Path, ...]] | None`：顺序准备 `folder` 对应的数据或结果。 调用：`self.ensure_required_steps`, `self.status_manager.ensure_status_file`, `self.store.find_files`, `self.store.folder_exists`。
- `M L100-L109` `ProcessingGuard.ensure_required_steps(self, folder_name: str) -> None`：确保 `required steps` 对应的数据或结果。 调用：`self.status_manager.is_step_done`。
- `M L111-L119` `ProcessingGuard.output_files_exist(self, folder_name: str, filenames: Iterable[str]) -> bool`：执行 `output files exist` 对应逻辑。 调用：`self.store.file_exists`。
- `M L121-L137` `ProcessingGuard.is_complete(self, folder_name: str, *, overwrite: bool, artifact_exists: bool=True, output_files: Iterable[str]=()) -> bool`：判断是否 `complete` 对应的数据或结果。 调用：`self.output_files_exist`, `self.status_manager.is_step_done`。
- `M L139-L143` `ProcessingGuard.step_done(self, folder_name: str) -> bool`：执行 `step done` 对应逻辑。 调用：`self.status_manager.is_step_done`。
- `M L145-L157` `ProcessingGuard.reconcile_existing(self, folder_name: str, *, overwrite: bool, artifact_exists: bool, detail: Any=None) -> ProcessResult | None`：执行 `reconcile existing` 对应逻辑。 调用：`self.mark_done`, `self.step_done`。
- `M L159-L164` `ProcessingGuard.mark_done(self, folder_name: str, detail: Any=None) -> None`：更新状态为 `done` 对应的数据或结果。 调用：`self.status_manager.mark_step_done`。
- `M L166-L174` `ProcessingGuard.record_failure(self, folder_name: str, error: Exception) -> None`：执行 `record failure` 对应逻辑。 调用：`exception_detail`, `self.status_manager.ensure_status_file`, `self.status_manager.mark_step_pending`, `self.store.folder_exists`。

## `src/before_traning/Lib/common/sequence.py`

职责：统一生成带定宽数字的稳定序列名称。

- `F L4-L17` `format_sequence_name(prefix: str, sequence: int, *, width: int=6) -> str`：执行 `format sequence name` 对应逻辑。

## `src/before_traning/Lib/tasks/flows.py`

职责：通用 direct/Prefect 循环执行 Pipeline API 与构建函数。
工程依赖：`before_traning.Lib.common.failures`, `before_traning.Lib.tasks.tasks`

- `C L18-L125` `TaskPipeline(Generic[SettingsT])` [CLASS]：封装 `TaskPipeline` 相关数据或行为。
- `M L19-L34` `TaskPipeline.__init__(self, registry: TaskRegistry[SettingsT], *, settings_loader: SettingsLoader[SettingsT], continue_on_error: ContinueOnError[SettingsT], flow_name: str)`：初始化实例依赖、配置和运行状态。
- `M L36-L50` `TaskPipeline._call_stage(self, stage: str, call: Callable[[], bool], *, continue_on_error: bool) -> bool`：执行 `call stage` 对应逻辑。 调用：`format_exception`。
- `M L52-L74` `TaskPipeline._run(self, settings: SettingsT | None, *, overrides: Mapping[str, bool | None] | None, only: Iterable[str] | None, use_prefect: bool) -> dict[str, bool]`：执行 `run` 对应逻辑。 调用：`self._call_stage`, `self.continue_on_error`, `self.registry.select`, `self.settings_loader`。
- `M L76-L87` `TaskPipeline._run_prefect(self, settings: SettingsT | None=None, overrides: Mapping[str, bool | None] | None=None, only: Iterable[str] | None=None) -> dict[str, bool]`：执行 `run prefect` 对应逻辑。 调用：`self._run`。
- `M L89-L100` `TaskPipeline.run_prefect(self, settings: SettingsT | None=None, *, overrides: Mapping[str, bool | None] | None=None, only: Iterable[str] | None=None) -> dict[str, bool]`：执行 `run prefect` 对应逻辑。 调用：`self._prefect_flow`。
- `M L102-L114` `TaskPipeline.run_direct(self, settings: SettingsT | None=None, *, overrides: Mapping[str, bool | None] | None=None, only: Iterable[str] | None=None) -> dict[str, bool]`：执行 `run direct` 对应逻辑。 调用：`self._run`。
- `M L116-L125` `TaskPipeline.__call__(self, settings: SettingsT | None=None, *, overrides: Mapping[str, bool | None] | None=None, only: Iterable[str] | None=None, use_prefect: bool=False) -> dict[str, bool]`：执行 `call` 对应逻辑。
- `F L128-L140` `build_task_pipeline(specs: Iterable[TaskSpec[SettingsT]], *, settings_loader: SettingsLoader[SettingsT], continue_on_error: ContinueOnError[SettingsT], flow_name: str) -> TaskPipeline[SettingsT]`：构建并返回 `task pipeline` 对应的数据或结果。 调用：`TaskPipeline`, `TaskRegistry`。

## `src/before_traning/Lib/tasks/tasks.py`

职责：通用 task 规格、注册器和循环 Prefect task 生成 API。

- `F L13-L16` `require_success(stage: str, success: bool) -> bool`：执行 `require success` 对应逻辑。
- `C L20-L32` `TaskSpec(Generic[SettingsT])` [CLASS]：封装 `TaskSpec` 相关数据或行为。
- `M L28-L32` `TaskSpec.default_enabled(self, settings: SettingsT) -> bool`：执行 `default enabled` 对应逻辑。
- `C L36-L38` `RegisteredTask(Generic[SettingsT])` [CLASS]：封装 `RegisteredTask` 相关数据或行为。
- `F L41-L50` `_build_prefect_task(spec: TaskSpec[SettingsT]) -> TaskCall[SettingsT]`：构建 `prefect task` 对应的数据或结果。
- `N L42-L43` `_build_prefect_task.run_registered_task(settings: SettingsT) -> bool`：执行 `run registered task` 对应逻辑。 调用：`require_success`。
- `C L53-L113` `TaskRegistry(Generic[SettingsT])` [CLASS]：封装 `TaskRegistry` 相关数据或行为。
- `M L54-L76` `TaskRegistry.__init__(self, specs: Iterable[TaskSpec[SettingsT]])`：初始化实例依赖、配置和运行状态。 调用：`RegisteredTask`, `_build_prefect_task`。
- `M L79-L80` `TaskRegistry.registered(self) -> tuple[RegisteredTask[SettingsT], ...]` [PROPERTY]：执行 `registered` 对应逻辑。
- `M L82-L113` `TaskRegistry.select(self, settings: SettingsT, *, overrides: Mapping[str, bool | None] | None=None, only: Iterable[str] | None=None) -> tuple[RegisteredTask[SettingsT], ...]`：执行 `select` 对应逻辑。 调用：`override_values.get`, `self._by_key.keys`, `spec.default_enabled`。

## `src/before_traning/Lib/tools/ffmpeg.py`

职责：提供 ffmpeg/ffprobe 参数构造与音频提取、裁切、默认去音频分段和裁剪高层 API。

- `F L43-L47` `_command_error_text(result: subprocess.CompletedProcess[str], unknown_error: str) -> str`：执行 `command error text` 对应逻辑。
- `F L50-L56` `_run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]` [PROCESS]：执行 `run command` 对应逻辑。 调用：`subprocess.run`。
- `F L59-L62` `run_ffmpeg(args: Sequence[str])` [PROCESS]：执行 `run ffmpeg` 对应逻辑。 调用：`_command_error_text`, `_run_command`。
- `F L65-L87` `build_extract_wav_args(source_path: Path, output_path: Path, *, sample_rate: int, from_video: bool) -> tuple[str, ...]`：构建并返回 `extract wav args` 对应的数据或结果。
- `F L90-L105` `extract_wav(source_path: Path, output_path: Path, *, sample_rate: int, from_video: bool) -> None` [IO-W PROCESS]：提取 `wav` 对应的数据或结果。 调用：`build_extract_wav_args`, `run_ffmpeg`。
- `F L108-L127` `build_trim_video_args(source_video_path: Path, output_video_path: Path, *, trim_start_seconds: float, trim_duration_seconds: float) -> tuple[str, ...]`：构建并返回 `trim video args` 对应的数据或结果。
- `F L130-L149` `trim_video(source_video_path: Path, output_video_path: Path, *, start_seconds: float, duration_seconds: float) -> None` [IO-W PROCESS]：执行 `trim video` 对应逻辑。 调用：`build_trim_video_args`, `run_ffmpeg`。
- `F L152-L173` `build_segment_video_args(source_video_path: Path, output_video_path: Path, *, trim_start_seconds: float, trim_duration_seconds: float, include_audio: bool=False) -> tuple[str, ...]`：构建并返回 `segment video args` 对应的数据或结果。
- `F L176-L197` `segment_video(source_video_path: Path, output_video_path: Path, *, start_seconds: float, end_seconds: float, include_audio: bool=False) -> None` [IO-W PROCESS]：执行 `segment video` 对应逻辑。 调用：`build_segment_video_args`, `run_ffmpeg`。
- `F L200-L219` `build_crop_video_args(source_video_path: Path, output_video_path: Path, *, crop_left: int, crop_top: int, crop_width: int, crop_height: int) -> tuple[str, ...]`：构建并返回 `crop video args` 对应的数据或结果。
- `F L222-L241` `crop_video(source_video_path: Path, output_video_path: Path, *, crop_left: int, crop_top: int, crop_width: int, crop_height: int) -> None` [IO-W PROCESS]：裁剪 `video` 对应的数据或结果。 调用：`build_crop_video_args`, `run_ffmpeg`。
- `F L244-L261` `run_ffprobe_json(args: Sequence[str], *, error_prefix: str) -> dict[str, Any]` [PROCESS]：执行 `run ffprobe json` 对应逻辑。 调用：`_command_error_text`, `_run_command`。
- `F L264-L286` `get_audio_stream_start_time(source_path: Path) -> float`：获取 `audio stream start time` 对应的数据或结果。 调用：`get`, `payload.get`, `run_ffprobe_json`。
- `F L289-L309` `get_media_duration_seconds(source_path: Path) -> float`：获取 `media duration seconds` 对应的数据或结果。 调用：`payload.get`, `payload.get.get`, `run_ffprobe_json`。
- `F L312-L333` `get_video_size(video_path: Path) -> tuple[int, int]`：获取 `video size` 对应的数据或结果。 调用：`payload.get`, `run_ffprobe_json`。

## `src/before_traning/Lib/video/av_processing/steps.py`

职责：可复用 AV 信号算法：采样、粗细相关、hit 校正和裁切窗口计算。
工程依赖：`before_traning.Lib.tools.ffmpeg`

- `C L13-L295` `AVCoreStepsMixin` [CLASS]：封装 `AVCoreStepsMixin` 相关数据或行为。
- `M L14-L20` `AVCoreStepsMixin._extract_audio_to_wav(self, source_path: Path, output_path: Path, from_video: bool)`：提取 `audio to wav` 对应的数据或结果。 调用：`extract_wav`。
- `M L22-L42` `AVCoreStepsMixin._load_wav_samples(self, wav_path: Path) -> np.ndarray` [IO-R]：加载 `wav samples` 对应的数据或结果。 调用：`wavfile.read`。
- `M L44-L50` `AVCoreStepsMixin._normalize_series(self, values: np.ndarray) -> np.ndarray`：规范化 `series` 对应的数据或结果。
- `M L52-L77` `AVCoreStepsMixin._build_feature_series(self, samples: np.ndarray, target_hz: int, mode: str='energy') -> np.ndarray`：构建 `feature series` 对应的数据或结果。 调用：`self._normalize_series`。
- `M L79-L91` `AVCoreStepsMixin._lowpass_samples(self, samples: np.ndarray) -> np.ndarray`：执行 `lowpass samples` 对应逻辑。
- `M L93-L99` `AVCoreStepsMixin._build_music_refine_series(self, samples: np.ndarray) -> np.ndarray`：构建 `music refine series` 对应的数据或结果。 调用：`self._build_feature_series`, `self._lowpass_samples`。
- `M L101-L128` `AVCoreStepsMixin._estimate_best_start_frame(self, long_series: np.ndarray, short_series: np.ndarray) -> tuple[float, float]`：估算 `best start frame` 对应的数据或结果。
- `M L130-L176` `AVCoreStepsMixin._estimate_offset_seconds(self, video_audio_samples: np.ndarray, song_audio_samples: np.ndarray) -> tuple[float, float, float]`：估算 `offset seconds` 对应的数据或结果。 调用：`self._build_feature_series`, `self._build_music_refine_series`, `self._estimate_best_start_frame`。
- `M L178-L191` `AVCoreStepsMixin._parse_verify_hit_times_ms(self, verify_path: Path) -> list[int]` [IO-R]：解析 `verify hit times ms` 对应的数据或结果。 调用：`verify_path.read_text`。
- `M L193-L206` `AVCoreStepsMixin._build_verify_click_train(self, hit_times_ms: list[int], length_frames: int) -> np.ndarray`：构建 `verify click train` 对应的数据或结果。 调用：`self._normalize_series`。
- `M L208-L257` `AVCoreStepsMixin._estimate_verify_adjustment_seconds(self, transient_series: np.ndarray, verify_path: Path, base_offset_seconds: float) -> tuple[float, dict[str, float]] | None`：估算 `verify adjustment seconds` 对应的数据或结果。 调用：`self._build_verify_click_train`, `self._normalize_series`, `self._parse_verify_hit_times_ms`。
- `M L259-L281` `AVCoreStepsMixin._validate_trim_window(self, offset_seconds: float, song_duration_seconds: float, video_duration_seconds: float) -> float`：校验 `trim window` 对应的数据或结果。
- `M L283-L295` `AVCoreStepsMixin._trim_video(self, source_video_path: Path, output_video_path: Path, trim_start_seconds: float, trim_duration_seconds: float)`：执行 `trim video` 对应逻辑。 调用：`trim_video`。

## `src/before_traning/Lib/video/clipping/geometry.py`

职责：按参考分辨率缩放裁剪矩形，并校验边界和编码偶数尺寸。
工程依赖：`before_traning.Lib.tools.ffmpeg`

- `C L8-L95` `ClipGeometryMixin` [CLASS]：封装 `ClipGeometryMixin` 相关数据或行为。
- `M L9-L10` `ClipGeometryMixin.get_video_size(self, video_path: Path) -> tuple[int, int]`：获取 `video size` 对应的数据或结果。
- `M L12-L18` `ClipGeometryMixin._scale_crop_coordinate(self, value: int, reference_size: int, video_size: int) -> int`：执行 `scale crop coordinate` 对应逻辑。
- `M L20-L75` `ClipGeometryMixin._resolve_scaled_crop(self, video_width: int, video_height: int) -> dict[str, int]`：解析并定位 `scaled crop` 对应的数据或结果。 调用：`self._scale_crop_coordinate`。
- `M L77-L92` `ClipGeometryMixin._validate_crop_bounds(self, video_path: Path) -> tuple[int, int, dict[str, int]]`：校验 `crop bounds` 对应的数据或结果。 调用：`self._resolve_scaled_crop`, `self.get_video_size`。
- `M L94-L95` `ClipGeometryMixin.describe_crop_for_video(self, video_path: Path) -> tuple[int, int, dict[str, int]]`：执行 `describe crop for video` 对应逻辑。 调用：`self._validate_crop_bounds`。

## `src/before_traning/Lib/video/segment_dataset.py`

职责：用 SQLite 管理视频片段索引、导出 CSV 并校验数据集文件完整性。
工程依赖：`before_traning.state.segment_schema`

- `C L67-L192` `SegmentDatasetManifest` [CLASS]：封装 `SegmentDatasetManifest` 相关数据或行为。
- `M L68-L80` `SegmentDatasetManifest.__init__(self, segment_root: Path, output_directories: Iterable[str], *, db_filename: str=SEGMENT_DB_FILENAME)` [DB IO-W]：初始化实例依赖、配置和运行状态。 调用：`self.segment_root.mkdir`。
- `M L82-L89` `SegmentDatasetManifest._records(self, folder_name: str) -> list[SegmentDatasetItem]` [DB]：执行 `records` 对应逻辑。 调用：`select`。
- `M L91-L95` `SegmentDatasetManifest.read_rows(self, folder_name: str) -> list[dict[str, str]]`：读取 `rows` 对应的数据或结果。 调用：`self._records`。
- `M L97-L129` `SegmentDatasetManifest.replace_folder(self, folder_name: str, rows: list[dict[str, object]]) -> None` [DB]：执行 `replace folder` 对应逻辑。 调用：`SegmentDatasetItem`, `select`。
- `M L131-L145` `SegmentDatasetManifest.write_table(self, output_directory: Path, rows: list[dict[str, object]]) -> Path` [IO-W]：写入 `table` 对应的数据或结果。
- `M L147-L152` `SegmentDatasetManifest.export_table(self, folder_name: str) -> Path`：导出 `table` 对应的数据或结果。 调用：`self.read_rows`, `self.write_table`。
- `M L154-L174` `SegmentDatasetManifest.import_existing_table(self, folder_name: str) -> bool` [IO-W]：导入 `existing table` 对应的数据或结果。 调用：`self.read_rows`, `self.replace_folder`。
- `M L176-L192` `SegmentDatasetManifest.output_complete(self, folder_name: str) -> bool`：执行 `output complete` 对应逻辑。 调用：`self.import_existing_table`, `self.read_rows`。
- `F L195-L199` `write_json_file(output_path: Path, payload: dict[str, object]) -> None` [IO-W]：写入 `json file` 对应的数据或结果。 调用：`output_path.write_text`。

## `src/before_traning/Lib/video/segmentation/planner.py`

职责：构建对象恰好归属一次的原子片段，支持稳定前置时间抖动，并将完整原子片段组合为长序列维度。
工程依赖：`before_traning.Lib.beatmap.hit_objects`, `before_traning.Lib.beatmap.standard`

- `C L41-L75` `SegmentPlan` [CLASS]：封装 `SegmentPlan` 相关数据或行为。
- `M L58-L59` `SegmentPlan.duration_seconds(self) -> float` [PROPERTY]：执行 `duration seconds` 对应逻辑。
- `M L62-L63` `SegmentPlan.pre_context_seconds(self) -> float` [PROPERTY]：执行 `pre context seconds` 对应逻辑。
- `M L66-L67` `SegmentPlan.post_context_seconds(self) -> float` [PROPERTY]：执行 `post context seconds` 对应逻辑。
- `M L70-L71` `SegmentPlan.clip_start_ms(self) -> int` [PROPERTY]：执行 `clip start ms` 对应逻辑。
- `M L74-L75` `SegmentPlan.clip_end_ms(self) -> int` [PROPERTY]：执行 `clip end ms` 对应逻辑。
- `F L78-L90` `circle_radius_from_size(circle_size: float) -> float`：执行 `circle radius from size` 对应逻辑。
- `F L93-L104` `circle_overlap_ratio(distance: float, radius: float) -> float`：执行 `circle overlap ratio` 对应逻辑。
- `F L107-L126` `_slider_polyline(slider: Slider) -> tuple[tuple[float, float], ...]`：执行 `slider polyline` 对应逻辑。
- `F L129-L136` `_object_polyline(hit_object: HitObject) -> tuple[tuple[float, float], ...]`：执行 `object polyline` 对应逻辑。 调用：`_slider_polyline`。
- `F L139-L157` `_point_to_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float`：执行 `point to segment distance` 对应逻辑。
- `F L160-L168` `_orientation(first: tuple[float, float], second: tuple[float, float], third: tuple[float, float]) -> float`：执行 `orientation` 对应逻辑。
- `F L171-L193` `_segments_intersect(first_start: tuple[float, float], first_end: tuple[float, float], second_start: tuple[float, float], second_end: tuple[float, float]) -> bool`：执行 `segments intersect` 对应逻辑。 调用：`_orientation`, `_point_to_segment_distance`。
- `F L196-L248` `_polyline_distance(first: tuple[tuple[float, float], ...], second: tuple[tuple[float, float], ...]) -> float`：执行 `polyline distance` 对应逻辑。 调用：`_point_to_segment_distance`, `_segments_intersect`。
- `F L251-L261` `hit_objects_overlap_ratio(first: HitObject, second: HitObject, *, circle_radius: float) -> float`：执行 `hit objects overlap ratio` 对应逻辑。 调用：`_object_polyline`, `_polyline_distance`, `circle_overlap_ratio`。
- `F L264-L333` `group_hit_objects(hit_objects: list[HitObject], overlap_merge_window_ms: int, *, circle_size: float=5.0, min_circle_overlap_ratio: float=0.5, priority_merge_window_ms: int=0, use_priority_merge: bool=True) -> list[list[HitObject]]`：执行 `group hit objects` 对应逻辑。 调用：`_object_polyline`, `_polyline_distance`, `circle_overlap_ratio`, `circle_radius_from_size`。
- `F L336-L356` `classify_hit_group(hit_group: list[HitObject]) -> SegmentCategory`：Classify by contained object types; mixed groups may contain many sliders。
- `F L359-L428` `_build_plan(hit_group: list[HitObject], object_indexes: list[int], *, dimension: SegmentDimension, source_plan_count: int, circle_size: float, circle_radius: float, approach_context_seconds: float, max_pre_context_seconds: float, pre_context_jitter_seconds: float, post_context_seconds: float, video_duration_seconds: float) -> SegmentPlan`：构建 `plan` 对应的数据或结果。 调用：`SegmentPlan`, `_stable_pre_context_jitter_seconds`, `classify_hit_group`。
- `F L431-L451` `_stable_pre_context_jitter_seconds(object_indexes: list[int], *, hit_start_ms: int, hit_end_ms: int, dimension: SegmentDimension, limit_seconds: float) -> float`：执行 `stable pre context jitter seconds` 对应逻辑。
- `F L454-L521` `build_segment_plans(hit_objects: list[HitObject], *, approach_preempt_ratio: float, circle_size: float, min_circle_overlap_ratio: float, priority_merge_window_ms: int, use_priority_merge: bool, approach_preempt_seconds: float, pre_context_jitter_seconds: float, post_context_seconds: float, video_duration_seconds: float) -> list[SegmentPlan]`：构建并返回 `segment plans` 对应的数据或结果。 调用：`_build_plan`, `circle_radius_from_size`, `group_hit_objects`。
- `F L524-L645` `build_long_sequence_plans(atomic_plans: list[SegmentPlan], *, approach_preempt_seconds: float, approach_preempt_ratio: float, pre_context_jitter_seconds: float, post_context_seconds: float, video_duration_seconds: float, max_objects: int, max_duration_seconds: float) -> list[SegmentPlan]`：构建并返回 `long sequence plans` 对应的数据或结果。 调用：`combined_plan`, `flush`。
- `N L560-L583` `build_long_sequence_plans.combined_plan(plans: list[SegmentPlan]) -> SegmentPlan`：执行 `combined plan` 对应逻辑。 调用：`_build_plan`。
- `N L585-L604` `build_long_sequence_plans.flush() -> None`：执行 `flush` 对应逻辑。 调用：`combined_plan`。

## `src/before_traning/Lib/video/segmentation/segmentation.py`

职责：根据显式参数调用 planner，返回原子与长序列计划集合。
工程依赖：`before_traning.Lib.beatmap.standard`, `before_traning.Lib.video.segmentation.planner`

- `C L14-L20` `SegmentPlanCollection` [CLASS]：封装 `SegmentPlanCollection` 相关数据或行为。
- `M L19-L20` `SegmentPlanCollection.all(self) -> tuple[SegmentPlan, ...]` [PROPERTY]：执行 `all` 对应逻辑。
- `F L23-L68` `plan_video_segments(beatmap: ParsedStandardBeatmap, *, video_duration_seconds: float, approach_preempt_ratio: float, pre_context_jitter_seconds: float, post_context_seconds: float, min_circle_overlap_ratio: float, priority_merge_window_ms: int, use_priority_merge: bool, build_long_sequences: bool, long_sequence_max_objects: int, long_sequence_max_duration_seconds: float) -> SegmentPlanCollection`：执行 `plan video segments` 对应逻辑。 调用：`SegmentPlanCollection`, `build_long_sequence_plans`, `build_segment_plans`。

## `src/before_traning/conf/field_groups.py`

职责：集中声明处理器字段组，负责批量赋值和处理器之间的参数转发。

- `F L72-L73` `group_values(config: SimpleNamespace, group: str) -> tuple[Any, ...]`：执行 `group values` 对应逻辑。
- `F L76-L78` `assign_group(target: Any, config: SimpleNamespace, group: str) -> None`：执行 `assign group` 对应逻辑。
- `F L81-L82` `forward_kwargs(source: Any, group: str) -> dict[str, Any]`：执行 `forward kwargs` 对应逻辑。

## `src/before_traning/conf/legacy_config.py`

职责：旧 builder API 的兼容层；把 Settings 展平、覆盖并按构造函数签名转发。
工程依赖：`before_traning.Lib.common.failures`, `before_traning.conf`, `before_traning.conf.artifacts`, `before_traning.conf.field_groups`

- `C L20-L21` `CheckDataConfigError(Exception)` [CLASS]：封装 `CheckDataConfigError` 相关数据或行为。
- `C L24-L37` `ConfigReader` [CLASS]：Small compatibility reader for legacy build_* helpers。
- `M L27-L31` `ConfigReader.__init__(self, path: Path | None=None)`：初始化实例依赖、配置和运行状态。 调用：`load_settings`, `self.settings.model_dump`。
- `M L33-L34` `ConfigReader.get(self, *_paths: tuple[str, ...]) -> Any`：执行 `get` 对应逻辑。
- `M L36-L37` `ConfigReader.read(self, *_args: Any, **_kwargs: Any) -> Any`：执行 `read` 对应逻辑。
- `F L52-L53` `load_config(config_path: Path | None=None) -> ConfigReader`：加载 `config` 对应的数据或结果。 调用：`ConfigReader`。
- `F L56-L57` `load_process_steps_config(config_path: Path | None=None) -> tuple[str, ...]`：加载 `process steps config` 对应的数据或结果。 调用：`load_settings`。
- `F L60-L69` `load_process_steps_config_or_default(config_path: Path | None=None, default_steps: Iterable[str] | None=None) -> tuple[str, ...]`：加载 `process steps config or default` 对应的数据或结果。 调用：`load_process_steps_config`。
- `F L72-L119` `settings_kwargs(settings: Settings, processor: str | None=None) -> dict[str, Any]`：执行 `settings kwargs` 对应逻辑。
- `F L122-L139` `_coerce_like(reference: Any, value: Any) -> Any`：执行 `coerce like` 对应逻辑。
- `F L142-L153` `settings_namespace(settings: Settings, processor: str | None=None, overrides: Mapping[str, Any] | None=None) -> SimpleNamespace`：执行 `settings namespace` 对应逻辑。 调用：`_coerce_like`, `settings_kwargs`。
- `F L156-L157` `_settings_kwargs(settings: Settings) -> dict[str, Any]`：执行 `settings kwargs` 对应逻辑。 调用：`settings_kwargs`。
- `F L160-L174` `_filter_builder_kwargs(builder: Callable[..., T], config: Mapping[str, Any]) -> dict[str, Any]`：筛选 `builder kwargs` 对应的数据或结果。
- `F L177-L183` `build_from_config(builder: Callable[..., T], _loaders: Iterable[Callable[[ConfigReader], Mapping[str, Any]]], config_path: Path | None=None) -> T`：构建并返回 `from config` 对应的数据或结果。 调用：`_filter_builder_kwargs`, `_settings_kwargs`, `load_settings`。
- `F L186-L200` `build_from_config_or_default(builder: Callable[..., T], loaders: Iterable[Callable[[ConfigReader], Mapping[str, Any]]], config_path: Path | None=None, default_builder: Callable[[], T] | None=None) -> T`：构建并返回 `from config or default` 对应的数据或结果。 调用：`build_from_config`, `format_exception`。

## `src/before_traning/conf/runtime.py`

职责：把 Prefect home 固定到仓库内可写目录。
工程依赖：`before_traning.conf.settings`

- `F L9-L12` `ensure_prefect_home(repo_root: Path=REPO_ROOT) -> Path`：确保 `prefect home` 对应的数据或结果。

## `src/before_traning/conf/settings.py`

职责：Pydantic 配置模型与 YAML/JSON 加载；解析相对路径、切片抖动和去音频配置，并兼容旧配置层级。

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
- `C L98-L161` `SegmentSettings(BaseModel)` [CLASS]：封装 `SegmentSettings` 相关数据或行为。
- `M L114-L119` `SegmentSettings._nonnegative_interval(cls, value: int) -> int` [VALIDATOR]：执行 `nonnegative interval` 对应逻辑。
- `M L123-L126` `SegmentSettings._long_sequence_object_limit(cls, value: int) -> int` [VALIDATOR]：执行 `long sequence object limit` 对应逻辑。
- `M L130-L133` `SegmentSettings._approach_ratio(cls, value: float) -> float` [VALIDATOR]：执行 `approach ratio` 对应逻辑。
- `M L137-L140` `SegmentSettings._overlap_ratio(cls, value: float) -> float` [VALIDATOR]：执行 `overlap ratio` 对应逻辑。
- `M L147-L152` `SegmentSettings._nonnegative_context(cls, value: float) -> float` [VALIDATOR]：执行 `nonnegative context` 对应逻辑。
- `M L156-L161` `SegmentSettings._positive_duration(cls, value: float) -> float` [VALIDATOR]：执行 `positive duration` 对应逻辑。
- `C L164-L176` `ProgressSettings(BaseModel)` [CLASS]：封装 `ProgressSettings` 相关数据或行为。
- `C L179-L212` `Settings(BaseSettings)` [CLASS]：封装 `Settings` 相关数据或行为。
- `M L199-L200` `Settings.target_root(self) -> Path` [PROPERTY]：执行 `target root` 对应逻辑。
- `M L203-L204` `Settings.overwrite(self) -> bool` [PROPERTY]：执行 `overwrite` 对应逻辑。
- `M L207-L208` `Settings.continue_on_error(self) -> bool` [PROPERTY]：执行 `continue on error` 对应逻辑。
- `M L211-L212` `Settings.global_offset_ms(self) -> float` [PROPERTY]：执行 `global offset ms` 对应逻辑。
- `F L215-L225` `_resolve_paths(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]`：解析并定位 `paths` 对应的数据或结果。 调用：`file_management.get`, `raw.get`。
- `F L228-L263` `_extract_nested(raw: dict[str, Any]) -> dict[str, Any]`：提取 `nested` 对应的数据或结果。 调用：`ProgressSettings`, `parameters.get`, `progress.get`, `raw.get`, `required_steps.get`, `status_steps.get`。
- `F L266-L283` `_read_config(config_path: Path) -> dict[str, Any]` [IO-R IO-W]：读取 `config` 对应的数据或结果。 调用：`SettingsError`, `json.load`。
- `F L286-L292` `load_settings(config_path: Path | None=None) -> Settings`：加载 `settings` 对应的数据或结果。 调用：`Settings`, `SettingsError`, `_extract_nested`, `_read_config`, `_resolve_paths`。

## `src/before_traning/core/audio/matching/matching.py`

职责：组合音频匹配处理器，并注入 AV 对齐算法能力。
工程依赖：`before_traning.Lib.beatmap.folder_store`, `before_traning.Lib.beatmap.manifest`, `before_traning.Lib.common.batch`, `before_traning.Lib.common.pathspec`, `before_traning.conf`, `before_traning.conf.defaults`, `before_traning.conf.legacy_config`, `before_traning.core.audio.matching.preflight`, `before_traning.core.audio.matching.steps`, `before_traning.core.audio.matching.wrapup`, `before_traning.core.video.av_processing`, `before_traning.state.process_status`

- `F L30-L31` `_load_audio_match_experiment_config(config: ConfigReader) -> dict[str, object]`：加载 `audio match experiment config` 对应的数据或结果。 调用：`read_config_values`。
- `F L34-L42` `build_audio_match_experiment_from_config_or_default(config_path: Path | None=None) -> 'AudioMatchExperiment'`：构建并返回 `audio match experiment from config or default` 对应的数据或结果。 调用：`build_from_config_or_default`。
- `C L45-L85` `AudioMatchExperiment(AudioMatchWrapUpMixin, AudioMatchStepsMixin, AudioMatchPreflightMixin)` [CLASS]：封装 `AudioMatchExperiment` 相关数据或行为。
- `M L50-L85` `AudioMatchExperiment.__init__(self, settings: Settings=DEFAULTS, status_manager: ProcessStatusManager | None=None, **overrides: object)`：初始化实例依赖、配置和运行状态。 调用：`BeatmapFolderStore`, `ManifestFolderWalker`, `ProcessStatusManager`, `VideoAVProcessor`, `assign_group`, `forward_kwargs`。
- `C L88-L89` `AudioMatchProcessor(AudioMatchExperiment)` [CLASS]：Task-aligned name for the audio-based video matching processor。
- `F L92-L94` `main()`：独立脚本入口，构建处理器并执行。 调用：`build_audio_match_experiment_from_config_or_default`, `experiment.run`。

## `src/before_traning/core/audio/matching/preflight.py`

职责：同步视频匹配状态，收集待匹配文件夹和候选视频。
工程依赖：`before_traning.Lib.common.failures`, `before_traning.Lib.common.pathspec`

- `C L9-L97` `AudioMatchPreflightMixin` [CLASS]：封装 `AudioMatchPreflightMixin` 相关数据或行为。
- `M L10-L15` `AudioMatchPreflightMixin._folder_has_video(self, folder_name: str) -> bool`：执行 `folder has video` 对应逻辑。 调用：`matches_name`, `self.store.get_folder_path`。
- `M L17-L45` `AudioMatchPreflightMixin._sync_video_matched_status(self, folder_name: str)`：同步 `video matched status` 对应的数据或结果。 调用：`failure_detail`, `self._folder_has_video`, `self.status_manager.ensure_status_file`, `self.status_manager.is_step_done`, `self.status_manager.mark_step_done`, `self.status_manager.mark_step_pending`。
- `M L47-L62` `AudioMatchPreflightMixin._pending_folder_names(self) -> list[str]`：执行 `pending folder names` 对应逻辑。 调用：`self._folder_has_video`, `self._sync_video_matched_status`, `self.store.file_exists`, `self.store.folder_exists`, `self.store.get_folder_path`, `self.walker.read_folder_names`。
- `M L64-L74` `AudioMatchPreflightMixin._candidate_folder_names(self, *, include_existing_video: bool) -> list[str]`：执行 `candidate folder names` 对应逻辑。 调用：`self._pending_folder_names`, `self.store.file_exists`, `self.store.folder_exists`, `self.walker.read_folder_names`。
- `M L76-L97` `AudioMatchPreflightMixin._candidate_videos(self, *, allow_fallback: bool) -> list[Path]`：执行 `candidate videos` 对应逻辑。 调用：`filter_files`, `self.video_root.exists`, `self.video_root.iterdir`, `self.walker.read_folder_names`。

## `src/before_traning/core/audio/matching/steps.py`

职责：调用 AV 信号 API 计算视频/歌曲配对得分并做一对一选择。
工程依赖：`before_traning.Lib.common.failures`

- `C L12-L194` `AudioMatchStepsMixin` [CLASS]：封装 `AudioMatchStepsMixin` 相关数据或行为。
- `M L13-L17` `AudioMatchStepsMixin._extract_samples(self, source_path: Path, *, from_video: bool) -> np.ndarray`：提取 `samples` 对应的数据或结果。 调用：`self.aligner._extract_audio_to_wav`, `self.aligner._load_wav_samples`。
- `M L19-L32` `AudioMatchStepsMixin._build_alignment_features(self, samples: np.ndarray) -> dict[str, np.ndarray]`：构建 `alignment features` 对应的数据或结果。 调用：`self.aligner._build_feature_series`, `self.aligner._build_music_refine_series`。
- `M L34-L69` `AudioMatchStepsMixin._estimate_offset_from_features(self, video_features: dict[str, np.ndarray], song_features: dict[str, np.ndarray]) -> tuple[float, float, float]`：估算 `offset from features` 对应的数据或结果。 调用：`self.aligner._estimate_best_start_frame`。
- `M L71-L79` `AudioMatchStepsMixin._result_sort_key(self, item: dict[str, Any]) -> tuple[float, float, float, float]`：执行 `result sort key` 对应逻辑。 调用：`item.get`。
- `M L81-L173` `AudioMatchStepsMixin._score_pairs(self, videos: list[Path], folder_names: list[str]) -> list[dict[str, Any]]`：执行 `score pairs` 对应逻辑。 调用：`exception_detail`, `format_failure`, `self._build_alignment_features`, `self._estimate_offset_from_features`, `self._extract_samples`, `self.aligner._estimate_verify_adjustment_seconds`。
- `M L175-L194` `AudioMatchStepsMixin._select_greedy_matches(self, pair_results: list[dict[str, Any]]) -> list[dict[str, Any]]`：选择 `greedy matches` 对应的数据或结果。

## `src/before_traning/core/audio/matching/wrapup.py`

职责：应用音频匹配结果，移动视频、回写状态并支持回滚。
工程依赖：`before_traning.Lib.common.failures`

- `C L9-L160` `AudioMatchWrapUpMixin` [CLASS]：封装 `AudioMatchWrapUpMixin` 相关数据或行为。
- `M L10-L26` `AudioMatchWrapUpMixin._print_greedy_matches(self, matches: list[dict[str, Any]])`：执行 `print greedy matches` 对应逻辑。
- `M L28-L133` `AudioMatchWrapUpMixin._apply_matches(self, matches: list[dict[str, Any]], pending_folder_names: list[str], candidate_videos: list[Path])` [IO-W]：应用 `matches` 对应的数据或结果。 调用：`exception_detail`, `failure_detail`, `self.status_manager.mark_step_done`, `self.status_manager.mark_step_pending`, `self.store.get_file_path`。
- `M L135-L160` `AudioMatchWrapUpMixin.run(self, *, apply_matches: bool=False, allow_fallback_videos: bool | None=None)`：执行该处理器的完整工作流。 调用：`self._apply_matches`, `self._candidate_folder_names`, `self._candidate_videos`, `self._print_greedy_matches`, `self._score_pairs`, `self._select_greedy_matches`。

## `src/before_traning/core/beatmap/beatmap.py`

职责：谱面阶段统一公开入口；集中导出三个处理器、阶段函数和 pipeline API。
工程依赖：`before_traning.conf`, `before_traning.core.beatmap.difficulty`, `before_traning.core.beatmap.importer`, `before_traning.core.beatmap.pipeline`, `before_traning.core.beatmap.verify`

- `F L21-L22` `run_beatmap(settings: Settings) -> dict[str, bool]`：执行 `run beatmap` 对应逻辑。 调用：`prepare_beatmaps`。

## `src/before_traning/core/beatmap/difficulty.py`

职责：完整难度实施；调用 ProcessingGuard、读取难度并更新 SQLite manifest。
工程依赖：`before_traning.Lib.beatmap.folder_store`, `before_traning.Lib.beatmap.osu_metadata`, `before_traning.Lib.common.batch`, `before_traning.Lib.common.processing`, `before_traning.conf`, `before_traning.core.beatmap.verify`, `before_traning.state.process_status`

- `C L18-L20` `DifficultyEntry` [CLASS]：封装 `DifficultyEntry` 相关数据或行为。
- `C L23-L115` `BeatmapDifficultyProcessor(FolderBatchProcessor)` [CLASS]：封装 `BeatmapDifficultyProcessor` 相关数据或行为。
- `M L24-L43` `BeatmapDifficultyProcessor.__init__(self, store: BeatmapFolderStore, *, status_manager: ProcessStatusManager | None=None)`：初始化实例依赖、配置和运行状态。 调用：`ProcessStatusManager`, `ProcessingGuard`, `super.__init__`。
- `M L45-L50` `BeatmapDifficultyProcessor.write_difficulty(self, folder_name: str, difficulty_value: float) -> None`：写入 `difficulty` 对应的数据或结果。 调用：`self.manifest.set_difficulty`。
- `M L52-L56` `BeatmapDifficultyProcessor.read_difficulty(self, folder_name: str) -> float`：读取 `difficulty` 对应的数据或结果。 调用：`self.manifest.difficulty_for`。
- `M L58-L90` `BeatmapDifficultyProcessor.process_one(self, folder_name: str, overwrite: bool=False) -> BatchProcessResult`：处理 manifest 中的单个内部谱面文件夹。 调用：`read_overall_difficulty`, `self.guard.mark_done`, `self.guard.prepare_folder`, `self.guard.reconcile_existing`, `self.manifest.difficulty_for`, `self.write_difficulty`。
- `M L92-L93` `BeatmapDifficultyProcessor.handle_failure(self, folder_name: str, error: Exception) -> None`：处理单文件夹失败并同步失败状态。 调用：`self.guard.record_failure`。
- `M L95-L115` `BeatmapDifficultyProcessor.list_difficulties(self, min_difficulty: float | None=None, max_difficulty: float | None=None) -> list[DifficultyEntry]`：列出 `difficulties` 对应的数据或结果。 调用：`DifficultyEntry`, `self.read_difficulty`, `self.walker.read_folder_names`。
- `F L121-L131` `export_difficulty(settings: Settings) -> bool`：导出 `difficulty` 对应的数据或结果。 调用：`BeatmapDifficultyProcessor`, `BeatmapDifficultyProcessor.run`, `build_store`。

## `src/before_traning/core/beatmap/importer.py`

职责：完整谱面导入实施；扫描 .osz、更新 manifest、写文件和导入状态。
工程依赖：`before_traning.Lib.beatmap.manifest`, `before_traning.Lib.beatmap.osz`, `before_traning.Lib.beatmap.package`, `before_traning.Lib.common.failures`, `before_traning.Lib.common.pathspec`, `before_traning.Lib.common.processing`, `before_traning.conf`, `before_traning.conf.field_groups`, `before_traning.conf.legacy_config`, `before_traning.state.process_status`

- `C L21-L183` `BeatmapImportProcessor` [CLASS]：封装 `BeatmapImportProcessor` 相关数据或行为。
- `M L22-L52` `BeatmapImportProcessor.__init__(self, settings: Settings=DEFAULT_SETTINGS, **overrides: object)`：初始化实例依赖、配置和运行状态。 调用：`PackageUpdater`, `ProcessStatusManager`, `assign_group`, `settings_namespace`, `suffix_spec`。
- `M L54-L59` `BeatmapImportProcessor._scan_single_osz(self, osz_path: Path) -> OsuEntry | None`：执行 `scan single osz` 对应逻辑。 调用：`read_osz_entry`。
- `M L61-L111` `BeatmapImportProcessor._scan_entries(self) -> list[OsuEntry]`：执行 `scan entries` 对应逻辑。 调用：`format_exception`, `matching_files`, `self._scan_single_osz`。
- `M L113-L127` `BeatmapImportProcessor._rebuild_manifest(self, entries: list[OsuEntry]) -> None`：执行 `rebuild manifest` 对应逻辑。 调用：`ManifestEntry`, `self.updater.replace_manifest`。
- `M L129-L161` `BeatmapImportProcessor._write_entries(self, entries: list[OsuEntry]) -> None` [IO-W]：写入 `entries` 对应的数据或结果。 调用：`self.status_manager.ensure_status_file`, `self.status_manager.mark_step_done`, `self.updater.create_folder_if_registered`, `self.updater.sync_folders_from_manifest`。
- `M L163-L183` `BeatmapImportProcessor.run(self) -> bool`：执行该处理器的完整工作流。 调用：`self._rebuild_manifest`, `self._scan_entries`, `self._write_entries`, `self.updater.find_unregistered_existing_folders`。
- `F L189-L192` `build_beatmap_import_processor_from_config_or_default(config_path: Path | None=None) -> BeatmapImportProcessor`：构建并返回 `beatmap import processor from config or default` 对应的数据或结果。 调用：`BeatmapImportProcessor`, `load_settings`。
- `F L195-L198` `build_osu_osz_processor_from_config_or_default(config_path: Path | None=None) -> BeatmapImportProcessor`：构建并返回 `osu osz processor from config or default` 对应的数据或结果。 调用：`build_beatmap_import_processor_from_config_or_default`。
- `F L201-L206` `import_beatmaps(settings: Settings) -> bool`：导入 `beatmaps` 对应的数据或结果。 调用：`BeatmapImportProcessor`, `BeatmapImportProcessor.run`。

## `src/before_traning/core/beatmap/pipeline.py`

职责：声明七阶段注册表与统一 Pipeline API，并用分组表选择谱面/视频阶段。
工程依赖：`before_traning.Lib.tasks`, `before_traning.conf`, `before_traning.core.beatmap.difficulty`, `before_traning.core.beatmap.importer`, `before_traning.core.beatmap.verify`, `before_traning.core.video.av`, `before_traning.core.video.clip`, `before_traning.core.video.match`, `before_traning.core.video.segment`

- `F L108-L112` `prepare_beatmaps(settings: Settings) -> dict[str, bool]`：顺序准备 `beatmaps` 对应的数据或结果。 调用：`TRAINING_PIPELINE.run_direct`。
- `C L115-L163` `TemporaryTrainingRunner` [CLASS]：封装 `TemporaryTrainingRunner` 相关数据或行为。
- `M L116-L117` `TemporaryTrainingRunner.__init__(self, config_path: Path | None=None)`：初始化实例依赖、配置和运行状态。 调用：`load_settings`。
- `M L119-L163` `TemporaryTrainingRunner.run(self, *, overwrite: bool=False, use_audio_match_experiment: bool=True, global_offset_ms: float | None=None, continue_on_error: bool=False, stage_overrides: Mapping[str, bool | None] | None=None, **legacy_stage_options: bool) -> dict[str, bool]`：执行该处理器的完整工作流。 调用：`TRAINING_PIPELINE.run_direct`, `self.settings.model_copy`, `self.settings.runtime.model_copy`, `self.settings.video_clip.model_copy`。

## `src/before_traning/core/beatmap/verify.py`

职责：完整 verify 实施；调用 ProcessingGuard、标准谱面缓存并导出 verify.txt。
工程依赖：`before_traning.Lib.beatmap.folder_store`, `before_traning.Lib.beatmap.osu_parser`, `before_traning.Lib.beatmap.standard`, `before_traning.Lib.common.batch`, `before_traning.Lib.common.processing`, `before_traning.conf`, `before_traning.state.process_status`

- `F L17-L21` `build_store(settings: Settings) -> BeatmapFolderStore`：构建并返回 `store` 对应的数据或结果。 调用：`BeatmapFolderStore`。
- `C L24-L87` `BeatmapVerifyExporter(FolderBatchProcessor)` [CLASS]：封装 `BeatmapVerifyExporter` 相关数据或行为。
- `M L25-L45` `BeatmapVerifyExporter.__init__(self, store: BeatmapFolderStore, *, status_manager: ProcessStatusManager | None=None)`：初始化实例依赖、配置和运行状态。 调用：`ProcessStatusManager`, `ProcessingGuard`, `VerifyOsuParser`, `super.__init__`。
- `M L47-L84` `BeatmapVerifyExporter.process_one(self, folder_name: str, overwrite: bool=False) -> BatchProcessResult`：处理 manifest 中的单个内部谱面文件夹。 调用：`load_standard_beatmap`, `self.guard.is_complete`, `self.guard.mark_done`, `self.guard.prepare_folder`, `self.guard.step_done`, `self.parser.objects_to_lines`。
- `M L86-L87` `BeatmapVerifyExporter.handle_failure(self, folder_name: str, error: Exception) -> None`：处理单文件夹失败并同步失败状态。 调用：`self.guard.record_failure`。
- `F L93-L97` `build_verify_exporter_from_config_or_default(config_path: Path | None=None) -> BeatmapVerifyExporter`：构建并返回 `verify exporter from config or default` 对应的数据或结果。 调用：`BeatmapVerifyExporter`, `build_store`, `load_settings`。
- `F L100-L103` `build_beatmap_verify_exporter_from_config_or_default(config_path: Path | None=None) -> BeatmapVerifyExporter`：构建并返回 `beatmap verify exporter from config or default` 对应的数据或结果。 调用：`build_verify_exporter_from_config_or_default`。
- `F L106-L113` `export_verify(settings: Settings=DEFAULT_SETTINGS) -> bool`：导出 `verify` 对应的数据或结果。 调用：`BeatmapVerifyExporter`, `BeatmapVerifyExporter.run`, `build_store`。

## `src/before_traning/core/pipeline.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`before_traning.Lib.tasks`, `before_traning.conf`, `before_traning.core.beatmap.difficulty`, `before_traning.core.beatmap.importer`, `before_traning.core.beatmap.verify`, `before_traning.core.video.av`, `before_traning.core.video.clip`, `before_traning.core.video.match`, `before_traning.core.video.segment`

- `C L91-L139` `TemporaryTrainingRunner` [CLASS]：封装 `TemporaryTrainingRunner` 相关数据或行为。
- `M L92-L93` `TemporaryTrainingRunner.__init__(self, config_path: Path | None=None)`：初始化实例依赖、配置和运行状态。 调用：`load_settings`。
- `M L95-L139` `TemporaryTrainingRunner.run(self, *, overwrite: bool=False, use_audio_match_experiment: bool=True, global_offset_ms: float | None=None, continue_on_error: bool=False, stage_overrides: Mapping[str, bool | None] | None=None, **legacy_stage_options: bool) -> dict[str, bool]`：执行该处理器的完整工作流。 调用：`TRAINING_PIPELINE.run_direct`, `self.settings.model_copy`, `self.settings.runtime.model_copy`, `self.settings.video_clip.model_copy`。

## `src/before_traning/core/video/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`before_traning.conf`, `before_traning.core.video.av`, `before_traning.core.video.clip`, `before_traning.core.video.match`, `before_traning.core.video.segment`

- `F L8-L11` `prepare_videos(settings: Settings) -> dict[str, bool]`：顺序准备 `videos` 对应的数据或结果。

## `src/before_traning/core/video/av.py`

职责：AV 对齐业务入口；把音频算法和状态参数传给 VideoAVProcessor。
工程依赖：`before_traning.conf`, `before_traning.core.video.av_processing`

- `F L13-L32` `av_correspondence(settings: Settings) -> bool`：执行 `av correspondence` 对应逻辑。 调用：`VideoAVProcessor`, `VideoAVProcessor.run`。

## `src/before_traning/core/video/av_processing/av_processing.py`

职责：组合 AV 处理器并初始化配置、存储和状态依赖。
工程依赖：`before_traning.Lib.beatmap.folder_store`, `before_traning.Lib.common.batch`, `before_traning.Lib.common.pathspec`, `before_traning.Lib.video.av_processing.steps`, `before_traning.conf`, `before_traning.conf.defaults`, `before_traning.conf.legacy_config`, `before_traning.core.video.av_processing.preflight`, `before_traning.core.video.av_processing.steps`, `before_traning.core.video.av_processing.wrapup`, `before_traning.state.process_status`

- `F L28-L29` `_load_av_correspondence_processor_config(config: ConfigReader) -> dict[str, object]`：加载 `av correspondence processor config` 对应的数据或结果。 调用：`read_config_values`。
- `F L32-L40` `build_av_correspondence_processor_from_config_or_default(config_path: Path | None=None) -> 'AVCorrespondenceProcessor'`：构建并返回 `av correspondence processor from config or default` 对应的数据或结果。 调用：`build_from_config_or_default`。
- `C L43-L86` `AVCorrespondenceProcessor(AVWrapUpMixin, AVProcessStepsMixin, AVCoreStepsMixin, AVPreflightMixin, FolderBatchProcessor)` [CLASS]：封装 `AVCorrespondenceProcessor` 相关数据或行为。
- `M L50-L86` `AVCorrespondenceProcessor.__init__(self, settings: Settings=DEFAULTS, status_manager: ProcessStatusManager | None=None, **overrides: object)`：初始化实例依赖、配置和运行状态。 调用：`BeatmapFolderStore`, `ProcessStatusManager`, `assign_group`, `self._ensure_status_steps_registered`, `self._validate_config`, `self.status_step.strip`。
- `C L89-L90` `VideoAVProcessor(AVCorrespondenceProcessor)` [CLASS]：Task-aligned name for the video AV correspondence processor。
- `F L93-L95` `main()`：独立脚本入口，构建处理器并执行。 调用：`build_av_correspondence_processor_from_config_or_default`, `processor.run`。

## `src/before_traning/core/video/av_processing/preflight.py`

职责：校验 AV 参数/状态步骤并定位阶段输入输出。
工程依赖：`before_traning.Lib.common.failures`, `before_traning.Lib.common.pathspec`

- `C L11-L109` `AVPreflightMixin` [CLASS]：封装 `AVPreflightMixin` 相关数据或行为。
- `M L12-L28` `AVPreflightMixin._validate_config(self, config)`：校验 `config` 对应的数据或结果。
- `M L30-L39` `AVPreflightMixin._ensure_status_steps_registered(self)`：确保 `status steps registered` 对应的数据或结果。
- `M L41-L59` `AVPreflightMixin._resolve_source_video_path(self, folder_name: str) -> Path`：解析并定位 `source video path` 对应的数据或结果。 调用：`filter_files`, `self.store.get_folder_path`。
- `M L61-L65` `AVPreflightMixin._resolve_song_audio_path(self, folder_name: str) -> Path`：解析并定位 `song audio path` 对应的数据或结果。 调用：`self.store.get_file_path`。
- `M L67-L68` `AVPreflightMixin._resolve_verify_path(self, folder_name: str) -> Path`：解析并定位 `verify path` 对应的数据或结果。 调用：`self.store.get_file_path`。
- `M L70-L97` `AVPreflightMixin._sync_output_status(self, folder_name: str) -> tuple[bool, bool]`：同步 `output status` 对应的数据或结果。 调用：`failure_detail`, `self.status_manager.is_step_done`, `self.status_manager.mark_step_done`, `self.status_manager.mark_step_pending`, `self.store.file_exists`, `self.store.get_file_path`。
- `M L99-L109` `AVPreflightMixin._ensure_required_steps_done(self, folder_name: str)`：确保 `required steps done` 对应的数据或结果。 调用：`self.status_manager.is_step_done`。

## `src/before_traning/core/video/av_processing/steps.py`

职责：执行单文件夹 AV 对齐阶段和状态推进。
工程依赖：`before_traning.Lib.common.batch`

- `C L9-L137` `AVProcessStepsMixin` [CLASS]：封装 `AVProcessStepsMixin` 相关数据或行为。
- `M L10-L137` `AVProcessStepsMixin.process_one(self, folder_name: str, overwrite: bool=False) -> BatchProcessResult`：处理 manifest 中的单个内部谱面文件夹。 调用：`self._build_feature_series`, `self._ensure_required_steps_done`, `self._estimate_offset_seconds`, `self._estimate_verify_adjustment_seconds`, `self._extract_audio_to_wav`, `self._load_wav_samples`。

## `src/before_traning/core/video/av_processing/wrapup.py`

职责：记录 AV 阶段进度、完成细节和失败状态。
工程依赖：`before_traning.Lib.common.failures`

- `C L6-L74` `AVWrapUpMixin` [CLASS]：封装 `AVWrapUpMixin` 相关数据或行为。
- `M L7-L11` `AVWrapUpMixin._update_progress(self, folder_name: str, stage: str, detail: dict | None=None)`：执行 `update progress` 对应逻辑。 调用：`self.status_manager.mark_step_pending`。
- `M L13-L14` `AVWrapUpMixin.progress_message(self, index: int, total: int, folder_name: str) -> str | None`：生成当前批处理进度文本。
- `M L16-L58` `AVWrapUpMixin._build_done_detail(self, *, source_video_path, output_video_path, song_audio_path, verify_path, raw_offset_seconds: float, verify_adjustment_seconds: float, global_offset_seconds: float, offset_seconds: float, trim_start_seconds: float, song_duration_seconds: float, score: float, coarse_score: float, verify_detail: dict[str, float] | None) -> dict`：构建 `done detail` 对应的数据或结果。
- `M L60-L65` `AVWrapUpMixin._mark_done(self, folder_name: str, **detail_kwargs)`：更新状态为 `done` 对应的数据或结果。 调用：`self._build_done_detail`, `self.status_manager.mark_step_done`。
- `M L67-L74` `AVWrapUpMixin.handle_failure(self, folder_name: str, error: Exception)`：处理单文件夹失败并同步失败状态。 调用：`exception_detail`, `self.status_manager.ensure_status_file`, `self.status_manager.mark_step_pending`, `self.store.folder_exists`。

## `src/before_traning/core/video/clip.py`

职责：固定区域裁剪业务入口。
工程依赖：`before_traning.conf`, `before_traning.core.video.clipping`

- `F L11-L16` `crop_video(settings: Settings) -> bool`：裁剪 `video` 对应的数据或结果。 调用：`VideoClipProcessor.from_settings`, `VideoClipProcessor.from_settings.run`。

## `src/before_traning/core/video/clipping/clipping.py`

职责：组合固定区域裁剪处理器并校验阶段配置。
工程依赖：`before_traning.Lib.beatmap.folder_store`, `before_traning.Lib.common.batch`, `before_traning.Lib.common.failures`, `before_traning.Lib.video.clipping.geometry`, `before_traning.conf`, `before_traning.conf.defaults`, `before_traning.conf.legacy_config`, `before_traning.core.video.clipping.preflight`, `before_traning.core.video.clipping.steps`, `before_traning.core.video.clipping.wrapup`, `before_traning.state.process_status`

- `F L24-L36` `build_fixed_region_video_crop_processor_from_config_or_default(config_path: Path | None=None) -> 'FixedRegionVideoCropProcessor'`：构建并返回 `fixed region video crop processor from config or default` 对应的数据或结果。 调用：`FixedRegionVideoCropProcessor`, `format_exception`, `load_settings`。
- `C L39-L112` `FixedRegionVideoCropProcessor(ClipWrapUpMixin, ClipStepsMixin, ClipGeometryMixin, ClipPreflightMixin, FolderBatchProcessor)` [CLASS]：封装 `FixedRegionVideoCropProcessor` 相关数据或行为。
- `M L47-L52` `FixedRegionVideoCropProcessor.from_settings(cls, settings: Settings, status_manager: ProcessStatusManager | None=None) -> 'FixedRegionVideoCropProcessor'`：从 Settings 创建处理器实例。
- `M L54-L112` `FixedRegionVideoCropProcessor.__init__(self, settings: Settings=DEFAULTS, status_manager: ProcessStatusManager | None=None, **overrides: object)`：初始化实例依赖、配置和运行状态。 调用：`BeatmapFolderStore`, `ProcessStatusManager`, `assign_group`, `group_values`, `self._ensure_status_steps_registered`, `settings_namespace`。
- `C L115-L116` `VideoClipProcessor(FixedRegionVideoCropProcessor)` [CLASS]：Task-aligned name for the fixed-region video clip processor。
- `F L119-L121` `main()`：独立脚本入口，构建处理器并执行。 调用：`build_fixed_region_video_crop_processor_from_config_or_default`, `processor.run`。

## `src/before_traning/core/video/clipping/preflight.py`

职责：校验裁剪状态步骤和单文件夹前置条件。

- `C L4-L32` `ClipPreflightMixin` [CLASS]：封装 `ClipPreflightMixin` 相关数据或行为。
- `M L5-L14` `ClipPreflightMixin._ensure_status_steps_registered(self)`：确保 `status steps registered` 对应的数据或结果。
- `M L16-L32` `ClipPreflightMixin._ensure_folder_ready(self, folder_name: str, overwrite: bool) -> bool`：确保 `folder ready` 对应的数据或结果。 调用：`self.status_manager.ensure_status_file`, `self.status_manager.is_step_done`, `self.store.folder_exists`。

## `src/before_traning/core/video/clipping/steps.py`

职责：调用通用 crop_video API 执行单文件夹原地裁剪。
工程依赖：`before_traning.Lib.common.batch`, `before_traning.Lib.tools.ffmpeg`

- `C L9-L45` `ClipStepsMixin` [CLASS]：封装 `ClipStepsMixin` 相关数据或行为。
- `M L10-L27` `ClipStepsMixin._crop_video_in_place(self, video_path: Path, crop_info: dict[str, int])` [IO-W]：执行 `crop video in place` 对应逻辑。 调用：`crop_video`, `temp_output_path.replace`。
- `M L29-L45` `ClipStepsMixin.process_one(self, folder_name: str, overwrite: bool=False) -> BatchProcessResult`：处理 manifest 中的单个内部谱面文件夹。 调用：`self._crop_video_in_place`, `self._ensure_folder_ready`, `self._mark_cropping`, `self._mark_done`, `self._validate_crop_bounds`, `self.store.get_file_path`。

## `src/before_traning/core/video/clipping/wrapup.py`

职责：记录裁剪进度、实际坐标和失败状态。
工程依赖：`before_traning.Lib.common.failures`

- `C L8-L71` `ClipWrapUpMixin` [CLASS]：封装 `ClipWrapUpMixin` 相关数据或行为。
- `M L9-L10` `ClipWrapUpMixin.progress_message(self, index: int, total: int, folder_name: str) -> str | None`：生成当前批处理进度文本。
- `M L12-L20` `ClipWrapUpMixin._reference_detail(self) -> dict[str, int]`：执行 `reference detail` 对应逻辑。
- `M L22-L37` `ClipWrapUpMixin._mark_cropping(self, folder_name: str, video_path: Path, crop_info: dict[str, int])`：更新状态为 `cropping` 对应的数据或结果。 调用：`self._reference_detail`, `self.status_manager.mark_step_pending`。
- `M L39-L58` `ClipWrapUpMixin._mark_done(self, folder_name: str, video_path: Path, video_width: int, video_height: int, crop_info: dict[str, int])`：更新状态为 `done` 对应的数据或结果。 调用：`self._reference_detail`, `self.status_manager.mark_step_done`。
- `M L60-L71` `ClipWrapUpMixin.handle_failure(self, folder_name: str, error: Exception)`：处理单文件夹失败并同步失败状态。 调用：`exception_detail`, `self._reference_detail`, `self.status_manager.ensure_status_file`, `self.status_manager.mark_step_pending`, `self.store.folder_exists`。

## `src/before_traning/core/video/match.py`

职责：视频匹配业务入口；处理“全部已有视频”的正常跳过情况。
工程依赖：`before_traning.conf`, `before_traning.core.video.matching`

- `F L11-L33` `match_videos(settings: Settings) -> bool`：匹配 `videos` 对应的数据或结果。 调用：`VideoMatchProcessor`, `VideoMatchProcessor.run`。

## `src/before_traning/core/video/matching/builders.py`

职责：视频顺序匹配器的兼容配置 builder。
工程依赖：`before_traning.Lib.common.batch`, `before_traning.conf.legacy_config`, `before_traning.core.video.matching.renamer`

- `F L14-L15` `_load_video_package_renamer_config(config: ConfigReader) -> dict[str, object]`：加载 `video package renamer config` 对应的数据或结果。 调用：`read_config_values`。
- `F L18-L26` `build_video_package_renamer_from_config_or_default(config_path: Path | None=None) -> VideoPackageRenamer`：构建并返回 `video package renamer from config or default` 对应的数据或结果。 调用：`build_from_config_or_default`。

## `src/before_traning/core/video/matching/matching.py`

职责：视频匹配策略入口；在音频匹配与时间顺序重命名之间切换。
工程依赖：`before_traning.conf`, `before_traning.conf.defaults`, `before_traning.conf.legacy_config`, `before_traning.core.video.matching.renamer`

- `C L9-L39` `VideoMatchProcessor` [CLASS]：Video matching entry point used by the video/match task。
- `M L12-L23` `VideoMatchProcessor.__init__(self, settings: Settings=DEFAULTS, **overrides: object)`：初始化实例依赖、配置和运行状态。 调用：`assign_group`, `settings_namespace`。
- `M L25-L39` `VideoMatchProcessor.run(self) -> None`：执行该处理器的完整工作流。 调用：`AudioMatchProcessor`, `AudioMatchProcessor.run`, `VideoMatchRenamer`, `VideoMatchRenamer.run`, `forward_kwargs`。

## `src/before_traning/core/video/matching/renamer.py`

职责：按录像时间与 manifest 顺序移动视频，并支持异常回滚。
工程依赖：`before_traning.Lib.beatmap.manifest`, `before_traning.Lib.common.failures`, `before_traning.Lib.common.pathspec`, `before_traning.conf`, `before_traning.conf.defaults`, `before_traning.conf.legacy_config`, `before_traning.state.process_status`

- `C L22-L187` `VideoPackageRenamer` [CLASS]：封装 `VideoPackageRenamer` 相关数据或行为。
- `M L23-L46` `VideoPackageRenamer.__init__(self, settings: Settings=DEFAULTS, status_manager: ProcessStatusManager | None=None, **overrides: object)`：初始化实例依赖、配置和运行状态。 调用：`ManifestFolderWalker`, `ProcessStatusManager`, `assign_group`, `settings_namespace`, `suffix_spec`。
- `M L48-L52` `VideoPackageRenamer._folder_has_video(self, folder_path: Path) -> bool`：执行 `folder has video` 对应逻辑。 调用：`matches_name`。
- `M L54-L69` `VideoPackageRenamer._parse_video_time(self, path: Path) -> datetime`：解析 `video time` 对应的数据或结果。
- `M L71-L82` `VideoPackageRenamer._list_videos_in_time_order(self) -> list[Path]`：列出 `videos in time order` 对应的数据或结果。 调用：`filter_files`, `self._parse_video_time`, `self.video_root.exists`, `self.video_root.iterdir`。
- `M L84-L115` `VideoPackageRenamer._pending_folder_names(self) -> list[str]`：执行 `pending folder names` 对应逻辑。 调用：`failure_detail`, `self._folder_has_video`, `self.status_manager.ensure_status_file`, `self.status_manager.is_step_done`, `self.status_manager.mark_step_done`, `self.status_manager.mark_step_pending`。
- `M L117-L142` `VideoPackageRenamer._build_rename_plan(self) -> list[tuple[str, Path, Path]]`：构建 `rename plan` 对应的数据或结果。 调用：`self._list_videos_in_time_order`, `self._pending_folder_names`。
- `M L144-L187` `VideoPackageRenamer.run(self)` [IO-W]：执行该处理器的完整工作流。 调用：`exception_detail`, `self._build_rename_plan`, `self.status_manager.mark_step_done`, `self.status_manager.mark_step_pending`。
- `C L190-L191` `VideoMatchRenamer(VideoPackageRenamer)` [CLASS]：Task-aligned name for sequence-based video matching。

## `src/before_traning/core/video/pipeline.py`

职责：顺序组合视频匹配、AV 对齐、裁剪和谱面切分。
工程依赖：`before_traning.conf`, `before_traning.core.beatmap.pipeline`

- `F L10-L14` `prepare_videos(settings: Settings) -> dict[str, bool]`：顺序准备 `videos` 对应的数据或结果。 调用：`TRAINING_PIPELINE.run_direct`。

## `src/before_traning/core/video/segment.py`

职责：最终谱面视频切分处理器；映射设置、调度分类、生成产物并更新状态。
工程依赖：`before_traning.Lib.beatmap.folder_store`, `before_traning.Lib.beatmap.osu_parser`, `before_traning.Lib.beatmap.standard`, `before_traning.Lib.common.batch`, `before_traning.Lib.common.failures`, `before_traning.Lib.common.sequence`, `before_traning.Lib.tools.ffmpeg`, `before_traning.Lib.video.segment_dataset`, `before_traning.Lib.video.segmentation`, `before_traning.Lib.video.segmentation.planner`, `before_traning.conf`, `before_traning.state.process_status`

- `F L50-L57` `_overlap_merge_window_ms(beatmap: ParsedStandardBeatmap, settings: Settings) -> int`：执行 `overlap merge window ms` 对应逻辑。
- `F L60-L63` `_output_directory_name(plan: SegmentPlan) -> str`：执行 `output directory name` 对应逻辑。
- `F L66-L71` `_segment_directory_name(index: int, plan: SegmentPlan) -> str`：执行 `segment directory name` 对应逻辑。 调用：`format_sequence_name`。
- `F L74-L154` `_beatmap_payload(*, folder_name: str, source_osu_path: Path, segment_id: str, beatmap: ParsedStandardBeatmap, plan: SegmentPlan, settings: Settings, parser: VerifyOsuParser) -> dict[str, object]`：执行 `beatmap payload` 对应逻辑。 调用：`_overlap_merge_window_ms`, `parser.hit_object_to_dict`。
- `F L157-L248` `_segment_row(*, segment_id: str, output_directory_name: str, directory_name: str, beatmap: ParsedStandardBeatmap, plan: SegmentPlan, settings: Settings) -> dict[str, object]`：执行 `segment row` 对应逻辑。 调用：`_overlap_merge_window_ms`。
- `C L251-L500` `VideoSegmentationProcessor(FolderBatchProcessor)` [CLASS]：封装 `VideoSegmentationProcessor` 相关数据或行为。
- `M L252-L297` `VideoSegmentationProcessor.__init__(self, settings: Settings, status_manager: ProcessStatusManager | None=None)`：初始化实例依赖、配置和运行状态。 调用：`BeatmapFolderStore`, `ProcessStatusManager`, `SegmentDatasetManifest`, `VerifyOsuParser`, `self.store.recover_atomic_outputs`, `self.walker.manifest.export_table`。
- `M L299-L305` `VideoSegmentationProcessor.progress_message(self, index: int, total: int, folder_name: str) -> str`：生成当前批处理进度文本。
- `M L307-L491` `VideoSegmentationProcessor.process_one(self, folder_name: str, overwrite: bool=False) -> BatchProcessResult`：处理 manifest 中的单个内部谱面文件夹。 调用：`_beatmap_payload`, `_output_directory_name`, `_segment_directory_name`, `_segment_row`, `failure_detail`, `format_sequence_name`。
- `M L493-L500` `VideoSegmentationProcessor.handle_failure(self, folder_name: str, error: Exception) -> None`：处理单文件夹失败并同步失败状态。 调用：`exception_detail`, `self.status_manager.ensure_status_file`, `self.status_manager.mark_step_pending`, `self.store.folder_exists`。
- `F L503-L510` `segment_videos(settings: Settings) -> bool`：执行 `segment videos` 对应逻辑。 调用：`VideoSegmentationProcessor`, `VideoSegmentationProcessor.run`。

## `src/before_traning/main.py`

职责：Typer CLI 入口；合并命令行覆盖项，选择 direct/Prefect runner，并渲染阶段结果。
工程依赖：`before_traning.conf`, `before_traning.core.beatmap.pipeline`

- `F L25-L26` `_resolve(default: bool, override: bool | None) -> bool`：执行 `resolve` 对应逻辑。
- `F L29-L30` `_skip(default: bool, skip_flag: bool) -> bool`：执行 `skip` 对应逻辑。
- `F L33-L60` `_settings(config: Path | None, overwrite: bool | None=None, continue_on_error: bool | None=None, use_audio_match_experiment: bool | None=None, global_offset_ms: float | None=None) -> Settings`：执行 `settings` 对应逻辑。 调用：`_resolve`, `load_settings`。
- `F L63-L72` `_render(results: dict[str, bool], elapsed: float)`：执行 `render` 对应逻辑。
- `F L75-L83` `run_training_pipeline(settings: Settings, **stages: bool | None) -> dict[str, bool]`：执行 `run training pipeline` 对应逻辑。 调用：`os.environ.get`。
- `F L86-L87` `pipeline_exit_code(results: dict[str, bool]) -> int`：执行 `pipeline exit code` 对应逻辑。
- `F L90-L130` `run_data_workflow(*, config: Path | None=None, overwrite: bool | None=None, continue_on_error: bool | None=None, skip_get_files: bool=False, skip_verify_export: bool=False, skip_difficulty_export: bool=False, skip_video_match: bool=False, skip_av_correspondence: bool=False, skip_clip: bool=False, skip_segment: bool=False, use_audio_match_experiment: bool | None=None, global_offset_ms: float | None=None) -> dict[str, bool]`：执行 `run data workflow` 对应逻辑。 调用：`_settings`, `_skip`, `run_training_pipeline`。
- `F L133-L148` `run_verify_workflow(*, config: Path | None=None, overwrite: bool | None=None, continue_on_error: bool | None=None) -> dict[str, bool]`：执行 `run verify workflow` 对应逻辑。 调用：`_settings`, `run_training_pipeline`。
- `F L151-L167` `run_match_workflow(*, config: Path | None=None, overwrite: bool | None=None, continue_on_error: bool | None=None, use_audio_match_experiment: bool | None=None) -> dict[str, bool]`：执行 `run match workflow` 对应逻辑。 调用：`_settings`, `run_training_pipeline`。
- `F L170-L186` `run_clip_workflow(*, config: Path | None=None, overwrite: bool | None=None, continue_on_error: bool | None=None, global_offset_ms: float | None=None) -> dict[str, bool]`：执行 `run clip workflow` 对应逻辑。 调用：`_settings`, `run_training_pipeline`。
- `F L189-L204` `run_segment_workflow(*, config: Path | None=None, overwrite: bool | None=None, continue_on_error: bool | None=None) -> dict[str, bool]`：执行 `run segment workflow` 对应逻辑。 调用：`_settings`, `run_training_pipeline`。
- `F L207-L208` `run_default_workflow() -> dict[str, bool]`：执行 `run default workflow` 对应逻辑。 调用：`load_settings`, `run_training_pipeline`。
- `F L211-L213` `_render_workflow_result(results: dict[str, bool], started_at: float) -> int`：执行 `render workflow result` 对应逻辑。 调用：`_render`, `pipeline_exit_code`。
- `F L217-L249` `run_command(config: Path | None=typer.Option(None, '--config', help='config.yaml/json path.'), overwrite: bool | None=typer.Option(None, '--overwrite/--no-overwrite'), continue_on_error: bool | None=typer.Option(None, '--continue-on-error/--stop-on-error'), skip_get_files: bool=typer.Option(False, '--skip-get-files'), skip_verify_export: bool=typer.Option(False, '--skip-verify-export'), skip_difficulty_export: bool=typer.Option(False, '--skip-difficulty-export'), skip_video_match: bool=typer.Option(False, '--skip-video-match'), skip_av_correspondence: bool=typer.Option(False, '--skip-av-correspondence'), skip_clip: bool=typer.Option(False, '--skip-clip'), skip_segment: bool=typer.Option(False, '--skip-segment'), use_audio_match_experiment: bool | None=typer.Option(None, '--use-audio-match-experiment/--disable-audio-match-experiment'), global_offset_ms: float | None=typer.Option(None, '--global-offset-ms'))` [CLI]：执行 `run command` 对应逻辑。 调用：`_render_workflow_result`, `run_data_workflow`。
- `F L253-L264` `verify_command(config: Path | None=typer.Option(None, '--config', help='config.yaml/json path.'), overwrite: bool | None=typer.Option(None, '--overwrite/--no-overwrite'), continue_on_error: bool | None=typer.Option(None, '--continue-on-error/--stop-on-error'))` [CLI]：执行 `verify command` 对应逻辑。 调用：`_render_workflow_result`, `run_verify_workflow`。
- `F L268-L284` `match_command(config: Path | None=typer.Option(None, '--config', help='config.yaml/json path.'), overwrite: bool | None=typer.Option(None, '--overwrite/--no-overwrite'), continue_on_error: bool | None=typer.Option(None, '--continue-on-error/--stop-on-error'), use_audio_match_experiment: bool | None=typer.Option(None, '--use-audio-match-experiment/--disable-audio-match-experiment'))` [CLI]：匹配 `command` 对应的数据或结果。 调用：`_render_workflow_result`, `run_match_workflow`。
- `F L288-L301` `clip_command(config: Path | None=typer.Option(None, '--config', help='config.yaml/json path.'), overwrite: bool | None=typer.Option(None, '--overwrite/--no-overwrite'), continue_on_error: bool | None=typer.Option(None, '--continue-on-error/--stop-on-error'), global_offset_ms: float | None=typer.Option(None, '--global-offset-ms'))` [CLI]：执行 `clip command` 对应逻辑。 调用：`_render_workflow_result`, `run_clip_workflow`。
- `F L305-L318` `segment_command(config: Path | None=typer.Option(None, '--config', help='config.yaml/json path.'), overwrite: bool | None=typer.Option(None, '--overwrite/--no-overwrite'), continue_on_error: bool | None=typer.Option(None, '--continue-on-error/--stop-on-error'))` [CLI]：执行 `segment command` 对应逻辑。 调用：`_render_workflow_result`, `run_segment_workflow`。
- `F L322-L326` `default_command(ctx: typer.Context)` [CLI]：执行 `default command` 对应逻辑。 调用：`_render_workflow_result`, `run_default_workflow`。
- `F L329-L337` `main(argv: list[str] | None=None) -> int`：独立脚本入口，构建处理器并执行。

## `src/before_traning/state/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。

- `F L7-L12` `__getattr__(name: str)`：执行 `getattr` 对应逻辑。

## `src/before_traning/state/manifest_schema.py`

职责：SQLModel 训练包 manifest 与谱面解析缓存表。

- `C L9-L24` `PackageManifestItem(SQLModel)` [CLASS]：封装 `PackageManifestItem` 相关数据或行为。
- `C L27-L39` `BeatmapDataRecord(SQLModel)` [CLASS]：封装 `BeatmapDataRecord` 相关数据或行为。

## `src/before_traning/state/process_status.py`

职责：按谱面文件夹读写 SQLite 处理状态，并迁移旧 process_status.json。
工程依赖：`before_traning.Lib.beatmap.manifest`, `before_traning.conf`, `before_traning.state.manifest_schema`, `before_traning.state.status_schema`

- `C L25-L208` `ProcessStatusManager` [CLASS]：封装 `ProcessStatusManager` 相关数据或行为。
- `M L26-L48` `ProcessStatusManager.__init__(self, target_root: str, manifest_filename: str=MANIFEST_DB_FILENAME, status_filename: str='process_status.json', process_steps: Iterable[str] | None=None, db_filename: str=STATUS_DB_FILENAME)` [DB]：初始化实例依赖、配置和运行状态。 调用：`ManifestFolderWalker`, `load_settings`, `normalize_process_steps`。
- `M L50-L56` `ProcessStatusManager._normalize_folder_name(self, folder_name: str) -> str`：规范化 `folder name` 对应的数据或结果。
- `M L58-L59` `ProcessStatusManager._registered_names(self) -> set[str]`：执行 `registered names` 对应逻辑。 调用：`self.walker.read_folder_names`。
- `M L61-L66` `ProcessStatusManager._assert_registered(self, folder_name: str)`：执行 `assert registered` 对应逻辑。 调用：`self._normalize_folder_name`, `self._registered_names`。
- `M L68-L74` `ProcessStatusManager._require_existing_folder(self, folder_name: str) -> Path`：执行 `require existing folder` 对应逻辑。 调用：`self._assert_registered`, `self._normalize_folder_name`。
- `M L76-L77` `ProcessStatusManager._default_status(self) -> dict[str, Any]`：执行 `default status` 对应逻辑。 调用：`default_status`。
- `M L79-L81` `ProcessStatusManager._validate_step(self, step: str)`：校验 `step` 对应的数据或结果。
- `M L83-L84` `ProcessStatusManager._normalize_status(self, raw_status: dict[str, Any] | None) -> dict[str, Any]`：规范化 `status` 对应的数据或结果。 调用：`normalize_status`。
- `M L86-L97` `ProcessStatusManager._select_record(self, session: Session, folder_name: str, step: str) -> ProcessStepStatus | None` [DB]：选择 `record` 对应的数据或结果。 调用：`select`。
- `M L99-L105` `ProcessStatusManager._has_records(self, folder_name: str) -> bool` [DB]：执行 `has records` 对应逻辑。 调用：`select`。
- `M L107-L121` `ProcessStatusManager._migrate_legacy_status_key(self, folder_name: str) -> None` [DB]：执行 `migrate legacy status key` 对应逻辑。 调用：`select`, `self.walker.source_name_for`。
- `M L123-L128` `ProcessStatusManager._load_legacy_json(self, folder_name: str) -> dict[str, Any] | None` [IO-R IO-W]：加载 `legacy json` 对应的数据或结果。 调用：`json.load`, `self.get_status_path`。
- `M L130-L132` `ProcessStatusManager.get_status_path(self, folder_name: str) -> Path`：获取 `status path` 对应的数据或结果。 调用：`self._require_existing_folder`。
- `M L134-L158` `ProcessStatusManager.load_status(self, folder_name: str) -> dict[str, Any]` [DB IO-W]：加载 `status` 对应的数据或结果。 调用：`decode_detail`, `select`, `self._default_status`, `self._has_records`, `self._load_legacy_json`, `self._migrate_legacy_status_key`。
- `M L160-L178` `ProcessStatusManager.save_status(self, folder_name: str, status: dict[str, Any])` [DB]：执行 `save status` 对应逻辑。 调用：`ProcessStepStatus`, `encode_detail`, `self._normalize_folder_name`, `self._normalize_status`, `self._require_existing_folder`, `self._select_record`。
- `M L180-L183` `ProcessStatusManager.ensure_status_file(self, folder_name: str) -> dict[str, Any]`：确保 `status file` 对应的数据或结果。 调用：`self.load_status`, `self.save_status`。
- `M L185-L188` `ProcessStatusManager.is_step_done(self, folder_name: str, step: str) -> bool`：判断是否 `step done` 对应的数据或结果。 调用：`self._validate_step`, `self.load_status`。
- `M L190-L196` `ProcessStatusManager.mark_step_done(self, folder_name: str, step: str, detail: Any=None)`：更新状态为 `step done` 对应的数据或结果。 调用：`self._validate_step`, `self.load_status`, `self.save_status`。
- `M L198-L204` `ProcessStatusManager.mark_step_pending(self, folder_name: str, step: str, detail: Any=None)`：更新状态为 `step pending` 对应的数据或结果。 调用：`self._validate_step`, `self.load_status`, `self.save_status`。
- `M L206-L208` `ProcessStatusManager.get_steps_summary(self, folder_name: str) -> dict[str, bool]`：获取 `steps summary` 对应的数据或结果。 调用：`self.load_status`。

## `src/before_traning/state/segment_schema.py`

职责：SQLModel 视频片段数据集索引表。

- `C L9-L27` `SegmentDatasetItem(SQLModel)` [CLASS]：封装 `SegmentDatasetItem` 相关数据或行为。

## `src/before_traning/state/status_schema.py`

职责：独立的 SQLModel 状态表、处理步骤规范化和 detail JSON 编解码。

- `C L22-L34` `ProcessStepStatus(SQLModel)` [CLASS]：封装 `ProcessStepStatus` 相关数据或行为。
- `F L37-L56` `normalize_process_steps(process_steps: Iterable[str]) -> tuple[str, ...]`：规范化 `process steps` 对应的数据或结果。
- `F L59-L69` `default_status(process_steps: Iterable[str]) -> dict[str, Any]`：执行 `default status` 对应逻辑。
- `F L72-L94` `normalize_status(raw_status: dict[str, Any] | None, process_steps: Iterable[str]) -> dict[str, Any]`：规范化 `status` 对应的数据或结果。 调用：`default_status`, `raw_status.get`, `raw_step.get`, `raw_steps.get`。
- `F L97-L100` `encode_detail(detail: Any) -> str | None`：执行 `encode detail` 对应逻辑。
- `F L103-L109` `decode_detail(detail_json: str | None) -> Any`：执行 `decode detail` 对应逻辑。

## `src/before_traning/tests/full_checks/runner.py`

职责：before_traning 全面检测统一入口；运行 full_checks 下的 pytest。
工程依赖：`package.checks`

- `F L14-L37` `run_full_checks() -> StartupCheckReport`：执行 `run full checks` 对应逻辑。 调用：`_run_pytest`, `_tail`。
- `F L40-L52` `_run_pytest(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]` [PROCESS]：执行 `run pytest` 对应逻辑。 调用：`env.get`, `subprocess.run`。
- `F L55-L56` `_tail(text: str, *, max_lines: int=80) -> str`：执行 `tail` 对应逻辑。

## `src/before_traning/tests/full_checks/test_cli_adapters.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`before_traning`, `before_traning.conf`

- `C L13-L71` `BeforeTrainingCliAdapterTests(unittest.TestCase)` [CLASS]：封装 `BeforeTrainingCliAdapterTests` 相关数据或行为。
- `M L14-L35` `BeforeTrainingCliAdapterTests.test_business_workflow_calls_pipeline_without_typer(self) -> None`：执行 `test business workflow calls pipeline without typer` 对应逻辑。 调用：`Settings`, `before_main.run_data_workflow`, `self.assertEqual`, `self.assertFalse`, `self.assertTrue`。
- `M L37-L61` `BeforeTrainingCliAdapterTests.test_run_cli_passes_arguments_to_business_function(self) -> None`：执行 `test run cli passes arguments to business function` 对应逻辑。 调用：`self.assertEqual`, `self.assertTrue`。
- `M L63-L71` `BeforeTrainingCliAdapterTests.test_run_cli_uses_pipeline_result_as_exit_code(self) -> None`：执行 `test run cli uses pipeline result as exit code` 对应逻辑。 调用：`self.assertEqual`。

## `src/before_traning/tests/full_checks/test_segmentation_planner.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`before_traning.Lib.beatmap.hit_objects`, `before_traning.Lib.tools.ffmpeg`, `before_traning.Lib.video.segmentation.planner`

- `C L11-L50` `SegmentPlannerTests(unittest.TestCase)` [CLASS]：封装 `SegmentPlannerTests` 相关数据或行为。
- `M L12-L30` `SegmentPlannerTests._plans(self)`：执行 `plans` 对应逻辑。 调用：`Circle`, `build_segment_plans`。
- `M L32-L39` `SegmentPlannerTests.test_pre_context_jitter_is_stable_and_varied(self) -> None`：执行 `test pre context jitter is stable and varied` 对应逻辑。 调用：`self._plans`, `self.assertEqual`, `self.assertGreater`, `self.assertTrue`。
- `M L41-L50` `SegmentPlannerTests.test_segment_video_strips_audio_by_default(self) -> None`：执行 `test segment video strips audio by default` 对应逻辑。 调用：`build_segment_video_args`, `self.assertIn`, `self.assertNotIn`。

## `src/before_traning/tests/startup_checks/items.py`

职责：before_traning 启动检测项；包含配置、阶段注册、分段规划和 raw-data 决策信号。
工程依赖：`before_traning.Lib.beatmap.hit_objects`, `before_traning.Lib.tools.ffmpeg`, `before_traning.Lib.video.segmentation.planner`, `before_traning.conf`, `before_traning.core.beatmap.pipeline`, `before_traning.tests.startup_checks.samples`, `package.checks`

- `F L17-L34` `check_settings_load(config_path: Path | None=None) -> tuple[StartupCheckResult, Settings]`：执行 `check settings load` 对应逻辑。 调用：`load_settings`。
- `F L37-L67` `check_pipeline_tasks(_settings: Settings | None=None) -> tuple[StartupCheckResult, None]`：执行 `check pipeline tasks` 对应逻辑。
- `F L70-L123` `check_segment_planner_contract(_settings: Settings | None=None) -> tuple[StartupCheckResult, None]`：执行 `check segment planner contract` 对应逻辑。 调用：`Circle`, `build_segment_plans`, `build_segment_video_args`。
- `F L126-L156` `check_raw_training_inputs(settings: Settings, *, matched_manifest_path: Path=DEFAULT_MATCHED_MANIFEST, run_match_probe: bool=True, min_match_score: float=0.1) -> tuple[StartupCheckResult, None]`：执行 `check raw training inputs` 对应逻辑。 调用：`inspect_before_training_samples`, `inspection.as_dict`。

## `src/before_traning/tests/startup_checks/runner.py`

职责：before_traning 启动检测统一入口；按顺序运行配置、pipeline、分段契约和原始数据扫描检测。
工程依赖：`before_traning.conf`, `before_traning.tests.startup_checks.items`, `before_traning.tests.startup_checks.samples`, `package.checks`

- `F L16-L71` `run_startup_checks(config_path: Path | None=None, *, matched_manifest_path: Path=DEFAULT_MATCHED_MANIFEST, run_match_probe: bool=True, min_match_score: float=0.1) -> StartupCheckReport`：执行 `run startup checks` 对应逻辑。 调用：`check_raw_training_inputs`, `check_settings_load`。

## `src/before_traning/tests/startup_checks/samples.py`

职责：只读扫描未匹配原始 .osz、候选视频、已导入待匹配样本和已匹配清单。
工程依赖：`before_traning.Lib.beatmap.osz`, `before_traning.conf`, `before_traning.state.status_schema`

- `F L23-L24` `_utc_now() -> str` [IO-W]：执行 `utc now` 对应逻辑。 调用：`datetime.now.replace`。
- `C L28-L48` `RawBeatmapCandidate` [CLASS]：封装 `RawBeatmapCandidate` 相关数据或行为。
- `M L37-L38` `RawBeatmapCandidate.identity(self) -> tuple[str, str, int]` [PROPERTY]：执行 `identity` 对应逻辑。
- `M L40-L48` `RawBeatmapCandidate.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L52-L64` `VideoCandidate` [CLASS]：封装 `VideoCandidate` 相关数据或行为。
- `M L58-L64` `VideoCandidate.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L68-L78` `BeforeManifestItem` [CLASS]：封装 `BeforeManifestItem` 相关数据或行为。
- `M L77-L78` `BeforeManifestItem.raw_identity(self) -> tuple[str, str | None, int | None]` [PROPERTY]：执行 `raw identity` 对应逻辑。
- `C L82-L102` `PendingImportedSample` [CLASS]：封装 `PendingImportedSample` 相关数据或行为。
- `M L91-L92` `PendingImportedSample.sample_key(self) -> str` [PROPERTY]：执行 `sample key` 对应逻辑。
- `M L94-L102` `PendingImportedSample.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L106-L165` `MatchedSample` [CLASS]：封装 `MatchedSample` 相关数据或行为。
- `M L121-L122` `MatchedSample.identity(self) -> tuple[str, str | None, int | None]` [PROPERTY]：执行 `identity` 对应逻辑。
- `M L124-L131` `MatchedSample.matches_raw_candidate(self, candidate: RawBeatmapCandidate) -> bool`：执行 `matches raw candidate` 对应逻辑。
- `M L133-L147` `MatchedSample.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `M L150-L165` `MatchedSample.from_mapping(cls, raw: Mapping[str, Any]) -> 'MatchedSample'`：执行 `from mapping` 对应逻辑。 调用：`_optional_float`, `_optional_int`, `_optional_str`, `_utc_now`, `raw.get`。
- `C L169-L220` `MatchedSampleManifest` [CLASS]：封装 `MatchedSampleManifest` 相关数据或行为。
- `M L174-L185` `MatchedSampleManifest.load(cls, path: Path | None=None) -> 'MatchedSampleManifest'` [IO-R]：执行 `load` 对应逻辑。 调用：`MatchedSample.from_mapping`, `manifest_path.read_text`, `payload.get`。
- `M L187-L203` `MatchedSampleManifest.merged(self, samples: Iterable[MatchedSample]) -> 'MatchedSampleManifest'`：执行 `merged` 对应逻辑。 调用：`MatchedSampleManifest`。
- `M L205-L217` `MatchedSampleManifest.save(self) -> None` [IO-W]：执行 `save` 对应逻辑。 调用：`_json_ready`, `_utc_now`, `sample.as_dict`, `self.path.parent.mkdir`, `self.path.with_name`, `tmp_path.replace`。
- `M L219-L220` `MatchedSampleManifest.matches_raw_candidate(self, candidate: RawBeatmapCandidate) -> bool`：执行 `matches raw candidate` 对应逻辑。 调用：`sample.matches_raw_candidate`。
- `C L224-L260` `MatchProbePair` [CLASS]：封装 `MatchProbePair` 相关数据或行为。
- `M L242-L260` `MatchProbePair.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L264-L285` `MatchProbeReport` [CLASS]：封装 `MatchProbeReport` 相关数据或行为。
- `M L272-L273` `MatchProbeReport.ok(self) -> bool` [PROPERTY]：执行 `ok` 对应逻辑。
- `M L275-L285` `MatchProbeReport.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`item.as_dict`。
- `C L289-L350` `BeforeTrainingSampleInspection` [CLASS]：封装 `BeforeTrainingSampleInspection` 相关数据或行为。
- `M L301-L302` `BeforeTrainingSampleInspection.has_unmatched_samples(self) -> bool` [PROPERTY]：执行 `has unmatched samples` 对应逻辑。
- `M L305-L306` `BeforeTrainingSampleInspection.has_video_candidates(self) -> bool` [PROPERTY]：执行 `has video candidates` 对应逻辑。
- `M L309-L314` `BeforeTrainingSampleInspection.should_run_before_traning(self) -> bool` [PROPERTY]：执行 `should run before traning` 对应逻辑。
- `M L317-L326` `BeforeTrainingSampleInspection.reason(self) -> str` [PROPERTY]：执行 `reason` 对应逻辑。
- `M L328-L350` `BeforeTrainingSampleInspection.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`item.as_dict`, `self.match_probe.as_dict`。
- `F L353-L391` `inspect_before_training_samples(settings: Settings, *, matched_manifest_path: Path | None=None, run_match_probe: bool=True, min_match_score: float=0.1) -> BeforeTrainingSampleInspection`：执行 `inspect before training samples` 对应逻辑。 调用：`BeforeTrainingSampleInspection`, `MatchedSampleManifest.load`, `_filter_unmatched_raw_candidates`, `_pending_imported_samples`, `_read_before_state`, `_recover_matched_samples`。
- `F L394-L402` `recover_matched_sample_manifest(settings: Settings, *, matched_manifest_path: Path | None=None) -> MatchedSampleManifest`：执行 `recover matched sample manifest` 对应逻辑。 调用：`MatchedSampleManifest.load`, `_read_before_state`, `_recover_matched_samples`, `manifest.merged`。
- `F L405-L428` `probe_before_training_matches(settings: Settings, *, raw_unmatched: Iterable[RawBeatmapCandidate], pending_imported: Iterable[PendingImportedSample], videos: Iterable[VideoCandidate], min_match_score: float) -> MatchProbeReport`：执行 `probe before training matches` 对应逻辑。 调用：`MatchProbeReport`, `_run_match_probe`。
- `F L431-L505` `_run_match_probe(settings: Settings, raw_unmatched: tuple[RawBeatmapCandidate, ...], pending_imported: tuple[PendingImportedSample, ...], videos: tuple[VideoCandidate, ...], min_match_score: float) -> MatchProbeReport`：执行 `run match probe` 对应逻辑。 调用：`MatchProbeReport`, `_StartupAudioAligner`, `_audio_samples_from_path`, `_build_probe_features`, `_probe_target_from_pending`, `_probe_target_from_raw`。
- `C L440-L441` `_run_match_probe._StartupAudioAligner(AVCoreStepsMixin)` [CLASS]：封装 `StartupAudioAligner` 相关数据或行为。
- `F L508-L530` `_probe_target_from_raw(settings: Settings, candidate: RawBeatmapCandidate) -> dict[str, Any]`：执行 `probe target from raw` 对应逻辑。 调用：`read_osz_entry`。
- `F L533-L545` `_probe_target_from_pending(candidate: PendingImportedSample) -> dict[str, Any]`：执行 `probe target from pending` 对应逻辑。
- `F L548-L616` `_score_probe_pair(aligner: Any, target: Mapping[str, Any], target_features: Mapping[str, Any], video: VideoCandidate, video_features: Mapping[str, Any]) -> MatchProbePair`：执行 `score probe pair` 对应逻辑。 调用：`MatchProbePair`, `_optional_float`, `_optional_int`, `_optional_str`, `aligner._estimate_best_start_frame`, `aligner._estimate_verify_adjustment_seconds`。
- `F L619-L631` `_select_greedy_probe_matches(pairs: Iterable[MatchProbePair]) -> tuple[MatchProbePair, ...]`：选择 `greedy probe matches` 对应的数据或结果。
- `F L634-L646` `_probe_sort_key(pair: MatchProbePair) -> tuple[float, float, float, float]`：执行 `probe sort key` 对应逻辑。
- `F L649-L662` `_build_probe_features(aligner: Any, samples: Any) -> dict[str, Any]`：构建 `probe features` 对应的数据或结果。 调用：`aligner._build_feature_series`, `aligner._build_music_refine_series`。
- `F L665-L676` `_target_audio_samples(aligner: Any, target: Mapping[str, Any]) -> Any` [IO-W]：执行 `target audio samples` 对应逻辑。 调用：`_audio_samples_from_path`, `target.get`。
- `F L679-L683` `_audio_samples_from_path(aligner: Any, path: Path, *, from_video: bool) -> Any`：执行 `audio samples from path` 对应逻辑。 调用：`aligner._extract_audio_to_wav`, `aligner._load_wav_samples`。
- `F L686-L719` `_scan_raw_beatmap_candidates(settings: Settings) -> tuple[list[RawBeatmapCandidate], list[str]]`：执行 `scan raw beatmap candidates` 对应逻辑。 调用：`RawBeatmapCandidate`, `read_osz_entry`。
- `F L722-L739` `_scan_video_candidates(settings: Settings) -> list[VideoCandidate]`：执行 `scan video candidates` 对应逻辑。 调用：`VideoCandidate`。
- `F L742-L761` `_filter_unmatched_raw_candidates(raw_candidates: Iterable[RawBeatmapCandidate], manifest_items: Iterable[BeforeManifestItem], known_matched: Iterable[MatchedSample]) -> tuple[RawBeatmapCandidate, ...]`：筛选 `unmatched raw candidates` 对应的数据或结果。 调用：`sample.matches_raw_candidate`。
- `F L764-L782` `_read_before_state(settings: Settings) -> tuple[tuple[BeforeManifestItem, ...], dict[tuple[str, str], dict[str, Any]], tuple[str, ...]]`：读取 `before state` 对应的数据或结果。 调用：`_read_manifest_items`, `_read_status_rows`。
- `F L785-L807` `_read_manifest_items(db_path: Path) -> tuple[BeforeManifestItem, ...]`：读取 `manifest items` 对应的数据或结果。 调用：`BeforeManifestItem`, `_optional_int`, `_optional_str`, `_sqlite_table_exists`。
- `F L810-L827` `_read_status_rows(db_path: Path) -> dict[tuple[str, str], dict[str, Any]]`：读取 `status rows` 对应的数据或结果。 调用：`_sqlite_table_exists`, `decode_detail`。
- `F L830-L835` `_sqlite_table_exists(connection: sqlite3.Connection, table_name: str) -> bool`：执行 `sqlite table exists` 对应逻辑。
- `F L838-L869` `_recover_matched_samples(settings: Settings, manifest_items: Iterable[BeforeManifestItem], status_rows: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[MatchedSample, ...]`：执行 `recover matched samples` 对应逻辑。 调用：`MatchedSample`, `_detail_float`, `_first_folder_video`, `_folder_has_video`, `_status_detail`, `_status_done`。
- `F L872-L901` `_pending_imported_samples(settings: Settings, manifest_items: Iterable[BeforeManifestItem], status_rows: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[PendingImportedSample, ...]`：执行 `pending imported samples` 对应逻辑。 调用：`PendingImportedSample`, `_folder_has_video`, `_status_done`。
- `F L904-L905` `_folder_has_video(settings: Settings, folder_name: str) -> bool`：执行 `folder has video` 对应逻辑。 调用：`_first_folder_video`。
- `F L908-L916` `_first_folder_video(settings: Settings, folder_name: str) -> Path | None`：执行 `first folder video` 对应逻辑。
- `F L919-L925` `_status_done(status_rows: Mapping[tuple[str, str], Mapping[str, Any]], folder_name: str, step: str) -> bool`：执行 `status done` 对应逻辑。 调用：`row.get`, `status_rows.get`。
- `F L928-L934` `_status_detail(status_rows: Mapping[tuple[str, str], Mapping[str, Any]], folder_name: str, step: str) -> Any`：执行 `status detail` 对应逻辑。 调用：`row.get`, `status_rows.get`。
- `F L937-L941` `_video_path_from_detail(detail: Any) -> Path | None`：执行 `video path from detail` 对应逻辑。 调用：`detail.get`。
- `F L944-L947` `_detail_float(detail: Any, key: str) -> float | None`：执行 `detail float` 对应逻辑。 调用：`_optional_float`, `detail.get`。
- `F L950-L954` `_optional_str(value: Any) -> str | None`：执行 `optional str` 对应逻辑。
- `F L957-L960` `_optional_int(value: Any) -> int | None`：执行 `optional int` 对应逻辑。
- `F L963-L969` `_optional_float(value: Any) -> float | None`：执行 `optional float` 对应逻辑。
- `F L972-L981` `_json_ready(value: Any) -> Any`：执行 `json ready` 对应逻辑。 调用：`_json_ready`。
