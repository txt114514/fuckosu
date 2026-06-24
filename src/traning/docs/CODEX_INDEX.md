# traning Codex Index

> 自动生成文件，请勿手工修改。运行 `python project_index/build_index.py` 重建。

面向 Codex 的低 token 工程导航；先按阶段定位，再读取命中的源码。

## 调用分层

```text
main.py -> core/decision/pipeline.py:TRAINING_STAGES
        -> core/dataset_import (训练集导入、preflight、Dataset/DataLoader)
        -> core/spatial (空间训练与单帧推理流程)
        -> core/temporal (候选缓存窗口与时序训练 smoke)
        -> core/decision (候选缓存与决策编排)
        -> core/optimization (评分、错误归因、参数搜索修改)
        -> core/result_export (结果可视化与图集导出)
        -> core/model_export (训练模型导出与迁移边界)
        -> lib/data | lib/models | lib/training | lib/metrics | lib/runtime | lib/visualization | lib/compat
        -> state (run / experiment / checkpoint metadata)
```

## Core 入口

| key | Core 入口 | 当前状态 |
|---|---|---|
| `dataset_import` | `core/dataset_import` | 训练集导入、检查、Dataset/DataLoader 已实现 |
| `spatial` | `core/spatial` | 空间训练和单帧推理已实现 |
| `temporal` | `core/temporal` | 候选缓存窗口和首版训练 smoke 已实现 |
| `decision` | `core/decision` | 候选缓存和训练阶段编排已实现 |
| `optimization` | `core/optimization` | 评分、归因和参数搜索修改已实现 |
| `result_export` | `core/result_export` | 结果可视化和图集导出已实现 |
| `model_export` | `core/model_export` | 训练模型导出迁移边界已建立 |

快速查询：`python project_index/build_index.py --lookup 符号名`。

## 符号索引

覆盖 `116` 个 Python 文件、`503` 个命名函数/方法、`146` 个类。匿名 lambda 不单独列出。

图例：`F` 模块函数，`M` 方法，`N` 嵌套函数，`C` 类；`IO-R/IO-W` 文件读写，`DB` 数据库，`PROCESS` 外部进程。

## `src/traning/conf/settings.py`

职责：训练配置模型与 YAML 加载；解析数据集路径、item 划分、颜色 cue、候选缓存、点击频率上限并校验采样和分块参数。

- `C L26-L27` `SettingsError(Exception)` [CLASS]：封装 `SettingsError` 相关数据或行为。
- `C L30-L32` `RuntimeSettings(BaseModel)` [CLASS]：封装 `RuntimeSettings` 相关数据或行为。
- `C L35-L46` `InputSettings(BaseModel)` [CLASS]：封装 `InputSettings` 相关数据或行为。
- `M L43-L46` `InputSettings._positive_dimension(cls, value: int) -> int` [VALIDATOR]：执行 `positive dimension` 对应逻辑。
- `C L49-L79` `TilingConfig(BaseModel)` [CLASS]：封装 `TilingConfig` 相关数据或行为。
- `M L59-L62` `TilingConfig._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `M L66-L69` `TilingConfig._nonnegative_overlap(cls, value: int) -> int` [VALIDATOR]：执行 `nonnegative overlap` 对应逻辑。
- `M L72-L79` `TilingConfig.validate_tiling(self) -> TilingConfig` [VALIDATOR]：校验 `tiling` 对应的数据或结果。
- `C L82-L98` `LocalEncoderConfig(BaseModel)` [CLASS]：封装 `LocalEncoderConfig` 相关数据或行为。
- `M L95-L98` `LocalEncoderConfig._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `C L101-L119` `GlobalEncoderConfig(BaseModel)` [CLASS]：封装 `GlobalEncoderConfig` 相关数据或行为。
- `M L116-L119` `GlobalEncoderConfig._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `C L122-L142` `FusionConfig(BaseModel)` [CLASS]：封装 `FusionConfig` 相关数据或行为。
- `M L133-L136` `FusionConfig._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `M L139-L142` `FusionConfig.validate_attention_shape(self) -> FusionConfig` [VALIDATOR]：校验 `attention shape` 对应的数据或结果。
- `C L145-L157` `TemporalConfig(BaseModel)` [CLASS]：封装 `TemporalConfig` 相关数据或行为。
- `M L154-L157` `TemporalConfig._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `C L160-L196` `MemoryConfig(BaseModel)` [CLASS]：封装 `MemoryConfig` 相关数据或行为。
- `M L179-L182` `MemoryConfig._finite_nonnegative_memory(cls, value: float) -> float` [VALIDATOR]：执行 `finite nonnegative memory` 对应逻辑。
- `M L186-L189` `MemoryConfig._positive_memory(cls, value: float) -> float` [VALIDATOR]：执行 `positive memory` 对应逻辑。
- `M L193-L196` `MemoryConfig._optional_positive_memory(cls, value: float | None) -> float | None` [VALIDATOR]：执行 `optional positive memory` 对应逻辑。
- `C L199-L200` `SMETConfig(BaseModel)` [CLASS]：封装 `SMETConfig` 相关数据或行为。
- `C L203-L247` `CandidateCacheSettings(BaseModel)` [CLASS]：封装 `CandidateCacheSettings` 相关数据或行为。
- `M L225-L228` `CandidateCacheSettings._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `M L232-L235` `CandidateCacheSettings._probability(cls, value: float) -> float` [VALIDATOR]：执行 `probability` 对应逻辑。
- `M L244-L247` `CandidateCacheSettings._nonnegative_float(cls, value: float) -> float` [VALIDATOR]：执行 `nonnegative float` 对应逻辑。
- `C L250-L286` `LoaderSettings(BaseModel)` [CLASS]：封装 `LoaderSettings` 相关数据或行为。
- `M L261-L264` `LoaderSettings._positive_batch_size(cls, value: int) -> int` [VALIDATOR]：执行 `positive batch size` 对应逻辑。
- `M L268-L271` `LoaderSettings._nonnegative_workers(cls, value: int) -> int` [VALIDATOR]：执行 `nonnegative workers` 对应逻辑。
- `M L275-L278` `LoaderSettings._optional_positive_prefetch(cls, value: int | None) -> int | None` [VALIDATOR]：执行 `optional positive prefetch` 对应逻辑。
- `M L281-L286` `LoaderSettings.validate_worker_options(self) -> LoaderSettings` [VALIDATOR]：校验 `worker options` 对应的数据或结果。
- `C L289-L297` `EvaluationSettings(BaseModel)` [CLASS]：封装 `EvaluationSettings` 相关数据或行为。
- `M L294-L297` `EvaluationSettings._nonnegative_interval(cls, value: float) -> float` [VALIDATOR]：执行 `nonnegative interval` 对应逻辑。
- `C L300-L315` `VisualizationSettings(BaseModel)` [CLASS]：封装 `VisualizationSettings` 相关数据或行为。
- `M L312-L315` `VisualizationSettings._positive_interval(cls, value: int) -> int` [VALIDATOR]：执行 `positive interval` 对应逻辑。
- `C L318-L395` `DataInputSettings(BaseModel)` [CLASS]：封装 `DataInputSettings` 相关数据或行为。
- `M L340-L343` `DataInputSettings._positive_fps(cls, value: float) -> float` [VALIDATOR]：执行 `positive fps` 对应逻辑。
- `M L347-L350` `DataInputSettings._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `M L354-L357` `DataInputSettings._optional_positive_integer(cls, value: int | None) -> int | None` [VALIDATOR]：执行 `optional positive integer` 对应逻辑。
- `M L361-L364` `DataInputSettings._nonnegative_visibility(cls, value: float) -> float` [VALIDATOR]：执行 `nonnegative visibility` 对应逻辑。
- `M L375-L381` `DataInputSettings._unique_nonempty_strings(cls, value: tuple[str, ...]) -> tuple[str, ...]` [VALIDATOR]：执行 `unique nonempty strings` 对应逻辑。
- `M L384-L389` `DataInputSettings.validate_item_splits(self) -> DataInputSettings` [VALIDATOR]：校验 `item splits` 对应的数据或结果。
- `M L391-L395` `DataInputSettings.validate_tiling(self) -> None`：校验 `tiling` 对应的数据或结果。
- `C L398-L437` `Settings(BaseSettings)` [CLASS]：封装 `Settings` 相关数据或行为。
- `M L424-L437` `Settings.settings_customise_sources(cls, settings_cls: type[BaseSettings], init_settings: PydanticBaseSettingsSource, env_settings: PydanticBaseSettingsSource, dotenv_settings: PydanticBaseSettingsSource, file_secret_settings: PydanticBaseSettingsSource) -> tuple[PydanticBaseSettingsSource, ...]`：执行 `settings customise sources` 对应逻辑。
- `F L440-L449` `_read_config(config_path: Path) -> dict[str, Any]` [IO-R]：读取 `config` 对应的数据或结果。 调用：`SettingsError`。
- `F L452-L478` `_resolve_paths(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]`：解析并定位 `paths` 对应的数据或结果。
- `F L481-L490` `load_settings(config_path: Path | None=None) -> Settings`：加载 `settings` 对应的数据或结果。 调用：`Settings`, `SettingsError`, `_read_config`, `_resolve_paths`, `settings.data_input.validate_tiling`, `settings.tiling.validate_tiling`。

## `src/traning/core/dataset_import/data_input.py`

职责：训练集导入模块公开门面；提供检查、Dataset 和 DataLoader。
工程依赖：`traning.conf`, `traning.core.dataset_import.loader`, `traning.core.dataset_import.preflight`

- `C L10-L26` `DataInputModule` [CLASS]：封装 `DataInputModule` 相关数据或行为。
- `M L11-L12` `DataInputModule.__init__(self, settings: Settings)`：初始化实例依赖、配置和运行状态。
- `M L14-L15` `DataInputModule.inspect(self, *, split: DataSplit='all') -> DataInputReport`：执行 `inspect` 对应逻辑。 调用：`inspect_data_input`。
- `M L17-L18` `DataInputModule.dataset(self, *, split: DataSplit='train')`：执行 `dataset` 对应逻辑。 调用：`build_dataset`。
- `M L20-L26` `DataInputModule.dataloader(self, *, split: DataSplit='train', shuffle: bool | None=None) -> DataLoader`：执行 `dataloader` 对应逻辑。 调用：`build_dataloader`。
- `F L29-L34` `check_data_input(settings: Settings | None=None, *, split: DataSplit='all') -> DataInputReport`：执行 `check data input` 对应逻辑。 调用：`DataInputModule`, `DataInputModule.inspect`, `load_settings`。

## `src/traning/core/dataset_import/loader.py`

职责：把配置映射为 SegmentFrameDataset 与 PyTorch DataLoader。
工程依赖：`traning.conf`, `traning.core.dataset_import.preflight`, `traning.lib.data`

- `F L10-L32` `build_dataset(settings: Settings, *, split: DataSplit='train') -> SegmentFrameDataset`：构建并返回 `dataset` 对应的数据或结果。 调用：`SegmentFrameDataset`, `discover_data_input`。
- `F L35-L56` `build_dataloader(settings: Settings, *, split: DataSplit='train', shuffle: bool | None=None) -> DataLoader`：构建并返回 `dataloader` 对应的数据或结果。 调用：`build_dataset`。

## `src/traning/core/dataset_import/preflight.py`

职责：扫描训练片段并生成数量、类别、维度和问题报告。
工程依赖：`traning.conf`, `traning.lib.data`, `traning.lib.data.models`

- `C L13-L25` `DataInputReport` [CLASS]：封装 `DataInputReport` 相关数据或行为。
- `M L24-L25` `DataInputReport.ok(self) -> bool` [PROPERTY]：执行 `ok` 对应逻辑。
- `F L28-L34` `_combine_item_filters(base_items: tuple[str, ...], split_items: tuple[str, ...]) -> tuple[str, ...]`：执行 `combine item filters` 对应逻辑。
- `F L37-L42` `_split_items(config, split: DataSplit) -> tuple[str, ...]`：执行 `split items` 对应逻辑。
- `F L45-L69` `discover_data_input(settings: Settings, *, split: DataSplit='all') -> DiscoveryResult`：执行 `discover data input` 对应逻辑。 调用：`DatasetIssue`, `DiscoveryResult`, `_combine_item_filters`, `_split_items`, `discover_segments`。
- `F L72-L101` `inspect_data_input(settings: Settings, *, split: DataSplit='all') -> DataInputReport`：执行 `inspect data input` 对应逻辑。 调用：`DataInputReport`, `discover_data_input`。

## `src/traning/core/decision/generator.py`

职责：离线空间候选缓存生成器；逐帧调用空间推理并写 JSONL/manifest/temporal target。
工程依赖：`package.coordinates`, `traning.conf`, `traning.core.dataset_import`, `traning.core.spatial`, `traning.lib.training`, `traning.lib.training.spatial_decode`

- `C L27-L51` `CandidateCacheBuildResult` [CLASS]：封装 `CandidateCacheBuildResult` 相关数据或行为。
- `M L39-L51` `CandidateCacheBuildResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `F L54-L170` `generate_candidate_cache(settings: Settings, *, output_dir: Path, device: torch.device, split: DataSplit='train', max_frames: int | None=None, patch_limit: int | None=None, max_candidates: int | None=None, score_threshold: float | None=None, nms_radius_px: float | None=None, slider_threshold: float | None=None, max_slider_paths: int | None=None, dataset: Sequence[Mapping[str, Any]] | None=None) -> CandidateCacheBuildResult` [IO-W]：执行 `generate candidate cache` 对应逻辑。 调用：`CandidateCacheBuildResult`, `build_candidate_cache_record`, `build_dataset`, `run_spatial_frame_inference`。
- `F L173-L234` `build_candidate_cache_record(sample: Mapping[str, Any], candidates: Sequence[SpatialCandidate], slider_paths: Sequence[SliderPathCandidate], *, frame_width: int, frame_height: int, device: str, patches_processed: int, frame_channels: int, save_dtype: str, low_confidence_threshold: float, close_score_margin: float, slider_attach_distance_px: float, action_window_ms: float=25.0) -> dict[str, Any]`：构建并返回 `candidate cache record` 对应的数据或结果。 调用：`_build_temporal_target`, `_candidate_ambiguity_reasons`, `_cast_embedding`, `_nearest_slider_path`, `candidate_rows.append`, `slider_path_to_dict`。
- `F L237-L276` `_build_temporal_target(sample: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], *, frame_width: int, frame_height: int, action_window_ms: float) -> dict[str, Any]`：构建 `temporal target` 对应的数据或结果。 调用：`_nearest_candidate`, `_optional_float`, `_select_temporal_object`。
- `F L279-L308` `_select_temporal_object(objects: object, *, timestamp_ms: float, action_window_ms: float) -> dict[str, Any] | None`：选择 `temporal object` 对应的数据或结果。 调用：`_temporal_target_for_object`。
- `F L311-L350` `_temporal_target_for_object(item: Mapping[str, Any], *, timestamp_ms: float, action_window_ms: float) -> dict[str, Any] | None`：执行 `temporal target for object` 对应逻辑。 调用：`_object_kind`, `_object_osu_point`, `_optional_float`。
- `F L353-L364` `_object_osu_point(item: Mapping[str, Any]) -> tuple[float, float] | None`：执行 `object osu point` 对应逻辑。 调用：`_object_kind`。
- `F L367-L373` `_object_kind(item: Mapping[str, Any]) -> str`：执行 `object kind` 对应逻辑。
- `F L376-L391` `_nearest_candidate(candidates: Sequence[Mapping[str, Any]], point: tuple[float, float]) -> Mapping[str, Any] | None`：执行 `nearest candidate` 对应逻辑。 调用：`_optional_float`, `_point_distance`。
- `F L394-L413` `_candidate_ambiguity_reasons(index: int, candidates: Sequence[SpatialCandidate], slider_path: SliderPathCandidate | None, *, low_confidence_threshold: float, close_score_margin: float) -> tuple[str, ...]`：执行 `candidate ambiguity reasons` 对应逻辑。 调用：`_has_close_neighbor`, `reasons.append`。
- `F L416-L429` `_has_close_neighbor(index: int, candidates: Sequence[SpatialCandidate], *, margin: float) -> bool`：执行 `has close neighbor` 对应逻辑。
- `F L432-L449` `_nearest_slider_path(candidate: SpatialCandidate, paths: Sequence[SliderPathCandidate], *, max_distance: float) -> SliderPathCandidate | None`：执行 `nearest slider path` 对应逻辑。 调用：`_distance_to_polyline`。
- `F L452-L463` `_distance_to_polyline(point: tuple[float, float], polyline: Sequence[tuple[float, float]]) -> float`：执行 `distance to polyline` 对应逻辑。 调用：`_point_distance`, `_point_to_segment_distance`。
- `F L466-L481` `_point_to_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float`：执行 `point to segment distance` 对应逻辑。 调用：`_point_distance`。
- `F L484-L488` `_point_distance(first: tuple[float, float], second: tuple[float, float]) -> float`：执行 `point distance` 对应逻辑。
- `F L491-L497` `_cast_embedding(values: Sequence[float], save_dtype: str) -> list[float]`：执行 `cast embedding` 对应逻辑。
- `F L500-L503` `_optional_float(value: Any) -> float | None`：执行 `optional float` 对应逻辑。

## `src/traning/core/decision/pipeline.py`

职责：声明训练阶段注册表；当前登记 dataset_import，后续扩展空间、决策、时序和导出阶段。
工程依赖：`traning.conf`, `traning.core.dataset_import`

- `C L11-L13` `TrainingStage` [CLASS]：封装 `TrainingStage` 相关数据或行为。
- `F L19-L21` `run_pipeline(settings: Settings | None=None) -> dict[str, object]`：执行 `run pipeline` 对应逻辑。 调用：`load_settings`, `stage.run`。

## `src/traning/core/decision/runner.py`

职责：加载 temporal checkpoint 和候选缓存，导出逐帧动作决策 JSONL。
工程依赖：`traning.conf`, `traning.core.temporal`, `traning.lib.models`, `traning.lib.runtime`

- `C L33-L53` `TemporalDecisionRunResult` [CLASS]：封装 `TemporalDecisionRunResult` 相关数据或行为。
- `M L43-L53` `TemporalDecisionRunResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `F L56-L137` `run_temporal_decision(settings: Settings, *, cache_dir: Path, checkpoint_path: Path, output_dir: Path, device: torch.device) -> TemporalDecisionRunResult` [IO-W]：执行 `run temporal decision` 对应逻辑。 调用：`CausalTemporalModel`, `CudaRuntimeConfig`, `TemporalCandidateWindowDataset.from_cache_dir`, `TemporalDecisionRunResult`, `_decision_row`, `_load_checkpoint`。
- `F L140-L156` `_load_checkpoint(checkpoint_path: Path, *, device: torch.device) -> Mapping[str, Any]`：加载 `checkpoint` 对应的数据或结果。 调用：`torch.load`。
- `F L159-L199` `_decision_row(window: TemporalWindow, frame_index: int, output) -> dict[str, Any]`：执行 `decision row` 对应逻辑。

## `src/traning/core/model_export/artifact.py`

职责：导出 inference/resume PyTorch artifact，写 manifest、文件 sha256 和版本信息。

- `C L18-L30` `ArtifactFile` [CLASS]：封装 `ArtifactFile` 相关数据或行为。
- `M L24-L30` `ArtifactFile.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`self.path.as_posix`。
- `C L34-L52` `ModelArtifactSpec` [CLASS]：封装 `ModelArtifactSpec` 相关数据或行为。
- `M L47-L52` `ModelArtifactSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L56-L70` `ModelArtifactResult` [CLASS]：封装 `ModelArtifactResult` 相关数据或行为。
- `M L63-L70` `ModelArtifactResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`item.as_dict`。
- `F L73-L78` `_sha256(path: Path) -> str` [IO-R IO-W]：执行 `sha256` 对应逻辑。
- `F L81-L91` `_copy_file(source: Path, destination: Path, role: str) -> ArtifactFile` [IO-W]：执行 `copy file` 对应逻辑。 调用：`ArtifactFile`, `_sha256`。
- `F L94-L101` `_copy_optional(files: list[ArtifactFile], source: Path | None, destination: Path, role: str) -> None`：执行 `copy optional` 对应逻辑。 调用：`_copy_file`, `files.append`。
- `F L104-L124` `_write_readme(path: Path, spec: ModelArtifactSpec) -> ArtifactFile` [IO-W]：写入 `readme` 对应的数据或结果。 调用：`ArtifactFile`, `_sha256`。
- `F L127-L133` `_manifest_file(item: ArtifactFile, artifact_dir: Path) -> dict[str, Any]`：执行 `manifest file` 对应逻辑。 调用：`item.as_dict`。
- `F L136-L205` `export_model_artifact(spec: ModelArtifactSpec) -> ModelArtifactResult` [IO-W]：导出 `model artifact` 对应的数据或结果。 调用：`ArtifactFile`, `ModelArtifactResult`, `_copy_file`, `_copy_optional`, `_manifest_file`, `_sha256`。
- `F L208-L224` `validate_model_artifact(manifest_path: Path) -> tuple[str, ...]` [IO-R]：校验 `model artifact` 对应的数据或结果。 调用：`_sha256`, `issues.append`。

## `src/traning/core/optimization/attribution/analyzer.py`

职责：汇总 trial 错误域、错误 tag 和 hard examples。
工程依赖：`traning.core.optimization.scoring`

- `C L15-L37` `HardExample` [CLASS]：封装 `HardExample` 相关数据或行为。
- `M L26-L37` `HardExample.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L41-L57` `AttributionSummary` [CLASS]：封装 `AttributionSummary` 相关数据或行为。
- `M L48-L57` `AttributionSummary.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`example.as_dict`。
- `F L60-L70` `_click_severity(click) -> float`：执行 `click severity` 对应逻辑。
- `F L73-L86` `_unresolved_example(sample: SampleScoreReport, target_id: str) -> HardExample`：执行 `unresolved example` 对应逻辑。 调用：`HardExample`。
- `F L89-L155` `analyze_trial_attribution(report: TrialScoreReport, *, max_hard_examples: int=32) -> AttributionSummary`：执行 `analyze trial attribution` 对应逻辑。 调用：`AttributionSummary`, `HardExample`, `_click_severity`, `_unresolved_example`, `hard_examples.append`。

## `src/traning/core/optimization/parameter_search/curriculum.py`

职责：实现连续通过门槛、子项目 gate 和课程晋级检查结果。
工程依赖：`traning.core.optimization.scoring`

- `C L12-L23` `SubprojectPassRule` [CLASS]：封装 `SubprojectPassRule` 相关数据或行为。
- `M L17-L23` `SubprojectPassRule.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L35-L55` `SubprojectGateResult` [CLASS]：封装 `SubprojectGateResult` 相关数据或行为。
- `M L43-L55` `SubprojectGateResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L59-L70` `CurriculumGateResult` [CLASS]：封装 `CurriculumGateResult` 相关数据或行为。
- `M L63-L70` `CurriculumGateResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`result.as_dict`, `self.subprojects.items`。
- `F L73-L99` `_gate_subproject(subproject: str, samples: Sequence[SampleScoreReport], rule: SubprojectPassRule) -> SubprojectGateResult`：执行 `gate subproject` 对应逻辑。 调用：`SubprojectGateResult`。
- `F L102-L123` `evaluate_curriculum_gate(samples: Sequence[SampleScoreReport], *, rules: Mapping[str, SubprojectPassRule]=DEFAULT_CURRICULUM_RULES) -> CurriculumGateResult`：执行 `evaluate curriculum gate` 对应逻辑。 调用：`CurriculumGateResult`, `_gate_subproject`, `append`。

## `src/traning/core/optimization/parameter_search/executor.py`

职责：根据优化计划创建 trial 记录、低预算训练 job、checkpoint 继承和 JSONL 记录。
工程依赖：`traning.core.optimization.attribution`, `traning.core.optimization.parameter_search.curriculum`, `traning.core.optimization.parameter_search.hard_examples`, `traning.core.optimization.parameter_search.planner`, `traning.core.optimization.scoring`, `traning.state`

- `C L34-L56` `TrainingJobSpec` [CLASS]：封装 `TrainingJobSpec` 相关数据或行为。
- `M L43-L56` `TrainingJobSpec.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`self.parameters.model_dump`。
- `C L60-L84` `OptimizationExecution` [CLASS]：封装 `OptimizationExecution` 相关数据或行为。
- `M L72-L84` `OptimizationExecution.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`self.job.as_dict`, `self.trial.model_dump`。
- `C L88-L103` `OptimizationExecutorConfig` [CLASS]：封装 `OptimizationExecutorConfig` 相关数据或行为。
- `M L97-L103` `OptimizationExecutorConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L106-L129` `JsonlTrialStore` [CLASS]：封装 `JsonlTrialStore` 相关数据或行为。
- `M L107-L108` `JsonlTrialStore.__init__(self, path: Path) -> None`：初始化实例依赖、配置和运行状态。
- `M L110-L120` `JsonlTrialStore.append(self, execution: OptimizationExecution) -> None` [IO-W]：执行 `append` 对应逻辑。 调用：`execution.as_dict`, `self.path.parent.mkdir`。
- `M L122-L129` `JsonlTrialStore.load(self) -> tuple[dict[str, Any], ...]` [IO-R]：执行 `load` 对应逻辑。 调用：`records.append`, `self.path.exists`, `self.path.read_text`, `self.path.read_text.splitlines`。
- `F L132-L158` `_apply_section_updates(base: Mapping[str, object], updates: Mapping[str, Any]) -> dict[str, object]`：应用 `section updates` 对应的数据或结果。
- `F L161-L181` `_apply_parameter_updates(parameters: TrialParameters, updates: Mapping[str, Mapping[str, Any]]) -> TrialParameters`：应用 `parameter updates` 对应的数据或结果。 调用：`TrialParameters`, `_apply_section_updates`。
- `F L184-L188` `_budget_steps(config: OptimizationExecutorConfig, rung: int) -> int`：执行 `budget steps` 对应逻辑。
- `F L191-L193` `_next_trial_id(source_trial_id: str, rung: int, stage: CurriculumStage) -> str`：执行 `next trial id` 对应逻辑。
- `F L196-L260` `execute_optimization_plan(report: TrialScoreReport, attribution: AttributionSummary, plan: OptimizationPlan, *, base_parameters: TrialParameters | None=None, parent_checkpoint_path: Path | None=None, config: OptimizationExecutorConfig=OptimizationExecutorConfig(), store: JsonlTrialStore | None=None) -> OptimizationExecution`：执行 `execute optimization plan` 对应逻辑。 调用：`JsonlTrialStore`, `OptimizationExecution`, `TrainingJobSpec`, `TrialMetadata`, `_apply_parameter_updates`, `_budget_steps`。

## `src/traning/core/optimization/parameter_search/hard_examples.py`

职责：把归因 hard examples 转换为样本采样权重计划。
工程依赖：`traning.core.optimization.attribution`

- `C L12-L23` `HardExampleSamplingPlan` [CLASS]：封装 `HardExampleSamplingPlan` 相关数据或行为。
- `M L16-L23` `HardExampleSamplingPlan.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`self.reasons.items`。
- `F L26-L55` `build_hard_example_sampling_plan(attribution: AttributionSummary, *, base_weight: float=1.0, severity_multiplier: float=1.5, max_examples: int=128) -> HardExampleSamplingPlan`：构建并返回 `hard example sampling plan` 对应的数据或结果。 调用：`HardExampleSamplingPlan`, `append`。

## `src/traning/core/optimization/parameter_search/planner.py`

职责：根据评分、归因、历史 trial 和资源指标生成下一轮参数调整计划。
工程依赖：`traning.core.optimization.attribution`, `traning.core.optimization.scoring`, `traning.state`

- `C L17-L47` `ASHAConfig` [CLASS]：封装 `ASHAConfig` 相关数据或行为。
- `M L31-L47` `ASHAConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L51-L61` `ParameterSearchConfig` [CLASS]：封装 `ParameterSearchConfig` 相关数据或行为。
- `M L57-L61` `ParameterSearchConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L65-L80` `TrialHistoryEntry` [CLASS]：封装 `TrialHistoryEntry` 相关数据或行为。
- `M L73-L80` `TrialHistoryEntry.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L84-L115` `OptimizationPlan` [CLASS]：封装 `OptimizationPlan` 相关数据或行为。
- `M L98-L115` `OptimizationPlan.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`self.parameter_updates.items`。
- `F L126-L128` `_next_stage(stage: CurriculumStage) -> CurriculumStage`：执行 `next stage` 对应逻辑。
- `F L131-L136` `_quantile(values: Sequence[float], quantile: float) -> float`：执行 `quantile` 对应逻辑。
- `F L139-L170` `_asha_action(report: TrialScoreReport, history: Sequence[TrialHistoryEntry], *, current_stage: CurriculumStage, rung: int, config: ASHAConfig) -> tuple[ASHAAction, tuple[str, ...]]`：执行 `asha action` 对应逻辑。 调用：`_quantile`, `reasons.append`。
- `F L173-L185` `_priority_domains(attribution: AttributionSummary) -> tuple[str, ...]`：执行 `priority domains` 对应逻辑。
- `F L188-L203` `_hard_example_keys(attribution: AttributionSummary, *, limit: int) -> tuple[str, ...]`：执行 `hard example keys` 对应逻辑。 调用：`keys.append`。
- `F L206-L212` `_set_update(updates: dict[str, dict[str, Any]], section: str, name: str, value: Any) -> None`：执行 `set update` 对应逻辑。
- `F L215-L239` `_apply_domain_updates(updates: dict[str, dict[str, Any]], attribution: AttributionSummary, reasons: list[str]) -> None`：应用 `domain updates` 对应的数据或结果。 调用：`_set_update`, `reasons.append`。
- `F L242-L265` `_apply_overall_updates(updates: dict[str, dict[str, Any]], report: TrialScoreReport, config: ParameterSearchConfig, reasons: list[str]) -> None`：应用 `overall updates` 对应的数据或结果。 调用：`_set_update`, `reasons.append`。
- `F L268-L315` `plan_next_trial(report: TrialScoreReport, attribution: AttributionSummary, *, history: Sequence[TrialHistoryEntry]=(), current_stage: CurriculumStage=CurriculumStage.BASIC, rung: int=0, config: ParameterSearchConfig=ParameterSearchConfig()) -> OptimizationPlan`：执行 `plan next trial` 对应逻辑。 调用：`OptimizationPlan`, `_apply_domain_updates`, `_apply_overall_updates`, `_asha_action`, `_hard_example_keys`, `_next_stage`。

## `src/traning/core/optimization/scoring/evaluator.py`

职责：按 point-slider-v2 和 click-sequence-v1 聚合 sample/trial 级质量分。
工程依赖：`traning.lib.metrics`, `traning.state`

- `C L22-L41` `TrialScoreSpec` [CLASS]：封装 `TrialScoreSpec` 相关数据或行为。
- `M L30-L41` `TrialScoreSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L45-L65` `SampleScoringInput` [CLASS]：封装 `SampleScoringInput` 相关数据或行为。
- `M L54-L65` `SampleScoringInput.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L69-L100` `SampleScoreReport` [CLASS]：封装 `SampleScoreReport` 相关数据或行为。
- `M L85-L100` `SampleScoreReport.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L104-L154` `TrialScoreReport` [CLASS]：封装 `TrialScoreReport` 相关数据或行为。
- `M L114-L115` `TrialScoreReport.target_count(self) -> int` [PROPERTY]：执行 `target count` 对应逻辑。
- `M L118-L119` `TrialScoreReport.hit_count(self) -> int` [PROPERTY]：执行 `hit count` 对应逻辑。
- `M L122-L123` `TrialScoreReport.miss_count(self) -> int` [PROPERTY]：执行 `miss count` 对应逻辑。
- `M L126-L127` `TrialScoreReport.unresolved_count(self) -> int` [PROPERTY]：执行 `unresolved count` 对应逻辑。
- `M L130-L131` `TrialScoreReport.frequency_limited_count(self) -> int` [PROPERTY]：执行 `frequency limited count` 对应逻辑。
- `M L134-L138` `TrialScoreReport.passed(self) -> bool` [PROPERTY]：执行 `passed` 对应逻辑。
- `M L140-L154` `TrialScoreReport.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`sample.as_dict`, `self.parameters.model_dump`。
- `F L157-L158` `_clamp01(value: float) -> float`：执行 `clamp01` 对应逻辑。
- `F L161-L162` `_safe_rate(count: int, total: int) -> float`：执行 `safe rate` 对应逻辑。
- `F L165-L172` `_resolved_object_score(sequence: SequenceScore, target_count: int) -> float`：执行 `resolved object score` 对应逻辑。 调用：`_clamp01`。
- `F L175-L215` `score_sample(sample: SampleScoringInput, *, spec: TrialScoreSpec=TrialScoreSpec()) -> SampleScoreReport`：执行 `score sample` 对应逻辑。 调用：`SampleScoreReport`, `_clamp01`, `_resolved_object_score`, `_safe_rate`, `score_click_sequence`。
- `F L218-L249` `score_trial(trial_id: str, samples: Sequence[SampleScoringInput], *, parameters: TrialParameters | None=None, metrics: Mapping[str, float] | None=None, spec: TrialScoreSpec=TrialScoreSpec()) -> TrialScoreReport`：执行 `score trial` 对应逻辑。 调用：`TrialParameters`, `TrialScoreReport`, `_clamp01`, `score_sample`。

## `src/traning/core/optimization/scoring/gallery.py`

职责：把 TrialScoreReport 转换为结果导出使用的 BatchGalleryRequest。
工程依赖：`traning.core.optimization.scoring.evaluator`, `traning.state`

- `F L14-L29` `_representative_click(sample: SampleScoreReport)`：执行 `representative click` 对应逻辑。
- `F L32-L62` `_frame_evaluation(sample: SampleScoreReport) -> FrameEvaluation`：执行 `frame evaluation` 对应逻辑。 调用：`FrameEvaluation`, `_representative_click`。
- `F L65-L88` `build_batch_gallery_request(report: TrialScoreReport, *, batch_id: str | None=None, random_seed: int=2026) -> BatchGalleryRequest`：Build the result-export request directly from optimization scoring。 调用：`BatchGalleryRequest`, `TrialGalleryEvaluation`, `_frame_evaluation`。

## `src/traning/core/result_export/preview.py`

职责：组装 Dataset、单帧点击标注和批次最佳参数图集。
工程依赖：`traning.conf`, `traning.core.dataset_import`, `traning.core.result_export.service`, `traning.lib.visualization`, `traning.state`

- `F L16-L38` `visualize_click_label(settings: Settings, *, segment_index: int=0, object_index: int=0, output_path: Path | None=None, show_window: bool | None=None) -> VisualizationResult`：执行 `visualize click label` 对应逻辑。 调用：`OptionalTrainingVisualizer`, `build_dataset`, `select_click_frame`, `visualizer.visualize`。
- `F L41-L55` `save_annotation_gallery(settings: Settings, request: BatchGalleryRequest, *, output_root: Path | None=None, samples_per_group: int | None=None) -> GalleryResult`：执行 `save annotation gallery` 对应逻辑。 调用：`OptionalTrainingVisualizer`, `build_dataset`, `visualizer.save_gallery`。

## `src/traning/core/result_export/service.py`

职责：可选可视化故障隔离、一次性告警和训练步频率控制。
工程依赖：`traning.conf`, `traning.lib.data`, `traning.lib.visualization`, `traning.state`

- `C L22-L187` `OptionalTrainingVisualizer` [CLASS]：Best-effort visualization that never raises into training code。
- `M L25-L30` `OptionalTrainingVisualizer.__init__(self, settings: VisualizationSettings)`：初始化实例依赖、配置和运行状态。
- `M L32-L36` `OptionalTrainingVisualizer._warning_once(self, message: str) -> str | None`：执行 `warning once` 对应逻辑。
- `M L38-L107` `OptionalTrainingVisualizer.visualize(self, sample: dict[str, Any], *, target_source_index: int | None=None, output_path: Path | None=None, force: bool=False, show_window: bool | None=None) -> VisualizationResult`：执行 `visualize` 对应逻辑。 调用：`VisualizationResult`, `allocate_output_identity`, `launch_image_window`, `render_annotated_frame`, `save_annotated_frame`, `self._default_output_path`。
- `M L109-L128` `OptionalTrainingVisualizer.maybe_visualize_step(self, sample: dict[str, Any], *, global_step: int, target_source_index: int | None=None) -> VisualizationResult`：执行 `maybe visualize step` 对应逻辑。 调用：`VisualizationResult`, `self._warning_once`, `self.visualize`。
- `M L130-L175` `OptionalTrainingVisualizer.save_gallery(self, dataset: SegmentFrameDataset, request: BatchGalleryRequest, *, output_root: Path | None=None, samples_per_group: int | None=None) -> GalleryResult`：执行 `save gallery` 对应逻辑。 调用：`GalleryResult`, `save_best_trial_gallery`, `self._warning_once`。
- `M L177-L187` `OptionalTrainingVisualizer._default_output_path(self, sample: dict[str, Any], output_identity: OutputIdentity) -> Path` [IO-W]：执行 `default output path` 对应逻辑。

## `src/traning/core/spatial/spatial_inference.py`

职责：单帧空间推理处理器；显式分离 GPU 前向与 CPU 画布融合、候选解码和输出缓存。
工程依赖：`traning.conf`, `traning.lib.data`, `traning.lib.models`, `traning.lib.runtime`, `traning.lib.training.spatial_decode`

- `C L49-L80` `SpatialFrameInferenceResult` [CLASS]：封装 `SpatialFrameInferenceResult` 相关数据或行为。
- `M L60-L80` `SpatialFrameInferenceResult.as_summary(self) -> dict[str, Any]`：执行 `as summary` 对应逻辑。
- `F L83-L196` `run_spatial_frame_inference(settings: Settings, sample: Mapping[str, Any], *, device: torch.device, max_candidates: int=16, score_threshold: float=0.0, nms_radius_px: float=32.0, slider_threshold: float=0.5, max_slider_paths: int=16, slider_min_cells: int=4, slider_path_points: int=32, patch_limit: int | None=None) -> SpatialFrameInferenceResult`：Run one-frame spatial inference with explicit GPU/CPU work separation。 调用：`CudaRuntimeConfig`, `PatchStream`, `SpatialFrameInferenceResult`, `SpatialPredictionCanvas`, `_model_frame`, `autocast_context`。
- `F L199-L213` `spatial_candidate_to_dict(candidate: SpatialCandidate) -> dict[str, Any]`：执行 `spatial candidate to dict` 对应逻辑。
- `F L216-L230` `slider_path_to_dict(path: SliderPathCandidate) -> dict[str, Any]`：执行 `slider path to dict` 对应逻辑。
- `F L233-L239` `_model_frame(image: torch.Tensor, *, settings: Settings) -> torch.Tensor`：执行 `model frame` 对应逻辑。 调用：`append_color_cues`。

## `src/traning/core/spatial/spatial_trainer.py`

职责：首版单帧空间训练循环；冻结 global、串行 patch 前向和逐 patch backward。
工程依赖：`traning.conf`, `traning.core.dataset_import`, `traning.lib.data`, `traning.lib.models`, `traning.lib.runtime`, `traning.lib.training.losses`, `traning.lib.training.spatial_targets`

- `C L30-L62` `SpatialTrainingResult` [CLASS]：封装 `SpatialTrainingResult` 相关数据或行为。
- `M L46-L62` `SpatialTrainingResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `F L65-L232` `run_spatial_training(settings: Settings, *, device: torch.device, run_dir: Path, split: DataSplit='train', max_steps: int=1, learning_rate: float=0.0001, patch_limit: int | None=None, dataset: Sequence[dict[str, Any]] | None=None) -> SpatialTrainingResult` [IO-W]：Run the first-version single-frame spatial training loop。 调用：`CudaRuntimeConfig`, `PatchStream`, `SpatialTrainingResult`, `_normalize_frame`, `_write_summary`, `append_color_cues`。
- `F L235-L240` `_normalize_frame(frame: torch.Tensor) -> torch.Tensor`：规范化 `frame` 对应的数据或结果。
- `F L243-L249` `_write_summary(result: SpatialTrainingResult) -> None` [IO-W]：写入 `summary` 对应的数据或结果。 调用：`result.as_dict`。

## `src/traning/core/temporal/dataset.py`

职责：读取候选缓存 JSONL，按 sample_key 生成固定长度时序窗口、mask 和动作监督。
工程依赖：`traning.core.decision`, `traning.lib.models`

- `C L25-L44` `TemporalFeatureSpec` [CLASS]：封装 `TemporalFeatureSpec` 相关数据或行为。
- `M L30-L36` `TemporalFeatureSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L39-L40` `TemporalFeatureSpec.candidate_feature_dim(self) -> int` [PROPERTY]：执行 `candidate feature dim` 对应逻辑。
- `M L43-L44` `TemporalFeatureSpec.frame_feature_dim(self) -> int` [PROPERTY]：执行 `frame feature dim` 对应逻辑。
- `C L48-L61` `TemporalWindow` [CLASS]：封装 `TemporalWindow` 相关数据或行为。
- `C L64-L145` `TemporalCandidateWindowDataset(Dataset[TemporalWindow])` [CLASS]：封装 `TemporalCandidateWindowDataset` 相关数据或行为。
- `M L65-L87` `TemporalCandidateWindowDataset.__init__(self, records: Sequence[Mapping[str, Any]], *, sequence_length: int, feature_spec: TemporalFeatureSpec, stride: int | None=None, drop_short: bool=False) -> None`：初始化实例依赖、配置和运行状态。 调用：`self._build_windows`。
- `M L90-L114` `TemporalCandidateWindowDataset.from_cache_dir(cls, cache_dir: Path, *, sequence_length: int, candidate_slots: int, embedding_dim: int | None=None, stride: int | None=None, drop_short: bool=False) -> TemporalCandidateWindowDataset`：执行 `from cache dir` 对应逻辑。 调用：`TemporalFeatureSpec`, `_infer_embedding_dim`, `load_candidate_cache_records`。
- `M L116-L117` `TemporalCandidateWindowDataset.__len__(self) -> int`：执行 `len` 对应逻辑。
- `M L119-L120` `TemporalCandidateWindowDataset.__getitem__(self, index: int) -> TemporalWindow`：执行 `getitem` 对应逻辑。
- `M L122-L145` `TemporalCandidateWindowDataset._build_windows(self, *, records: Sequence[Mapping[str, Any]], drop_short: bool) -> list[TemporalWindow]`：构建 `windows` 对应的数据或结果。 调用：`_encode_window`, `_group_records_by_sample`, `windows.append`。
- `F L148-L178` `load_candidate_cache_records(cache_dir: Path) -> tuple[dict[str, Any], ...]` [IO-R]：加载 `candidate cache records` 对应的数据或结果。 调用：`records.append`。
- `F L181-L196` `_group_records_by_sample(records: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]`：执行 `group records by sample` 对应逻辑。 调用：`_optional_string`, `current.append`, `groups.append`。
- `F L199-L207` `_record_sort_key(record: Mapping[str, Any]) -> tuple[str, int, float]`：执行 `record sort key` 对应逻辑。 调用：`_optional_float`, `_optional_int`, `_optional_string`。
- `F L210-L311` `_encode_window(records: Sequence[Mapping[str, Any]], *, sequence_length: int, spec: TemporalFeatureSpec) -> TemporalWindow`：执行 `encode window` 对应逻辑。 调用：`TemporalWindow`, `_action_id_from_target`, `_encode_candidate`, `_optional_float`, `_optional_int`, `_optional_string`。
- `F L314-L319` `_action_id_from_target(target: Mapping[str, Any]) -> int`：执行 `action id from target` 对应逻辑。
- `F L322-L332` `_selected_candidate_slot(target: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> int`：执行 `selected candidate slot` 对应逻辑。 调用：`_optional_int`。
- `F L335-L363` `_encode_candidate(candidate: Mapping[str, Any], *, record: Mapping[str, Any], spec: TemporalFeatureSpec) -> torch.Tensor`：执行 `encode candidate` 对应逻辑。 调用：`_candidate_embedding`, `_float_field`, `_optional_float`, `_optional_string`。
- `F L366-L376` `_sorted_candidates(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]`：执行 `sorted candidates` 对应逻辑。 调用：`_float_field`。
- `F L379-L387` `_candidate_embedding(candidate: Mapping[str, Any], embedding_dim: int) -> list[float]`：执行 `candidate embedding` 对应逻辑。
- `F L390-L396` `_infer_embedding_dim(records: Sequence[Mapping[str, Any]]) -> int`：执行 `infer embedding dim` 对应逻辑。 调用：`_sorted_candidates`。
- `F L399-L403` `_float_field(candidate: Mapping[str, Any], key: str) -> float`：执行 `float field` 对应逻辑。
- `F L406-L407` `_optional_string(value: Any) -> str | None`：执行 `optional string` 对应逻辑。
- `F L410-L411` `_optional_int(value: Any) -> int | None`：执行 `optional int` 对应逻辑。
- `F L414-L417` `_optional_float(value: Any) -> float | None`：执行 `optional float` 对应逻辑。

## `src/traning/core/temporal/trainer.py`

职责：因果 GRU 时序训练入口；消费候选窗口并写 summary/checkpoint。
工程依赖：`traning.conf`, `traning.core.temporal.dataset`, `traning.lib.models`, `traning.lib.runtime`

- `C L34-L70` `TemporalTrainingResult` [CLASS]：封装 `TemporalTrainingResult` 相关数据或行为。
- `M L52-L70` `TemporalTrainingResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `F L73-L193` `run_temporal_training(settings: Settings, *, cache_dir: Path, device: torch.device, run_dir: Path, max_steps: int=1, learning_rate: float=0.0001, sequence_length: int | None=None, candidate_slots: int | None=None, dataset: Sequence[TemporalWindow] | None=None) -> TemporalTrainingResult`：执行 `run temporal training` 对应逻辑。 调用：`CausalTemporalModel`, `CudaRuntimeConfig`, `TemporalCandidateWindowDataset.from_cache_dir`, `TemporalTrainingResult`, `_compute_temporal_loss`, `_window_to_device`。
- `F L196-L215` `_window_to_device(window: TemporalWindow, *, device: torch.device) -> dict[str, torch.Tensor]`：执行 `window to device` 对应逻辑。 调用：`tensor_to_device`。
- `F L218-L264` `_compute_temporal_loss(outputs, *, action_target: torch.Tensor, selected_candidate_target: torch.Tensor, xy_target: torch.Tensor, time_offset_target: torch.Tensor, frame_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]`：执行 `compute temporal loss` 对应逻辑。
- `F L267-L272` `_write_summary(result: TemporalTrainingResult) -> None` [IO-W]：写入 `summary` 对应的数据或结果。 调用：`result.as_dict`。
- `F L275-L299` `_write_checkpoint(result: TemporalTrainingResult, *, model: torch.nn.Module, hidden_size: int, layers: int) -> None` [IO-W]：写入 `checkpoint` 对应的数据或结果。

## `src/traning/lib/compat/legacy_imports.py`

职责：集中登记旧训练模块路径到新 lib/core/environment 路径的兼容别名。

- `F L111-L119` `install_legacy_training_aliases(aliases: Mapping[str, str]=LEGACY_MODULE_ALIASES) -> None`：执行 `install legacy training aliases` 对应逻辑。 调用：`_bind_parent_attribute`。
- `F L122-L128` `_bind_parent_attribute(module_name: str, module: object) -> None`：执行 `bind parent attribute` 对应逻辑。

## `src/traning/lib/data/annotation.py`

职责：beatmap.json 的 Pydantic 契约和按帧可见 HitObject 筛选。

- `C L10-L30` `HitObjectAnnotation(BaseModel)` [CLASS]：封装 `HitObjectAnnotation` 相关数据或行为。
- `M L26-L30` `HitObjectAnnotation._valid_end(cls, value: int, info: Any) -> int` [VALIDATOR]：执行 `valid end` 对应逻辑。
- `C L33-L37` `DifficultyAnnotation(BaseModel)` [CLASS]：封装 `DifficultyAnnotation` 相关数据或行为。
- `C L40-L54` `SourceAnnotation(BaseModel)` [CLASS]：封装 `SourceAnnotation` 相关数据或行为。
- `M L50-L54` `SourceAnnotation._valid_clip_end(cls, value: int, info: Any) -> int` [VALIDATOR]：执行 `valid clip end` 对应逻辑。
- `C L57-L70` `SegmentAnnotation(BaseModel)` [CLASS]：封装 `SegmentAnnotation` 相关数据或行为。
- `M L69-L70` `SegmentAnnotation.duration_ms(self) -> int` [PROPERTY]：执行 `duration ms` 对应逻辑。
- `F L73-L78` `load_annotation(path: Path) -> SegmentAnnotation` [IO-R]：加载 `annotation` 对应的数据或结果。
- `F L81-L94` `visible_hit_objects(annotation: SegmentAnnotation, timestamp_ms: float, *, visibility_post_ms: float) -> tuple[HitObjectAnnotation, ...]`：执行 `visible hit objects` 对应逻辑。

## `src/traning/lib/data/collate.py`

职责：组装图像批次并保留可变长度样本元数据。

- `F L8-L21` `collate_frame_samples(samples: list[dict[str, Any]]) -> dict[str, Any]`：执行 `collate frame samples` 对应逻辑。

## `src/traning/lib/data/color_cues.py`

职责：从 RGB 帧派生 osu 色号、白色数字/内纹和目标相关边缘输入 cue。

- `F L23-L28` `color_cue_channel_count(mode: ColorCueMode) -> int`：执行 `color cue channel count` 对应逻辑。
- `F L31-L37` `append_color_cues(frame: torch.Tensor, *, mode: ColorCueMode) -> torch.Tensor`：Append deterministic osu! color/number cues to a normalized CHW RGB frame。 调用：`extract_osu_basic_color_cues`。
- `F L40-L57` `extract_osu_basic_color_cues(frame: torch.Tensor) -> torch.Tensor`：Return palette, white-glyph and object-edge cue maps for one CHW RGB frame。 调用：`_object_edge_response`, `_palette_response`, `_white_glyph_response`。
- `F L60-L78` `_palette_response(rgb: torch.Tensor, *, saturation: torch.Tensor, value: torch.Tensor) -> torch.Tensor`：执行 `palette response` 对应逻辑。
- `F L81-L88` `_white_glyph_response(*, saturation: torch.Tensor, value: torch.Tensor) -> torch.Tensor`：执行 `white glyph response` 对应逻辑。
- `F L91-L115` `_object_edge_response(rgb: torch.Tensor, *, object_prior: torch.Tensor) -> torch.Tensor`：执行 `object edge response` 对应逻辑。

## `src/traning/lib/data/coordinates.py`

职责：Patch local/global 与 image/feature-grid 坐标转换辅助函数。
工程依赖：`traning.lib.data.patch_stream`

- `F L6-L9` `local_to_global(meta: PatchMeta, x: float, y: float) -> tuple[float, float]`：Convert patch-local image coordinates to full-frame image coordinates。
- `F L12-L15` `global_to_local(meta: PatchMeta, x: float, y: float) -> tuple[float, float]`：Convert full-frame image coordinates to patch-local image coordinates。
- `F L18-L29` `global_to_patch_indices(metas: tuple[PatchMeta, ...], x: float, y: float) -> tuple[int, ...]`：Return patch indices whose valid image area contains a full-frame point。
- `F L32-L42` `image_to_feature_grid(x: float, y: float, *, stride: int) -> tuple[float, float]`：Map image-pixel coordinates to a stride-based feature grid。
- `F L45-L55` `feature_grid_to_image(gx: float, gy: float, *, stride: int) -> tuple[float, float]`：Map stride-based feature-grid coordinates back to image pixels。

## `src/traning/lib/data/dataset.py`

职责：按片段帧索引解码原分辨率 RGB Tensor、可变长度标签和 difficulty 派生参数。
工程依赖：`traning.lib.data.annotation`, `traning.lib.data.models`, `traning.lib.data.sampling`, `traning.lib.data.video_reader`

- `C L14-L87` `SegmentFrameDataset(Dataset[dict[str, Any]])` [CLASS]：封装 `SegmentFrameDataset` 相关数据或行为。
- `M L15-L36` `SegmentFrameDataset.__init__(self, records: tuple[SegmentRecord, ...], *, sample_fps: float, frame_step: int=1, max_frames_per_segment: int | None=None, visibility_post_ms: float=100.0, normalize_images: bool=True)`：初始化实例依赖、配置和运行状态。 调用：`build_frame_references`。
- `M L38-L39` `SegmentFrameDataset.__len__(self) -> int`：执行 `len` 对应逻辑。
- `M L41-L44` `SegmentFrameDataset._video_reader(self) -> VideoReader`：执行 `video reader` 对应逻辑。 调用：`VideoReader`。
- `M L46-L82` `SegmentFrameDataset.__getitem__(self, index: int) -> dict[str, Any]`：执行 `getitem` 对应逻辑。 调用：`self._video_reader`, `self._video_reader.read_frame_at`, `visible_hit_objects`。
- `M L84-L87` `SegmentFrameDataset.__getstate__(self) -> dict[str, Any]`：执行 `getstate` 对应逻辑。

## `src/traning/lib/data/discovery.py`

职责：发现 video.mp4 与 beatmap.json 配对并构建稳定片段记录。
工程依赖：`traning.lib.data.annotation`, `traning.lib.data.models`

- `F L9-L71` `discover_segments(dataset_root: Path, *, dimensions: tuple[str, ...]=(), categories: tuple[str, ...]=(), include_items: tuple[str, ...]=(), exclude_items: tuple[str, ...]=(), max_segments: int | None=None) -> DiscoveryResult`：执行 `discover segments` 对应逻辑。 调用：`DatasetIssue`, `DiscoveryResult`, `SegmentRecord`, `issues.append`, `load_annotation`, `records.append`。

## `src/traning/lib/data/models.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data.annotation`

- `C L10-L18` `SegmentRecord` [CLASS]：封装 `SegmentRecord` 相关数据或行为。
- `C L22-L24` `DatasetIssue` [CLASS]：封装 `DatasetIssue` 相关数据或行为。
- `C L28-L30` `DiscoveryResult` [CLASS]：封装 `DiscoveryResult` 相关数据或行为。
- `C L34-L37` `FrameReference` [CLASS]：封装 `FrameReference` 相关数据或行为。

## `src/traning/lib/data/patch_stream.py`

职责：基于现有 tiling 窗口生成固定尺寸 CHW patch、padding 和含 padded 尺寸的 PatchMeta 元数据。
工程依赖：`traning.lib.data.tiling`

- `C L13-L46` `PatchMeta` [CLASS]：Full-frame coordinates for one CHW patch。
- `M L33-L34` `PatchMeta.width(self) -> int` [PROPERTY]：执行 `width` 对应逻辑。
- `M L37-L38` `PatchMeta.height(self) -> int` [PROPERTY]：执行 `height` 对应逻辑。
- `M L41-L42` `PatchMeta.padded_width(self) -> int` [PROPERTY]：执行 `padded width` 对应逻辑。
- `M L45-L46` `PatchMeta.padded_height(self) -> int` [PROPERTY]：执行 `padded height` 对应逻辑。
- `C L49-L176` `PatchStream` [CLASS]：Generate padded CHW patches on CPU without invoking model code。
- `M L52-L73` `PatchStream.__init__(self, *, patch_width: int=512, patch_height: int=512, overlap_x: int=128, overlap_y: int=128, pin_memory: bool=False, padding_value: float=0.0) -> None`：初始化实例依赖、配置和运行状态。
- `M L75-L105` `PatchStream.metas(self, *, frame_width: int, frame_height: int) -> tuple[PatchMeta, ...]`：Return deterministic patch metadata covering the full frame。 调用：`PatchMeta`, `build_patch_windows`, `self._validate_coverage`。
- `M L107-L111` `PatchStream.count(self, frame: torch.Tensor) -> int`：Return the number of patches that ``iter_patches`` would emit。 调用：`self._shape`, `self.metas`。
- `M L113-L142` `PatchStream.iter_patches(self, frame: torch.Tensor) -> Iterator[tuple[torch.Tensor, PatchMeta]]`：Yield ``(patch, meta)`` pairs from a CHW image tensor。 调用：`self._shape`, `self.metas`。
- `M L144-L149` `PatchStream.to_device(self, patch: torch.Tensor, device: torch.device | str) -> torch.Tensor`：Move a patch to a device using non-blocking transfer when possible。
- `M L152-L158` `PatchStream._shape(frame: torch.Tensor) -> tuple[int, int, int]`：执行 `shape` 对应逻辑。
- `M L161-L176` `PatchStream._validate_coverage(metas: tuple[PatchMeta, ...], *, frame_width: int, frame_height: int) -> None`：校验 `coverage` 对应的数据或结果。

## `src/traning/lib/data/sampling.py`

职责：根据片段时长、FPS 和步长建立帧引用表。
工程依赖：`traning.lib.data.models`

- `F L8-L31` `build_frame_references(records: tuple[SegmentRecord, ...], *, sample_fps: float, frame_step: int, max_frames_per_segment: int | None) -> tuple[FrameReference, ...]`：构建并返回 `frame references` 对应的数据或结果。 调用：`FrameReference`。

## `src/traning/lib/data/synthetic_structures.py`

职责：生成跨 patch 圆环、边界圆、slider、spinner 和噪声合成测试图像。

- `C L9-L16` `SyntheticStructure` [CLASS]：Small synthetic image bundle for model and fusion smoke tests。
- `F L19-L24` `_coordinate_grid(width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]`：执行 `coordinate grid` 对应逻辑。
- `F L27-L28` `_image_from_mask(mask: torch.Tensor, *, channels: int=3) -> torch.Tensor`：执行 `image from mask` 对应逻辑。
- `F L31-L49` `make_cross_patch_ring(*, width: int=768, height: int=768, center: tuple[float, float]=(384.0, 384.0), radius: float=210.0, thickness: float=8.0) -> SyntheticStructure`：Create a ring whose circumference crosses four 512px patches。 调用：`SyntheticStructure`, `_coordinate_grid`, `_image_from_mask`。
- `F L52-L68` `make_boundary_circle(*, width: int=768, height: int=512, center: tuple[float, float]=(512.0, 256.0), radius: float=48.0) -> SyntheticStructure`：Create a filled circle centered on a typical patch boundary。 调用：`SyntheticStructure`, `_coordinate_grid`, `_image_from_mask`。
- `F L71-L97` `make_cross_patch_slider(*, width: int=1152, height: int=512, start: tuple[float, float]=(120.0, 256.0), end: tuple[float, float]=(1032.0, 256.0), thickness: float=12.0) -> SyntheticStructure`：Create a long straight slider spanning multiple patch windows。 调用：`SyntheticStructure`, `_coordinate_grid`, `_image_from_mask`。
- `F L100-L114` `make_spinner(*, width: int=768, height: int=768, center: tuple[float, float]=(384.0, 384.0), radius: float=260.0) -> SyntheticStructure`：Create a large spinner-like disk with a bright rim。 调用：`SyntheticStructure`, `_coordinate_grid`。
- `F L117-L128` `make_noise_background(*, width: int=512, height: int=512, seed: int=2026) -> SyntheticStructure`：Create deterministic noise for background robustness smoke tests。 调用：`SyntheticStructure`。

## `src/traning/lib/data/tiling.py`

职责：构建覆盖完整画面的重叠 patch 窗口并返回 Tensor 视图。

- `C L10-L22` `PatchWindow` [CLASS]：封装 `PatchWindow` 相关数据或行为。
- `M L17-L18` `PatchWindow.right(self) -> int` [PROPERTY]：执行 `right` 对应逻辑。
- `M L21-L22` `PatchWindow.bottom(self) -> int` [PROPERTY]：执行 `bottom` 对应逻辑。
- `F L25-L38` `_axis_starts(size: int, patch_size: int, overlap: int) -> tuple[int, ...]`：执行 `axis starts` 对应逻辑。 调用：`starts.append`。
- `F L41-L61` `build_patch_windows(image_width: int, image_height: int, *, patch_width: int, patch_height: int, overlap_x: int, overlap_y: int) -> tuple[PatchWindow, ...]`：构建并返回 `patch windows` 对应的数据或结果。 调用：`PatchWindow`, `_axis_starts`。
- `F L64-L78` `iter_patches(image: Tensor, windows: tuple[PatchWindow, ...]) -> Iterator[tuple[PatchWindow, Tensor]]`：执行 `iter patches` 对应逻辑。

## `src/traning/lib/data/video_reader.py`

职责：带有限打开文件缓存的 OpenCV 视频帧读取器。

- `C L10-L56` `VideoReader` [CLASS]：封装 `VideoReader` 相关数据或行为。
- `M L11-L15` `VideoReader.__init__(self, max_open_videos: int=4)`：初始化实例依赖、配置和运行状态。
- `M L17-L28` `VideoReader._capture(self, path: Path) -> cv2.VideoCapture`：执行 `capture` 对应逻辑。 调用：`self._captures.pop`, `self._captures.popitem`。
- `M L30-L36` `VideoReader.read_frame(self, path: Path, frame_index: int) -> np.ndarray` [IO-R]：读取 `frame` 对应的数据或结果。 调用：`self._capture`。
- `M L38-L48` `VideoReader.read_frame_at(self, path: Path, timestamp_ms: float) -> np.ndarray` [IO-R]：读取 `frame at` 对应的数据或结果。 调用：`self._capture`。
- `M L50-L53` `VideoReader.close(self) -> None`：执行 `close` 对应逻辑。 调用：`self._captures.clear`, `self._captures.values`。
- `M L55-L56` `VideoReader.__del__(self) -> None`：执行 `del` 对应逻辑。 调用：`self.close`。

## `src/traning/lib/metrics/scoring.py`

职责：实现点与 slider 的空间、时间、1.5x 膨胀路径覆盖和组合评分。

- `C L13-L65` `ScoreSpec` [CLASS]：封装 `ScoreSpec` 相关数据或行为。
- `M L29-L56` `ScoreSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L59-L60` `ScoreSpec.maximum_coefficient(self) -> float` [PROPERTY]：执行 `maximum coefficient` 对应逻辑。
- `M L63-L65` `ScoreSpec.maximum_raw_score(self) -> float` [PROPERTY]：执行 `maximum raw score` 对应逻辑。
- `C L69-L73` `CombinedScore` [CLASS]：封装 `CombinedScore` 相关数据或行为。
- `C L77-L82` `PointScore` [CLASS]：封装 `PointScore` 相关数据或行为。
- `C L86-L93` `PathScore` [CLASS]：封装 `PathScore` 相关数据或行为。
- `C L97-L101` `SliderScore` [CLASS]：封装 `SliderScore` 相关数据或行为。
- `F L104-L112` `_interpolate(value: float, start: float, end: float, start_score: float, end_score: float) -> float`：执行 `interpolate` 对应逻辑。
- `F L115-L137` `spatial_coefficient(distance_ratio: float, *, spec: ScoreSpec=ScoreSpec()) -> float`：执行 `spatial coefficient` 对应逻辑。
- `F L140-L182` `temporal_coefficient(time_error_ms: float, *, spec: ScoreSpec=ScoreSpec()) -> float`：执行 `temporal coefficient` 对应逻辑。 调用：`_interpolate`。
- `F L185-L201` `combine_coefficients(spatial: float, temporal: float, *, spec: ScoreSpec=ScoreSpec()) -> CombinedScore`：执行 `combine coefficients` 对应逻辑。 调用：`CombinedScore`。
- `F L204-L235` `score_point(reference_xy: Point, predicted_xy: Point, *, circle_radius: float, reference_time_ms: float, predicted_time_ms: float, spec: ScoreSpec=ScoreSpec()) -> PointScore`：执行 `score point` 对应逻辑。 调用：`PointScore`, `combine_coefficients`, `spatial_coefficient`, `temporal_coefficient`。
- `F L238-L257` `_point_to_segment_distance(point: Point, start: Point, end: Point) -> float`：执行 `point to segment distance` 对应逻辑。
- `F L260-L266` `_minimum_distance(point: Point, path: PathPoints) -> float`：执行 `minimum distance` 对应逻辑。 调用：`_point_to_segment_distance`。
- `F L269-L283` `_densify_path(path: PathPoints, *, maximum_step: float) -> PathPoints`：执行 `densify path` 对应逻辑。
- `F L286-L297` `_directed_path_statistics(source: PathPoints, target: PathPoints, *, distance_limit: float) -> tuple[float, float]`：Measure source centerline samples inside the dilated target corridor。 调用：`_minimum_distance`。
- `F L300-L369` `score_slider_path(reference_path: PathPoints, predicted_path: PathPoints, *, circle_radius: float, spec: ScoreSpec=ScoreSpec()) -> PathScore`：执行 `score slider path` 对应逻辑。 调用：`PathScore`, `_densify_path`, `_directed_path_statistics`。
- `F L372-L420` `score_slider(reference_head_xy: Point | None, predicted_head_xy: Point | None, reference_path: PathPoints, predicted_path: PathPoints, *, circle_radius: float, reference_start_ms: float, predicted_start_ms: float, spec: ScoreSpec=ScoreSpec()) -> SliderScore`：执行 `score slider` 对应逻辑。 调用：`SliderScore`, `combine_coefficients`, `score_point`, `score_slider_path`。

## `src/traning/lib/metrics/sequence.py`

职责：按点击时间模拟目标一次性命中、重叠目标递进、最小点击间隔限制和错误归因。
工程依赖：`traning.lib.metrics.scoring`

- `C L34-L40` `SequenceScoreSpec` [CLASS]：封装 `SequenceScoreSpec` 相关数据或行为。
- `M L38-L40` `SequenceScoreSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L44-L64` `TargetObject` [CLASS]：封装 `TargetObject` 相关数据或行为。
- `M L54-L64` `TargetObject.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L68-L76` `PredictedClick` [CLASS]：封装 `PredictedClick` 相关数据或行为。
- `M L74-L76` `PredictedClick.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L80-L85` `TargetResolution` [CLASS]：封装 `TargetResolution` 相关数据或行为。
- `C L89-L103` `ClickEvaluation` [CLASS]：封装 `ClickEvaluation` 相关数据或行为。
- `M L102-L103` `ClickEvaluation.frequency_limited(self) -> bool` [PROPERTY]：执行 `frequency limited` 对应逻辑。
- `C L107-L122` `SequenceScore` [CLASS]：封装 `SequenceScore` 相关数据或行为。
- `M L113-L114` `SequenceScore.hit_count(self) -> int` [PROPERTY]：执行 `hit count` 对应逻辑。
- `M L117-L118` `SequenceScore.miss_count(self) -> int` [PROPERTY]：执行 `miss count` 对应逻辑。
- `M L121-L122` `SequenceScore.frequency_limited_count(self) -> int` [PROPERTY]：执行 `frequency limited count` 对应逻辑。
- `F L125-L131` `_target_sort_key(target: TargetObject) -> tuple[float, int, str]`：执行 `target sort key` 对应逻辑。
- `F L134-L161` `_score_target(target: TargetObject, click: PredictedClick, *, circle_radius: float, spec: ScoreSpec) -> PointScore | SliderScore`：执行 `score target` 对应逻辑。 调用：`score_point`, `score_slider`。
- `F L164-L165` `_score_value(score: PointScore | SliderScore) -> float`：执行 `score value` 对应逻辑。
- `F L168-L171` `_spatial_passed(score: PointScore | SliderScore, spec: ScoreSpec) -> bool`：执行 `spatial passed` 对应逻辑。
- `F L174-L176` `_temporal_passed(score: PointScore | SliderScore, spec: ScoreSpec) -> bool`：执行 `temporal passed` 对应逻辑。
- `F L179-L182` `_spatial_error(score: PointScore | SliderScore) -> float`：执行 `spatial error` 对应逻辑。
- `F L185-L189` `_temporal_error_ms(target: TargetObject, click: PredictedClick) -> float`：执行 `temporal error ms` 对应逻辑。
- `F L192-L199` `_spatial_excess(score: PointScore | SliderScore, spec: ScoreSpec) -> float`：执行 `spatial excess` 对应逻辑。
- `F L202-L209` `_temporal_excess(score: PointScore | SliderScore, spec: ScoreSpec) -> float`：执行 `temporal excess` 对应逻辑。
- `F L212-L249` `_error_attribution(target: TargetObject, click: PredictedClick, score: PointScore | SliderScore, *, spec: ScoreSpec) -> tuple[ErrorDomain, tuple[ErrorTag, ...], float, float]`：执行 `error attribution` 对应逻辑。 调用：`_spatial_error`, `_spatial_excess`, `_spatial_passed`, `_temporal_error_ms`, `_temporal_excess`, `_temporal_passed`。
- `F L252-L273` `_best_scored_target(targets: tuple[TargetObject, ...], click: PredictedClick, *, circle_radius: float, spec: ScoreSpec) -> tuple[TargetObject, PointScore | SliderScore] | None`：执行 `best scored target` 对应逻辑。 调用：`_score_target`, `_score_value`。
- `F L276-L430` `score_click_sequence(targets: tuple[TargetObject, ...], clicks: tuple[PredictedClick, ...], *, circle_radius: float, spec: SequenceScoreSpec=SequenceScoreSpec()) -> SequenceScore`：执行 `score click sequence` 对应逻辑。 调用：`ClickEvaluation`, `SequenceScore`, `TargetResolution`, `_best_scored_target`, `_error_attribution`, `_score_target`。

## `src/traning/lib/models/gated_sparse_fusion.py`

职责：纯 PyTorch grid_sample 全局门控注入与稀疏跨区域采样融合。
工程依赖：`traning.lib.data`, `traning.lib.models.local_encoder`

- `C L14-L17` `FusedPatchFeatures` [CLASS]：封装 `FusedPatchFeatures` 相关数据或行为。
- `F L20-L49` `_base_grid(meta: PatchMeta, *, height: int, width: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor`：执行 `base grid` 对应逻辑。
- `F L52-L76` `sample_global_feature(global_feature: torch.Tensor, patch_meta: PatchMeta, local_feature_shape: tuple[int, int]) -> torch.Tensor`：Sample full-frame global features at one patch feature-grid alignment。 调用：`_base_grid`。
- `C L79-L221` `GatedSparseFusion(nn.Module)` [CLASS]：Fuse local patch features with sparse low-resolution global context。
- `M L82-L127` `GatedSparseFusion.__init__(self, *, local_channels: int, global_channels: int, hidden_dim: int=96, heads: int=4, sampling_points: int=4, layers: int=2, enabled: bool=True) -> None`：初始化实例依赖、配置和运行状态。 调用：`super.__init__`。
- `M L129-L166` `GatedSparseFusion.forward(self, *, local_features: LocalFeatures, global_features: torch.Tensor, patch_meta: PatchMeta) -> FusedPatchFeatures`：执行 `forward` 对应逻辑。 调用：`FusedPatchFeatures`, `sample_global_feature`, `self._sparse_context`, `self.context_project`, `self.gate_project`, `self.refinement`。
- `M L168-L221` `GatedSparseFusion._sparse_context(self, *, local: torch.Tensor, global_features: torch.Tensor, patch_meta: PatchMeta) -> torch.Tensor`：执行 `sparse context` 对应逻辑。 调用：`_base_grid`, `self.global_project`, `self.offset_predictor`, `self.weight_predictor`, `self.weight_predictor.view`。

## `src/traning/lib/models/global_encoder.py`

职责：无网络依赖的低分辨率完整画面全局 CNN encoder。

- `C L11-L17` `GlobalFeatures` [CLASS]：Low-resolution full-frame context features in BCHW layout。
- `F L20-L24` `_group_count(channels: int) -> int`：执行 `group count` 对应逻辑。
- `C L27-L53` `_ConvBlock(nn.Module)` [CLASS]：封装 `ConvBlock` 相关数据或行为。
- `M L28-L50` `_ConvBlock.__init__(self, in_channels: int, out_channels: int, *, stride: int) -> None`：初始化实例依赖、配置和运行状态。 调用：`_group_count`, `super.__init__`。
- `M L52-L53` `_ConvBlock.forward(self, x: torch.Tensor) -> torch.Tensor`：执行 `forward` 对应逻辑。 调用：`self.block`。
- `C L56-L116` `LightweightGlobalEncoder(nn.Module)` [CLASS]：Offline low-resolution full-frame encoder for global object context。
- `M L59-L90` `LightweightGlobalEncoder.__init__(self, *, in_channels: int=3, input_height: int=360, input_width: int=640, feature_channels: int=64, backbone: str='lightweight_cnn', pretrained: bool=False, frozen: bool=False) -> None`：初始化实例依赖、配置和运行状态。 调用：`_ConvBlock`, `self.parameters`, `super.__init__`。
- `M L92-L116` `LightweightGlobalEncoder.forward(self, frame: torch.Tensor) -> GlobalFeatures`：执行 `forward` 对应逻辑。 调用：`GlobalFeatures`, `self.stage16`, `self.stage2`, `self.stage4`, `self.stage8`。

## `src/traning/lib/models/global_structure_head.py`

职责：全局对象性、圆心、圆环、slider、spinner、粗半径和 context token 预测头。

- `C L11-L18` `GlobalStructurePrediction` [CLASS]：封装 `GlobalStructurePrediction` 相关数据或行为。
- `C L21-L65` `GlobalStructureHead(nn.Module)` [CLASS]：Predict coarse full-frame object structure from global features。
- `M L24-L50` `GlobalStructureHead.__init__(self, in_channels: int, *, hidden_channels: int | None=None, context_dim: int | None=None) -> None`：初始化实例依赖、配置和运行状态。 调用：`super.__init__`。
- `M L52-L65` `GlobalStructureHead.forward(self, features: torch.Tensor) -> GlobalStructurePrediction`：执行 `forward` 对应逻辑。 调用：`GlobalStructurePrediction`, `self.center_heatmap`, `self.coarse_radius`, `self.context_projection`, `self.objectness`, `self.ring_likelihood`。

## `src/traning/lib/models/local_encoder.py`

职责：小显存高分辨率局部 CNN；GroupNorm、depthwise separable residual block 和 stride-8 pyramid。

- `C L12-L21` `LocalFeatures` [CLASS]：High-resolution patch features。
- `F L24-L28` `_group_count(channels: int) -> int`：执行 `group count` 对应逻辑。
- `C L31-L48` `DepthwiseSeparableConv(nn.Module)` [CLASS]：封装 `DepthwiseSeparableConv` 相关数据或行为。
- `M L32-L45` `DepthwiseSeparableConv.__init__(self, in_channels: int, out_channels: int, *, stride: int=1) -> None`：初始化实例依赖、配置和运行状态。 调用：`_group_count`, `super.__init__`。
- `M L47-L48` `DepthwiseSeparableConv.forward(self, x: torch.Tensor) -> torch.Tensor`：执行 `forward` 对应逻辑。 调用：`self.act`, `self.depthwise`, `self.norm`, `self.pointwise`。
- `C L51-L76` `SeparableResidualBlock(nn.Module)` [CLASS]：封装 `SeparableResidualBlock` 相关数据或行为。
- `M L52-L73` `SeparableResidualBlock.__init__(self, in_channels: int, out_channels: int, *, stride: int=1) -> None`：初始化实例依赖、配置和运行状态。 调用：`DepthwiseSeparableConv`, `_group_count`, `super.__init__`。
- `M L75-L76` `SeparableResidualBlock.forward(self, x: torch.Tensor) -> torch.Tensor`：执行 `forward` 对应逻辑。 调用：`self.act`, `self.conv1`, `self.conv2`, `self.skip`。
- `C L79-L140` `SmallLocalEncoder(nn.Module)` [CLASS]：Small-channel local CNN for serial high-resolution patch training。
- `M L82-L117` `SmallLocalEncoder.__init__(self, *, in_channels: int=3, stem_channels: int=8, feature_channels: int=48, output_stride: int=8, gradient_checkpointing: bool=False) -> None`：初始化实例依赖、配置和运行状态。 调用：`SeparableResidualBlock`, `_group_count`, `super.__init__`。
- `M L119-L126` `SmallLocalEncoder._maybe_checkpoint(self, module: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor`：执行 `maybe checkpoint` 对应逻辑。
- `M L128-L140` `SmallLocalEncoder.forward(self, patch: torch.Tensor) -> LocalFeatures`：执行 `forward` 对应逻辑。 调用：`LocalFeatures`, `self._maybe_checkpoint`, `self.p2_project`, `self.p4_project`, `self.p8_project`, `self.stem`。

## `src/traning/lib/models/object_heads.py`

职责：空间多任务 dense prediction head 和对象类型表。
工程依赖：`traning.lib.models.outputs`

- `F L22-L26` `_group_count(channels: int) -> int`：执行 `group count` 对应逻辑。
- `C L29-L88` `SpatialPredictionHead(nn.Module)` [CLASS]：Multi-task dense heads for one fused high-resolution patch feature map。
- `M L32-L69` `SpatialPredictionHead.__init__(self, in_channels: int, *, hidden_channels: int | None=None, embedding_dim: int=96, object_type_count: int=len(OBJECT_TYPE_NAMES)) -> None`：初始化实例依赖、配置和运行状态。 调用：`_group_count`, `super.__init__`。
- `M L71-L88` `SpatialPredictionHead.forward(self, features: torch.Tensor) -> SpatialPrediction`：执行 `forward` 对应逻辑。 调用：`SpatialPrediction`, `self.trunk`。

## `src/traning/lib/models/outputs.py`

职责：空间预测与因果动作预测 dataclass 契约。

- `C L9-L21` `SpatialPrediction` [CLASS]：Dense spatial predictions on a patch feature grid。
- `C L25-L33` `ActionPrediction` [CLASS]：Causal action prediction for one frame step。

## `src/traning/lib/models/stack.py`

职责：从 Settings 统一构建 local/global/structure/fusion/spatial head 模型栈。
工程依赖：`traning.conf`, `traning.lib.data`, `traning.lib.models.gated_sparse_fusion`, `traning.lib.models.global_encoder`, `traning.lib.models.global_structure_head`, `traning.lib.models.local_encoder`, `traning.lib.models.object_heads`

- `F L14-L55` `build_model_stack(settings: Settings) -> dict[str, torch.nn.Module]`：Build the shared local/global/fusion/spatial model stack from settings。 调用：`GatedSparseFusion`, `GlobalStructureHead`, `LightweightGlobalEncoder`, `SmallLocalEncoder`, `SpatialPredictionHead`, `color_cue_channel_count`。

## `src/traning/lib/models/temporal_model.py`

职责：因果 GRU 时序模型；提供 initial_state 与 step 流式接口。
工程依赖：`traning.lib.models.outputs`

- `C L9-L95` `CausalTemporalModel(nn.Module)` [CLASS]：Causal GRU action head for streaming frame-by-frame inference。
- `M L12-L34` `CausalTemporalModel.__init__(self, *, input_size: int, hidden_size: int=256, layers: int=2, candidate_slots: int=64, action_classes: int=4) -> None`：初始化实例依赖、配置和运行状态。 调用：`super.__init__`。
- `M L36-L53` `CausalTemporalModel.initial_state(self, batch_size: int, device: torch.device | str, *, dtype: torch.dtype | None=None) -> torch.Tensor`：执行 `initial state` 对应逻辑。 调用：`self.parameters`。
- `M L55-L82` `CausalTemporalModel.step(self, current_features: torch.Tensor, previous_state: torch.Tensor) -> tuple[ActionPrediction, torch.Tensor]`：执行 `step` 对应逻辑。 调用：`ActionPrediction`, `next_states.append`, `self.action_head`, `self.candidate_head`, `self.time_head`, `self.xy_head`。
- `M L84-L95` `CausalTemporalModel.forward(self, sequence: torch.Tensor) -> tuple[list[ActionPrediction], torch.Tensor]`：执行 `forward` 对应逻辑。 调用：`outputs.append`, `self.initial_state`, `self.step`。

## `src/traning/lib/runtime/memory.py`

职责：统一 CUDA/runtime memory policy；管理 AMP、GradScaler、channels-last、TF32、显存/RAM预算、显存快照和 OOM 建议。

- `C L13-L18` `MemorySnapshot` [CLASS]：封装 `MemorySnapshot` 相关数据或行为。
- `C L22-L50` `RuntimeMemoryBudget` [CLASS]：封装 `RuntimeMemoryBudget` 相关数据或行为。
- `M L36-L50` `RuntimeMemoryBudget.as_dict(self) -> dict[str, float | str | None]`：执行 `as dict` 对应逻辑。
- `C L54-L58` `CudaRuntimeConfig` [CLASS]：封装 `CudaRuntimeConfig` 相关数据或行为。
- `C L62-L69` `CudaRuntimeState` [CLASS]：封装 `CudaRuntimeState` 相关数据或行为。
- `F L72-L174` `enforce_runtime_memory_budget(*, device: torch.device, max_vram_gib: float, reserve_vram_gib: float, max_ram_gib: float | None, reserve_ram_gib: float, set_cuda_fraction: bool=True) -> RuntimeMemoryBudget`：Validate CPU/CUDA budgets and reserve headroom for the host system。 调用：`RuntimeMemoryBudget`, `_finite`。
- `F L177-L186` `resolve_amp_dtype(device: torch.device, amp_dtype: str) -> torch.dtype | None`：解析并定位 `amp dtype` 对应的数据或结果。
- `F L190-L197` `autocast_context(device: torch.device, amp_dtype: str) -> Iterator[None]`：执行 `autocast context` 对应逻辑。 调用：`resolve_amp_dtype`。
- `F L200-L233` `configure_torch_runtime(*, device: torch.device, amp_dtype: str, runtime: CudaRuntimeConfig=CudaRuntimeConfig()) -> CudaRuntimeState`：Apply CUDA runtime defaults used by training and smoke tests。 调用：`CudaRuntimeState`, `amp_uses_grad_scaler`, `resolve_amp_dtype`。
- `F L236-L239` `amp_uses_grad_scaler(device: torch.device, amp_dtype: str) -> bool`：执行 `amp uses grad scaler` 对应逻辑。 调用：`resolve_amp_dtype`。
- `F L242-L255` `create_grad_scaler(*, device: torch.device, amp_dtype: str, mode: str='auto') -> torch.amp.GradScaler`：执行 `create grad scaler` 对应逻辑。 调用：`amp_uses_grad_scaler`。
- `F L258-L267` `module_to_device(module: nn.Module, device: torch.device, *, channels_last: bool) -> nn.Module`：执行 `module to device` 对应逻辑。
- `F L270-L280` `maybe_compile_module(module: nn.Module, *, enabled: bool, mode: str='default') -> nn.Module`：执行 `maybe compile module` 对应逻辑。
- `F L283-L296` `tensor_to_device(tensor: torch.Tensor, device: torch.device, *, channels_last: bool, non_blocking: bool=True) -> torch.Tensor`：执行 `tensor to device` 对应逻辑。
- `F L299-L314` `collect_memory_snapshot() -> MemorySnapshot`：执行 `collect memory snapshot` 对应逻辑。 调用：`MemorySnapshot`。
- `F L317-L350` `format_oom_guidance(*, patch_size: tuple[int, int], global_size: tuple[int, int], batch_size: int, amp_dtype: str, config_path: str | None) -> str`：执行 `format oom guidance` 对应逻辑。 调用：`collect_memory_snapshot`。
- `F L353-L354` `_finite(value: float) -> bool`：执行 `finite` 对应逻辑。

## `src/traning/lib/training/feature_canvas.py`

职责：detached CPU feature canvas；按 patch 元数据累计融合特征。
工程依赖：`traning.lib.data`

- `C L13-L81` `FeatureCanvas` [CLASS]：CPU accumulation canvas for detached patch features。
- `M L22-L28` `FeatureCanvas.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L30-L72` `FeatureCanvas.write_patch(self, features: torch.Tensor, meta: PatchMeta, *, weight: torch.Tensor | None=None) -> None`：Accumulate one detached CHW or 1CHW patch feature tensor on CPU。
- `M L74-L77` `FeatureCanvas.to_tensor(self) -> torch.Tensor`：Return the weighted average canvas as a detached CPU tensor。 调用：`self._weights.clamp_min`。
- `M L80-L81` `FeatureCanvas.weights(self) -> torch.Tensor` [PROPERTY]：执行 `weights` 对应逻辑。

## `src/traning/lib/training/losses.py`

职责：空间多任务损失、全局局部一致性、跨 patch embedding 和时序一致性损失。
工程依赖：`traning.lib.models.outputs`

- `C L12-L24` `LossWeights` [CLASS]：封装 `LossWeights` 相关数据或行为。
- `C L28-L37` `SpatialLossTargets` [CLASS]：封装 `SpatialLossTargets` 相关数据或行为。
- `F L40-L53` `_masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor`：执行 `masked smooth l1` 对应逻辑。
- `F L56-L117` `compute_spatial_loss(prediction: SpatialPrediction, target: SpatialLossTargets, *, weights: LossWeights=LossWeights()) -> dict[str, torch.Tensor]`：Compute first-version dense multi-task spatial losses。 调用：`_masked_smooth_l1`。
- `F L120-L147` `cosine_embedding_consistency_loss(embeddings: torch.Tensor, object_ids: torch.Tensor, *, margin: float=0.4) -> torch.Tensor`：Pull embeddings for the same object together and push others apart。 调用：`pieces.append`。
- `F L150-L159` `global_local_consistency_loss(local_logits: torch.Tensor, sampled_global_logits: torch.Tensor) -> torch.Tensor`：Encourage local dense predictions to agree with sampled global context。
- `F L162-L175` `temporal_consistency_loss(current: torch.Tensor, previous: torch.Tensor, *, mask: torch.Tensor | None=None) -> torch.Tensor`：Penalize abrupt dense prediction changes between neighboring frames。

## `src/traning/lib/training/spatial_decode.py`

职责：把 patch dense 空间预测融合为全图概率画布，并解码 Top-K 空间候选和首版 slider 连通域路径。
工程依赖：`traning.lib.data`, `traning.lib.models`

- `C L15-L29` `SpatialPredictionMaps` [CLASS]：封装 `SpatialPredictionMaps` 相关数据或行为。
- `C L33-L46` `SpatialCandidate` [CLASS]：封装 `SpatialCandidate` 相关数据或行为。
- `C L50-L62` `SliderPathCandidate` [CLASS]：封装 `SliderPathCandidate` 相关数据或行为。
- `C L65-L166` `SpatialPredictionCanvas` [CLASS]：CPU canvas for fusing detached dense spatial predictions across patches。
- `M L68-L100` `SpatialPredictionCanvas.__init__(self, *, frame_width: int, frame_height: int, stride: int, object_type_count: int=len(OBJECT_TYPE_NAMES), embedding_dim: int, dtype: torch.dtype=torch.float32, feather_edges: bool=True) -> None`：初始化实例依赖、配置和运行状态。
- `M L102-L140` `SpatialPredictionCanvas.write_patch(self, prediction: SpatialPrediction, meta: PatchMeta) -> None`：写入 `patch` 对应的数据或结果。 调用：`_patch_weight`, `_prediction_to_payload`, `_write_region`。
- `M L142-L166` `SpatialPredictionCanvas.to_maps(self) -> SpatialPredictionMaps`：执行 `to maps` 对应逻辑。 调用：`SpatialPredictionMaps`, `self._values.items`, `self._weights.clamp_min`, `self._weights.clone`。
- `F L169-L231` `decode_spatial_candidates(maps: SpatialPredictionMaps, *, max_candidates: int=32, score_threshold: float=0.05, nms_radius_px: float=32.0) -> tuple[SpatialCandidate, ...]`：执行 `decode spatial candidates` 对应逻辑。 调用：`SpatialCandidate`, `_is_suppressed`, `selected.append`。
- `F L234-L272` `decode_slider_paths(maps: SpatialPredictionMaps, *, threshold: float=0.5, min_cells: int=4, max_paths: int=16, sample_points: int=32, continuity_threshold: float=0.75) -> tuple[SliderPathCandidate, ...]`：Recover first-version slider path candidates from the fused CPU canvas。 调用：`_connected_components`, `_decode_slider_component`, `paths.append`。
- `F L275-L297` `_prediction_to_payload(prediction: SpatialPrediction, *, dtype: torch.dtype) -> dict[str, torch.Tensor]`：执行 `prediction to payload` 对应逻辑。
- `F L300-L327` `_write_region(meta: PatchMeta, *, feature_height: int, feature_width: int, frame_height: int, frame_width: int, stride: int) -> tuple[slice, slice, slice, slice] | None`：写入 `region` 对应的数据或结果。
- `F L330-L341` `_patch_weight(height: int, width: int, *, dtype: torch.dtype, feather_edges: bool) -> torch.Tensor`：执行 `patch weight` 对应逻辑。 调用：`_hann_axis`。
- `F L344-L347` `_hann_axis(size: int, *, dtype: torch.dtype) -> torch.Tensor`：执行 `hann axis` 对应逻辑。
- `F L350-L363` `_is_suppressed(selected: list[SpatialCandidate], *, x: float, y: float, radius: float) -> bool`：判断是否 `suppressed` 对应的数据或结果。
- `F L366-L388` `_connected_components(mask: torch.Tensor) -> tuple[tuple[tuple[int, int], ...], ...]`：执行 `connected components` 对应逻辑。 调用：`_neighbor_cells`, `component.append`, `components.append`, `queue.append`。
- `F L391-L437` `_decode_slider_component(maps: SpatialPredictionMaps, *, component_id: int, component: tuple[tuple[int, int], ...], sample_points: int, continuity_threshold: float) -> SliderPathCandidate`：执行 `decode slider component` 对应逻辑。 调用：`SliderPathCandidate`, `_cell_to_xy`, `_component_bbox`, `_component_degree`, `_orient_slider_cells`, `_sample_polyline`。
- `F L440-L456` `_neighbor_cells(cell: tuple[int, int], *, height: int, width: int) -> tuple[tuple[int, int], ...]`：执行 `neighbor cells` 对应逻辑。 调用：`neighbors.append`。
- `F L459-L477` `_component_neighbors(cell: tuple[int, int], component: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]`：执行 `component neighbors` 对应逻辑。
- `F L480-L484` `_component_degree(cell: tuple[int, int], component: set[tuple[int, int]]) -> int`：执行 `component degree` 对应逻辑。 调用：`_component_neighbors`。
- `F L487-L492` `_select_component_endpoints(component: tuple[tuple[int, int], ...], endpoints: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], tuple[int, int]]`：选择 `component endpoints` 对应的数据或结果。 调用：`_farthest_pair`。
- `F L495-L502` `_farthest_pair(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], tuple[int, int]]`：执行 `farthest pair` 对应逻辑。 调用：`_cell_distance_squared`。
- `F L505-L509` `_cell_distance_squared(first: tuple[int, int], second: tuple[int, int]) -> int`：执行 `cell distance squared` 对应逻辑。
- `F L512-L537` `_shortest_component_path(component: set[tuple[int, int]], *, start: tuple[int, int], end: tuple[int, int]) -> tuple[tuple[int, int], ...]`：执行 `shortest component path` 对应逻辑。 调用：`_component_neighbors`, `path.append`, `queue.append`。
- `F L540-L558` `_orient_slider_cells(cells: tuple[tuple[int, int], ...], maps: SpatialPredictionMaps) -> tuple[tuple[int, int], ...]`：执行 `orient slider cells` 对应逻辑。
- `F L561-L568` `_cell_to_xy(cell: tuple[int, int], maps: SpatialPredictionMaps) -> tuple[float, float]`：执行 `cell to xy` 对应逻辑。
- `F L571-L610` `_sample_polyline(points: tuple[tuple[float, float], ...], *, sample_points: int) -> tuple[tuple[float, float], ...]`：执行 `sample polyline` 对应逻辑。 调用：`distances.append`, `sampled.append`。
- `F L613-L623` `_component_bbox(component: tuple[tuple[int, int], ...], maps: SpatialPredictionMaps) -> tuple[float, float, float, float]`：执行 `component bbox` 对应逻辑。
- `F L626-L643` `_slider_ambiguity_reasons(*, endpoint_count: int, branch_points: int, continuity: float, continuity_threshold: float, polyline: tuple[tuple[float, float], ...]) -> tuple[str, ...]`：执行 `slider ambiguity reasons` 对应逻辑。 调用：`reasons.append`。

## `src/traning/lib/training/spatial_targets.py`

职责：把单帧 osu 标注按 PatchMeta 光栅化为空间多任务 dense loss target。
工程依赖：`package.coordinates`, `traning.lib.data`, `traning.lib.models`, `traning.lib.training.losses`

- `F L22-L99` `build_spatial_loss_targets(sample: Mapping[str, Any], patch_meta: PatchMeta, feature_size: Sequence[int], *, device: torch.device | str | None=None, dtype: torch.dtype=torch.float32) -> SpatialLossTargets`：Rasterize one frame sample into dense targets for one patch feature grid。 调用：`SpatialLossTargets`, `_empty_targets`, `_finite_float`, `_normalize_feature_size`, `_object_kind`, `_paint_circle`。
- `F L102-L108` `_normalize_feature_size(feature_size: Sequence[int]) -> tuple[int, int]`：规范化 `feature size` 对应的数据或结果。
- `F L111-L164` `_empty_targets(*, feature_height: int, feature_width: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]`：执行 `empty targets` 对应逻辑。
- `F L167-L195` `_patch_grid(patch_meta: PatchMeta, *, feature_height: int, feature_width: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor | float]`：执行 `patch grid` 对应逻辑。
- `F L198-L205` `_finite_float(value: Any, default: float) -> float`：执行 `finite float` 对应逻辑。
- `F L208-L216` `_object_kind(item: Mapping[str, Any]) -> str` [IO-W]：执行 `object kind` 对应逻辑。
- `F L219-L226` `_set_type(target: dict[str, torch.Tensor], mask: torch.Tensor, object_type: str) -> None`：执行 `set type` 对应逻辑。
- `F L229-L235` `_set_heatmap_max(tensor: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> None`：执行 `set heatmap max` 对应逻辑。
- `F L238-L244` `_point_to_local(point: tuple[float, float], transform: OsuVideoTransform, grid: Mapping[str, torch.Tensor | float]) -> tuple[float, float]`：执行 `point to local` 对应逻辑。
- `F L247-L264` `_object_points(item: Mapping[str, Any]) -> tuple[tuple[float, float], ...]`：执行 `object points` 对应逻辑。 调用：`_distance`, `_finite_float`, `points.append`。
- `F L267-L268` `_distance(first: tuple[float, float], second: tuple[float, float]) -> float`：执行 `distance` 对应逻辑。
- `F L271-L281` `_distance_to_point(grid: Mapping[str, torch.Tensor | float], *, local_x: float, local_y: float) -> torch.Tensor`：执行 `distance to point` 对应逻辑。
- `F L284-L307` `_paint_center(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], *, local_x: float, local_y: float, radius: float, object_type: str) -> None`：执行 `paint center` 对应逻辑。 调用：`_distance_to_point`, `_set_heatmap_max`, `_set_type`, `_write_offset`。
- `F L310-L337` `_write_offset(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], *, local_x: float, local_y: float) -> None`：写入 `offset` 对应的数据或结果。
- `F L340-L379` `_paint_circle(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], item: Mapping[str, Any], *, transform: OsuVideoTransform, hit_radius: float, timestamp_ms: float, preempt_ms: float) -> None`：执行 `paint circle` 对应逻辑。 调用：`_distance_to_point`, `_finite_float`, `_object_points`, `_paint_center`, `_point_to_local`, `_set_heatmap_max`。
- `F L382-L430` `_paint_slider(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], item: Mapping[str, Any], *, transform: OsuVideoTransform, hit_radius: float) -> None`：执行 `paint slider` 对应逻辑。 调用：`_finite_float`, `_object_points`, `_paint_center`, `_paint_repeat_points`, `_paint_slider_body`, `_point_to_local`。
- `F L433-L476` `_paint_slider_body(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], points: tuple[tuple[float, float], ...], *, tube_radius: float) -> None`：执行 `paint slider body` 对应逻辑。 调用：`_set_heatmap_max`, `_set_type`, `_unoriented_direction`。
- `F L479-L481` `_unoriented_direction(vx: float, vy: float) -> tuple[float, float]`：执行 `unoriented direction` 对应逻辑。
- `F L484-L501` `_paint_repeat_points(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], points: tuple[tuple[float, float], ...], *, repeats: int, radius: float) -> None`：执行 `paint repeat points` 对应逻辑。 调用：`_distance_to_point`, `_set_type`, `repeat_points.append`。
- `F L504-L530` `_paint_spinner(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], *, transform: OsuVideoTransform) -> None`：执行 `paint spinner` 对应逻辑。 调用：`_paint_center`, `_set_heatmap_max`, `_set_type`。

## `src/traning/lib/visualization/display.py`

职责：通过独立 ffplay 子进程把标注图片显示到主机 X11。

- `F L9-L50` `launch_image_window(image_path: Path, *, title: str, ffplay_binary: str='ffplay', display: str | None=None, previous_process: subprocess.Popen[bytes] | None=None) -> subprocess.Popen[bytes]`：执行 `launch image window` 对应逻辑。

## `src/traning/lib/visualization/gallery.py`

职责：选择批次最高分 trial，并按通过状态和六个子项目随机保存标注帧图集。
工程依赖：`traning.lib.data`, `traning.lib.visualization.output_identity`, `traning.lib.visualization.render`, `traning.state.gallery_schema`

- `F L34-L36` `_safe_name(value: str) -> str`：执行 `safe name` 对应逻辑。
- `F L39-L43` `_subproject_for_record(record: SegmentRecord) -> str`：执行 `subproject for record` 对应逻辑。
- `F L46-L58` `_frame_lookup(dataset: SegmentFrameDataset) -> dict[tuple[str, int], tuple[int, str]]`：执行 `frame lookup` 对应逻辑。 调用：`_subproject_for_record`。
- `F L61-L65` `_metric_lines(metrics: Mapping[str, float]) -> tuple[str, ...]`：执行 `metric lines` 对应逻辑。
- `F L68-L223` `save_best_trial_gallery(dataset: SegmentFrameDataset, request: BatchGalleryRequest, *, output_root: Path, samples_per_group: int=10) -> tuple[Path, int, tuple[str, ...]]` [IO-W]：执行 `save best trial gallery` 对应逻辑。 调用：`_frame_lookup`, `_metric_lines`, `_safe_name`, `allocate_output_identity`, `append`, `issues.append`。

## `src/traning/lib/visualization/models.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L18-L25` `VisualizationResult` [CLASS]：封装 `VisualizationResult` 相关数据或行为。
- `M L24-L25` `VisualizationResult.succeeded(self) -> bool` [PROPERTY]：执行 `succeeded` 对应逻辑。
- `C L29-L38` `GalleryResult` [CLASS]：封装 `GalleryResult` 相关数据或行为。
- `M L37-L38` `GalleryResult.succeeded(self) -> bool` [PROPERTY]：执行 `succeeded` 对应逻辑。
- `C L42-L47` `SelectedFrame` [CLASS]：封装 `SelectedFrame` 相关数据或行为。

## `src/traning/lib/visualization/output_identity.py`

职责：为 traning_example 输出分配进程安全的递增次数和 UTC 时间标识。

- `C L14-L21` `OutputIdentity` [CLASS]：封装 `OutputIdentity` 相关数据或行为。
- `M L20-L21` `OutputIdentity.prefix(self) -> str` [PROPERTY]：执行 `prefix` 对应逻辑。
- `F L24-L28` `_read_counter(path: Path) -> int` [IO-R]：读取 `counter` 对应的数据或结果。
- `F L31-L37` `_existing_max_sequence(output_root: Path) -> int`：执行 `existing max sequence` 对应逻辑。
- `F L40-L63` `allocate_output_identity(output_root: Path) -> OutputIdentity` [IO-W]：执行 `allocate output identity` 对应逻辑。 调用：`OutputIdentity`, `_existing_max_sequence`, `_read_counter`。

## `src/traning/lib/visualization/render.py`

职责：把帧 Tensor、osu 标签和共享坐标变换渲染为标注图片。
工程依赖：`package`

- `F L21-L30` `_image_from_tensor(image: torch.Tensor) -> Image.Image`：执行 `image from tensor` 对应逻辑。
- `F L33-L39` `_point(transform: OsuVideoTransform, x: float, y: float) -> tuple[int, int]`：执行 `point` 对应逻辑。
- `F L42-L51` `_draw_cross(draw: ImageDraw.ImageDraw, point: tuple[int, int], color: tuple[int, int, int], size: int=12, width: int=3) -> None`：执行 `draw cross` 对应逻辑。
- `F L54-L61` `_is_target(hit_object: Mapping[str, Any], target_source_index: int | None) -> bool`：判断是否 `target` 对应的数据或结果。
- `F L64-L92` `_draw_circle(draw: ImageDraw.ImageDraw, hit_object: Mapping[str, Any], transform: OsuVideoTransform, radius: int, target_source_index: int | None) -> tuple[int, int] | None`：执行 `draw circle` 对应逻辑。 调用：`_draw_cross`, `_is_target`, `_point`。
- `F L95-L147` `_draw_slider(draw: ImageDraw.ImageDraw, hit_object: Mapping[str, Any], transform: OsuVideoTransform, radius: int, target_source_index: int | None) -> tuple[int, int] | None`：执行 `draw slider` 对应逻辑。 调用：`_draw_cross`, `_is_target`, `_point`。
- `F L150-L248` `render_annotated_frame(sample: Mapping[str, Any], *, target_source_index: int | None=None, include_all_objects: bool=False, predicted_osu_xy: tuple[float, float] | None=None, metadata_lines: Sequence[str]=()) -> Image.Image`：执行 `render annotated frame` 对应逻辑。 调用：`_draw_circle`, `_draw_cross`, `_draw_slider`, `_image_from_tensor`, `_is_target`, `_point`。
- `F L251-L254` `save_annotated_frame(image: Image.Image, output_path: Path) -> Path` [IO-W]：执行 `save annotated frame` 对应逻辑。

## `src/traning/lib/visualization/selection.py`

职责：根据 HitObject 起始时间反推最接近的采样帧。
工程依赖：`traning.lib.data.dataset`, `traning.lib.visualization.models`

- `F L7-L47` `select_click_frame(dataset: SegmentFrameDataset, *, segment_index: int, object_index: int=0) -> SelectedFrame`：选择 `click frame` 对应的数据或结果。 调用：`SelectedFrame`。

## `src/traning/main.py`

职责：Typer CLI；执行数据检查、样本预览、空间训练/推理 smoke、候选缓存和训练阶段注册表。
工程依赖：`traning.conf`, `traning.core.dataset_import`, `traning.core.decision`, `traning.core.decision.pipeline`, `traning.core.result_export`, `traning.core.spatial`, `traning.core.temporal`, `traning.lib.data`, `traning.lib.models`, `traning.lib.runtime`, `traning.state`

- `F L58-L70` `_render_report(report) -> None`：执行 `render report` 对应逻辑。
- `F L73-L76` `_format_bool(value: bool | None) -> str`：执行 `format bool` 对应逻辑。
- `F L79-L82` `_format_gib(value: float | None) -> str`：执行 `format gib` 对应逻辑。
- `F L85-L122` `_render_env_report(report) -> None`：执行 `render env report` 对应逻辑。 调用：`_format_bool`, `_format_gib`。
- `F L125-L129` `_run_dir(kind: str, *, root: Path | None=None) -> Path` [IO-W]：执行 `run dir` 对应逻辑。
- `F L132-L141` `_select_device(device: str) -> torch.device`：选择 `device` 对应的数据或结果。
- `F L144-L147` `_load_image_tensor(path: Path) -> torch.Tensor` [IO-W]：加载 `image tensor` 对应的数据或结果。
- `F L150-L151` `_build_model_stack(settings) -> dict[str, torch.nn.Module]`：构建 `model stack` 对应的数据或结果。 调用：`build_model_stack`。
- `F L154-L298` `_execute_model_smoke(*, config: Path | None, device: torch.device, backward: bool) -> dict[str, Any]`：执行 `execute model smoke` 对应逻辑。 调用：`CudaRuntimeConfig`, `PatchStream`, `_build_model_stack`, `append_color_cues`, `autocast_context`, `collect_memory_snapshot`。
- `F L301-L308` `_render_dict_table(title: str, values: dict[str, Any]) -> None`：执行 `render dict table` 对应逻辑。
- `F L311-L324` `_compact_slider_path(path: dict[str, Any]) -> dict[str, Any]`：执行 `compact slider path` 对应逻辑。
- `F L328-L336` `data_check(config: Path | None=typer.Option(None, '--config'), split: DataSplit=typer.Option('all', '--split')) -> None` [CLI]：执行 `data check` 对应逻辑。 调用：`_render_report`, `inspect_data_input`, `load_settings`。
- `F L340-L359` `env_check(strict: bool=typer.Option(False, '--strict/--no-strict', help='Exit non-zero when required runtime dependencies are missing.'), require_cuda: bool=typer.Option(False, '--require-cuda/--no-require-cuda', help='Treat CUDA unavailability as a failure in strict mode.')) -> None` [CLI]：执行 `env check` 对应逻辑。 调用：`_render_env_report`。
- `F L363-L395` `data_preview(index: int=typer.Option(0, '--index', min=0), split: DataSplit=typer.Option('train', '--split'), config: Path | None=typer.Option(None, '--config')) -> None` [CLI]：执行 `data preview` 对应逻辑。 调用：`build_dataset`, `build_patch_windows`, `load_settings`。
- `F L399-L403` `run(config: Path | None=typer.Option(None, '--config')) -> None` [CLI]：执行该处理器的完整工作流。 调用：`_render_report`, `load_settings`, `run_pipeline`。
- `F L407-L432` `model_smoke(config: Path=typer.Option(Path('configs/model_small_vram.yaml'), '--config'), device: str=typer.Option('cpu', '--device', help='cpu, cuda, or auto. CPU is the default smoke path.'), backward: bool=typer.Option(True, '--backward/--no-backward', help='Run backward and optimizer step in addition to forward.')) -> None` [CLI IO-W]：执行 `model smoke` 对应逻辑。 调用：`_execute_model_smoke`, `_render_dict_table`, `_run_dir`, `_select_device`。
- `F L436-L494` `spatial_decode_smoke(config: Path=typer.Option(Path('configs/model_small_vram.yaml'), '--config'), split: DataSplit=typer.Option('train', '--split'), index: int=typer.Option(0, '--index', min=0), device: str=typer.Option('cpu', '--device'), max_candidates: int=typer.Option(16, '--max-candidates', min=1), score_threshold: float=typer.Option(0.0, '--score-threshold', min=0.0), nms_radius_px: float=typer.Option(32.0, '--nms-radius-px', min=0.0), slider_threshold: float=typer.Option(0.5, '--slider-threshold', min=0.0, max=1.0), max_slider_paths: int=typer.Option(16, '--max-slider-paths', min=1), patch_limit: int | None=typer.Option(None, '--patch-limit', min=1)) -> None` [CLI IO-W]：执行 `spatial decode smoke` 对应逻辑。 调用：`_compact_slider_path`, `_render_dict_table`, `_run_dir`, `_select_device`, `build_dataset`, `load_settings`。
- `F L498-L539` `build_candidate_cache(config: Path=typer.Option(Path('configs/model_small_vram.yaml'), '--config'), split: DataSplit=typer.Option('train', '--split'), device: str=typer.Option('cpu', '--device'), max_frames: int | None=typer.Option(None, '--max-frames', min=1), patch_limit: int | None=typer.Option(None, '--patch-limit', min=1), max_candidates: int | None=typer.Option(None, '--max-candidates', min=1), score_threshold: float | None=typer.Option(None, '--score-threshold', min=0.0, max=1.0), nms_radius_px: float | None=typer.Option(None, '--nms-radius-px', min=0.0), slider_threshold: float | None=typer.Option(None, '--slider-threshold', min=0.0, max=1.0), max_slider_paths: int | None=typer.Option(None, '--max-slider-paths', min=1), output: Path | None=typer.Option(None, '--output')) -> None` [CLI]：构建并返回 `candidate cache` 对应的数据或结果。 调用：`_render_dict_table`, `_run_dir`, `_select_device`, `generate_candidate_cache`, `load_settings`, `result.as_dict`。
- `F L543-L567` `memory_profile(config: Path=typer.Option(Path('configs/model_small_vram.yaml'), '--config'), device: str=typer.Option('cuda', '--device', help='cuda, cpu, or auto. CUDA is the default for memory profiling.')) -> None` [CLI IO-W]：执行 `memory profile` 对应逻辑。 调用：`_execute_model_smoke`, `_render_dict_table`, `_run_dir`, `_select_device`。
- `F L571-L599` `visualize_patches(input_image: Path=typer.Option(..., '--input', exists=True, file_okay=True, dir_okay=False, readable=True), output: Path | None=typer.Option(None, '--output'), config: Path=typer.Option(Path('configs/model_small_vram.yaml'), '--config')) -> None` [CLI IO-W]：执行 `visualize patches` 对应逻辑。 调用：`PatchStream`, `_run_dir`, `load_settings`, `stream.metas`。
- `F L603-L690` `visualize_fusion(input_image: Path=typer.Option(..., '--input', exists=True, file_okay=True, dir_okay=False, readable=True), output: Path | None=typer.Option(None, '--output'), config: Path=typer.Option(Path('configs/model_small_vram.yaml'), '--config'), device: str=typer.Option('cpu', '--device')) -> None` [CLI IO-W]：执行 `visualize fusion` 对应逻辑。 调用：`CudaRuntimeConfig`, `PatchStream`, `_build_model_stack`, `_load_image_tensor`, `_run_dir`, `_select_device`。
- `F L693-L713` `_training_placeholder(stage: str, config: Path) -> None` [IO-W]：执行 `training placeholder` 对应逻辑。 调用：`_run_dir`, `load_settings`。
- `F L717-L762` `train_spatial(config: Path=typer.Option(Path('configs/model_small_vram.yaml'), '--config'), split: DataSplit=typer.Option('train', '--split'), device: str=typer.Option('auto', '--device', help='cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.'), max_steps: int=typer.Option(1, '--max-steps', min=1), learning_rate: float=typer.Option(0.0001, '--lr', min=1e-08), patch_limit: int | None=typer.Option(None, '--patch-limit', min=1)) -> None` [CLI]：执行 `train spatial` 对应逻辑。 调用：`_render_dict_table`, `_run_dir`, `_select_device`, `format_oom_guidance`, `load_settings`, `result.as_dict`。
- `F L766-L769` `train_fusion(config: Path=typer.Option(Path('configs/model_small_vram.yaml'), '--config')) -> None` [CLI]：执行 `train fusion` 对应逻辑。 调用：`_training_placeholder`。
- `F L773-L807` `train_temporal(config: Path=typer.Option(Path('configs/model_small_vram.yaml'), '--config'), cache: Path=typer.Option(..., '--cache', exists=True, file_okay=False, dir_okay=True, readable=True, help='Candidate cache directory containing manifest.json and frames.jsonl.'), device: str=typer.Option('auto', '--device', help='cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.'), max_steps: int=typer.Option(1, '--max-steps', min=1), learning_rate: float=typer.Option(0.0001, '--lr', min=1e-08), sequence_length: int | None=typer.Option(None, '--sequence-length', min=1), candidate_slots: int | None=typer.Option(None, '--candidate-slots', min=1)) -> None` [CLI]：执行 `train temporal` 对应逻辑。 调用：`_render_dict_table`, `_run_dir`, `_select_device`, `load_settings`, `result.as_dict`, `run_temporal_training`。
- `F L811-L848` `run_decision(config: Path=typer.Option(Path('configs/model_small_vram.yaml'), '--config'), cache: Path=typer.Option(..., '--cache', exists=True, file_okay=False, dir_okay=True, readable=True, help='Candidate cache directory containing manifest.json and frames.jsonl.'), checkpoint: Path=typer.Option(..., '--checkpoint', exists=True, file_okay=True, dir_okay=False, readable=True, help='Temporal checkpoint produced by train-temporal.'), output: Path | None=typer.Option(None, '--output'), device: str=typer.Option('auto', '--device', help='cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.')) -> None` [CLI]：执行 `run decision` 对应逻辑。 调用：`_render_dict_table`, `_run_dir`, `_select_device`, `load_settings`, `result.as_dict`, `run_temporal_decision`。
- `F L852-L871` `visualize_label(segment_index: int=typer.Option(0, '--segment-index', min=0), object_index: int=typer.Option(0, '--object-index', min=0), output: Path | None=typer.Option(None, '--output'), show: bool=typer.Option(False, '--show/--no-show'), config: Path | None=typer.Option(None, '--config')) -> None` [CLI]：执行 `visualize label` 对应逻辑。 调用：`load_settings`, `visualize_click_label`。
- `F L875-L909` `save_gallery(results: Path=typer.Option(..., '--results', exists=True, file_okay=True, dir_okay=False, readable=True), output_root: Path | None=typer.Option(None, '--output-root'), samples_per_group: int | None=typer.Option(None, '--samples-per-group', min=1), config: Path | None=typer.Option(None, '--config')) -> None` [CLI]：执行 `save gallery` 对应逻辑。 调用：`load_batch_gallery_request`, `load_settings`, `save_annotation_gallery`。

## `src/traning/state/checkpoint_schema.py`

职责：检查点 lineage 与模型/优化器/scheduler/AMP 恢复契约。
工程依赖：`traning.state.experiment_schema`

- `C L10-L27` `CheckpointMetadata(BaseModel)` [CLASS]：封装 `CheckpointMetadata` 相关数据或行为。
- `M L24-L27` `CheckpointMetadata._nonnegative_integer(cls, value: int) -> int` [VALIDATOR]：执行 `nonnegative integer` 对应逻辑。

## `src/traning/state/experiment_schema.py`

职责：三层参数、TPE/随机来源、ASHA trial、课程和独立评估契约。

- `C L8-L10` `SearchMethod(StrEnum)` [CLASS]：封装 `SearchMethod` 相关数据或行为。
- `C L13-L19` `TrialStatus(StrEnum)` [CLASS]：封装 `TrialStatus` 相关数据或行为。
- `C L22-L26` `CurriculumStage(StrEnum)` [CLASS]：封装 `CurriculumStage` 相关数据或行为。
- `C L29-L32` `TrialParameters(BaseModel)` [CLASS]：封装 `TrialParameters` 相关数据或行为。
- `C L35-L56` `TrialMetadata(BaseModel)` [CLASS]：封装 `TrialMetadata` 相关数据或行为。
- `M L53-L56` `TrialMetadata._nonnegative_integer(cls, value: int) -> int` [VALIDATOR]：执行 `nonnegative integer` 对应逻辑。
- `C L59-L65` `EvaluationRunMetadata(BaseModel)` [CLASS]：封装 `EvaluationRunMetadata` 相关数据或行为。
- `C L68-L78` `ExperimentMetadata(BaseModel)` [CLASS]：封装 `ExperimentMetadata` 相关数据或行为。

## `src/traning/state/gallery_schema.py`

职责：批次 trial 分数、稳定帧引用、错误归因和最佳参数图集输入契约。
工程依赖：`traning.state.experiment_schema`

- `C L24-L49` `FrameEvaluation(BaseModel)` [CLASS]：封装 `FrameEvaluation` 相关数据或行为。
- `M L39-L42` `FrameEvaluation._nonnegative_frame_index(cls, value: int) -> int` [VALIDATOR]：执行 `nonnegative frame index` 对应逻辑。
- `M L46-L49` `FrameEvaluation._finite_optional_metric(cls, value: float | None) -> float | None` [VALIDATOR]：执行 `finite optional metric` 对应逻辑。
- `C L52-L65` `TrialGalleryEvaluation(BaseModel)` [CLASS]：封装 `TrialGalleryEvaluation` 相关数据或行为。
- `M L62-L65` `TrialGalleryEvaluation._finite_score(cls, value: float) -> float` [VALIDATOR]：执行 `finite score` 对应逻辑。
- `C L68-L94` `BatchGalleryRequest(BaseModel)` [CLASS]：封装 `BatchGalleryRequest` 相关数据或行为。
- `M L75-L81` `BatchGalleryRequest._require_trials(cls, value: tuple[TrialGalleryEvaluation, ...]) -> tuple[TrialGalleryEvaluation, ...]` [VALIDATOR]：执行 `require trials` 对应逻辑。
- `M L84-L90` `BatchGalleryRequest._require_one_score_version(self) -> BatchGalleryRequest` [VALIDATOR]：执行 `require one score version` 对应逻辑。
- `M L93-L94` `BatchGalleryRequest.best_trial(self) -> TrialGalleryEvaluation` [PROPERTY]：执行 `best trial` 对应逻辑。
- `F L97-L102` `load_batch_gallery_request(path: Path) -> BatchGalleryRequest` [IO-R]：加载 `batch gallery request` 对应的数据或结果。

## `src/traning/state/run_state.py`

职责：保存 trial、课程阶段、rung、预算和全局步数的运行状态。
工程依赖：`traning.state.experiment_schema`

- `C L9-L16` `RunState` [CLASS]：封装 `RunState` 相关数据或行为。

## `src/traning/tests/test_candidate_cache.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.decision`, `traning.lib.training`, `traning.lib.training.spatial_decode`

- `F L22-L41` `_candidate(*, score: float=0.55, object_type: str='slider_head') -> SpatialCandidate`：执行 `candidate` 对应逻辑。 调用：`SpatialCandidate`。
- `F L44-L58` `_slider_path(*, ambiguous: bool=False) -> SliderPathCandidate`：执行 `slider path` 对应逻辑。 调用：`SliderPathCandidate`。
- `C L61-L138` `CandidateCacheTests(unittest.TestCase)` [CLASS]：封装 `CandidateCacheTests` 相关数据或行为。
- `M L62-L104` `CandidateCacheTests.test_record_keeps_embedding_and_candidate_ambiguity(self) -> None`：执行 `test record keeps embedding and candidate ambiguity` 对应逻辑。 调用：`_candidate`, `_slider_path`, `build_candidate_cache_record`, `self.assertEqual`, `self.assertIn`。
- `M L106-L138` `CandidateCacheTests.test_generate_candidate_cache_writes_manifest_and_jsonl(self) -> None` [IO-R]：执行 `test generate candidate cache writes manifest and jsonl` 对应逻辑。 调用：`Settings`, `_candidate`, `_slider_path`, `generate_candidate_cache`, `self.assertEqual`。

## `src/traning/tests/test_causal_temporal.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.models`

- `C L10-L37` `CausalTemporalTests(unittest.TestCase)` [CLASS]：封装 `CausalTemporalTests` 相关数据或行为。
- `M L11-L21` `CausalTemporalTests.test_future_frames_do_not_change_past_outputs(self) -> None`：执行 `test future frames do not change past outputs` 对应逻辑。 调用：`CausalTemporalModel`, `self.assertTrue`。
- `M L23-L31` `CausalTemporalTests.test_reset_state_repeats_output(self) -> None`：执行 `test reset state repeats output` 对应逻辑。 调用：`CausalTemporalModel`, `model.initial_state`, `model.step`, `self.assertTrue`。
- `M L33-L37` `CausalTemporalTests.test_batch_size_one_runs(self) -> None`：执行 `test batch size one runs` 对应逻辑。 调用：`CausalTemporalModel`, `model.initial_state`, `model.step`, `self.assertEqual`。

## `src/traning/tests/test_color_cues.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.lib.data`, `traning.lib.models`

- `C L16-L52` `ColorCueTests(unittest.TestCase)` [CLASS]：封装 `ColorCueTests` 相关数据或行为。
- `M L17-L28` `ColorCueTests.test_osu_basic_cues_highlight_colored_target_and_white_number(self) -> None`：执行 `test osu basic cues highlight colored target and white number` 对应逻辑。 调用：`extract_osu_basic_color_cues`, `self.assertEqual`, `self.assertGreater`, `self.assertLess`。
- `M L30-L37` `ColorCueTests.test_append_color_cues_is_configurable(self) -> None`：执行 `test append color cues is configurable` 对应逻辑。 调用：`append_color_cues`, `color_cue_channel_count`, `self.assertEqual`, `self.assertIs`。
- `M L39-L52` `ColorCueTests.test_model_stack_accepts_augmented_input_channels(self) -> None`：执行 `test model stack accepts augmented input channels` 对应逻辑。 调用：`Settings`, `build_model_stack`, `self.assertEqual`。

## `src/traning/tests/test_coordinates.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`

- `C L15-L42` `CoordinateTests(unittest.TestCase)` [CLASS]：封装 `CoordinateTests` 相关数据或行为。
- `M L16-L30` `CoordinateTests.test_local_global_round_trip(self) -> None`：执行 `test local global round trip` 对应逻辑。 调用：`PatchMeta`, `global_to_local`, `local_to_global`, `self.assertEqual`。
- `M L32-L37` `CoordinateTests.test_global_to_patch_indices_returns_all_overlaps(self) -> None`：执行 `test global to patch indices returns all overlaps` 对应逻辑。 调用：`PatchMeta`, `global_to_patch_indices`, `self.assertEqual`。
- `M L39-L42` `CoordinateTests.test_feature_grid_round_trip(self) -> None`：执行 `test feature grid round trip` 对应逻辑。 调用：`feature_grid_to_image`, `image_to_feature_grid`, `self.assertEqual`。

## `src/traning/tests/test_cross_patch_ring.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`

- `C L11-L36` `CrossPatchRingTests(unittest.TestCase)` [CLASS]：封装 `CrossPatchRingTests` 相关数据或行为。
- `M L12-L36` `CrossPatchRingTests.test_ring_is_visible_from_multiple_patches_with_global_context(self) -> None`：执行 `test ring is visible from multiple patches with global context` 对应逻辑。 调用：`PatchStream`, `make_cross_patch_ring`, `sample_global_feature`, `self.assertGreaterEqual`, `self.assertTrue`, `stream.metas`。

## `src/traning/tests/test_cross_patch_slider.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`

- `C L11-L36` `CrossPatchSliderTests(unittest.TestCase)` [CLASS]：封装 `CrossPatchSliderTests` 相关数据或行为。
- `M L12-L36` `CrossPatchSliderTests.test_slider_spans_multiple_patches_with_shared_global_context(self) -> None`：执行 `test slider spans multiple patches with shared global context` 对应逻辑。 调用：`PatchStream`, `make_cross_patch_slider`, `sample_global_feature`, `self.assertGreater`, `self.assertGreaterEqual`, `stream.metas`。

## `src/traning/tests/test_cuda_config.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`

- `C L10-L33` `CudaConfigTests(unittest.TestCase)` [CLASS]：封装 `CudaConfigTests` 相关数据或行为。
- `M L11-L22` `CudaConfigTests.test_memory_defaults_enable_cuda_optimized_runtime(self) -> None`：执行 `test memory defaults enable cuda optimized runtime` 对应逻辑。 调用：`MemoryConfig`, `self.assertEqual`, `self.assertFalse`, `self.assertTrue`。
- `M L24-L33` `CudaConfigTests.test_loader_worker_options_require_workers(self) -> None`：执行 `test loader worker options require workers` 对应逻辑。 调用：`LoaderSettings`, `self.assertEqual`, `self.assertRaises`, `self.assertTrue`。

## `src/traning/tests/test_cuda_optimization.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.runtime`

- `C L20-L90` `CudaOptimizationTests(unittest.TestCase)` [CLASS]：封装 `CudaOptimizationTests` 相关数据或行为。
- `M L21-L30` `CudaOptimizationTests.test_cpu_runtime_keeps_cuda_only_options_inactive(self) -> None`：执行 `test cpu runtime keeps cuda only options inactive` 对应逻辑。 调用：`CudaRuntimeConfig`, `configure_torch_runtime`, `self.assertEqual`, `self.assertFalse`。
- `M L32-L39` `CudaOptimizationTests.test_grad_scaler_auto_is_disabled_without_fp16_cuda(self) -> None`：执行 `test grad scaler auto is disabled without fp16 cuda` 对应逻辑。 调用：`amp_uses_grad_scaler`, `create_grad_scaler`, `self.assertFalse`。
- `M L41-L48` `CudaOptimizationTests.test_tensor_to_device_preserves_cpu_contiguous_layout(self) -> None`：执行 `test tensor to device preserves cpu contiguous layout` 对应逻辑。 调用：`self.assertTrue`, `tensor_to_device`。
- `M L50-L61` `CudaOptimizationTests.test_cpu_memory_budget_reports_system_reserve(self) -> None`：执行 `test cpu memory budget reports system reserve` 对应逻辑。 调用：`enforce_runtime_memory_budget`, `self.assertEqual`, `self.assertGreater`, `self.assertIsNone`。
- `M L63-L72` `CudaOptimizationTests.test_cpu_memory_budget_rejects_unavailable_reserve(self) -> None`：执行 `test cpu memory budget rejects unavailable reserve` 对应逻辑。 调用：`enforce_runtime_memory_budget`, `self.assertRaises`。
- `M L74-L90` `CudaOptimizationTests.test_cuda_channels_last_when_available(self) -> None`：执行 `test cuda channels last when available` 对应逻辑。 调用：`module_to_device`, `self.assertTrue`, `self.skipTest`, `tensor_to_device`。

## `src/traning/tests/test_discovery.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`

- `F L11-L36` `_write_segment(root: Path, item_name: str, segment_id: str) -> None` [IO-W]：写入 `segment` 对应的数据或结果。
- `C L39-L60` `DiscoverySplitTests(unittest.TestCase)` [CLASS]：封装 `DiscoverySplitTests` 相关数据或行为。
- `M L40-L49` `DiscoverySplitTests.test_include_items_filters_records_before_loading(self) -> None`：执行 `test include items filters records before loading` 对应逻辑。 调用：`_write_segment`, `discover_segments`, `self.assertEqual`。
- `M L51-L60` `DiscoverySplitTests.test_exclude_items_removes_records(self) -> None`：执行 `test exclude items removes records` 对应逻辑。 调用：`_write_segment`, `discover_segments`, `self.assertEqual`。

## `src/traning/tests/test_env_check.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L11-L24` `EnvironmentCheckTests(unittest.TestCase)` [CLASS]：封装 `EnvironmentCheckTests` 相关数据或行为。
- `M L12-L17` `EnvironmentCheckTests.test_collect_environment_report_is_non_destructive(self) -> None`：执行 `test collect environment report is non destructive` 对应逻辑。 调用：`self.assertIsNotNone`, `self.assertTrue`。
- `M L19-L24` `EnvironmentCheckTests.test_required_package_specs_are_reported(self) -> None`：执行 `test required package specs are reported` 对应逻辑。 调用：`self.assertTrue`。

## `src/traning/tests/test_gallery_schema.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.state`

- `C L10-L54` `BatchGalleryRequestTests(unittest.TestCase)` [CLASS]：封装 `BatchGalleryRequestTests` 相关数据或行为。
- `M L11-L32` `BatchGalleryRequestTests.test_frame_evaluation_accepts_error_attribution(self) -> None`：执行 `test frame evaluation accepts error attribution` 对应逻辑。 调用：`self.assertEqual`。
- `M L34-L54` `BatchGalleryRequestTests.test_trials_must_share_score_version(self) -> None`：执行 `test trials must share score version` 对应逻辑。 调用：`self.assertRaises`。

## `src/traning/tests/test_gated_fusion.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`, `traning.lib.models.local_encoder`

- `C L12-L33` `GatedFusionTests(unittest.TestCase)` [CLASS]：封装 `GatedFusionTests` 相关数据或行为。
- `M L13-L33` `GatedFusionTests.test_forward_and_backward(self) -> None`：执行 `test forward and backward` 对应逻辑。 调用：`GatedSparseFusion`, `LocalFeatures`, `PatchMeta`, `self.assertEqual`, `self.assertIsNotNone`。

## `src/traning/tests/test_global_encoder.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.models`

- `C L10-L27` `GlobalEncoderTests(unittest.TestCase)` [CLASS]：封装 `GlobalEncoderTests` 相关数据或行为。
- `M L11-L23` `GlobalEncoderTests.test_lightweight_encoder_and_structure_head(self) -> None`：执行 `test lightweight encoder and structure head` 对应逻辑。 调用：`GlobalStructureHead`, `LightweightGlobalEncoder`, `self.assertEqual`。
- `M L25-L27` `GlobalEncoderTests.test_non_default_backbone_requires_external_setup(self) -> None`：执行 `test non default backbone requires external setup` 对应逻辑。 调用：`LightweightGlobalEncoder`, `self.assertRaises`。

## `src/traning/tests/test_global_sampling.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`

- `C L11-L19` `GlobalSamplingTests(unittest.TestCase)` [CLASS]：封装 `GlobalSamplingTests` 相关数据或行为。
- `M L12-L19` `GlobalSamplingTests.test_patch_position_changes_sampled_context(self) -> None`：执行 `test patch position changes sampled context` 对应逻辑。 调用：`PatchMeta`, `sample_global_feature`, `self.assertLess`。

## `src/traning/tests/test_local_encoder.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.models`

- `C L10-L23` `LocalEncoderTests(unittest.TestCase)` [CLASS]：封装 `LocalEncoderTests` 相关数据或行为。
- `M L11-L23` `LocalEncoderTests.test_forward_shapes_and_backward(self) -> None`：执行 `test forward shapes and backward` 对应逻辑。 调用：`SmallLocalEncoder`, `self.assertEqual`, `self.assertIn`, `self.assertIsNotNone`。

## `src/traning/tests/test_memory_smoke.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`, `traning.lib.runtime`

- `C L23-L93` `MemorySmokeTests(unittest.TestCase)` [CLASS]：封装 `MemorySmokeTests` 相关数据或行为。
- `M L24-L82` `MemorySmokeTests.run_smoke(self, device: torch.device) -> None`：执行 `run smoke` 对应逻辑。 调用：`CudaRuntimeConfig`, `GatedSparseFusion`, `PatchMeta`, `SmallLocalEncoder`, `SpatialPredictionHead`, `autocast_context`。
- `M L84-L85` `MemorySmokeTests.test_cpu_forward_backward_step(self) -> None`：执行 `test cpu forward backward step` 对应逻辑。 调用：`self.run_smoke`。
- `M L87-L93` `MemorySmokeTests.test_cuda_forward_backward_step_when_available(self) -> None`：执行 `test cuda forward backward step when available` 对应逻辑。 调用：`collect_memory_snapshot`, `self.assertIsNotNone`, `self.run_smoke`, `self.skipTest`。

## `src/traning/tests/test_model_export.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.model_export`

- `C L15-L43` `ModelExportTests(unittest.TestCase)` [CLASS]：封装 `ModelExportTests` 相关数据或行为。
- `M L16-L43` `ModelExportTests.test_export_model_artifact_copies_files_and_validates_hashes(self) -> None` [IO-W]：执行 `test export model artifact copies files and validates hashes` 对应逻辑。 调用：`ModelArtifactSpec`, `export_model_artifact`, `self.assertEqual`, `self.assertTrue`, `validate_model_artifact`。

## `src/traning/tests/test_optimization_module.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.optimization`, `traning.core.optimization.parameter_search`, `traning.lib.metrics`, `traning.state`

- `F L27-L35` `_circle_target(target_id: str='circle-1') -> TargetObject`：执行 `circle target` 对应逻辑。 调用：`TargetObject`。
- `C L38-L200` `OptimizationModuleTests(unittest.TestCase)` [CLASS]：封装 `OptimizationModuleTests` 相关数据或行为。
- `M L39-L55` `OptimizationModuleTests.test_score_trial_aggregates_point_slider_sequence_rules(self) -> None`：执行 `test score trial aggregates point slider sequence rules` 对应逻辑。 调用：`PredictedClick`, `SampleScoringInput`, `_circle_target`, `score_trial`, `self.assertAlmostEqual`, `self.assertEqual`。
- `M L57-L74` `OptimizationModuleTests.test_attribution_groups_temporal_and_decision_errors(self) -> None`：执行 `test attribution groups temporal and decision errors` 对应逻辑。 调用：`PredictedClick`, `SampleScoringInput`, `_circle_target`, `analyze_trial_attribution`, `score_trial`, `self.assertEqual`。
- `M L76-L111` `OptimizationModuleTests.test_parameter_plan_uses_attribution_and_asha_thresholds(self) -> None`：执行 `test parameter plan uses attribution and asha thresholds` 对应逻辑。 调用：`ParameterSearchConfig`, `PredictedClick`, `SampleScoringInput`, `TrialHistoryEntry`, `_circle_target`, `analyze_trial_attribution`。
- `M L113-L129` `OptimizationModuleTests.test_gallery_request_is_built_from_trial_score_report(self) -> None`：执行 `test gallery request is built from trial score report` 对应逻辑。 调用：`PredictedClick`, `SampleScoringInput`, `_circle_target`, `build_batch_gallery_request`, `score_trial`, `self.assertEqual`。
- `M L131-L168` `OptimizationModuleTests.test_curriculum_gate_and_hard_example_sampling(self) -> None`：执行 `test curriculum gate and hard example sampling` 对应逻辑。 调用：`PredictedClick`, `SampleScoringInput`, `_circle_target`, `analyze_trial_attribution`, `build_hard_example_sampling_plan`, `evaluate_curriculum_gate`。
- `M L170-L200` `OptimizationModuleTests.test_execute_optimization_plan_records_trial_and_job(self) -> None` [IO-W]：执行 `test execute optimization plan records trial and job` 对应逻辑。 调用：`OptimizationExecutorConfig`, `PredictedClick`, `SampleScoringInput`, `_circle_target`, `analyze_trial_attribution`, `execute_optimization_plan`。

## `src/traning/tests/test_patch_stream.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`

- `C L10-L46` `PatchStreamTests(unittest.TestCase)` [CLASS]：封装 `PatchStreamTests` 相关数据或行为。
- `M L11-L23` `PatchStreamTests.assert_full_coverage(self, width: int, height: int) -> None`：执行 `assert full coverage` 对应逻辑。 调用：`PatchStream`, `self.assertEqual`, `self.assertNotIn`, `self.assertTrue`, `stream.iter_patches`。
- `M L25-L27` `PatchStreamTests.test_common_resolutions_are_fully_covered(self) -> None`：执行 `test common resolutions are fully covered` 对应逻辑。 调用：`self.assert_full_coverage`。
- `M L29-L30` `PatchStreamTests.test_odd_dimensions_are_fully_covered(self) -> None`：执行 `test odd dimensions are fully covered` 对应逻辑。 调用：`self.assert_full_coverage`。
- `M L32-L42` `PatchStreamTests.test_small_image_is_padded(self) -> None`：执行 `test small image is padded` 对应逻辑。 调用：`PatchStream`, `self.assertEqual`, `self.assertTrue`, `stream.iter_patches`。
- `M L44-L46` `PatchStreamTests.test_invalid_overlap_raises(self) -> None`：执行 `test invalid overlap raises` 对应逻辑。 调用：`PatchStream`, `self.assertRaises`。

## `src/traning/tests/test_scoring.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.metrics`

- `C L14-L79` `PointScoringTests(unittest.TestCase)` [CLASS]：封装 `PointScoringTests` 相关数据或行为。
- `M L15-L16` `PointScoringTests.setUp(self) -> None`：执行 `setUp` 对应逻辑。 调用：`ScoreSpec`。
- `M L18-L24` `PointScoringTests.test_spatial_bonus_clamps_inside_sixty_percent(self) -> None`：执行 `test spatial bonus clamps inside sixty percent` 对应逻辑。 调用：`self.assertEqual`, `self.assertGreater`, `spatial_coefficient`。
- `M L26-L32` `PointScoringTests.test_spatial_comfort_and_zero_boundaries(self) -> None`：执行 `test spatial comfort and zero boundaries` 对应逻辑。 调用：`self.assertEqual`, `self.assertGreater`, `self.assertLessEqual`, `spatial_coefficient`。
- `M L34-L45` `PointScoringTests.test_temporal_boundaries_follow_v2_bands(self) -> None`：执行 `test temporal boundaries follow v2 bands` 对应逻辑。 调用：`self.assertEqual`, `self.assertGreater`, `self.assertLess`, `temporal_coefficient`。
- `M L47-L79` `PointScoringTests.test_point_pass_requires_space_and_time(self) -> None`：执行 `test point pass requires space and time` 对应逻辑。 调用：`score_point`, `self.assertAlmostEqual`, `self.assertFalse`, `self.assertTrue`。
- `C L82-L162` `SliderScoringTests(unittest.TestCase)` [CLASS]：封装 `SliderScoringTests` 相关数据或行为。
- `M L83-L95` `SliderScoringTests.test_slider_uses_first_path_point_as_missing_head(self) -> None`：执行 `test slider uses first path point as missing head` 对应逻辑。 调用：`score_slider`, `self.assertEqual`, `self.assertTrue`。
- `M L97-L139` `SliderScoringTests.test_slider_requires_bidirectional_path_match(self) -> None`：执行 `test slider requires bidirectional path match` 对应逻辑。 调用：`ScoreSpec`, `score_slider`, `self.assertEqual`, `self.assertFalse`, `self.assertLessEqual`, `self.assertTrue`。
- `M L141-L162` `SliderScoringTests.test_slider_corridor_uses_one_point_five_radius(self) -> None`：执行 `test slider corridor uses one point five radius` 对应逻辑。 调用：`score_slider`, `self.assertFalse`, `self.assertTrue`。

## `src/traning/tests/test_sequence_scoring.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.metrics`

- `C L13-L155` `ClickSequenceScoringTests(unittest.TestCase)` [CLASS]：封装 `ClickSequenceScoringTests` 相关数据或行为。
- `M L14-L41` `ClickSequenceScoringTests.test_first_passing_hit_resolves_target_once(self) -> None`：执行 `test first passing hit resolves target once` 对应逻辑。 调用：`PredictedClick`, `TargetObject`, `score_click_sequence`, `self.assertEqual`。
- `M L43-L64` `ClickSequenceScoringTests.test_failed_hit_keeps_target_active_for_later_click(self) -> None`：执行 `test failed hit keeps target active for later click` 对应逻辑。 调用：`PredictedClick`, `TargetObject`, `score_click_sequence`, `self.assertEqual`, `self.assertIn`。
- `M L66-L84` `ClickSequenceScoringTests.test_early_click_is_attributed_to_temporal_parameters(self) -> None`：执行 `test early click is attributed to temporal parameters` 对应逻辑。 调用：`PredictedClick`, `TargetObject`, `score_click_sequence`, `self.assertEqual`。
- `M L86-L117` `ClickSequenceScoringTests.test_overlapping_targets_resolve_by_earliest_active_target(self) -> None`：执行 `test overlapping targets resolve by earliest active target` 对应逻辑。 调用：`PredictedClick`, `TargetObject`, `score_click_sequence`, `self.assertEqual`。
- `M L119-L155` `ClickSequenceScoringTests.test_click_frequency_limit_blocks_high_rate_hits(self) -> None`：执行 `test click frequency limit blocks high rate hits` 对应逻辑。 调用：`PredictedClick`, `SequenceScoreSpec`, `TargetObject`, `score_click_sequence`, `self.assertEqual`, `self.assertTrue`。

## `src/traning/tests/test_spatial_decode.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`, `traning.lib.training`

- `F L18-L39` `_prediction(*, height: int=16, width: int=16, embedding_dim: int=4) -> SpatialPrediction`：执行 `prediction` 对应逻辑。 调用：`SpatialPrediction`。
- `C L42-L170` `SpatialDecodeTests(unittest.TestCase)` [CLASS]：封装 `SpatialDecodeTests` 相关数据或行为。
- `M L43-L77` `SpatialDecodeTests.test_canvas_decodes_global_candidate_with_offset_and_type(self) -> None`：执行 `test canvas decodes global candidate with offset and type` 对应逻辑。 调用：`PatchMeta`, `SpatialPredictionCanvas`, `_prediction`, `canvas.to_maps`, `canvas.write_patch`, `decode_spatial_candidates`。
- `M L79-L104` `SpatialDecodeTests.test_padding_region_is_not_written_to_global_canvas(self) -> None`：执行 `test padding region is not written to global canvas` 对应逻辑。 调用：`PatchMeta`, `SpatialPredictionCanvas`, `_prediction`, `canvas.to_maps`, `canvas.write_patch`, `decode_spatial_candidates`。
- `M L106-L130` `SpatialDecodeTests.test_decode_applies_nms(self) -> None`：执行 `test decode applies nms` 对应逻辑。 调用：`PatchMeta`, `SpatialPredictionCanvas`, `_prediction`, `canvas.to_maps`, `canvas.write_patch`, `decode_spatial_candidates`。
- `M L132-L170` `SpatialDecodeTests.test_decode_slider_paths_recovers_ordered_polyline(self) -> None`：执行 `test decode slider paths recovers ordered polyline` 对应逻辑。 调用：`SpatialPredictionMaps`, `decode_slider_paths`, `self.assertAlmostEqual`, `self.assertEqual`, `self.assertFalse`, `self.assertLess`。

## `src/traning/tests/test_spatial_inference.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.spatial`

- `F L16-L58` `_tiny_settings() -> Settings`：执行 `tiny settings` 对应逻辑。 调用：`Settings`。
- `C L61-L91` `SpatialInferenceTests(unittest.TestCase)` [CLASS]：封装 `SpatialInferenceTests` 相关数据或行为。
- `M L62-L91` `SpatialInferenceTests.test_cpu_single_frame_inference_reports_cpu_gpu_split(self) -> None`：执行 `test cpu single frame inference reports cpu gpu split` 对应逻辑。 调用：`_tiny_settings`, `result.as_summary`, `run_spatial_frame_inference`, `self.assertEqual`, `self.assertIn`, `self.assertLessEqual`。

## `src/traning/tests/test_spatial_model.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.models`

- `C L10-L23` `SpatialModelTests(unittest.TestCase)` [CLASS]：封装 `SpatialModelTests` 相关数据或行为。
- `M L11-L23` `SpatialModelTests.test_prediction_head_outputs_all_required_tasks(self) -> None`：执行 `test prediction head outputs all required tasks` 对应逻辑。 调用：`SpatialPredictionHead`, `self.assertEqual`。

## `src/traning/tests/test_spatial_targets.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`, `traning.lib.training`

- `C L10-L86` `SpatialTargetTests(unittest.TestCase)` [CLASS]：封装 `SpatialTargetTests` 相关数据或行为。
- `M L11-L45` `SpatialTargetTests.test_circle_target_contains_center_and_approach_ring(self) -> None`：执行 `test circle target contains center and approach ring` 对应逻辑。 调用：`PatchMeta`, `build_spatial_loss_targets`, `self.assertGreater`, `self.assertIn`。
- `M L47-L72` `SpatialTargetTests.test_slider_target_contains_body_direction_head_and_tail(self) -> None`：执行 `test slider target contains body direction head and tail` 对应逻辑。 调用：`PatchMeta`, `build_spatial_loss_targets`, `self.assertGreater`, `self.assertIn`, `self.assertLess`。
- `M L74-L86` `SpatialTargetTests.test_spinner_target_marks_valid_patch_area(self) -> None`：执行 `test spinner target marks valid patch area` 对应逻辑。 调用：`PatchMeta`, `build_spatial_loss_targets`, `self.assertGreater`, `self.assertIn`, `self.assertLess`。

## `src/traning/tests/test_spatial_trainer.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.spatial`

- `C L14-L85` `SpatialTrainerTests(unittest.TestCase)` [CLASS]：封装 `SpatialTrainerTests` 相关数据或行为。
- `M L15-L85` `SpatialTrainerTests.test_cpu_single_step_with_synthetic_sample(self) -> None`：执行 `test cpu single step with synthetic sample` 对应逻辑。 调用：`Settings`, `run_spatial_training`, `self.assertEqual`, `self.assertTrue`。

## `src/traning/tests/test_temporal_dataset.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.decision`, `traning.core.temporal`

- `F L18-L37` `_record(sample_key: str, frame_index: int, *, candidates: list[dict] | None=None, temporal_target: dict | None=None) -> dict`：执行 `record` 对应逻辑。
- `F L40-L65` `_candidate(score: float, *, x: float=25.0, y: float=10.0, candidate_id: int=0) -> dict`：执行 `candidate` 对应逻辑。
- `F L68-L82` `_write_cache(path: Path, records: list[dict]) -> None` [IO-W]：写入 `cache` 对应的数据或结果。
- `C L85-L147` `TemporalDatasetTests(unittest.TestCase)` [CLASS]：封装 `TemporalDatasetTests` 相关数据或行为。
- `M L86-L93` `TemporalDatasetTests.test_loads_candidate_cache_records(self) -> None`：执行 `test loads candidate cache records` 对应逻辑。 调用：`_candidate`, `_record`, `_write_cache`, `load_candidate_cache_records`, `self.assertEqual`。
- `M L95-L117` `TemporalDatasetTests.test_encodes_fixed_windows_without_crossing_samples(self) -> None`：执行 `test encodes fixed windows without crossing samples` 对应逻辑。 调用：`TemporalCandidateWindowDataset`, `TemporalFeatureSpec`, `_candidate`, `_record`, `self.assertEqual`, `self.assertFalse`。
- `M L119-L147` `TemporalDatasetTests.test_uses_explicit_temporal_target_when_present(self) -> None`：执行 `test uses explicit temporal target when present` 对应逻辑。 调用：`TemporalCandidateWindowDataset`, `TemporalFeatureSpec`, `_candidate`, `_record`, `self.assertEqual`, `self.assertTrue`。

## `src/traning/tests/test_temporal_decision.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.decision`, `traning.core.temporal`

- `F L15-L53` `_record(frame_index: int) -> dict`：执行 `record` 对应逻辑。
- `F L56-L65` `_write_cache(path: Path) -> None` [IO-W]：写入 `cache` 对应的数据或结果。 调用：`_record`。
- `C L68-L103` `TemporalDecisionTests(unittest.TestCase)` [CLASS]：封装 `TemporalDecisionTests` 相关数据或行为。
- `M L69-L103` `TemporalDecisionTests.test_train_then_run_decision(self) -> None` [IO-R]：执行 `test train then run decision` 对应逻辑。 调用：`Settings`, `_write_cache`, `run_temporal_decision`, `run_temporal_training`, `self.assertEqual`, `self.assertIn`。

## `src/traning/tests/test_temporal_trainer.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.decision`, `traning.core.temporal`

- `F L15-L45` `_record(frame_index: int) -> dict`：执行 `record` 对应逻辑。
- `F L48-L62` `_write_cache(path: Path) -> None` [IO-W]：写入 `cache` 对应的数据或结果。 调用：`_record`。
- `C L65-L90` `TemporalTrainerTests(unittest.TestCase)` [CLASS]：封装 `TemporalTrainerTests` 相关数据或行为。
- `M L66-L90` `TemporalTrainerTests.test_cpu_temporal_training_smoke(self) -> None`：执行 `test cpu temporal training smoke` 对应逻辑。 调用：`Settings`, `_write_cache`, `run_temporal_training`, `self.assertEqual`, `self.assertGreater`, `self.assertTrue`。
