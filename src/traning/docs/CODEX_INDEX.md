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
        -> core/optimization (评分、错误归因、参数搜索、SQLite 记录和多目标排序)
        -> core/result_export (结果可视化与图集导出)
        -> core/model_export (训练模型导出与迁移边界)
        -> start/checks (完整训练启动前自检)
tests/startup_checks/runner.py -> settings/runtime/data/core startup checks
tests/full_checks/runner.py -> full pytest checks
        -> lib/data | lib/models | lib/training | lib/metrics | lib/runtime | lib/visualization
        -> state (run / experiment / checkpoint metadata)
```

## Core 入口

| key | Core 入口 | 当前状态 |
|---|---|---|
| `dataset_import` | `core/dataset_import` | 训练集导入、检查、Dataset/DataLoader 已实现 |
| `spatial` | `core/spatial` | 空间训练和单帧推理已实现 |
| `temporal` | `core/temporal` | 候选缓存窗口和首版训练 smoke 已实现 |
| `decision` | `core/decision` | 候选缓存和训练阶段编排已实现 |
| `optimization` | `core/optimization` | 评分、归因、参数搜索、SQLite trial store 和多目标排序已实现 |
| `result_export` | `core/result_export` | 结果可视化和图集导出已实现 |
| `model_export` | `core/model_export` | 训练模型导出迁移边界已建立 |

快速查询：`python project_index/build_index.py --lookup 符号名`。

## 符号索引

覆盖 `141` 个 Python 文件、`801` 个命名函数/方法、`188` 个类。匿名 lambda 不单独列出。

图例：`F` 模块函数，`M` 方法，`N` 嵌套函数，`C` 类；`IO-R/IO-W` 文件读写，`DB` 数据库，`PROCESS` 外部进程。

## `src/traning/conf/settings.py`

职责：训练配置模型与 YAML 加载；解析数据集路径、item 划分、颜色 cue、候选缓存、点击频率上限并校验采样和分块参数。
工程依赖：`package.coordinates`

- `C L27-L28` `SettingsError(Exception)` [CLASS]：封装 `SettingsError` 相关数据或行为。
- `C L31-L33` `RuntimeSettings(BaseModel)` [CLASS]：封装 `RuntimeSettings` 相关数据或行为。
- `C L36-L47` `InputSettings(BaseModel)` [CLASS]：封装 `InputSettings` 相关数据或行为。
- `M L44-L47` `InputSettings._positive_dimension(cls, value: int) -> int` [VALIDATOR]：执行 `positive dimension` 对应逻辑。
- `C L50-L68` `PlayfieldRectSettings(BaseModel)` [CLASS]：封装 `PlayfieldRectSettings` 相关数据或行为。
- `M L58-L61` `PlayfieldRectSettings._finite_number(cls, value: float) -> float` [VALIDATOR]：执行 `finite number` 对应逻辑。
- `M L65-L68` `PlayfieldRectSettings._positive_dimension(cls, value: float) -> float` [VALIDATOR]：执行 `positive dimension` 对应逻辑。
- `C L71-L84` `CoordinateTransformSettings(BaseModel)` [CLASS]：封装 `CoordinateTransformSettings` 相关数据或行为。
- `M L77-L84` `CoordinateTransformSettings.validate_transform(self) -> CoordinateTransformSettings` [VALIDATOR]：校验 `transform` 对应的数据或结果。
- `C L87-L117` `TilingConfig(BaseModel)` [CLASS]：封装 `TilingConfig` 相关数据或行为。
- `M L97-L100` `TilingConfig._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `M L104-L107` `TilingConfig._nonnegative_overlap(cls, value: int) -> int` [VALIDATOR]：执行 `nonnegative overlap` 对应逻辑。
- `M L110-L117` `TilingConfig.validate_tiling(self) -> TilingConfig` [VALIDATOR]：校验 `tiling` 对应的数据或结果。
- `C L120-L136` `LocalEncoderConfig(BaseModel)` [CLASS]：封装 `LocalEncoderConfig` 相关数据或行为。
- `M L133-L136` `LocalEncoderConfig._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `C L139-L157` `GlobalEncoderConfig(BaseModel)` [CLASS]：封装 `GlobalEncoderConfig` 相关数据或行为。
- `M L154-L157` `GlobalEncoderConfig._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `C L160-L180` `FusionConfig(BaseModel)` [CLASS]：封装 `FusionConfig` 相关数据或行为。
- `M L171-L174` `FusionConfig._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `M L177-L180` `FusionConfig.validate_attention_shape(self) -> FusionConfig` [VALIDATOR]：校验 `attention shape` 对应的数据或结果。
- `C L183-L195` `TemporalConfig(BaseModel)` [CLASS]：封装 `TemporalConfig` 相关数据或行为。
- `M L192-L195` `TemporalConfig._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `C L198-L209` `TemporalLossWeights(BaseModel)` [CLASS]：封装 `TemporalLossWeights` 相关数据或行为。
- `M L206-L209` `TemporalLossWeights._finite_nonnegative(cls, value: float) -> float` [VALIDATOR]：执行 `finite nonnegative` 对应逻辑。
- `C L212-L222` `SpatialConsistencyLossWeights(BaseModel)` [CLASS]：封装 `SpatialConsistencyLossWeights` 相关数据或行为。
- `M L219-L222` `SpatialConsistencyLossWeights._finite_nonnegative(cls, value: float) -> float` [VALIDATOR]：执行 `finite nonnegative` 对应逻辑。
- `C L225-L231` `TrainingSettings(BaseModel)` [CLASS]：封装 `TrainingSettings` 相关数据或行为。
- `C L234-L270` `MemoryConfig(BaseModel)` [CLASS]：封装 `MemoryConfig` 相关数据或行为。
- `M L253-L256` `MemoryConfig._finite_nonnegative_memory(cls, value: float) -> float` [VALIDATOR]：执行 `finite nonnegative memory` 对应逻辑。
- `M L260-L263` `MemoryConfig._positive_memory(cls, value: float) -> float` [VALIDATOR]：执行 `positive memory` 对应逻辑。
- `M L267-L270` `MemoryConfig._optional_positive_memory(cls, value: float | None) -> float | None` [VALIDATOR]：执行 `optional positive memory` 对应逻辑。
- `C L273-L298` `SMETConfig(BaseModel)` [CLASS]：封装 `SMETConfig` 相关数据或行为。
- `M L281-L284` `SMETConfig._sparsity(cls, value: float) -> float` [VALIDATOR]：执行 `sparsity` 对应逻辑。
- `M L288-L291` `SMETConfig._positive_interval(cls, value: int) -> int` [VALIDATOR]：执行 `positive interval` 对应逻辑。
- `M L295-L298` `SMETConfig._density(cls, value: float) -> float` [VALIDATOR]：执行 `density` 对应逻辑。
- `C L301-L352` `CandidateCacheSettings(BaseModel)` [CLASS]：封装 `CandidateCacheSettings` 相关数据或行为。
- `M L329-L332` `CandidateCacheSettings._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `M L336-L339` `CandidateCacheSettings._probability(cls, value: float) -> float` [VALIDATOR]：执行 `probability` 对应逻辑。
- `M L349-L352` `CandidateCacheSettings._nonnegative_float(cls, value: float) -> float` [VALIDATOR]：执行 `nonnegative float` 对应逻辑。
- `C L355-L393` `OptimizationSettings(BaseModel)` [CLASS]：封装 `OptimizationSettings` 相关数据或行为。
- `M L378-L381` `OptimizationSettings._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `M L385-L393` `OptimizationSettings._finite_objective_weights(cls, value: dict[str, float]) -> dict[str, float]` [VALIDATOR]：执行 `finite objective weights` 对应逻辑。
- `C L396-L432` `LoaderSettings(BaseModel)` [CLASS]：封装 `LoaderSettings` 相关数据或行为。
- `M L407-L410` `LoaderSettings._positive_batch_size(cls, value: int) -> int` [VALIDATOR]：执行 `positive batch size` 对应逻辑。
- `M L414-L417` `LoaderSettings._nonnegative_workers(cls, value: int) -> int` [VALIDATOR]：执行 `nonnegative workers` 对应逻辑。
- `M L421-L424` `LoaderSettings._optional_positive_prefetch(cls, value: int | None) -> int | None` [VALIDATOR]：执行 `optional positive prefetch` 对应逻辑。
- `M L427-L432` `LoaderSettings.validate_worker_options(self) -> LoaderSettings` [VALIDATOR]：校验 `worker options` 对应的数据或结果。
- `C L435-L443` `EvaluationSettings(BaseModel)` [CLASS]：封装 `EvaluationSettings` 相关数据或行为。
- `M L440-L443` `EvaluationSettings._nonnegative_interval(cls, value: float) -> float` [VALIDATOR]：执行 `nonnegative interval` 对应逻辑。
- `C L446-L461` `VisualizationSettings(BaseModel)` [CLASS]：封装 `VisualizationSettings` 相关数据或行为。
- `M L458-L461` `VisualizationSettings._positive_interval(cls, value: int) -> int` [VALIDATOR]：执行 `positive interval` 对应逻辑。
- `C L464-L551` `DataInputSettings(BaseModel)` [CLASS]：封装 `DataInputSettings` 相关数据或行为。
- `M L488-L491` `DataInputSettings._positive_fps(cls, value: float) -> float` [VALIDATOR]：执行 `positive fps` 对应逻辑。
- `M L495-L498` `DataInputSettings._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `M L502-L505` `DataInputSettings._optional_positive_integer(cls, value: int | None) -> int | None` [VALIDATOR]：执行 `optional positive integer` 对应逻辑。
- `M L509-L512` `DataInputSettings._nonnegative_visibility(cls, value: float) -> float` [VALIDATOR]：执行 `nonnegative visibility` 对应逻辑。
- `M L524-L530` `DataInputSettings._unique_nonempty_strings(cls, value: tuple[str, ...]) -> tuple[str, ...]` [VALIDATOR]：执行 `unique nonempty strings` 对应逻辑。
- `M L533-L545` `DataInputSettings.validate_item_splits(self) -> DataInputSettings` [VALIDATOR]：校验 `item splits` 对应的数据或结果。
- `M L547-L551` `DataInputSettings.validate_tiling(self) -> None`：校验 `tiling` 对应的数据或结果。
- `C L554-L598` `Settings(BaseSettings)` [CLASS]：封装 `Settings` 相关数据或行为。
- `M L585-L598` `Settings.settings_customise_sources(cls, settings_cls: type[BaseSettings], init_settings: PydanticBaseSettingsSource, env_settings: PydanticBaseSettingsSource, dotenv_settings: PydanticBaseSettingsSource, file_secret_settings: PydanticBaseSettingsSource) -> tuple[PydanticBaseSettingsSource, ...]`：执行 `settings customise sources` 对应逻辑。
- `F L601-L610` `_read_config(config_path: Path) -> dict[str, Any]` [IO-R]：读取 `config` 对应的数据或结果。 调用：`SettingsError`。
- `F L613-L654` `_resolve_paths(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]`：解析并定位 `paths` 对应的数据或结果。
- `F L657-L666` `load_settings(config_path: Path | None=None) -> Settings`：加载 `settings` 对应的数据或结果。 调用：`Settings`, `SettingsError`, `_read_config`, `_resolve_paths`, `settings.data_input.validate_tiling`, `settings.tiling.validate_tiling`。

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
工程依赖：`package.coordinates`, `traning.conf`, `traning.core.dataset_import.preflight`, `traning.lib.data`

- `F L11-L44` `build_dataset(settings: Settings, *, split: DataSplit='train') -> SegmentFrameDataset`：构建并返回 `dataset` 对应的数据或结果。 调用：`SegmentFrameDataset`, `discover_data_input`。
- `F L47-L68` `build_dataloader(settings: Settings, *, split: DataSplit='train', shuffle: bool | None=None) -> DataLoader`：构建并返回 `dataloader` 对应的数据或结果。 调用：`build_dataset`。

## `src/traning/core/dataset_import/preflight.py`

职责：读取 split manifest 或旧 item 配置，扫描训练片段并生成数量、类别、维度和问题报告。
工程依赖：`package.dataset_split`, `traning.conf`, `traning.lib.data`, `traning.lib.data.models`

- `C L15-L28` `DataInputReport` [CLASS]：封装 `DataInputReport` 相关数据或行为。
- `M L27-L28` `DataInputReport.ok(self) -> bool` [PROPERTY]：执行 `ok` 对应逻辑。
- `F L31-L37` `_combine_item_filters(base_items: tuple[str, ...], split_items: tuple[str, ...]) -> tuple[str, ...]`：执行 `combine item filters` 对应逻辑。
- `F L40-L55` `_split_items(config, split: DataSplit) -> tuple[str, ...]`：执行 `split items` 对应逻辑。
- `F L58-L82` `discover_data_input(settings: Settings, *, split: DataSplit='all') -> DiscoveryResult`：执行 `discover data input` 对应逻辑。 调用：`DatasetIssue`, `DiscoveryResult`, `_combine_item_filters`, `_split_items`, `discover_segments`。
- `F L85-L117` `inspect_data_input(settings: Settings, *, split: DataSplit='all') -> DataInputReport`：执行 `inspect data input` 对应逻辑。 调用：`DataInputReport`, `_distribution_and_topology`, `discover_data_input`。
- `F L120-L166` `_distribution_and_topology(records) -> tuple[dict[str, object], tuple[str, ...]]`：执行 `distribution and topology` 对应逻辑。 调用：`_high_density_windows`, `_kind`, `_slider_topology_issues`, `_summary`, `inter_object_intervals.append`, `slider_durations.append`。
- `F L169-L172` `_summary(values: list[float]) -> dict[str, float | int | None]`：执行 `summary` 对应逻辑。
- `F L175-L181` `_kind(value: str) -> str`：执行 `kind` 对应逻辑。
- `F L184-L190` `_high_density_windows(objects) -> int`：执行 `high density windows` 对应逻辑。
- `F L193-L210` `_slider_topology_issues(record, item) -> tuple[str, ...]`：执行 `slider topology issues` 对应逻辑。 调用：`_self_intersects`, `_touches_branch`, `issues.append`。
- `F L213-L221` `_self_intersects(path: tuple[tuple[float, float], ...]) -> bool`：执行 `self intersects` 对应逻辑。 调用：`_segments_intersect`。
- `F L224-L231` `_touches_branch(path: tuple[tuple[float, float], ...]) -> bool`：执行 `touches branch` 对应逻辑。
- `F L234-L238` `_segments_intersect(a, b, c, d) -> bool`：执行 `segments intersect` 对应逻辑。 调用：`orient`。
- `N L235-L236` `_segments_intersect.orient(p, q, r) -> float`：执行 `orient` 对应逻辑。

## `src/traning/core/decision/generator.py`

职责：离线空间候选缓存生成器；逐帧调用空间推理并写 JSONL/manifest/temporal target。
工程依赖：`traning.conf`, `traning.core.dataset_import`, `traning.core.spatial`, `traning.lib.coordinates`, `traning.lib.training`, `traning.lib.training.spatial_decode`, `traning.state.versioning`

- `C L28-L52` `CandidateCacheBuildResult` [CLASS]：封装 `CandidateCacheBuildResult` 相关数据或行为。
- `M L40-L52` `CandidateCacheBuildResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `F L55-L174` `generate_candidate_cache(settings: Settings, *, output_dir: Path, device: torch.device, split: DataSplit='train', max_frames: int | None=None, patch_limit: int | None=None, max_candidates: int | None=None, score_threshold: float | None=None, nms_radius_px: float | None=None, slider_threshold: float | None=None, max_slider_paths: int | None=None, dataset: Sequence[Mapping[str, Any]] | None=None) -> CandidateCacheBuildResult` [IO-W]：执行 `generate candidate cache` 对应逻辑。 调用：`CandidateCacheBuildResult`, `build_candidate_cache_record`, `build_dataset`, `run_spatial_frame_inference`, `version_manifest`。
- `F L177-L278` `build_candidate_cache_record(sample: Mapping[str, Any], candidates: Sequence[SpatialCandidate], slider_paths: Sequence[SliderPathCandidate], *, frame_width: int, frame_height: int, device: str, patches_processed: int, frame_channels: int, save_dtype: str, low_confidence_threshold: float, close_score_margin: float, slider_attach_distance_px: float, action_window_ms: float=25.0, settings: Settings | None=None) -> dict[str, Any]`：构建并返回 `candidate cache record` 对应的数据或结果。 调用：`_apply_candidate_reviews`, `_apply_local_refinement`, `_build_temporal_target`, `_candidate_ambiguity_reasons`, `_cast_embedding`, `_nearest_slider_path`。
- `F L281-L329` `_build_temporal_target(sample: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], *, frame_width: int, frame_height: int, action_window_ms: float, settings: Settings | None=None) -> dict[str, Any]`：构建 `temporal target` 对应的数据或结果。 调用：`_nearest_candidate`, `_optional_float`, `_select_temporal_object`, `transform_from_settings_or_sample`。
- `F L332-L361` `_select_temporal_object(objects: object, *, timestamp_ms: float, action_window_ms: float) -> dict[str, Any] | None`：选择 `temporal object` 对应的数据或结果。 调用：`_temporal_target_for_object`。
- `F L364-L423` `_temporal_target_for_object(item: Mapping[str, Any], *, timestamp_ms: float, action_window_ms: float) -> dict[str, Any] | None`：执行 `temporal target for object` 对应逻辑。 调用：`_click_duration_ms`, `_is_release_frame`, `_object_kind`, `_object_osu_point`, `_optional_float`, `_repeat_boundaries`。
- `F L426-L427` `_click_duration_ms(action_window_ms: float) -> float`：执行 `click duration ms` 对应逻辑。
- `F L430-L437` `_is_release_frame(timestamp_ms: float, *, start_ms: float, action_window_ms: float) -> bool`：判断是否 `release frame` 对应的数据或结果。 调用：`_click_duration_ms`。
- `F L440-L461` `_repeat_boundaries(item: Mapping[str, Any], *, start_ms: float, end_ms: float) -> tuple[tuple[float, str, tuple[float, float]], ...]`：执行 `repeat boundaries` 对应逻辑。 调用：`_object_osu_point`, `_optional_float`, `_slider_tail_point`, `boundaries.append`。
- `F L464-L470` `_slider_tail_point(item: Mapping[str, Any]) -> tuple[float, float] | None`：执行 `slider tail point` 对应逻辑。 调用：`_object_osu_point`。
- `F L473-L484` `_object_osu_point(item: Mapping[str, Any]) -> tuple[float, float] | None`：执行 `object osu point` 对应逻辑。 调用：`_object_kind`。
- `F L487-L493` `_object_kind(item: Mapping[str, Any]) -> str`：执行 `object kind` 对应逻辑。
- `F L496-L511` `_nearest_candidate(candidates: Sequence[Mapping[str, Any]], point: tuple[float, float]) -> Mapping[str, Any] | None`：执行 `nearest candidate` 对应逻辑。 调用：`_optional_float`, `_point_distance`。
- `F L514-L533` `_candidate_ambiguity_reasons(index: int, candidates: Sequence[SpatialCandidate], slider_path: SliderPathCandidate | None, *, low_confidence_threshold: float, close_score_margin: float) -> tuple[str, ...]`：执行 `candidate ambiguity reasons` 对应逻辑。 调用：`_has_close_neighbor`, `reasons.append`。
- `F L536-L582` `_apply_candidate_reviews(rows: list[dict[str, Any]], *, slider_rows: Sequence[Mapping[str, Any]], frame_width: int, frame_height: int, enabled: bool, max_candidates: int) -> None`：应用 `candidate reviews` 对应的数据或结果。 调用：`_review_candidate_locally`。
- `F L585-L634` `_apply_local_refinement(rows: list[dict[str, Any]], *, slider_rows: Sequence[Mapping[str, Any]], frame_width: int, frame_height: int, enabled: bool, top_k: int, radius_px: float) -> None`：应用 `local refinement` 对应的数据或结果。 调用：`_refined_candidate_xy`。
- `F L637-L669` `_review_candidate_locally(row: Mapping[str, Any], *, slider_rows: Sequence[Mapping[str, Any]], frame_width: int, frame_height: int) -> dict[str, Any]`：执行 `review candidate locally` 对应逻辑。 调用：`_distance_to_polyline`, `_local_evidence_score`, `_optional_float`, `_polyline_from_row`, `_row_slider_path`。
- `F L672-L686` `_local_evidence_score(row: Mapping[str, Any]) -> float`：执行 `local evidence score` 对应逻辑。 调用：`_optional_float`。
- `F L689-L707` `_refined_candidate_xy(row: Mapping[str, Any], *, slider_rows: Sequence[Mapping[str, Any]], current_xy: tuple[float, float], radius_px: float) -> tuple[float, float]`：执行 `refined candidate xy` 对应逻辑。 调用：`_nearest_polyline_point`, `_point_distance`, `_point_from_row`, `_polyline_from_row`, `_row_slider_path`。
- `F L710-L720` `_row_slider_path(row: Mapping[str, Any], slider_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None`：执行 `row slider path` 对应逻辑。
- `F L723-L730` `_point_from_row(value: Any) -> tuple[float, float] | None`：执行 `point from row` 对应逻辑。
- `F L733-L742` `_polyline_from_row(row: Mapping[str, Any]) -> tuple[tuple[float, float], ...]`：执行 `polyline from row` 对应逻辑。 调用：`_point_from_row`。
- `F L745-L761` `_nearest_polyline_point(point: tuple[float, float], polyline: Sequence[tuple[float, float]]) -> tuple[float, float] | None`：执行 `nearest polyline point` 对应逻辑。 调用：`_point_distance`, `_project_point_to_segment`。
- `F L764-L779` `_project_point_to_segment(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]`：执行 `project point to segment` 对应逻辑。
- `F L782-L795` `_has_close_neighbor(index: int, candidates: Sequence[SpatialCandidate], *, margin: float) -> bool`：执行 `has close neighbor` 对应逻辑。
- `F L798-L815` `_nearest_slider_path(candidate: SpatialCandidate, paths: Sequence[SliderPathCandidate], *, max_distance: float) -> SliderPathCandidate | None`：执行 `nearest slider path` 对应逻辑。 调用：`_distance_to_polyline`。
- `F L818-L829` `_distance_to_polyline(point: tuple[float, float], polyline: Sequence[tuple[float, float]]) -> float`：执行 `distance to polyline` 对应逻辑。 调用：`_point_distance`, `_point_to_segment_distance`。
- `F L832-L847` `_point_to_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float`：执行 `point to segment distance` 对应逻辑。 调用：`_point_distance`。
- `F L850-L854` `_point_distance(first: tuple[float, float], second: tuple[float, float]) -> float`：执行 `point distance` 对应逻辑。
- `F L857-L863` `_cast_embedding(values: Sequence[float], save_dtype: str) -> list[float]`：执行 `cast embedding` 对应逻辑。
- `F L866-L869` `_optional_float(value: Any) -> float | None`：执行 `optional float` 对应逻辑。

## `src/traning/core/decision/pipeline.py`

职责：声明完整训练阶段表；先调用 start.checks 自检，再串接 data-check、空间训练、候选缓存、时序训练和决策导出。
工程依赖：`start.checks`, `traning.conf`, `traning.core.dataset_import`, `traning.core.decision.generator`, `traning.core.decision.runner`, `traning.core.optimization`, `traning.core.result_export`, `traning.core.spatial`, `traning.core.temporal`, `traning.lib.metrics`, `traning.state.versioning`

- `C L51-L53` `TrainingStage` [CLASS]：封装 `TrainingStage` 相关数据或行为。
- `C L57-L105` `FullTrainingRunConfig` [CLASS]：封装 `FullTrainingRunConfig` 相关数据或行为。
- `M L83-L105` `FullTrainingRunConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L109-L161` `FullTrainingEvaluationResult` [CLASS]：封装 `FullTrainingEvaluationResult` 相关数据或行为。
- `M L135-L161` `FullTrainingEvaluationResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L165-L222` `FullTrainingRunResult` [CLASS]：封装 `FullTrainingRunResult` 相关数据或行为。
- `M L176-L187` `FullTrainingRunResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`_data_input_report_dict`, `self.candidate_cache.as_dict`, `self.decision.as_dict`, `self.evaluation.as_dict`, `self.spatial.as_dict`, `self.startup_checks.as_dict`。
- `M L189-L222` `FullTrainingRunResult.as_summary(self) -> dict[str, Any]`：执行 `as summary` 对应逻辑。
- `F L235-L516` `run_full_training_pipeline(settings: Settings, *, config: FullTrainingRunConfig) -> FullTrainingRunResult` [IO-W]：执行 `run full training pipeline` 对应逻辑。 调用：`FullTrainingRunResult`, `_category_scores_from_report`, `_evaluate_training_outputs`, `_full_training_parameter_snapshot`, `_json_ready`, `_report_resource`。
- `F L519-L670` `_evaluate_training_outputs(settings: Settings, *, config: FullTrainingRunConfig, candidate_cache: CandidateCacheBuildResult, spatial: SpatialTrainingResult, temporal: TemporalTrainingResult, decision: TemporalDecisionRunResult) -> FullTrainingEvaluationResult` [IO-W]：执行 `evaluate training outputs` 对应逻辑。 调用：`OptimizationExecutorConfig`, `ParameterSearchConfig`, `SequenceScoreSpec`, `TrialScoreSpec`, `_evaluation_result_from_score`, `_json_ready`。
- `F L673-L714` `_evaluation_result_from_score(score_result: DecisionOutputScoreResult, *, report_path: Path, gallery_request_path: Path, gallery_status: str, gallery_output_dir: Path | None, gallery_saved_frame_count: int, attribution_path: Path | None, optimization_plan_path: Path | None, next_job_path: Path | None, gallery_warning: str | None, asha_action: str | None, asha_reasons: tuple[str, ...]) -> FullTrainingEvaluationResult`：执行 `evaluation result from score` 对应逻辑。 调用：`FullTrainingEvaluationResult`。
- `F L717-L727` `run_pipeline(settings: Settings | None=None, *, config: FullTrainingRunConfig | None=None) -> FullTrainingRunResult`：执行 `run pipeline` 对应逻辑。 调用：`FullTrainingRunConfig`, `_device_from_settings`, `load_settings`, `run_full_training_pipeline`。
- `F L730-L742` `_data_input_report_dict(report: DataInputReport) -> dict[str, Any]`：执行 `data input report dict` 对应逻辑。
- `F L745-L748` `_device_from_settings(settings: Settings) -> torch.device`：执行 `device from settings` 对应逻辑。
- `F L751-L763` `_json_ready(value: Any) -> Any`：执行 `json ready` 对应逻辑。 调用：`_json_ready`。
- `F L766-L809` `_full_training_parameter_snapshot(settings: Settings, *, config: FullTrainingRunConfig, spatial: SpatialTrainingResult, candidate_cache: CandidateCacheBuildResult, temporal: TemporalTrainingResult, decision: TemporalDecisionRunResult, evaluation: FullTrainingEvaluationResult) -> dict[str, Any]`：执行 `full training parameter snapshot` 对应逻辑。 调用：`_json_ready`, `_training_parameter_config_snapshot`。
- `F L812-L869` `_training_parameter_config_snapshot(settings: Settings, *, config: FullTrainingRunConfig) -> dict[str, Any]`：执行 `training parameter config snapshot` 对应逻辑。 调用：`_json_ready`。
- `F L872-L885` `_trial_outcome(evaluation: FullTrainingEvaluationResult) -> tuple[str, str, str, str | None]`：执行 `trial outcome` 对应逻辑。
- `F L888-L917` `_report_stage(reporter: TrainingReporter, stage_id: str, name: str, status: str, *, processed: int=0, total: int | None=None, output_path: Path | None=None, warnings: int=0, blocks_training: bool=False, error_reason: str | None=None, score: float | None=None, threshold: float | None=None) -> None`：执行 `report stage` 对应逻辑。 调用：`reporter.update_pipeline_stage`。
- `F L920-L924` `_report_resource(reporter: TrainingReporter) -> None`：执行 `report resource` 对应逻辑。
- `F L927-L944` `_category_scores_from_report(report_path: Path) -> dict[str, float]` [IO-R]：执行 `category scores from report` 对应逻辑。 调用：`groups.setdefault.append`。

## `src/traning/core/decision/runner.py`

职责：加载 temporal checkpoint 和候选缓存，导出逐帧动作决策 JSONL。
工程依赖：`traning.conf`, `traning.core.temporal`, `traning.lib.models`, `traning.lib.runtime`

- `C L33-L53` `TemporalDecisionRunResult` [CLASS]：封装 `TemporalDecisionRunResult` 相关数据或行为。
- `M L43-L53` `TemporalDecisionRunResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `F L56-L137` `run_temporal_decision(settings: Settings, *, cache_dir: Path, checkpoint_path: Path, output_dir: Path, device: torch.device) -> TemporalDecisionRunResult` [IO-W]：执行 `run temporal decision` 对应逻辑。 调用：`CausalTemporalModel`, `CudaRuntimeConfig`, `TemporalCandidateWindowDataset.from_cache_dir`, `TemporalDecisionRunResult`, `_decision_row`, `_load_checkpoint`。
- `F L140-L156` `_load_checkpoint(checkpoint_path: Path, *, device: torch.device) -> Mapping[str, Any]`：加载 `checkpoint` 对应的数据或结果。 调用：`torch.load`。
- `F L159-L199` `_decision_row(window: TemporalWindow, frame_index: int, output) -> dict[str, Any]`：执行 `decision row` 对应逻辑。

## `src/traning/core/full_flow/orchestrator.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`start.flow`, `start.samples`, `traning.conf`, `traning.core.full_flow.result`, `traning.core.full_flow.stages`, `traning.core.model_export`, `traning.core.training_inheritance`, `traning.core.training_ramp`, `traning.state.versioning`

- `C L47-L81` `FullFlowConfig` [CLASS]：封装 `FullFlowConfig` 相关数据或行为。
- `C L85-L113` `_FlowRuntime` [CLASS]：封装 `FlowRuntime` 相关数据或行为。
- `M L100-L101` `_FlowRuntime.state_path(self) -> Path` [PROPERTY]：执行 `state path` 对应逻辑。
- `M L104-L105` `_FlowRuntime.manifest_path(self) -> Path` [PROPERTY]：执行 `manifest path` 对应逻辑。
- `M L108-L109` `_FlowRuntime.report_json_path(self) -> Path` [PROPERTY]：执行 `report json path` 对应逻辑。
- `M L112-L113` `_FlowRuntime.report_markdown_path(self) -> Path` [PROPERTY]：执行 `report markdown path` 对应逻辑。
- `F L116-L196` `run_full_flow(config: FullFlowConfig) -> FullFlowResult`：执行 `run full flow` 对应逻辑。 调用：`_FlowRuntime`, `_base_manifest`, `_init_layout`, `_initial_stage_states`, `_mark_failed`, `_mark_interrupted`。
- `F L199-L256` `load_full_flow_status(output_root: Path=DEFAULT_FULL_FLOW_ROOT, *, run_id: str | None=None) -> FullFlowResult`：加载 `full flow status` 对应的数据或结果。 调用：`FullFlowResult`, `FullFlowStageState`, `_read_json`。
- `F L259-L337` `_run_startup_section(_runtime: _FlowRuntime, *, reporter) -> None`：执行 `run startup section` 对应逻辑。 调用：`_finish_stage`, `_persist`, `_select_device_name`, `_start_stage`, `_write_json`, `startup.as_dict`。
- `F L340-L388` `_run_resume_section(_runtime: _FlowRuntime, *, reporter)`：执行 `run resume section` 对应逻辑。 调用：`_finish_stage`, `_persist`, `_start_stage`, `_write_json`, `inheritance.as_dict`, `load_inheritance_package`。
- `F L391-L452` `_run_ramp_section(_runtime: _FlowRuntime, *, inheritance, reporter) -> None`：执行 `run ramp section` 对应逻辑。 调用：`_finish_stage`, `_last_training_record`, `_persist`, `_select_device_name`, `_stage_enabled`, `_stage_forced`。
- `F L455-L459` `_run_finalize_section(_runtime: _FlowRuntime) -> None`：执行 `run finalize section` 对应逻辑。 调用：`_finish_export_stage`, `_finish_inheritance_stage`, `_last_training_record`, `load_settings`。
- `F L462-L503` `_finish_export_stage(_runtime: _FlowRuntime, *, settings, record: Mapping[str, Any] | None) -> None`：执行 `finish export stage` 对应逻辑。 调用：`ModelArtifactSpec`, `_record_extra_files`, `_record_path`, `_report_full_flow_stage`, `collect_code_version`, `export_model_artifact`。
- `F L506-L545` `_finish_inheritance_stage(_runtime: _FlowRuntime, *, settings, record: Mapping[str, Any] | None) -> None`：执行 `finish inheritance stage` 对应逻辑。 调用：`_record_extra_files`, `_record_path`, `_report_full_flow_stage`, `create_inheritance_package`, `package.as_dict`, `stage.mark_finished`。
- `F L548-L563` `_mark_plan(_runtime: _FlowRuntime) -> None`：更新状态为 `plan` 对应的数据或结果。 调用：`_report_full_flow_stage`, `_selected_stage_ids`, `_stage_forced`, `state.mark_finished`。
- `F L566-L579` `_mark_training_skipped_for_dry_run(_runtime: _FlowRuntime) -> None`：更新状态为 `training skipped for dry run` 对应的数据或结果。 调用：`_report_full_flow_stage`, `mark_finished`。
- `F L582-L589` `_mark_failed(_runtime: _FlowRuntime, error: Exception) -> None`：更新状态为 `failed` 对应的数据或结果。 调用：`_report_full_flow_stage`, `state.mark_finished`。
- `F L592-L596` `_mark_interrupted(_runtime: _FlowRuntime, reason: str) -> None`：更新状态为 `interrupted` 对应的数据或结果。 调用：`_report_full_flow_stage`, `state.mark_finished`。
- `F L599-L674` `_persist(_runtime: _FlowRuntime, *, status: str, stop_reason: str | None=None) -> None`：执行 `persist` 对应逻辑。 调用：`_report_full_flow_stage`, `_write_json`, `_write_reports`, `report_stage.mark_finished`, `report_stage.mark_started`, `stage.as_dict`。
- `F L677-L697` `_result(_runtime: _FlowRuntime, *, status: str) -> FullFlowResult`：执行 `result` 对应逻辑。 调用：`FullFlowResult`, `utc_now`。
- `F L700-L721` `_base_manifest(config: FullFlowConfig, _runtime: _FlowRuntime) -> dict[str, Any]`：执行 `base manifest` 对应逻辑。 调用：`_dataset_fingerprint`, `_file_sha256`, `_selected_stage_ids`, `collect_code_version`, `collect_code_version.as_dict`, `load_settings`。
- `F L724-L728` `_initial_stage_states() -> dict[str, FullFlowStageState]`：执行 `initial stage states` 对应逻辑。 调用：`FullFlowStageState`。
- `F L731-L748` `_publish_initial_dashboard_stages(_runtime: _FlowRuntime) -> None`：执行 `publish initial dashboard stages` 对应逻辑。 调用：`_selected_stage_ids`, `reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L751-L753` `_report_resource_snapshot(reporter: TrainingReporter) -> None`：执行 `report resource snapshot` 对应逻辑。
- `F L756-L777` `_start_stage(_runtime: _FlowRuntime, stage_id: str, reporter) -> None`：执行 `start stage` 对应逻辑。 调用：`_phase_for_full_flow_stage`, `_report_full_flow_stage`, `_stage_enabled`, `mark_finished`, `reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L780-L800` `_finish_stage(_runtime: _FlowRuntime, stage_id: str, status, *, result: Mapping[str, Any] | None=None, warnings: tuple[str, ...]=(), artifacts: tuple[str, ...]=(), restored: bool=False) -> None`：执行 `finish stage` 对应逻辑。 调用：`_report_full_flow_stage`, `_stage_forced`, `mark_finished`。
- `F L803-L834` `_report_full_flow_stage(_runtime: _FlowRuntime, stage_id: str) -> None`：执行 `report full flow stage` 对应逻辑。 调用：`_dashboard_status`, `_optional_int`, `_phase_for_full_flow_stage`, `reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L837-L850` `_dashboard_status(status: str) -> str`：执行 `dashboard status` 对应逻辑。
- `F L853-L879` `_phase_for_full_flow_stage(stage_id: str) -> PipelinePhase`：执行 `phase for full flow stage` 对应逻辑。
- `F L882-L888` `_optional_int(value: object, *, default: int | None=None) -> int | None`：执行 `optional int` 对应逻辑。
- `F L891-L892` `_stage_enabled(_runtime: _FlowRuntime, stage_id: str) -> bool`：执行 `stage enabled` 对应逻辑。 调用：`_selected_stage_ids`。
- `F L895-L896` `_stage_forced(config: FullFlowConfig, stage_id: str) -> bool`：执行 `stage forced` 对应逻辑。 调用：`validate_stage_id`。
- `F L899-L908` `_selected_stage_ids(config: FullFlowConfig) -> tuple[str, ...]`：执行 `selected stage ids` 对应逻辑。 调用：`validate_stage_id`。
- `F L911-L926` `_validate_config(config: FullFlowConfig) -> None`：校验 `config` 对应的数据或结果。 调用：`_selected_stage_ids`, `validate_stage_id`。
- `F L929-L941` `_init_layout(output_dir: Path) -> None` [IO-W]：执行 `init layout` 对应逻辑。
- `F L944-L946` `_write_resolved_config(config_path: Path, output_dir: Path) -> None`：写入 `resolved config` 对应的数据或结果。
- `F L949-L975` `_write_reports(_runtime: _FlowRuntime, state: Mapping[str, Any]) -> None` [IO-W]：写入 `reports` 对应的数据或结果。 调用：`_write_json`, `lines.append`。
- `F L978-L997` `_last_training_record(ramp_manifest_path: Path | None) -> Mapping[str, Any] | None`：执行 `last training record` 对应逻辑。 调用：`_read_json`。
- `F L1000-L1009` `_record_path(record: Mapping[str, Any], keys: tuple[str, ...]) -> Path | None`：执行 `record path` 对应逻辑。
- `F L1012-L1019` `_record_extra_files(record: Mapping[str, Any]) -> dict[str, Path]`：执行 `record extra files` 对应逻辑。 调用：`_record_path`。
- `F L1022-L1029` `_select_device_name(device: str) -> str`：选择 `device name` 对应的数据或结果。
- `F L1032-L1042` `_dataset_fingerprint(settings) -> dict[str, Any]`：执行 `dataset fingerprint` 对应逻辑。
- `F L1045-L1050` `_file_sha256(path: Path) -> str` [IO-R IO-W]：执行 `file sha256` 对应逻辑。
- `F L1053-L1054` `_new_run_id() -> str`：执行 `new run id` 对应逻辑。
- `F L1057-L1058` `_read_json(path: Path) -> dict[str, Any]` [IO-R]：读取 `json` 对应的数据或结果。
- `F L1061-L1066` `_write_json(path: Path, value: Mapping[str, Any]) -> None` [IO-W]：写入 `json` 对应的数据或结果。 调用：`_json_ready`。
- `F L1069-L1080` `_json_ready(value: Any) -> Any`：执行 `json ready` 对应逻辑。 调用：`_json_ready`。

## `src/traning/core/full_flow/result.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.full_flow.stages`

- `F L11-L12` `utc_now() -> str` [IO-W]：执行 `utc now` 对应逻辑。
- `C L16-L53` `FullFlowStageState` [CLASS]：封装 `FullFlowStageState` 相关数据或行为。
- `M L28-L32` `FullFlowStageState.mark_started(self) -> None`：更新状态为 `started` 对应的数据或结果。 调用：`utc_now`。
- `M L34-L50` `FullFlowStageState.mark_finished(self, status: FullFlowStageStatus, *, result: Mapping[str, Any] | None=None, warnings: tuple[str, ...]=(), error: str | None=None, artifacts: tuple[str, ...]=(), restored: bool=False) -> None`：更新状态为 `finished` 对应的数据或结果。 调用：`utc_now`。
- `M L52-L53` `FullFlowStageState.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L57-L91` `FullFlowResult` [CLASS]：封装 `FullFlowResult` 相关数据或行为。
- `M L74-L91` `FullFlowResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`stage.as_dict`。

## `src/traning/core/full_flow/stages.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L23-L34` `FullFlowStageSpec` [CLASS]：封装 `FullFlowStageSpec` 相关数据或行为。
- `M L33-L34` `FullFlowStageSpec.as_dict(self) -> dict[str, object]`：执行 `as dict` 对应逻辑。
- `F L167-L168` `stage_ids() -> tuple[str, ...]`：执行 `stage ids` 对应逻辑。
- `F L171-L175` `validate_stage_id(stage_id: str) -> str` [IO-W]：校验 `stage id` 对应的数据或结果。

## `src/traning/core/model_export/artifact.py`

职责：导出 inference/resume PyTorch artifact，写 manifest、文件 sha256 和版本信息。
工程依赖：`traning.conf`, `traning.lib.data`, `traning.lib.models`, `traning.state.versioning`

- `C L27-L39` `ArtifactFile` [CLASS]：封装 `ArtifactFile` 相关数据或行为。
- `M L33-L39` `ArtifactFile.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`self.path.as_posix`。
- `C L43-L61` `ModelArtifactSpec` [CLASS]：封装 `ModelArtifactSpec` 相关数据或行为。
- `M L56-L61` `ModelArtifactSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L65-L79` `ModelArtifactResult` [CLASS]：封装 `ModelArtifactResult` 相关数据或行为。
- `M L72-L79` `ModelArtifactResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`item.as_dict`。
- `F L82-L87` `_sha256(path: Path) -> str` [IO-R IO-W]：执行 `sha256` 对应逻辑。
- `F L90-L100` `_copy_file(source: Path, destination: Path, role: str) -> ArtifactFile` [IO-W]：执行 `copy file` 对应逻辑。 调用：`ArtifactFile`, `_sha256`。
- `F L103-L110` `_copy_optional(files: list[ArtifactFile], source: Path | None, destination: Path, role: str) -> None`：执行 `copy optional` 对应逻辑。 调用：`_copy_file`, `files.append`。
- `F L113-L133` `_write_readme(path: Path, spec: ModelArtifactSpec) -> ArtifactFile` [IO-W]：写入 `readme` 对应的数据或结果。 调用：`ArtifactFile`, `_sha256`。
- `F L136-L142` `_manifest_file(item: ArtifactFile, artifact_dir: Path) -> dict[str, Any]`：执行 `manifest file` 对应逻辑。 调用：`item.as_dict`。
- `F L145-L215` `export_model_artifact(spec: ModelArtifactSpec) -> ModelArtifactResult` [IO-W]：导出 `model artifact` 对应的数据或结果。 调用：`ArtifactFile`, `ModelArtifactResult`, `_copy_file`, `_copy_optional`, `_manifest_file`, `_sha256`。
- `F L218-L235` `validate_model_artifact(manifest_path: Path | str) -> tuple[str, ...]` [IO-R]：校验 `model artifact` 对应的数据或结果。 调用：`_sha256`, `issues.append`。
- `F L238-L256` `migrate_settings_file(settings_path: Path | str) -> tuple[Path, dict[str, Any]]` [IO-R IO-W]：执行 `migrate settings file` 对应逻辑。 调用：`log.append`。
- `F L259-L272` `import_model_artifact(manifest_path: Path | str) -> dict[str, Any]` [IO-R]：导入 `model artifact` 对应的数据或结果。 调用：`load_settings`, `migrate_settings_file`, `validate_model_artifact`。
- `F L275-L321` `smoke_test_model_artifact(manifest_path: Path | str, *, device: str='cpu') -> dict[str, Any]`：执行 `smoke test model artifact` 对应逻辑。 调用：`CausalTemporalModel`, `append_color_cues`, `build_model_stack`, `import_model_artifact`, `load_settings`, `temporal.initial_state`。

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

职责：根据优化计划创建 trial 记录、低预算训练 job、checkpoint 继承和 JSONL/SQLite 记录。
工程依赖：`traning.core.optimization.attribution`, `traning.core.optimization.parameter_search.curriculum`, `traning.core.optimization.parameter_search.hard_examples`, `traning.core.optimization.parameter_search.planner`, `traning.core.optimization.scoring`, `traning.state`

- `C L35-L57` `TrainingJobSpec` [CLASS]：封装 `TrainingJobSpec` 相关数据或行为。
- `M L44-L57` `TrainingJobSpec.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`self.parameters.model_dump`。
- `C L61-L85` `OptimizationExecution` [CLASS]：封装 `OptimizationExecution` 相关数据或行为。
- `M L73-L85` `OptimizationExecution.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`self.job.as_dict`, `self.trial.model_dump`。
- `C L89-L104` `OptimizationExecutorConfig` [CLASS]：封装 `OptimizationExecutorConfig` 相关数据或行为。
- `M L98-L104` `OptimizationExecutorConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L107-L130` `JsonlTrialStore` [CLASS]：封装 `JsonlTrialStore` 相关数据或行为。
- `M L108-L109` `JsonlTrialStore.__init__(self, path: Path) -> None`：初始化实例依赖、配置和运行状态。
- `M L111-L121` `JsonlTrialStore.append(self, execution: OptimizationExecution) -> None` [IO-W]：执行 `append` 对应逻辑。 调用：`execution.as_dict`, `self.path.parent.mkdir`。
- `M L123-L130` `JsonlTrialStore.load(self) -> tuple[dict[str, Any], ...]` [IO-R]：执行 `load` 对应逻辑。 调用：`records.append`, `self.path.exists`, `self.path.read_text`, `self.path.read_text.splitlines`。
- `C L133-L205` `SQLiteTrialStore` [CLASS]：封装 `SQLiteTrialStore` 相关数据或行为。
- `M L134-L135` `SQLiteTrialStore.__init__(self, path: Path) -> None`：初始化实例依赖、配置和运行状态。
- `M L137-L161` `SQLiteTrialStore._connect(self) -> sqlite3.Connection` [IO-W]：执行 `connect` 对应逻辑。 调用：`self.path.parent.mkdir`。
- `M L163-L192` `SQLiteTrialStore.append(self, execution: OptimizationExecution) -> None`：执行 `append` 对应逻辑。 调用：`execution.as_dict`, `self._connect`。
- `M L194-L205` `SQLiteTrialStore.load(self) -> tuple[dict[str, Any], ...]`：执行 `load` 对应逻辑。 调用：`self._connect`, `self.path.exists`。
- `F L208-L218` `create_trial_store(*, backend: str, jsonl_path: Path, sqlite_path: Path) -> JsonlTrialStore | SQLiteTrialStore`：执行 `create trial store` 对应逻辑。 调用：`JsonlTrialStore`, `SQLiteTrialStore`。
- `F L221-L247` `_apply_section_updates(base: Mapping[str, object], updates: Mapping[str, Any]) -> dict[str, object]`：应用 `section updates` 对应的数据或结果。
- `F L250-L270` `_apply_parameter_updates(parameters: TrialParameters, updates: Mapping[str, Mapping[str, Any]]) -> TrialParameters`：应用 `parameter updates` 对应的数据或结果。 调用：`TrialParameters`, `_apply_section_updates`。
- `F L273-L277` `_budget_steps(config: OptimizationExecutorConfig, rung: int) -> int`：执行 `budget steps` 对应逻辑。
- `F L280-L282` `_next_trial_id(source_trial_id: str, rung: int, stage: CurriculumStage) -> str`：执行 `next trial id` 对应逻辑。
- `F L285-L350` `execute_optimization_plan(report: TrialScoreReport, attribution: AttributionSummary, plan: OptimizationPlan, *, base_parameters: TrialParameters | None=None, parent_checkpoint_path: Path | None=None, config: OptimizationExecutorConfig=OptimizationExecutorConfig(), store: JsonlTrialStore | SQLiteTrialStore | None=None) -> OptimizationExecution`：执行 `execute optimization plan` 对应逻辑。 调用：`JsonlTrialStore`, `OptimizationExecution`, `TrainingJobSpec`, `TrialMetadata`, `_apply_parameter_updates`, `_budget_steps`。

## `src/traning/core/optimization/parameter_search/hard_examples.py`

职责：把归因 hard examples 转换为样本采样权重计划。
工程依赖：`traning.core.optimization.attribution`

- `C L12-L23` `HardExampleSamplingPlan` [CLASS]：封装 `HardExampleSamplingPlan` 相关数据或行为。
- `M L16-L23` `HardExampleSamplingPlan.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`self.reasons.items`。
- `F L26-L55` `build_hard_example_sampling_plan(attribution: AttributionSummary, *, base_weight: float=1.0, severity_multiplier: float=1.5, max_examples: int=128) -> HardExampleSamplingPlan`：构建并返回 `hard example sampling plan` 对应的数据或结果。 调用：`HardExampleSamplingPlan`, `append`。

## `src/traning/core/optimization/parameter_search/objectives.py`

职责：计算 quality/VRAM/latency 多目标排序值和可复现 sort key。
工程依赖：`traning.core.optimization.scoring`

- `C L18-L43` `ObjectiveScore` [CLASS]：封装 `ObjectiveScore` 相关数据或行为。
- `M L23-L27` `ObjectiveScore.composite_score(self) -> float` [PROPERTY]：执行 `composite score` 对应逻辑。 调用：`self.values.get`, `self.weights.items`。
- `M L29-L36` `ObjectiveScore.as_dict(self) -> dict[str, object]`：执行 `as dict` 对应逻辑。 调用：`self.sort_key`。
- `M L38-L43` `ObjectiveScore.sort_key(self) -> tuple[float, float, float]`：执行 `sort key` 对应逻辑。 调用：`self.values.get`。
- `F L46-L52` `objective_values_from_report(report: TrialScoreReport) -> dict[str, float]`：执行 `objective values from report` 对应逻辑。
- `F L55-L63` `score_trial_objectives(report: TrialScoreReport, *, weights: Mapping[str, float] | None=None) -> ObjectiveScore`：执行 `score trial objectives` 对应逻辑。 调用：`ObjectiveScore`, `objective_values_from_report`。

## `src/traning/core/optimization/parameter_search/planner.py`

职责：根据评分、归因、历史 trial、资源指标和多目标分数生成下一轮参数调整计划。
工程依赖：`traning.core.optimization.attribution`, `traning.core.optimization.parameter_search.objectives`, `traning.core.optimization.scoring`, `traning.state`

- `C L21-L51` `ASHAConfig` [CLASS]：封装 `ASHAConfig` 相关数据或行为。
- `M L35-L51` `ASHAConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L55-L71` `ParameterSearchConfig` [CLASS]：封装 `ParameterSearchConfig` 相关数据或行为。
- `M L64-L71` `ParameterSearchConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`self.objective_weights.values`。
- `C L75-L90` `TrialHistoryEntry` [CLASS]：封装 `TrialHistoryEntry` 相关数据或行为。
- `M L83-L90` `TrialHistoryEntry.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L94-L133` `OptimizationPlan` [CLASS]：封装 `OptimizationPlan` 相关数据或行为。
- `M L112-L133` `OptimizationPlan.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`self.parameter_updates.items`。
- `F L144-L146` `_next_stage(stage: CurriculumStage) -> CurriculumStage`：执行 `next stage` 对应逻辑。
- `F L149-L154` `_quantile(values: Sequence[float], quantile: float) -> float`：执行 `quantile` 对应逻辑。
- `F L157-L188` `_asha_action(report: TrialScoreReport, history: Sequence[TrialHistoryEntry], *, current_stage: CurriculumStage, rung: int, config: ASHAConfig) -> tuple[ASHAAction, tuple[str, ...]]`：执行 `asha action` 对应逻辑。 调用：`_quantile`, `reasons.append`。
- `F L191-L203` `_priority_domains(attribution: AttributionSummary) -> tuple[str, ...]`：执行 `priority domains` 对应逻辑。
- `F L206-L221` `_hard_example_keys(attribution: AttributionSummary, *, limit: int) -> tuple[str, ...]`：执行 `hard example keys` 对应逻辑。 调用：`keys.append`。
- `F L224-L230` `_set_update(updates: dict[str, dict[str, Any]], section: str, name: str, value: Any) -> None`：执行 `set update` 对应逻辑。
- `F L233-L257` `_apply_domain_updates(updates: dict[str, dict[str, Any]], attribution: AttributionSummary, reasons: list[str]) -> None`：应用 `domain updates` 对应的数据或结果。 调用：`_set_update`, `reasons.append`。
- `F L260-L283` `_apply_overall_updates(updates: dict[str, dict[str, Any]], report: TrialScoreReport, config: ParameterSearchConfig, reasons: list[str]) -> None`：应用 `overall updates` 对应的数据或结果。 调用：`_set_update`, `reasons.append`。
- `F L286-L342` `plan_next_trial(report: TrialScoreReport, attribution: AttributionSummary, *, history: Sequence[TrialHistoryEntry]=(), current_stage: CurriculumStage=CurriculumStage.BASIC, rung: int=0, config: ParameterSearchConfig=ParameterSearchConfig()) -> OptimizationPlan`：执行 `plan next trial` 对应逻辑。 调用：`OptimizationPlan`, `_apply_domain_updates`, `_apply_overall_updates`, `_asha_action`, `_hard_example_keys`, `_next_stage`。

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

- `F L16-L24` `_metadata_point(value: object) -> tuple[float, float] | None`：执行 `metadata point` 对应逻辑。
- `F L27-L42` `_representative_click(sample: SampleScoreReport)`：执行 `representative click` 对应逻辑。
- `F L45-L52` `_unresolved_source_index(sample: SampleScoreReport) -> int | None`：执行 `unresolved source index` 对应逻辑。
- `F L55-L89` `_frame_evaluation(sample: SampleScoreReport) -> FrameEvaluation`：执行 `frame evaluation` 对应逻辑。 调用：`FrameEvaluation`, `_metadata_point`, `_representative_click`, `_unresolved_source_index`。
- `F L92-L117` `build_batch_gallery_request(report: TrialScoreReport, *, batch_id: str | None=None, random_seed: int=2026, metadata: dict[str, object] | None=None) -> BatchGalleryRequest`：Build the result-export request directly from optimization scoring。 调用：`BatchGalleryRequest`, `TrialGalleryEvaluation`, `_frame_evaluation`。

## `src/traning/core/optimization/scoring/run_outputs.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package.coordinates`, `traning.core.optimization.scoring.evaluator`, `traning.lib.coordinates`, `traning.lib.metrics`, `traning.state`

- `C L28-L51` `DecisionOutputScoreResult` [CLASS]：封装 `DecisionOutputScoreResult` 相关数据或行为。
- `M L36-L51` `DecisionOutputScoreResult.as_summary(self) -> dict[str, Any]`：执行 `as summary` 对应逻辑。
- `F L54-L117` `score_decision_outputs(*, parameter_group_id: str, candidate_cache_path: Path, decisions_path: Path, metrics: Mapping[str, float] | None=None, circle_radius: float=DEFAULT_CIRCLE_RADIUS_OSU, spec: TrialScoreSpec=TrialScoreSpec(), settings: Any | None=None) -> DecisionOutputScoreResult`：执行 `score decision outputs` 对应逻辑。 调用：`DecisionOutputScoreResult`, `TrialParameters`, `_frame_key`, `_read_jsonl`, `_sample_from_rows`, `samples.append`。
- `F L120-L157` `_sample_from_rows(cache_row: Mapping[str, Any], decision: Mapping[str, Any], *, parameter_group_id: str, circle_radius: float, settings: Any | None=None) -> SampleScoringInput`：执行 `sample from rows` 对应逻辑。 调用：`SampleScoringInput`, `_predicted_clicks`, `_prediction_video_xy`, `_safe_int`, `_subproject_from_sample_key`, `_target_objects`。
- `F L160-L200` `_target_objects(row: Mapping[str, Any], *, settings: Any | None=None) -> tuple[TargetObject, ...]`：执行 `target objects` 对应逻辑。 调用：`TargetObject`, `_point_pair`, `_safe_float`, `_safe_int`, `_video_to_osu_pair`。
- `F L203-L230` `_predicted_clicks(cache_row: Mapping[str, Any], decision: Mapping[str, Any], *, predicted_video_xy: tuple[float, float] | None=None, settings: Any | None=None) -> tuple[PredictedClick, ...]`：执行 `predicted clicks` 对应逻辑。 调用：`PredictedClick`, `_normalized_to_osu`, `_safe_float`, `_video_to_osu`。
- `F L233-L255` `_prediction_video_xy(cache_row: Mapping[str, Any], decision: Mapping[str, Any]) -> tuple[float, float] | None`：执行 `prediction video xy` 对应逻辑。 调用：`_safe_float`, `_safe_int`。
- `F L258-L267` `_video_to_osu_pair(value: object, row: Mapping[str, Any], *, settings: Any | None=None) -> tuple[float, float] | None`：执行 `video to osu pair` 对应逻辑。 调用：`_point_pair`, `_video_to_osu`。
- `F L270-L287` `_video_to_osu(x: float, y: float, row: Mapping[str, Any], *, settings: Any | None=None) -> tuple[float, float] | None`：执行 `video to osu` 对应逻辑。 调用：`_safe_int`, `transform_from_settings_or_sample`。
- `F L290-L294` `_normalized_to_osu(value: object) -> tuple[float, float] | None`：执行 `normalized to osu` 对应逻辑。 调用：`_point_pair`。
- `F L297-L306` `_point_pair(value: object) -> tuple[float, float] | None`：执行 `point pair` 对应逻辑。 调用：`_safe_float`。
- `F L309-L320` `_read_jsonl(path: Path) -> list[dict[str, Any]]` [IO-W]：读取 `jsonl` 对应的数据或结果。 调用：`rows.append`。
- `F L323-L326` `_frame_key(row: Mapping[str, Any]) -> tuple[str, int]`：执行 `frame key` 对应逻辑。 调用：`_safe_int`。
- `F L329-L334` `_subproject_from_sample_key(sample_key: str) -> str`：执行 `subproject from sample key` 对应逻辑。
- `F L337-L343` `_safe_float(value: object) -> float | None`：执行 `safe float` 对应逻辑。
- `F L346-L352` `_safe_int(value: object) -> int | None`：执行 `safe int` 对应逻辑。

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
- `M L130-L175` `OptionalTrainingVisualizer.save_gallery(self, dataset: SegmentFrameDataset, request: BatchGalleryRequest, *, output_root: Path | None=None, samples_per_group: int | None=None) -> GalleryResult`：执行 `save gallery` 对应逻辑。 调用：`GalleryResult`, `self._warning_once`。
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
工程依赖：`traning.conf`, `traning.core.dataset_import`, `traning.core.training_inheritance`, `traning.lib.data`, `traning.lib.models`, `traning.lib.runtime`, `traning.lib.training.losses`, `traning.lib.training.spatial_targets`

- `C L47-L85` `SpatialTrainingResult` [CLASS]：封装 `SpatialTrainingResult` 相关数据或行为。
- `M L64-L66` `SpatialTrainingResult.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L68-L85` `SpatialTrainingResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `F L88-L339` `run_spatial_training(settings: Settings, *, device: torch.device, run_dir: Path, split: DataSplit='train', max_steps: int=1, learning_rate: float=0.0001, patch_limit: int | None=None, dataset: Sequence[dict[str, Any]] | None=None, reporter: TrainingReporter=NullReporter(), resume_checkpoint_path: Path | None=None, resume_policy: str='none') -> SpatialTrainingResult` [IO-W]：Run the first-version single-frame spatial training loop。 调用：`CudaRuntimeConfig`, `PatchStream`, `SpatialTrainingResult`, `TrainingPosition`, `_add_spatial_consistency_losses`, `_normalize_frame`。
- `F L342-L373` `_add_spatial_consistency_losses(loss_dict: dict[str, torch.Tensor], *, prediction, target, weights) -> None`：执行 `add spatial consistency losses` 对应逻辑。 调用：`temporal_consistency_loss`。
- `F L376-L381` `_normalize_frame(frame: torch.Tensor) -> torch.Tensor`：规范化 `frame` 对应的数据或结果。
- `F L384-L390` `_write_summary(result: SpatialTrainingResult) -> None` [IO-W]：写入 `summary` 对应的数据或结果。 调用：`result.as_dict`。
- `F L393-L453` `_write_checkpoint(result: SpatialTrainingResult, *, modules: dict[str, torch.nn.Module], settings: Settings, optimizer: torch.optim.Optimizer, scaler, position: TrainingPosition, checkpoint_kind: str) -> None` [IO-W]：写入 `checkpoint` 对应的数据或结果。 调用：`atomic_torch_save_checkpoint`, `build_training_checkpoint`。
- `F L456-L523` `_restore_spatial_training_state(*, modules: dict[str, torch.nn.Module], optimizer: torch.optim.Optimizer, scaler, checkpoint_path: Path | None, policy: str, reporter: TrainingReporter) -> TrainingPosition`：执行 `restore spatial training state` 对应逻辑。 调用：`TrainingPosition`, `TrainingPosition.from_mapping`, `_optimizer_state_to_device`, `load_training_checkpoint`, `restore_module_state`, `restore_rng_state`。
- `F L526-L533` `_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None`：执行 `optimizer state to device` 对应逻辑。
- `F L536-L572` `_report_spatial_step(reporter: TrainingReporter, *, step: int, target: int, loss: float, sample: dict[str, Any], total_samples: int, generated_patches: int, device: torch.device) -> None`：执行 `report spatial step` 对应逻辑。 调用：`reporter.update_metrics`。

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
工程依赖：`traning.conf`, `traning.core.temporal.dataset`, `traning.core.training_inheritance`, `traning.lib.models`, `traning.lib.runtime`

- `C L51-L96` `TemporalTrainingResult` [CLASS]：封装 `TemporalTrainingResult` 相关数据或行为。
- `M L77-L96` `TemporalTrainingResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `F L99-L303` `run_temporal_training(settings: Settings, *, cache_dir: Path, device: torch.device, run_dir: Path, max_steps: int=1, learning_rate: float=0.0001, sequence_length: int | None=None, candidate_slots: int | None=None, dataset: Sequence[TemporalWindow] | None=None, reporter: TrainingReporter=NullReporter(), resume_checkpoint_path: Path | None=None, resume_policy: str='none') -> TemporalTrainingResult`：执行 `run temporal training` 对应逻辑。 调用：`CausalTemporalModel`, `CudaRuntimeConfig`, `TemporalCandidateWindowDataset.from_cache_dir`, `TemporalTrainingResult`, `TrainingPosition`, `_compute_temporal_loss`。
- `F L306-L325` `_window_to_device(window: TemporalWindow, *, device: torch.device) -> dict[str, torch.Tensor]`：执行 `window to device` 对应逻辑。 调用：`tensor_to_device`。
- `F L328-L380` `_compute_temporal_loss(outputs, *, action_target: torch.Tensor, selected_candidate_target: torch.Tensor, xy_target: torch.Tensor, time_offset_target: torch.Tensor, frame_mask: torch.Tensor, weights) -> tuple[torch.Tensor, dict[str, torch.Tensor]]`：执行 `compute temporal loss` 对应逻辑。
- `F L383-L388` `_write_summary(result: TemporalTrainingResult) -> None` [IO-W]：写入 `summary` 对应的数据或结果。 调用：`result.as_dict`。
- `F L391-L466` `_write_checkpoint(result: TemporalTrainingResult, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler, hidden_size: int, layers: int, position: TrainingPosition, checkpoint_kind: str) -> None` [IO-W]：写入 `checkpoint` 对应的数据或结果。 调用：`atomic_torch_save_checkpoint`, `build_training_checkpoint`。
- `F L469-L524` `_restore_temporal_training_state(*, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler, checkpoint_path: Path | None, policy: str, reporter: TrainingReporter) -> TrainingPosition`：执行 `restore temporal training state` 对应逻辑。 调用：`TrainingPosition`, `TrainingPosition.from_mapping`, `_optimizer_state_to_device`, `load_training_checkpoint`, `restore_module_state`, `restore_rng_state`。
- `F L527-L534` `_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None`：执行 `optimizer state to device` 对应逻辑。
- `F L537-L572` `_report_temporal_step(reporter: TrainingReporter, *, step: int, target: int, loss: float, window: TemporalWindow, total_windows: int, device: torch.device) -> None`：执行 `report temporal step` 对应逻辑。 调用：`reporter.update_metrics`。

## `src/traning/core/training_inheritance/checkpoint.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L18-L47` `TrainingPosition` [CLASS]：封装 `TrainingPosition` 相关数据或行为。
- `M L28-L29` `TrainingPosition.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `M L32-L47` `TrainingPosition.from_mapping(cls, raw: Mapping[str, Any] | None) -> TrainingPosition`：执行 `from mapping` 对应逻辑。
- `C L51-L76` `CheckpointRestorePlan` [CLASS]：封装 `CheckpointRestorePlan` 相关数据或行为。
- `M L63-L64` `CheckpointRestorePlan.enabled(self) -> bool` [PROPERTY]：执行 `enabled` 对应逻辑。
- `M L66-L76` `CheckpointRestorePlan.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`self.position.as_dict`。
- `F L79-L88` `capture_rng_state() -> dict[str, Any]`：执行 `capture rng state` 对应逻辑。
- `F L91-L108` `restore_rng_state(raw: Mapping[str, Any] | None) -> bool`：执行 `restore rng state` 对应逻辑。
- `F L111-L154` `build_training_checkpoint(*, checkpoint_kind: str, run_id: str, trial_id: str, models: Mapping[str, Any], optimizer: torch.optim.Optimizer | None, scheduler: Any | None, scaler: Any | None, position: TrainingPosition, score_state: Mapping[str, Any] | None=None, grade_state: Mapping[str, Any] | None=None, promotion_state: Mapping[str, Any] | None=None, dataset_state: Mapping[str, Any] | None=None, sampler_state: Mapping[str, Any] | None=None, resolved_config: Mapping[str, Any] | None=None, dataset_fingerprint: Mapping[str, Any] | None=None, extra: Mapping[str, Any] | None=None) -> dict[str, Any]`：构建并返回 `training checkpoint` 对应的数据或结果。 调用：`capture_rng_state`, `position.as_dict`。
- `F L157-L171` `atomic_torch_save_checkpoint(payload: Mapping[str, Any], path: Path, *, expected_kind: str) -> None` [IO-W]：执行 `atomic torch save checkpoint` 对应逻辑。 调用：`torch.load`, `validate_training_checkpoint`。
- `F L174-L178` `load_training_checkpoint(path: Path) -> dict[str, Any]`：加载 `training checkpoint` 对应的数据或结果。 调用：`torch.load`。
- `F L181-L203` `validate_training_checkpoint(payload: Mapping[str, Any], *, expected_kind: str | None=None) -> None`：校验 `training checkpoint` 对应的数据或结果。 调用：`TrainingPosition.from_mapping`。
- `F L206-L224` `restore_module_state(module: torch.nn.Module, state_dict: Mapping[str, Any], *, strict: bool) -> tuple[tuple[str, ...], tuple[str, ...]]`：执行 `restore module state` 对应逻辑。

## `src/traning/core/training_inheritance/manager.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.training_inheritance.checkpoint`, `traning.state.versioning`

- `C L26-L40` `InheritancePackage` [CLASS]：封装 `InheritancePackage` 相关数据或行为。
- `M L33-L40` `InheritancePackage.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L44-L70` `InheritanceLoadResult` [CLASS]：封装 `InheritanceLoadResult` 相关数据或行为。
- `M L57-L70` `InheritanceLoadResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `F L73-L131` `create_inheritance_package(*, output_dir: Path, settings: Settings, resolved_config_path: Path | None=None, latest_checkpoint_path: Path | None=None, best_checkpoint_path: Path | None=None, training_state: dict[str, Any] | None=None, score_state: dict[str, Any] | None=None, promotion_state: dict[str, Any] | None=None, artifacts: dict[str, Any] | None=None, stage_checkpoints: Mapping[str, Path | None] | None=None) -> InheritancePackage` [IO-W]：执行 `create inheritance package` 对应逻辑。 调用：`InheritancePackage`, `_copy_checkpoint`, `_dataset_fingerprint`, `_rng_state`, `_write_json`, `collect_code_version`。
- `F L134-L232` `load_inheritance_package(*, inherit_from: Path | str | None, current_settings: Settings, policy: ResumePolicy) -> InheritanceLoadResult` [IO-R]：加载 `inheritance package` 对应的数据或结果。 调用：`InheritanceLoadResult`, `_compatibility_reasons`, `_stage_checkpoint_paths`, `load_training_checkpoint`, `missing_fields.append`, `reasons.append`。
- `F L235-L248` `resolve_inheritance_path(value: Path | str | None) -> Path | None` [IO-R]：解析并定位 `inheritance path` 对应的数据或结果。
- `F L251-L263` `_compatibility_reasons(manifest: dict[str, Any], settings: Settings) -> list[str]`：执行 `compatibility reasons` 对应逻辑。 调用：`_comparable`, `_dataset_fingerprint`, `reasons.append`, `version_manifest`。
- `F L266-L277` `_stage_checkpoint_paths(root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]`：执行 `stage checkpoint paths` 对应逻辑。
- `F L280-L290` `_dataset_fingerprint(settings: Settings) -> dict[str, Any]`：执行 `dataset fingerprint` 对应逻辑。
- `F L293-L299` `_rng_state() -> dict[str, Any]`：执行 `rng state` 对应逻辑。
- `F L302-L307` `_comparable(value: Any) -> Any`：执行 `comparable` 对应逻辑。 调用：`_comparable`。
- `F L310-L315` `_copy_checkpoint(source: Path | None, destination: Path) -> Path | None` [IO-W]：执行 `copy checkpoint` 对应逻辑。
- `F L318-L323` `_write_json(path: Path, value: Any) -> None` [IO-W]：写入 `json` 对应的数据或结果。 调用：`_json_ready`。
- `F L326-L339` `_json_ready(value: Any) -> Any`：执行 `json ready` 对应逻辑。 调用：`_json_ready`, `value.as_dict`。

## `src/traning/core/training_ramp.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.dataset_import`, `traning.core.decision`, `traning.core.model_export`, `traning.state.versioning`

- `C L45-L67` `RampLevelSpec` [CLASS]：封装 `RampLevelSpec` 相关数据或行为。
- `M L56-L67` `RampLevelSpec.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L71-L91` `RampTarget` [CLASS]：封装 `RampTarget` 相关数据或行为。
- `M L80-L91` `RampTarget.as_level(self) -> RampLevelSpec`：执行 `as level` 对应逻辑。 调用：`RampLevelSpec`。
- `C L95-L113` `RampRunResult` [CLASS]：封装 `RampRunResult` 相关数据或行为。
- `M L104-L113` `RampRunResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L116-L117` `RampGateError(RuntimeError)` [CLASS]：封装 `RampGateError` 相关数据或行为。
- `F L120-L181` `run_training_ramp(*, config_path: Path, device: str, output_root: Path=DEFAULT_OUTPUT_ROOT, target_config_path: Path | None=None, run_id: str | None=None, auto_launch_full: bool=False, force_level: bool=False, max_levels: int | None=None, run_full_checks: bool=True, progress_ui: str='auto', progress_language: str='zh-CN', resume_policy: str='none', resume_stage_checkpoints: Mapping[str, Path] | None=None, full_gallery_output_root: Path | None=None, full_gallery_samples_per_group: int | None=None, reporter: TrainingReporter | None=None) -> RampRunResult`：执行 `run training ramp` 对应逻辑。 调用：`_init_layout`, `_run_training_ramp_with_reporter`。
- `F L184-L386` `_run_training_ramp_with_reporter(*, config_path: Path, device: str, target_config_path: Path | None, run_id: str, output_dir: Path, auto_launch_full: bool, force_level: bool, max_levels: int | None, run_full_checks: bool, reporter: TrainingReporter, resume_policy: str, resume_stage_checkpoints: Mapping[str, Path], full_gallery_output_root: Path | None, full_gallery_samples_per_group: int | None) -> RampRunResult`：执行 `run training ramp with reporter` 对应逻辑。 调用：`RampRunResult`, `_launch_full_training`, `_read_json`, `_report_full_training_finished`, `_report_full_training_started`, `_report_level_finished`。
- `F L389-L405` `ensure_full_target_config(*, source_config: Path, target_config: Path, output_dir: Path) -> tuple[Path, RampTarget]` [IO-W]：确保 `full target config` 对应的数据或结果。 调用：`_absolutize_config`, `_build_default_full_config`, `_read_yaml`, `_target_from_raw`, `_write_yaml`。
- `F L408-L437` `build_ramp_levels(target: RampTarget) -> list[RampLevelSpec]`：构建并返回 `ramp levels` 对应的数据或结果。 调用：`RampLevelSpec`, `_clip_level`, `_level_reaches_target`, `as_dict`, `clipped.as_dict`, `levels.append`。
- `F L440-L549` `_run_preflight(*, config_path: Path, device: str, output_dir: Path, run_full_checks: bool, reporter: TrainingReporter) -> dict[str, Any]` [IO-W PROCESS]：执行 `run preflight` 对应逻辑。 调用：`RampGateError`, `_write_json`, `inspect_data_input`, `load_settings`, `reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L552-L601` `_report_ramp_started(reporter: TrainingReporter, *, levels: list[RampLevelSpec], target: RampTarget, auto_launch_full: bool) -> None`：执行 `report ramp started` 对应逻辑。 调用：`reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L604-L652` `_report_level_started(reporter: TrainingReporter, *, level: RampLevelSpec, index: int, total_levels: int) -> None`：执行 `report level started` 对应逻辑。 调用：`_level_stage_id`, `_level_title`, `level.as_dict`, `reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L655-L738` `_report_level_finished(reporter: TrainingReporter, *, level: RampLevelSpec, index: int, total_levels: int, record: Mapping[str, Any], restored: bool=False) -> None`：执行 `report level finished` 对应逻辑。 调用：`_level_stage_id`, `_level_title`, `_record_gallery_path`, `_record_pass_threshold`, `_record_quality_score`, `reporter.update_metrics`。
- `F L741-L783` `_report_ramp_finished(reporter: TrainingReporter, *, levels: list[RampLevelSpec], readiness_path: Path, auto_launch_full: bool) -> None`：执行 `report ramp finished` 对应逻辑。 调用：`reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L786-L816` `_report_full_training_started(reporter: TrainingReporter, *, level: RampLevelSpec) -> None`：执行 `report full training started` 对应逻辑。 调用：`reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L819-L849` `_report_full_training_finished(reporter: TrainingReporter, *, record: Mapping[str, Any]) -> None`：执行 `report full training finished` 对应逻辑。 调用：`_summary_quality_score`, `reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L852-L912` `_report_ramp_failed(reporter: TrainingReporter, *, error: Exception, active_level: RampLevelSpec | None, active_index: int, completed_levels: int, total_levels: int) -> None`：执行 `report ramp failed` 对应逻辑。 调用：`_level_stage_id`, `_level_title`, `reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L915-L916` `_level_stage_id(level: RampLevelSpec) -> str`：执行 `level stage id` 对应逻辑。
- `F L919-L922` `_level_title(level: RampLevelSpec | None) -> str`：执行 `level title` 对应逻辑。
- `F L925-L930` `_record_quality_score(record: Mapping[str, Any]) -> float | None`：执行 `record quality score` 对应逻辑。
- `F L933-L938` `_record_pass_threshold(record: Mapping[str, Any]) -> float | None`：执行 `record pass threshold` 对应逻辑。
- `F L941-L943` `_summary_quality_score(summary: Mapping[str, Any]) -> float | None`：执行 `summary quality score` 对应逻辑。
- `F L946-L951` `_record_gallery_path(record: Mapping[str, Any]) -> str | None`：执行 `record gallery path` 对应逻辑。
- `F L954-L1072` `_run_level(*, level: RampLevelSpec, base_config: Path, level_dir: Path, device: str, reporter: TrainingReporter, resume_policy: str, resume_stage_checkpoints: Mapping[str, Path], gallery_output_root: Path | None, gallery_samples_per_group: int | None) -> dict[str, Any]` [IO-W]：执行 `run level` 对应逻辑。 调用：`FullTrainingRunConfig`, `ModelArtifactSpec`, `_gate_level`, `_level_title`, `_ramp_parameter_snapshot`, `_run_job_dry_run`。
- `F L1075-L1157` `_gate_level(*, level: RampLevelSpec, result, elapsed: float, artifact_path: Path, artifact_issues: tuple[str, ...], artifact_smoke: dict[str, Any], dry_run: dict[str, Any]) -> dict[str, Any]`：执行 `gate level` 对应逻辑。 调用：`RampGateError`, `_read_json`, `failures.append`, `level.as_dict`, `result.candidate_cache.as_dict`, `result.decision.as_dict`。
- `F L1160-L1200` `_ramp_parameter_snapshot(*, level: RampLevelSpec, record: Mapping[str, Any], config_path: Path, device: str, resume_policy: str, resume_stage_checkpoints: Mapping[str, Path]) -> dict[str, Any]`：执行 `ramp parameter snapshot` 对应逻辑。 调用：`level.as_dict`。
- `F L1203-L1245` `_launch_full_training(*, level: RampLevelSpec, config_path: Path, run_dir: Path, device: str, reporter: TrainingReporter, resume_policy: str, resume_stage_checkpoints: Mapping[str, Path], gallery_output_root: Path | None, gallery_samples_per_group: int | None) -> dict[str, Any]`：执行 `launch full training` 对应逻辑。 调用：`FullTrainingRunConfig`, `load_settings`, `result.as_summary`, `run_full_training_pipeline`。
- `F L1248-L1308` `_write_final_readiness(*, output_dir: Path, manifest: dict[str, Any], target: RampTarget, levels: list[RampLevelSpec], auto_launch_full: bool, failure: str | None=None) -> Path` [IO-W]：写入 `final readiness` 对应的数据或结果。 调用：`_full_command_text`, `_write_json`, `lines.append`。
- `F L1311-L1347` `_run_job_dry_run(*, job_path: Path | None, config_path: Path, level_dir: Path, device: str) -> dict[str, Any]` [IO-W PROCESS]：执行 `run job dry run` 对应逻辑。 调用：`_pythonpath_with_src`, `subprocess.run`。
- `F L1350-L1355` `_pythonpath_with_src() -> str`：执行 `pythonpath with src` 对应逻辑。 调用：`entries.append`。
- `F L1358-L1372` `_write_level_config(base_config: Path, level_dir: Path, level: RampLevelSpec) -> Path`：写入 `level config` 对应的数据或结果。 调用：`_absolutize_config`, `_read_yaml`, `_write_yaml`, `level.as_dict`。
- `F L1375-L1387` `_build_default_full_config(source: dict[str, Any]) -> dict[str, Any]`：构建 `default full config` 对应的数据或结果。 调用：`RampTarget`。
- `F L1390-L1405` `_target_from_raw(raw: dict[str, Any]) -> RampTarget`：执行 `target from raw` 对应逻辑。 调用：`RampTarget`。
- `F L1408-L1422` `_clip_level(level: RampLevelSpec, target: RampTarget) -> RampLevelSpec`：执行 `clip level` 对应逻辑。 调用：`RampLevelSpec`。
- `F L1425-L1433` `_level_reaches_target(level: RampLevelSpec, target: RampTarget) -> bool`：执行 `level reaches target` 对应逻辑。
- `F L1436-L1438` `_init_layout(output_dir: Path) -> None` [IO-W]：执行 `init layout` 对应逻辑。
- `F L1441-L1453` `_full_command_text(config_path: Path, target: RampTarget) -> str`：执行 `full command text` 对应逻辑。
- `F L1456-L1457` `_read_json(path: Path) -> dict[str, Any]` [IO-R]：读取 `json` 对应的数据或结果。
- `F L1460-L1465` `_write_json(path: Path, value: dict[str, Any]) -> None` [IO-W]：写入 `json` 对应的数据或结果。 调用：`_json_ready`。
- `F L1468-L1472` `_read_yaml(path: Path) -> dict[str, Any]` [IO-R]：读取 `yaml` 对应的数据或结果。
- `F L1475-L1480` `_write_yaml(path: Path, value: dict[str, Any]) -> None` [IO-W]：写入 `yaml` 对应的数据或结果。 调用：`_json_ready`。
- `F L1483-L1498` `_absolutize_config(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]`：执行 `absolutize config` 对应逻辑。
- `F L1501-L1510` `_json_ready(value: Any) -> Any`：执行 `json ready` 对应逻辑。 调用：`_json_ready`。

## `src/traning/lib/coordinates.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package.coordinates`

- `F L14-L58` `transform_from_settings_or_sample(settings: Any | None, sample: Mapping[str, Any] | None=None, *, frame_width: int | None=None, frame_height: int | None=None) -> tuple[OsuVideoTransform, CoordinateTransformSpec]`：Resolve the explicit playfield transform for original video pixels。 调用：`_sample_transform_spec`。
- `F L61-L84` `_sample_transform_spec(sample: Mapping[str, Any] | None) -> CoordinateTransformSpec | None`：执行 `sample transform spec` 对应逻辑。 调用：`PlayfieldRect.from_mapping`。

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

- `C L14-L90` `SegmentFrameDataset(Dataset[dict[str, Any]])` [CLASS]：封装 `SegmentFrameDataset` 相关数据或行为。
- `M L15-L38` `SegmentFrameDataset.__init__(self, records: tuple[SegmentRecord, ...], *, sample_fps: float, frame_step: int=1, max_frames_per_segment: int | None=None, visibility_post_ms: float=100.0, normalize_images: bool=True, coordinate_transform: dict[str, Any] | None=None)`：初始化实例依赖、配置和运行状态。 调用：`build_frame_references`。
- `M L40-L41` `SegmentFrameDataset.__len__(self) -> int`：执行 `len` 对应逻辑。
- `M L43-L46` `SegmentFrameDataset._video_reader(self) -> VideoReader`：执行 `video reader` 对应逻辑。 调用：`VideoReader`。
- `M L48-L85` `SegmentFrameDataset.__getitem__(self, index: int) -> dict[str, Any]`：执行 `getitem` 对应逻辑。 调用：`self._video_reader`, `self._video_reader.read_frame_at`, `visible_hit_objects`。
- `M L87-L90` `SegmentFrameDataset.__getstate__(self) -> dict[str, Any]`：执行 `getstate` 对应逻辑。

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

## `src/traning/lib/models/smet.py`

职责：动态 top-k 稀疏线性层和普通/稀疏 Linear 工厂。

- `C L8-L65` `DynamicSparseLinear(nn.Module)` [CLASS]：Linear layer with a deterministic top-k dynamic sparse weight mask。
- `M L11-L38` `DynamicSparseLinear.__init__(self, in_features: int, out_features: int, *, bias: bool=True, sparsity: float=0.5, update_interval: int=16, min_density: float=0.05) -> None`：初始化实例依赖、配置和运行状态。 调用：`self.refresh_mask`, `self.register_buffer`, `self.reset_parameters`, `super.__init__`。
- `M L41-L42` `DynamicSparseLinear.density(self) -> float` [PROPERTY]：执行 `density` 对应逻辑。
- `M L44-L49` `DynamicSparseLinear.reset_parameters(self) -> None`：执行 `reset parameters` 对应逻辑。
- `M L51-L52` `DynamicSparseLinear.refresh_mask(self) -> None`：执行 `refresh mask` 对应逻辑。 调用：`self._mask_from_weight`, `self._mask_from_weight.to`, `self.mask.copy_`。
- `M L54-L56` `DynamicSparseLinear.forward(self, input: torch.Tensor) -> torch.Tensor`：执行 `forward` 对应逻辑。 调用：`self._mask_from_weight`。
- `M L58-L65` `DynamicSparseLinear._mask_from_weight(self) -> torch.Tensor`：执行 `mask from weight` 对应逻辑。 调用：`self.weight.detach`, `self.weight.numel`。
- `F L68-L87` `maybe_sparse_linear(in_features: int, out_features: int, *, enabled: bool, bias: bool=True, sparsity: float=0.5, update_interval: int=16, min_density: float=0.05) -> nn.Module`：执行 `maybe sparse linear` 对应逻辑。 调用：`DynamicSparseLinear`。

## `src/traning/lib/models/stack.py`

职责：从 Settings 统一构建 local/global/structure/fusion/spatial head 模型栈。
工程依赖：`traning.conf`, `traning.lib.data`, `traning.lib.models.gated_sparse_fusion`, `traning.lib.models.global_encoder`, `traning.lib.models.global_structure_head`, `traning.lib.models.local_encoder`, `traning.lib.models.object_heads`

- `F L14-L55` `build_model_stack(settings: Settings) -> dict[str, torch.nn.Module]`：Build the shared local/global/fusion/spatial model stack from settings。 调用：`GatedSparseFusion`, `GlobalStructureHead`, `LightweightGlobalEncoder`, `SmallLocalEncoder`, `SpatialPredictionHead`, `color_cue_channel_count`。

## `src/traning/lib/models/temporal_model.py`

职责：因果 GRU 时序模型；提供 initial_state、step 流式接口和可选 SMET 稀疏 heads。
工程依赖：`traning.lib.models.outputs`, `traning.lib.models.smet`

- `C L10-L106` `CausalTemporalModel(nn.Module)` [CLASS]：Causal GRU action head for streaming frame-by-frame inference。
- `M L13-L45` `CausalTemporalModel.__init__(self, *, input_size: int, hidden_size: int=256, layers: int=2, candidate_slots: int=64, action_classes: int=4, smet_enabled: bool=False, smet_sparsity: float=0.5, smet_update_interval: int=16, smet_min_density: float=0.05) -> None`：初始化实例依赖、配置和运行状态。 调用：`maybe_sparse_linear`, `super.__init__`。
- `M L47-L64` `CausalTemporalModel.initial_state(self, batch_size: int, device: torch.device | str, *, dtype: torch.dtype | None=None) -> torch.Tensor`：执行 `initial state` 对应逻辑。 调用：`self.parameters`。
- `M L66-L93` `CausalTemporalModel.step(self, current_features: torch.Tensor, previous_state: torch.Tensor) -> tuple[ActionPrediction, torch.Tensor]`：执行 `step` 对应逻辑。 调用：`ActionPrediction`, `next_states.append`, `self.action_head`, `self.candidate_head`, `self.time_head`, `self.xy_head`。
- `M L95-L106` `CausalTemporalModel.forward(self, sequence: torch.Tensor) -> tuple[list[ActionPrediction], torch.Tensor]`：执行 `forward` 对应逻辑。 调用：`outputs.append`, `self.initial_state`, `self.step`。

## `src/traning/lib/runtime/memory.py`

职责：统一 CUDA/runtime memory policy；管理 AMP、GradScaler、channels-last、TF32、显存/RAM预算、显存快照和 OOM 建议。

- `C L15-L20` `MemorySnapshot` [CLASS]：封装 `MemorySnapshot` 相关数据或行为。
- `C L24-L52` `RuntimeMemoryBudget` [CLASS]：封装 `RuntimeMemoryBudget` 相关数据或行为。
- `M L38-L52` `RuntimeMemoryBudget.as_dict(self) -> dict[str, float | str | None]`：执行 `as dict` 对应逻辑。
- `C L56-L60` `CudaRuntimeConfig` [CLASS]：封装 `CudaRuntimeConfig` 相关数据或行为。
- `C L64-L71` `CudaRuntimeState` [CLASS]：封装 `CudaRuntimeState` 相关数据或行为。
- `F L74-L176` `enforce_runtime_memory_budget(*, device: torch.device, max_vram_gib: float, reserve_vram_gib: float, max_ram_gib: float | None, reserve_ram_gib: float, set_cuda_fraction: bool=True) -> RuntimeMemoryBudget`：Validate CPU/CUDA budgets and reserve headroom for the host system。 调用：`RuntimeMemoryBudget`, `_finite`。
- `F L179-L197` `resolve_amp_dtype(device: torch.device, amp_dtype: AmpDType) -> torch.dtype | None`：解析并定位 `amp dtype` 对应的数据或结果。
- `F L201-L208` `autocast_context(device: torch.device, amp_dtype: AmpDType) -> Iterator[None]`：执行 `autocast context` 对应逻辑。 调用：`resolve_amp_dtype`。
- `F L211-L244` `configure_torch_runtime(*, device: torch.device, amp_dtype: AmpDType, runtime: CudaRuntimeConfig=CudaRuntimeConfig()) -> CudaRuntimeState`：Apply CUDA runtime defaults used by training and smoke tests。 调用：`CudaRuntimeState`, `amp_uses_grad_scaler`, `resolve_amp_dtype`。
- `F L247-L250` `amp_uses_grad_scaler(device: torch.device, amp_dtype: AmpDType) -> bool`：执行 `amp uses grad scaler` 对应逻辑。 调用：`resolve_amp_dtype`。
- `F L253-L266` `create_grad_scaler(*, device: torch.device, amp_dtype: AmpDType, mode: str='auto') -> torch.amp.GradScaler`：执行 `create grad scaler` 对应逻辑。 调用：`amp_uses_grad_scaler`。
- `F L269-L278` `module_to_device(module: nn.Module, device: torch.device, *, channels_last: bool) -> nn.Module`：执行 `module to device` 对应逻辑。
- `F L281-L291` `maybe_compile_module(module: nn.Module, *, enabled: bool, mode: str='default') -> nn.Module`：执行 `maybe compile module` 对应逻辑。
- `F L294-L307` `tensor_to_device(tensor: torch.Tensor, device: torch.device, *, channels_last: bool, non_blocking: bool=True) -> torch.Tensor`：执行 `tensor to device` 对应逻辑。
- `F L310-L325` `collect_memory_snapshot() -> MemorySnapshot`：执行 `collect memory snapshot` 对应逻辑。 调用：`MemorySnapshot`。
- `F L328-L361` `format_oom_guidance(*, patch_size: tuple[int, int], global_size: tuple[int, int], batch_size: int, amp_dtype: str, config_path: str | None) -> str`：执行 `format oom guidance` 对应逻辑。 调用：`collect_memory_snapshot`。
- `F L364-L365` `_finite(value: float) -> bool`：执行 `finite` 对应逻辑。

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
工程依赖：`package.coordinates`, `traning.lib.coordinates`, `traning.lib.data`, `traning.lib.models`, `traning.lib.training.losses`

- `F L23-L103` `build_spatial_loss_targets(sample: Mapping[str, Any], patch_meta: PatchMeta, feature_size: Sequence[int], *, settings: Any | None=None, device: torch.device | str | None=None, dtype: torch.dtype=torch.float32) -> SpatialLossTargets`：Rasterize one frame sample into dense targets for one patch feature grid。 调用：`SpatialLossTargets`, `_empty_targets`, `_finite_float`, `_normalize_feature_size`, `_object_kind`, `_paint_circle`。
- `F L106-L112` `_normalize_feature_size(feature_size: Sequence[int]) -> tuple[int, int]`：规范化 `feature size` 对应的数据或结果。
- `F L115-L168` `_empty_targets(*, feature_height: int, feature_width: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]`：执行 `empty targets` 对应逻辑。
- `F L171-L199` `_patch_grid(patch_meta: PatchMeta, *, feature_height: int, feature_width: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor | float]`：执行 `patch grid` 对应逻辑。
- `F L202-L209` `_finite_float(value: Any, default: float) -> float`：执行 `finite float` 对应逻辑。
- `F L212-L220` `_object_kind(item: Mapping[str, Any]) -> str` [IO-W]：执行 `object kind` 对应逻辑。
- `F L223-L230` `_set_type(target: dict[str, torch.Tensor], mask: torch.Tensor, object_type: str) -> None`：执行 `set type` 对应逻辑。
- `F L233-L239` `_set_heatmap_max(tensor: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> None`：执行 `set heatmap max` 对应逻辑。
- `F L242-L248` `_point_to_local(point: tuple[float, float], transform: OsuVideoTransform, grid: Mapping[str, torch.Tensor | float]) -> tuple[float, float]`：执行 `point to local` 对应逻辑。
- `F L251-L268` `_object_points(item: Mapping[str, Any]) -> tuple[tuple[float, float], ...]`：执行 `object points` 对应逻辑。 调用：`_distance`, `_finite_float`, `points.append`。
- `F L271-L272` `_distance(first: tuple[float, float], second: tuple[float, float]) -> float`：执行 `distance` 对应逻辑。
- `F L275-L285` `_distance_to_point(grid: Mapping[str, torch.Tensor | float], *, local_x: float, local_y: float) -> torch.Tensor`：执行 `distance to point` 对应逻辑。
- `F L288-L311` `_paint_center(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], *, local_x: float, local_y: float, radius: float, object_type: str) -> None`：执行 `paint center` 对应逻辑。 调用：`_distance_to_point`, `_set_heatmap_max`, `_set_type`, `_write_offset`。
- `F L314-L341` `_write_offset(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], *, local_x: float, local_y: float) -> None`：写入 `offset` 对应的数据或结果。
- `F L344-L383` `_paint_circle(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], item: Mapping[str, Any], *, transform: OsuVideoTransform, hit_radius: float, timestamp_ms: float, preempt_ms: float) -> None`：执行 `paint circle` 对应逻辑。 调用：`_distance_to_point`, `_finite_float`, `_object_points`, `_paint_center`, `_point_to_local`, `_set_heatmap_max`。
- `F L386-L434` `_paint_slider(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], item: Mapping[str, Any], *, transform: OsuVideoTransform, hit_radius: float) -> None`：执行 `paint slider` 对应逻辑。 调用：`_finite_float`, `_object_points`, `_paint_center`, `_paint_repeat_points`, `_paint_slider_body`, `_point_to_local`。
- `F L437-L480` `_paint_slider_body(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], points: tuple[tuple[float, float], ...], *, tube_radius: float) -> None`：执行 `paint slider body` 对应逻辑。 调用：`_set_heatmap_max`, `_set_type`, `_unoriented_direction`。
- `F L483-L485` `_unoriented_direction(vx: float, vy: float) -> tuple[float, float]`：执行 `unoriented direction` 对应逻辑。
- `F L488-L505` `_paint_repeat_points(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], points: tuple[tuple[float, float], ...], *, repeats: int, radius: float) -> None`：执行 `paint repeat points` 对应逻辑。 调用：`_distance_to_point`, `_set_type`, `repeat_points.append`。
- `F L508-L534` `_paint_spinner(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], *, transform: OsuVideoTransform) -> None`：执行 `paint spinner` 对应逻辑。 调用：`_paint_center`, `_set_heatmap_max`, `_set_type`。

## `src/traning/lib/visualization/display.py`

职责：通过独立 ffplay 子进程把标注图片显示到主机 X11。

- `F L9-L50` `launch_image_window(image_path: Path, *, title: str, ffplay_binary: str='ffplay', display: str | None=None, previous_process: subprocess.Popen[bytes] | None=None) -> subprocess.Popen[bytes]`：执行 `launch image window` 对应逻辑。

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
工程依赖：`package`, `traning.lib.coordinates`

- `F L22-L31` `_image_from_tensor(image: torch.Tensor) -> Image.Image`：执行 `image from tensor` 对应逻辑。
- `F L34-L40` `_point(transform: OsuVideoTransform, x: float, y: float) -> tuple[int, int]`：执行 `point` 对应逻辑。
- `F L43-L52` `_draw_cross(draw: ImageDraw.ImageDraw, point: tuple[int, int], color: tuple[int, int, int], size: int=12, width: int=3) -> None`：执行 `draw cross` 对应逻辑。
- `F L55-L62` `_is_target(hit_object: Mapping[str, Any], target_source_index: int | None) -> bool`：判断是否 `target` 对应的数据或结果。
- `F L65-L93` `_draw_circle(draw: ImageDraw.ImageDraw, hit_object: Mapping[str, Any], transform: OsuVideoTransform, radius: int, target_source_index: int | None) -> tuple[int, int] | None`：执行 `draw circle` 对应逻辑。 调用：`_draw_cross`, `_is_target`, `_point`。
- `F L96-L148` `_draw_slider(draw: ImageDraw.ImageDraw, hit_object: Mapping[str, Any], transform: OsuVideoTransform, radius: int, target_source_index: int | None) -> tuple[int, int] | None`：执行 `draw slider` 对应逻辑。 调用：`_draw_cross`, `_is_target`, `_point`。
- `F L151-L265` `render_annotated_frame(sample: Mapping[str, Any], *, target_source_index: int | None=None, include_all_objects: bool=False, predicted_osu_xy: tuple[float, float] | None=None, predicted_video_xy: tuple[float, float] | None=None, metadata_lines: Sequence[str]=()) -> Image.Image`：执行 `render annotated frame` 对应逻辑。 调用：`_draw_circle`, `_draw_cross`, `_draw_slider`, `_image_from_tensor`, `_is_target`, `_point`。
- `F L268-L271` `save_annotated_frame(image: Image.Image, output_path: Path) -> Path` [IO-W]：执行 `save annotated frame` 对应逻辑。

## `src/traning/lib/visualization/selection.py`

职责：根据 HitObject 起始时间反推最接近的采样帧。
工程依赖：`traning.lib.data.dataset`, `traning.lib.visualization.models`

- `F L7-L47` `select_click_frame(dataset: SegmentFrameDataset, *, segment_index: int, object_index: int=0) -> SelectedFrame`：选择 `click frame` 对应的数据或结果。 调用：`SelectedFrame`。

## `src/traning/main.py`

职责：Typer CLI；执行环境/数据检查、端到端 run、模型 smoke、空间训练、候选缓存、时序训练、决策和结果导出。
工程依赖：`traning.conf`, `traning.core.dataset_import`, `traning.core.decision`, `traning.core.full_flow`, `traning.core.result_export`, `traning.core.spatial`, `traning.core.temporal`, `traning.core.training_inheritance`, `traning.core.training_ramp`, `traning.lib.data`, `traning.lib.models`, `traning.lib.runtime`, `traning.state`

- `C L78-L79` `CliParameterError(ValueError)` [CLASS]：Raised when a plain business entry receives an invalid CLI-like value。
- `F L82-L94` `_render_report(report) -> None`：执行 `render report` 对应逻辑。
- `F L97-L100` `_format_bool(value: bool | None) -> str`：执行 `format bool` 对应逻辑。
- `F L103-L106` `_format_gib(value: float | None) -> str`：执行 `format gib` 对应逻辑。
- `F L109-L146` `_render_env_report(report) -> None`：执行 `render env report` 对应逻辑。 调用：`_format_bool`, `_format_gib`。
- `F L149-L153` `_run_dir(kind: str, *, root: Path | None=None) -> Path` [IO-W]：执行 `run dir` 对应逻辑。
- `F L156-L165` `_select_device(device: str) -> torch.device`：选择 `device` 对应的数据或结果。 调用：`CliParameterError`。
- `F L168-L171` `_load_image_tensor(path: Path) -> torch.Tensor` [IO-W]：加载 `image tensor` 对应的数据或结果。
- `F L174-L175` `_build_model_stack(settings) -> dict[str, torch.nn.Module]`：构建 `model stack` 对应的数据或结果。 调用：`build_model_stack`。
- `F L178-L322` `_execute_model_smoke(*, config: Path | None, device: torch.device, backward: bool) -> dict[str, Any]`：执行 `execute model smoke` 对应逻辑。 调用：`CudaRuntimeConfig`, `PatchStream`, `_build_model_stack`, `append_color_cues`, `autocast_context`, `collect_memory_snapshot`。
- `F L325-L332` `_render_dict_table(title: str, values: dict[str, Any]) -> None`：执行 `render dict table` 对应逻辑。
- `F L335-L367` `_render_parameter_group_score(evaluation) -> None`：执行 `render parameter group score` 对应逻辑。
- `F L370-L383` `_compact_slider_path(path: dict[str, Any]) -> dict[str, Any]`：执行 `compact slider path` 对应逻辑。
- `F L386-L390` `_write_summary_txt(output_dir: Path, summary: dict[str, Any]) -> None` [IO-W]：写入 `summary txt` 对应的数据或结果。
- `F L393-L398` `_write_json_report(path: Path, payload: dict[str, Any]) -> None` [IO-W]：写入 `json report` 对应的数据或结果。
- `F L401-L408` `inspect_training_data(*, config: Path | None=None, split: DataSplit='all')`：执行 `inspect training data` 对应逻辑。 调用：`inspect_data_input`, `load_settings`。
- `F L411-L412` `collect_training_environment()`：执行 `collect training environment` 对应逻辑。
- `F L415-L446` `preview_training_sample(*, index: int=0, split: DataSplit='train', config: Path | None=None) -> dict[str, Any]`：执行 `preview training sample` 对应逻辑。 调用：`CliParameterError`, `build_dataset`, `build_patch_windows`, `load_settings`。
- `F L449-L548` `run_training(*, config: Path=DEFAULT_TRAINING_CONFIG, split: DataSplit='train', device: str='auto', spatial_max_steps: int=1, temporal_max_steps: int=1, spatial_learning_rate: float=0.0001, temporal_learning_rate: float=0.0001, patch_limit: int=1, cache_max_frames: int=1, sequence_length: int | None=None, candidate_slots: int | None=None, parameter_group_id: str='pg-0001', render_gallery: bool=True, gallery_output_root: Path | None=None, gallery_samples_per_group: int | None=None, progress_ui: str='auto', progress_language: str='zh-CN', inherit_from: Path | str | None=None, resume_policy: str='none')`：执行 `run training` 对应逻辑。 调用：`CliParameterError`, `FullTrainingRunConfig`, `_run_dir`, `_safe_create_inheritance_package`, `_select_device`, `_write_json_report`。
- `F L551-L587` `_safe_create_inheritance_package(*, run_dir: Path, settings, config: Path, result, reporter)`：执行 `safe create inheritance package` 对应逻辑。 调用：`create_inheritance_package`, `result.as_summary`, `result.evaluation.as_dict`。
- `F L590-L624` `run_training_job_spec(*, job: Path, config: Path=DEFAULT_TRAINING_CONFIG, device: str='auto', execute: bool=True)` [IO-R]：执行 `run training job spec` 对应逻辑。 调用：`CliParameterError`, `result.as_summary`, `run_training`。
- `F L627-L668` `run_training_ramp_job(*, config: Path=DEFAULT_TRAINING_CONFIG, device: str='auto', output_root: Path=Path('artifacts') / 'training_ramp', target_config: Path | None=None, run_id: str | None=None, auto_launch_full: bool=False, force_level: bool=False, max_levels: int | None=None, run_full_checks: bool=True, progress_ui: str='auto', progress_language: str='zh-CN', inherit_from: Path | str | None=None, resume_policy: str='none')`：执行 `run training ramp job` 对应逻辑。 调用：`CliParameterError`, `_select_device`, `load_inheritance_package`, `load_settings`, `run_training_ramp`。
- `F L671-L727` `run_full_flow_job(*, config: Path=DEFAULT_TRAINING_CONFIG, device: str='auto', mode: str='execute', output_root: Path=DEFAULT_FULL_FLOW_ROOT, target_config: Path | None=None, run_id: str | None=None, auto_launch_full: bool=False, force_level: bool=False, max_levels: int | None=None, run_full_checks: bool=True, progress_ui: str='auto', progress_language: str='zh-CN', inherit_from: Path | str | None=None, resume_policy: str='none', resume: bool=False, from_stage: str | None=None, until_stage: str | None=None, force_stages: tuple[str, ...]=(), skip_stages: tuple[str, ...]=())`：执行 `run full flow job` 对应逻辑。 调用：`CliParameterError`, `FullFlowConfig`, `load_full_flow_status`, `run_full_flow`。
- `F L730-L745` `run_model_smoke(*, config: Path=DEFAULT_TRAINING_CONFIG, device: str='cpu', backward: bool=True) -> dict[str, Any]`：执行 `run model smoke` 对应逻辑。 调用：`_execute_model_smoke`, `_run_dir`, `_select_device`, `_write_summary_txt`。
- `F L748-L804` `run_spatial_decode_smoke(*, config: Path=DEFAULT_TRAINING_CONFIG, split: DataSplit='train', index: int=0, device: str='cpu', max_candidates: int=16, score_threshold: float=0.0, nms_radius_px: float=32.0, slider_threshold: float=0.5, max_slider_paths: int=16, patch_limit: int | None=None) -> dict[str, Any]` [IO-W]：执行 `run spatial decode smoke` 对应逻辑。 调用：`CliParameterError`, `_compact_slider_path`, `_run_dir`, `_select_device`, `_write_summary_txt`, `build_dataset`。
- `F L807-L838` `run_candidate_cache_build(*, config: Path=DEFAULT_TRAINING_CONFIG, split: DataSplit='train', device: str='cpu', max_frames: int | None=None, patch_limit: int | None=None, max_candidates: int | None=None, score_threshold: float | None=None, nms_radius_px: float | None=None, slider_threshold: float | None=None, max_slider_paths: int | None=None, output: Path | None=None)`：执行 `run candidate cache build` 对应逻辑。 调用：`_run_dir`, `_select_device`, `generate_candidate_cache`, `load_settings`。
- `F L841-L855` `run_memory_profile(*, config: Path=DEFAULT_TRAINING_CONFIG, device: str='cuda') -> dict[str, Any]`：执行 `run memory profile` 对应逻辑。 调用：`_execute_model_smoke`, `_run_dir`, `_select_device`, `_write_summary_txt`。
- `F L858-L880` `visualize_patch_windows(*, input_image: Path, output: Path | None=None, config: Path=DEFAULT_TRAINING_CONFIG) -> Path` [IO-W]：执行 `visualize patch windows` 对应逻辑。 调用：`PatchStream`, `_run_dir`, `load_settings`, `stream.metas`。
- `F L883-L964` `visualize_fusion_context(*, input_image: Path, output: Path | None=None, config: Path=DEFAULT_TRAINING_CONFIG, device: str='cpu') -> Path` [IO-W]：执行 `visualize fusion context` 对应逻辑。 调用：`CudaRuntimeConfig`, `PatchStream`, `_build_model_stack`, `_load_image_tensor`, `_run_dir`, `_select_device`。
- `F L967-L987` `run_spatial_training_job(*, config: Path=DEFAULT_TRAINING_CONFIG, split: DataSplit='train', device: str='auto', max_steps: int=1, learning_rate: float=0.0001, patch_limit: int | None=None)`：执行 `run spatial training job` 对应逻辑。 调用：`_run_dir`, `_select_device`, `load_settings`, `run_spatial_training`。
- `F L990-L1004` `spatial_training_oom_guidance(config: Path) -> str`：执行 `spatial training oom guidance` 对应逻辑。 调用：`format_oom_guidance`, `load_settings`。
- `F L1007-L1029` `run_temporal_training_job(*, config: Path=DEFAULT_TRAINING_CONFIG, cache: Path, device: str='auto', max_steps: int=1, learning_rate: float=0.0001, sequence_length: int | None=None, candidate_slots: int | None=None)`：执行 `run temporal training job` 对应逻辑。 调用：`_run_dir`, `_select_device`, `load_settings`, `run_temporal_training`。
- `F L1032-L1049` `run_decision_job(*, config: Path=DEFAULT_TRAINING_CONFIG, cache: Path, checkpoint: Path, output: Path | None=None, device: str='auto')`：执行 `run decision job` 对应逻辑。 调用：`_run_dir`, `_select_device`, `load_settings`, `run_temporal_decision`。
- `F L1052-L1066` `run_label_visualization(*, segment_index: int=0, object_index: int=0, output: Path | None=None, show: bool=False, config: Path | None=None)`：执行 `run label visualization` 对应逻辑。 调用：`load_settings`, `visualize_click_label`。
- `F L1069-L1081` `run_gallery_export(*, results: Path, output_root: Path | None=None, samples_per_group: int | None=None, config: Path | None=None)`：执行 `run gallery export` 对应逻辑。 调用：`load_batch_gallery_request`, `load_settings`, `save_annotation_gallery`。
- `F L1084-L1085` `_raise_cli_parameter(error: CliParameterError) -> NoReturn`：执行 `raise cli parameter` 对应逻辑。
- `F L1089-L1096` `data_check(config: Path | None=typer.Option(None, '--config'), split: DataSplit=typer.Option('all', '--split')) -> None` [CLI]：执行 `data check` 对应逻辑。 调用：`_render_report`, `inspect_training_data`。
- `F L1100-L1119` `env_check(strict: bool=typer.Option(False, '--strict/--no-strict', help='Exit non-zero when required runtime dependencies are missing.'), require_cuda: bool=typer.Option(False, '--require-cuda/--no-require-cuda', help='Treat CUDA unavailability as a failure in strict mode.')) -> None` [CLI]：执行 `env check` 对应逻辑。 调用：`_render_env_report`, `collect_training_environment`。
- `F L1123-L1136` `data_preview(index: int=typer.Option(0, '--index', min=0), split: DataSplit=typer.Option('train', '--split'), config: Path | None=typer.Option(None, '--config')) -> None` [CLI]：执行 `data preview` 对应逻辑。 调用：`_raise_cli_parameter`, `preview_training_sample`。
- `F L1140-L1221` `run(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), split: DataSplit=typer.Option('train', '--split'), device: str=typer.Option('auto', '--device', help='cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.'), spatial_max_steps: int=typer.Option(1, '--spatial-max-steps', min=1), temporal_max_steps: int=typer.Option(1, '--temporal-max-steps', min=1), spatial_learning_rate: float=typer.Option(0.0001, '--spatial-lr', min=1e-08), temporal_learning_rate: float=typer.Option(0.0001, '--temporal-lr', min=1e-08), patch_limit: int=typer.Option(1, '--patch-limit', min=0, help='0 means process all patches in each frame.'), cache_max_frames: int=typer.Option(1, '--cache-max-frames', min=0, help='0 means no frame limit for candidate cache generation.'), sequence_length: int | None=typer.Option(None, '--sequence-length', min=1), candidate_slots: int | None=typer.Option(None, '--candidate-slots', min=1), parameter_group_id: str=typer.Option('pg-0001', '--parameter-group-id'), render_gallery: bool=typer.Option(True, '--render-gallery/--no-render-gallery', help='Render the best parameter group gallery after the training round.'), gallery_output_root: Path | None=typer.Option(None, '--gallery-output-root'), gallery_samples_per_group: int | None=typer.Option(None, '--gallery-samples-per-group', min=1), progress_ui: str=typer.Option('auto', '--progress-ui'), progress_language: str=typer.Option('zh-CN', '--progress-language'), inherit_from: str | None=typer.Option(None, '--inherit-from'), resume_policy: str=typer.Option('none', '--resume-policy'), resume: bool=typer.Option(False, '--resume')) -> None` [CLI]：执行该处理器的完整工作流。 调用：`_raise_cli_parameter`, `_render_dict_table`, `_render_parameter_group_score`, `result.as_summary`, `run_training`。
- `F L1225-L1247` `run_job(job: Path=typer.Option(..., '--job', exists=True, file_okay=True, dir_okay=False, readable=True), config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), device: str=typer.Option('auto', '--device'), execute: bool=typer.Option(True, '--execute/--dry-run')) -> None` [CLI]：执行 `run job` 对应逻辑。 调用：`_raise_cli_parameter`, `_render_dict_table`, `run_training_job_spec`。
- `F L1251-L1314` `full_flow(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), device: str=typer.Option('auto', '--device', help='cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.'), mode: str=typer.Option('execute', '--mode', help='execute, plan, dry-run, or status.'), output_root: Path=typer.Option(DEFAULT_FULL_FLOW_ROOT, '--output-root'), target_config: Path | None=typer.Option(None, '--target-config'), run_id: str | None=typer.Option(None, '--run-id'), auto_launch_full: bool=typer.Option(False, '--auto-launch-full/--no-auto-launch-full', help='Launch finite full training after ramp gates pass.'), force_level: bool=typer.Option(False, '--force-level/--resume-passed-levels'), max_levels: int | None=typer.Option(None, '--max-levels', min=1), run_full_checks: bool=typer.Option(True, '--run-full-checks/--skip-full-checks'), progress_ui: str=typer.Option('auto', '--progress-ui'), progress_language: str=typer.Option('zh-CN', '--progress-language'), inherit_from: str | None=typer.Option(None, '--inherit-from'), resume_policy: str=typer.Option('none', '--resume-policy'), resume: bool=typer.Option(False, '--resume'), from_stage: str | None=typer.Option(None, '--from-stage'), until_stage: str | None=typer.Option(None, '--until-stage'), force_stage: list[str] | None=typer.Option(None, '--force-stage'), skip_stage: list[str] | None=typer.Option(None, '--skip-stage')) -> None` [CLI]：执行 `full flow` 对应逻辑。 调用：`CliParameterError`, `_raise_cli_parameter`, `_render_dict_table`, `result.as_dict`, `run_full_flow_job`。
- `F L1318-L1326` `full_flow_status(output_root: Path=typer.Option(DEFAULT_FULL_FLOW_ROOT, '--output-root'), run_id: str | None=typer.Option(None, '--run-id')) -> None` [CLI]：执行 `full flow status` 对应逻辑。 调用：`CliParameterError`, `_raise_cli_parameter`, `_render_dict_table`, `load_full_flow_status`, `result.as_dict`。
- `F L1330-L1396` `ramp_to_full(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), device: str=typer.Option('auto', '--device', help='cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.'), output_root: Path=typer.Option(Path('artifacts') / 'training_ramp', '--output-root'), target_config: Path | None=typer.Option(None, '--target-config'), run_id: str | None=typer.Option(None, '--run-id', help='Resume or extend an existing ramp output run id.'), auto_launch_full: bool=typer.Option(False, '--auto-launch-full/--no-auto-launch-full', help='Launch the finite full training run after all ramp gates pass.'), force_level: bool=typer.Option(False, '--force-level/--resume-passed-levels', help='Re-run levels even when their level_state.json is already passed.'), max_levels: int | None=typer.Option(None, '--max-levels', min=1, help='Limit levels for controlled validation; omit for target ramp.'), run_full_checks: bool=typer.Option(True, '--run-full-checks/--skip-full-checks', help='Run full pytest checks during preflight.'), progress_ui: str=typer.Option('auto', '--progress-ui'), progress_language: str=typer.Option('zh-CN', '--progress-language'), inherit_from: str | None=typer.Option(None, '--inherit-from'), resume_policy: str=typer.Option('none', '--resume-policy'), resume: bool=typer.Option(False, '--resume')) -> None` [CLI]：执行 `ramp to full` 对应逻辑。 调用：`_raise_cli_parameter`, `_render_dict_table`, `result.as_dict`, `run_training_ramp_job`。
- `F L1400-L1421` `model_smoke(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), device: str=typer.Option('cpu', '--device', help='cpu, cuda, or auto. CPU is the default smoke path.'), backward: bool=typer.Option(True, '--backward/--no-backward', help='Run backward and optimizer step in addition to forward.')) -> None` [CLI]：执行 `model smoke` 对应逻辑。 调用：`_raise_cli_parameter`, `_render_dict_table`, `run_model_smoke`。
- `F L1425-L1452` `spatial_decode_smoke(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), split: DataSplit=typer.Option('train', '--split'), index: int=typer.Option(0, '--index', min=0), device: str=typer.Option('cpu', '--device'), max_candidates: int=typer.Option(16, '--max-candidates', min=1), score_threshold: float=typer.Option(0.0, '--score-threshold', min=0.0), nms_radius_px: float=typer.Option(32.0, '--nms-radius-px', min=0.0), slider_threshold: float=typer.Option(0.5, '--slider-threshold', min=0.0, max=1.0), max_slider_paths: int=typer.Option(16, '--max-slider-paths', min=1), patch_limit: int | None=typer.Option(None, '--patch-limit', min=1)) -> None` [CLI]：执行 `spatial decode smoke` 对应逻辑。 调用：`_raise_cli_parameter`, `_render_dict_table`, `run_spatial_decode_smoke`。
- `F L1456-L1495` `build_candidate_cache(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), split: DataSplit=typer.Option('train', '--split'), device: str=typer.Option('cpu', '--device'), max_frames: int | None=typer.Option(None, '--max-frames', min=1), patch_limit: int | None=typer.Option(None, '--patch-limit', min=1), max_candidates: int | None=typer.Option(None, '--max-candidates', min=1), score_threshold: float | None=typer.Option(None, '--score-threshold', min=0.0, max=1.0), nms_radius_px: float | None=typer.Option(None, '--nms-radius-px', min=0.0), slider_threshold: float | None=typer.Option(None, '--slider-threshold', min=0.0, max=1.0), max_slider_paths: int | None=typer.Option(None, '--max-slider-paths', min=1), output: Path | None=typer.Option(None, '--output')) -> None` [CLI]：构建并返回 `candidate cache` 对应的数据或结果。 调用：`_raise_cli_parameter`, `_render_dict_table`, `result.as_dict`, `run_candidate_cache_build`。
- `F L1499-L1516` `memory_profile(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), device: str=typer.Option('cuda', '--device', help='cuda, cpu, or auto. CUDA is the default for memory profiling.')) -> None` [CLI]：执行 `memory profile` 对应逻辑。 调用：`_raise_cli_parameter`, `_render_dict_table`, `_select_device`, `run_memory_profile`。
- `F L1520-L1537` `visualize_patches(input_image: Path=typer.Option(..., '--input', exists=True, file_okay=True, dir_okay=False, readable=True), output: Path | None=typer.Option(None, '--output'), config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config')) -> None` [CLI]：执行 `visualize patches` 对应逻辑。 调用：`visualize_patch_windows`。
- `F L1541-L1563` `visualize_fusion(input_image: Path=typer.Option(..., '--input', exists=True, file_okay=True, dir_okay=False, readable=True), output: Path | None=typer.Option(None, '--output'), config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), device: str=typer.Option('cpu', '--device')) -> None` [CLI]：执行 `visualize fusion` 对应逻辑。 调用：`_raise_cli_parameter`, `visualize_fusion_context`。
- `F L1567-L1594` `train_spatial(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), split: DataSplit=typer.Option('train', '--split'), device: str=typer.Option('auto', '--device', help='cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.'), max_steps: int=typer.Option(1, '--max-steps', min=1), learning_rate: float=typer.Option(0.0001, '--lr', min=1e-08), patch_limit: int | None=typer.Option(None, '--patch-limit', min=1)) -> None` [CLI]：执行 `train spatial` 对应逻辑。 调用：`_raise_cli_parameter`, `_render_dict_table`, `result.as_dict`, `run_spatial_training_job`, `spatial_training_oom_guidance`。
- `F L1598-L1631` `train_temporal(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), cache: Path=typer.Option(..., '--cache', exists=True, file_okay=False, dir_okay=True, readable=True, help='Candidate cache directory containing manifest.json and frames.jsonl.'), device: str=typer.Option('auto', '--device', help='cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.'), max_steps: int=typer.Option(1, '--max-steps', min=1), learning_rate: float=typer.Option(0.0001, '--lr', min=1e-08), sequence_length: int | None=typer.Option(None, '--sequence-length', min=1), candidate_slots: int | None=typer.Option(None, '--candidate-slots', min=1)) -> None` [CLI]：执行 `train temporal` 对应逻辑。 调用：`_raise_cli_parameter`, `_render_dict_table`, `result.as_dict`, `run_temporal_training_job`。
- `F L1635-L1672` `run_decision(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), cache: Path=typer.Option(..., '--cache', exists=True, file_okay=False, dir_okay=True, readable=True, help='Candidate cache directory containing manifest.json and frames.jsonl.'), checkpoint: Path=typer.Option(..., '--checkpoint', exists=True, file_okay=True, dir_okay=False, readable=True, help='Temporal checkpoint produced by train-temporal.'), output: Path | None=typer.Option(None, '--output'), device: str=typer.Option('auto', '--device', help='cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.')) -> None` [CLI]：执行 `run decision` 对应逻辑。 调用：`_raise_cli_parameter`, `_render_dict_table`, `result.as_dict`, `run_decision_job`。
- `F L1676-L1695` `visualize_label(segment_index: int=typer.Option(0, '--segment-index', min=0), object_index: int=typer.Option(0, '--object-index', min=0), output: Path | None=typer.Option(None, '--output'), show: bool=typer.Option(False, '--show/--no-show'), config: Path | None=typer.Option(None, '--config')) -> None` [CLI]：执行 `visualize label` 对应逻辑。 调用：`run_label_visualization`。
- `F L1699-L1733` `save_gallery(results: Path=typer.Option(..., '--results', exists=True, file_okay=True, dir_okay=False, readable=True), output_root: Path | None=typer.Option(None, '--output-root'), samples_per_group: int | None=typer.Option(None, '--samples-per-group', min=1), config: Path | None=typer.Option(None, '--config')) -> None` [CLI]：执行 `save gallery` 对应逻辑。 调用：`run_gallery_export`。

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

- `C L24-L60` `FrameEvaluation(BaseModel)` [CLASS]：封装 `FrameEvaluation` 相关数据或行为。
- `M L40-L43` `FrameEvaluation._nonnegative_frame_index(cls, value: int) -> int` [VALIDATOR]：执行 `nonnegative frame index` 对应逻辑。
- `M L47-L53` `FrameEvaluation._finite_optional_point(cls, value: tuple[float, float] | None) -> tuple[float, float] | None` [VALIDATOR]：执行 `finite optional point` 对应逻辑。
- `M L57-L60` `FrameEvaluation._finite_optional_metric(cls, value: float | None) -> float | None` [VALIDATOR]：执行 `finite optional metric` 对应逻辑。
- `C L63-L76` `TrialGalleryEvaluation(BaseModel)` [CLASS]：封装 `TrialGalleryEvaluation` 相关数据或行为。
- `M L73-L76` `TrialGalleryEvaluation._finite_score(cls, value: float) -> float` [VALIDATOR]：执行 `finite score` 对应逻辑。
- `C L79-L106` `BatchGalleryRequest(BaseModel)` [CLASS]：封装 `BatchGalleryRequest` 相关数据或行为。
- `M L87-L93` `BatchGalleryRequest._require_trials(cls, value: tuple[TrialGalleryEvaluation, ...]) -> tuple[TrialGalleryEvaluation, ...]` [VALIDATOR]：执行 `require trials` 对应逻辑。
- `M L96-L102` `BatchGalleryRequest._require_one_score_version(self) -> BatchGalleryRequest` [VALIDATOR]：执行 `require one score version` 对应逻辑。
- `M L105-L106` `BatchGalleryRequest.best_trial(self) -> TrialGalleryEvaluation` [PROPERTY]：执行 `best trial` 对应逻辑。
- `F L109-L114` `load_batch_gallery_request(path: Path) -> BatchGalleryRequest` [IO-R]：加载 `batch gallery request` 对应的数据或结果。

## `src/traning/state/run_state.py`

职责：保存 trial、课程阶段、rung、预算和全局步数的运行状态。
工程依赖：`traning.state.experiment_schema`

- `C L9-L16` `RunState` [CLASS]：封装 `RunState` 相关数据或行为。

## `src/traning/state/versioning.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package.coordinates`

- `C L18-L24` `CodeVersion` [CLASS]：封装 `CodeVersion` 相关数据或行为。
- `M L23-L24` `CodeVersion.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `F L27-L44` `collect_code_version(repo_root: Path | None=None) -> CodeVersion`：执行 `collect code version` 对应逻辑。 调用：`CodeVersion`。
- `F L47-L60` `dataset_version(settings: Any) -> str`：执行 `dataset version` 对应逻辑。
- `F L63-L71` `version_manifest(settings: Any) -> dict[str, Any]`：执行 `version manifest` 对应逻辑。 调用：`collect_code_version`, `collect_code_version.as_dict`, `dataset_version`。
- `F L74-L93` `ensure_compatible_versions(left: Mapping[str, Any], right: Mapping[str, Any], *, override: bool=False) -> tuple[bool, tuple[str, ...]]`：确保 `compatible versions` 对应的数据或结果。

## `src/traning/tests/full_checks/runner.py`

职责：traning 全面检测统一入口；运行 full_checks 下的 pytest。
工程依赖：`package.checks`

- `F L14-L37` `run_full_checks() -> StartupCheckReport`：执行 `run full checks` 对应逻辑。 调用：`_run_pytest`, `_tail`。
- `F L40-L52` `_run_pytest(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]` [PROCESS]：执行 `run pytest` 对应逻辑。 调用：`subprocess.run`。
- `F L55-L56` `_tail(text: str, *, max_lines: int=80) -> str`：执行 `tail` 对应逻辑。

## `src/traning/tests/full_checks/test_candidate_cache.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.decision`, `traning.lib.training`, `traning.lib.training.spatial_decode`

- `F L22-L41` `_candidate(*, score: float=0.55, object_type: str='slider_head') -> SpatialCandidate`：执行 `candidate` 对应逻辑。 调用：`SpatialCandidate`。
- `F L44-L58` `_slider_path(*, ambiguous: bool=False) -> SliderPathCandidate`：执行 `slider path` 对应逻辑。 调用：`SliderPathCandidate`。
- `C L61-L169` `CandidateCacheTests(unittest.TestCase)` [CLASS]：封装 `CandidateCacheTests` 相关数据或行为。
- `M L62-L104` `CandidateCacheTests.test_record_keeps_embedding_and_candidate_ambiguity(self) -> None`：执行 `test record keeps embedding and candidate ambiguity` 对应逻辑。 调用：`_candidate`, `_slider_path`, `build_candidate_cache_record`, `self.assertEqual`, `self.assertIn`。
- `M L106-L138` `CandidateCacheTests.test_generate_candidate_cache_writes_manifest_and_jsonl(self) -> None` [IO-R]：执行 `test generate candidate cache writes manifest and jsonl` 对应逻辑。 调用：`Settings`, `_candidate`, `_slider_path`, `generate_candidate_cache`, `self.assertEqual`。
- `M L140-L169` `CandidateCacheTests.test_local_consistency_review_resolves_supported_ambiguity(self) -> None`：执行 `test local consistency review resolves supported ambiguity` 对应逻辑。 调用：`Settings`, `_candidate`, `_slider_path`, `build_candidate_cache_record`, `self.assertEqual`, `self.assertFalse`。

## `src/traning/tests/full_checks/test_causal_temporal.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.models`

- `C L10-L145` `CausalTemporalTests(unittest.TestCase)` [CLASS]：封装 `CausalTemporalTests` 相关数据或行为。
- `M L11-L21` `CausalTemporalTests.test_future_frames_do_not_change_past_outputs(self) -> None`：执行 `test future frames do not change past outputs` 对应逻辑。 调用：`CausalTemporalModel`, `self.assertTrue`。
- `M L23-L31` `CausalTemporalTests.test_reset_state_repeats_output(self) -> None`：执行 `test reset state repeats output` 对应逻辑。 调用：`CausalTemporalModel`, `model.initial_state`, `model.step`, `self.assertTrue`。
- `M L33-L37` `CausalTemporalTests.test_batch_size_one_runs(self) -> None`：执行 `test batch size one runs` 对应逻辑。 调用：`CausalTemporalModel`, `model.initial_state`, `model.step`, `self.assertEqual`。
- `M L39-L53` `CausalTemporalTests.test_smet_sparse_heads_run(self) -> None`：执行 `test smet sparse heads run` 对应逻辑。 调用：`CausalTemporalModel`, `self.assertEqual`, `self.assertTrue`。
- `M L55-L81` `CausalTemporalTests.test_smet_sparse_heads_backward_after_dynamic_updates(self) -> None`：执行 `test smet sparse heads backward after dynamic updates` 对应逻辑。 调用：`CausalTemporalModel`, `self.assertIsNotNone`, `self.assertTrue`。
- `M L83-L107` `CausalTemporalTests.test_smet_training_forward_does_not_mutate_mask_buffers(self) -> None`：执行 `test smet training forward does not mutate mask buffers` 对应逻辑。 调用：`CausalTemporalModel`, `self.assertEqual`。
- `M L109-L121` `CausalTemporalTests.test_mutating_future_window_does_not_change_prefix(self) -> None`：执行 `test mutating future window does not change prefix` 对应逻辑。 调用：`CausalTemporalModel`, `self.assertTrue`。
- `M L123-L145` `CausalTemporalTests.test_segmented_execution_matches_continuous_and_batch_isolated(self) -> None`：执行 `test segmented execution matches continuous and batch isolated` 对应逻辑。 调用：`CausalTemporalModel`, `model.initial_state`, `model.step`, `segmented.append`, `self.assertTrue`。

## `src/traning/tests/full_checks/test_cli_adapters.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning`, `traning.core.decision`

- `C L15-L96` `TrainingCliAdapterTests(unittest.TestCase)` [CLASS]：封装 `TrainingCliAdapterTests` 相关数据或行为。
- `M L16-L43` `TrainingCliAdapterTests.test_business_run_training_calls_pipeline_without_typer(self) -> None`：执行 `test business run training calls pipeline without typer` 对应逻辑。 调用：`self.assertEqual`, `self.assertIs`, `self.assertIsInstance`, `self.assertIsNone`, `training_main.run_training`。
- `M L45-L85` `TrainingCliAdapterTests.test_run_cli_passes_arguments_to_business_function(self) -> None`：执行 `test run cli passes arguments to business function` 对应逻辑。 调用：`self.assertEqual`。
- `M L87-L96` `TrainingCliAdapterTests.test_cli_parameter_error_maps_to_typer_exit(self) -> None`：执行 `test cli parameter error maps to typer exit` 对应逻辑。 调用：`self.assertEqual`, `self.assertIn`, `training_main.CliParameterError`。

## `src/traning/tests/full_checks/test_color_cues.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.lib.data`, `traning.lib.models`

- `C L16-L52` `ColorCueTests(unittest.TestCase)` [CLASS]：封装 `ColorCueTests` 相关数据或行为。
- `M L17-L28` `ColorCueTests.test_osu_basic_cues_highlight_colored_target_and_white_number(self) -> None`：执行 `test osu basic cues highlight colored target and white number` 对应逻辑。 调用：`extract_osu_basic_color_cues`, `self.assertEqual`, `self.assertGreater`, `self.assertLess`。
- `M L30-L37` `ColorCueTests.test_append_color_cues_is_configurable(self) -> None`：执行 `test append color cues is configurable` 对应逻辑。 调用：`append_color_cues`, `color_cue_channel_count`, `self.assertEqual`, `self.assertIs`。
- `M L39-L52` `ColorCueTests.test_model_stack_accepts_augmented_input_channels(self) -> None`：执行 `test model stack accepts augmented input channels` 对应逻辑。 调用：`Settings`, `build_model_stack`, `self.assertEqual`。

## `src/traning/tests/full_checks/test_coordinates.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`

- `C L15-L42` `CoordinateTests(unittest.TestCase)` [CLASS]：封装 `CoordinateTests` 相关数据或行为。
- `M L16-L30` `CoordinateTests.test_local_global_round_trip(self) -> None`：执行 `test local global round trip` 对应逻辑。 调用：`PatchMeta`, `global_to_local`, `local_to_global`, `self.assertEqual`。
- `M L32-L37` `CoordinateTests.test_global_to_patch_indices_returns_all_overlaps(self) -> None`：执行 `test global to patch indices returns all overlaps` 对应逻辑。 调用：`PatchMeta`, `global_to_patch_indices`, `self.assertEqual`。
- `M L39-L42` `CoordinateTests.test_feature_grid_round_trip(self) -> None`：执行 `test feature grid round trip` 对应逻辑。 调用：`feature_grid_to_image`, `image_to_feature_grid`, `self.assertEqual`。

## `src/traning/tests/full_checks/test_cross_patch_ring.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`

- `C L11-L36` `CrossPatchRingTests(unittest.TestCase)` [CLASS]：封装 `CrossPatchRingTests` 相关数据或行为。
- `M L12-L36` `CrossPatchRingTests.test_ring_is_visible_from_multiple_patches_with_global_context(self) -> None`：执行 `test ring is visible from multiple patches with global context` 对应逻辑。 调用：`PatchStream`, `make_cross_patch_ring`, `sample_global_feature`, `self.assertGreaterEqual`, `self.assertTrue`, `stream.metas`。

## `src/traning/tests/full_checks/test_cross_patch_slider.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`

- `C L11-L36` `CrossPatchSliderTests(unittest.TestCase)` [CLASS]：封装 `CrossPatchSliderTests` 相关数据或行为。
- `M L12-L36` `CrossPatchSliderTests.test_slider_spans_multiple_patches_with_shared_global_context(self) -> None`：执行 `test slider spans multiple patches with shared global context` 对应逻辑。 调用：`PatchStream`, `make_cross_patch_slider`, `sample_global_feature`, `self.assertGreater`, `self.assertGreaterEqual`, `stream.metas`。

## `src/traning/tests/full_checks/test_cuda_config.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`

- `C L10-L33` `CudaConfigTests(unittest.TestCase)` [CLASS]：封装 `CudaConfigTests` 相关数据或行为。
- `M L11-L22` `CudaConfigTests.test_memory_defaults_enable_cuda_optimized_runtime(self) -> None`：执行 `test memory defaults enable cuda optimized runtime` 对应逻辑。 调用：`MemoryConfig`, `self.assertEqual`, `self.assertFalse`, `self.assertTrue`。
- `M L24-L33` `CudaConfigTests.test_loader_worker_options_require_workers(self) -> None`：执行 `test loader worker options require workers` 对应逻辑。 调用：`LoaderSettings`, `self.assertEqual`, `self.assertRaises`, `self.assertTrue`。

## `src/traning/tests/full_checks/test_cuda_optimization.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.runtime`

- `C L21-L100` `CudaOptimizationTests(unittest.TestCase)` [CLASS]：封装 `CudaOptimizationTests` 相关数据或行为。
- `M L22-L31` `CudaOptimizationTests.test_cpu_runtime_keeps_cuda_only_options_inactive(self) -> None`：执行 `test cpu runtime keeps cuda only options inactive` 对应逻辑。 调用：`CudaRuntimeConfig`, `configure_torch_runtime`, `self.assertEqual`, `self.assertFalse`。
- `M L33-L40` `CudaOptimizationTests.test_grad_scaler_auto_is_disabled_without_fp16_cuda(self) -> None`：执行 `test grad scaler auto is disabled without fp16 cuda` 对应逻辑。 调用：`amp_uses_grad_scaler`, `create_grad_scaler`, `self.assertFalse`。
- `M L42-L49` `CudaOptimizationTests.test_resolved_amp_dtype_values_can_be_reused(self) -> None`：执行 `test resolved amp dtype values can be reused` 对应逻辑。 调用：`amp_uses_grad_scaler`, `resolve_amp_dtype`, `self.assertEqual`, `self.assertIsNone`, `self.assertTrue`。
- `M L51-L58` `CudaOptimizationTests.test_tensor_to_device_preserves_cpu_contiguous_layout(self) -> None`：执行 `test tensor to device preserves cpu contiguous layout` 对应逻辑。 调用：`self.assertTrue`, `tensor_to_device`。
- `M L60-L71` `CudaOptimizationTests.test_cpu_memory_budget_reports_system_reserve(self) -> None`：执行 `test cpu memory budget reports system reserve` 对应逻辑。 调用：`enforce_runtime_memory_budget`, `self.assertEqual`, `self.assertGreater`, `self.assertIsNone`。
- `M L73-L82` `CudaOptimizationTests.test_cpu_memory_budget_rejects_unavailable_reserve(self) -> None`：执行 `test cpu memory budget rejects unavailable reserve` 对应逻辑。 调用：`enforce_runtime_memory_budget`, `self.assertRaises`。
- `M L84-L100` `CudaOptimizationTests.test_cuda_channels_last_when_available(self) -> None`：执行 `test cuda channels last when available` 对应逻辑。 调用：`module_to_device`, `self.assertTrue`, `self.skipTest`, `tensor_to_device`。

## `src/traning/tests/full_checks/test_dataset_split_manifest.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package.dataset_split`, `traning.conf`, `traning.core.dataset_import.preflight`, `traning.lib.data.models`

- `F L14-L17` `_segment(root: Path, item_name: str, segment_id: str) -> None` [IO-W]：执行 `segment` 对应逻辑。
- `C L20-L55` `DatasetSplitManifestTests(unittest.TestCase)` [CLASS]：封装 `DatasetSplitManifestTests` 相关数据或行为。
- `M L21-L55` `DatasetSplitManifestTests.test_discovery_uses_split_manifest_when_present(self) -> None`：执行 `test discovery uses split manifest when present` 对应逻辑。 调用：`DiscoveryResult`, `Settings`, `_segment`, `discover_data_input`, `self.assertEqual`。

## `src/traning/tests/full_checks/test_decision_output_scoring.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.optimization`

- `C L11-L86` `DecisionOutputScoringTests(unittest.TestCase)` [CLASS]：封装 `DecisionOutputScoringTests` 相关数据或行为。
- `M L12-L86` `DecisionOutputScoringTests.test_scores_parameter_group_from_cache_and_decisions(self) -> None` [IO-W]：执行 `test scores parameter group from cache and decisions` 对应逻辑。 调用：`build_batch_gallery_request`, `result.as_summary`, `score_decision_outputs`, `self.assertEqual`, `self.assertGreater`。

## `src/traning/tests/full_checks/test_discovery.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`

- `F L11-L36` `_write_segment(root: Path, item_name: str, segment_id: str) -> None` [IO-W]：写入 `segment` 对应的数据或结果。
- `C L39-L60` `DiscoverySplitTests(unittest.TestCase)` [CLASS]：封装 `DiscoverySplitTests` 相关数据或行为。
- `M L40-L49` `DiscoverySplitTests.test_include_items_filters_records_before_loading(self) -> None`：执行 `test include items filters records before loading` 对应逻辑。 调用：`_write_segment`, `discover_segments`, `self.assertEqual`。
- `M L51-L60` `DiscoverySplitTests.test_exclude_items_removes_records(self) -> None`：执行 `test exclude items removes records` 对应逻辑。 调用：`_write_segment`, `discover_segments`, `self.assertEqual`。

## `src/traning/tests/full_checks/test_env_check.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L11-L24` `EnvironmentCheckTests(unittest.TestCase)` [CLASS]：封装 `EnvironmentCheckTests` 相关数据或行为。
- `M L12-L17` `EnvironmentCheckTests.test_collect_environment_report_is_non_destructive(self) -> None`：执行 `test collect environment report is non destructive` 对应逻辑。 调用：`self.assertIsNotNone`, `self.assertTrue`。
- `M L19-L24` `EnvironmentCheckTests.test_required_package_specs_are_reported(self) -> None`：执行 `test required package specs are reported` 对应逻辑。 调用：`self.assertTrue`。

## `src/traning/tests/full_checks/test_full_flow.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.full_flow`, `traning.core.full_flow.orchestrator`, `traning.core.full_flow.result`

- `C L27-L303` `FullFlowTests(unittest.TestCase)` [CLASS]：封装 `FullFlowTests` 相关数据或行为。
- `M L28-L33` `FullFlowTests.test_stage_ids_are_unique_and_ordered(self) -> None`：执行 `test stage ids are unique and ordered` 对应逻辑。 调用：`self.assertEqual`, `self.assertTrue`, `stage_ids`。
- `M L35-L59` `FullFlowTests.test_plan_mode_writes_manifest_state_and_reports(self) -> None`：执行 `test plan mode writes manifest state and reports` 对应逻辑。 调用：`FullFlowConfig`, `load_full_flow_status`, `run_full_flow`, `self.assertEqual`, `self.assertTrue`。
- `M L61-L73` `FullFlowTests.test_critical_stages_cannot_be_skipped(self) -> None`：执行 `test critical stages cannot be skipped` 对应逻辑。 调用：`FullFlowConfig`, `run_full_flow`, `self.assertRaises`。
- `M L75-L97` `FullFlowTests.test_force_stage_is_reported_in_plan_manifest(self) -> None` [IO-R]：执行 `test force stage is reported in plan manifest` 对应逻辑。 调用：`FullFlowConfig`, `run_full_flow`, `self.assertEqual`, `self.assertIn`, `self.assertTrue`。
- `M L99-L112` `FullFlowTests.test_force_stage_cannot_conflict_with_skip(self) -> None`：执行 `test force stage cannot conflict with skip` 对应逻辑。 调用：`FullFlowConfig`, `run_full_flow`, `self.assertRaises`。
- `M L114-L127` `FullFlowTests.test_force_stage_must_be_inside_selected_range(self) -> None`：执行 `test force stage must be inside selected range` 对应逻辑。 调用：`FullFlowConfig`, `run_full_flow`, `self.assertRaises`。
- `M L129-L163` `FullFlowTests.test_finish_stage_updates_dashboard_reporter(self) -> None`：执行 `test finish stage updates dashboard reporter` 对应逻辑。 调用：`FullFlowConfig`, `_FlowRuntime`, `_finish_stage`, `_initial_stage_states`, `self.assertEqual`, `utc_now`。
- `M L165-L196` `FullFlowTests.test_initial_dashboard_stages_are_published_as_pending(self) -> None`：执行 `test initial dashboard stages are published as pending` 对应逻辑。 调用：`FullFlowConfig`, `_FlowRuntime`, `_initial_stage_states`, `_publish_initial_dashboard_stages`, `self.assertEqual`, `self.assertTrue`。
- `M L198-L256` `FullFlowTests.test_full_flow_reports_initial_resource_snapshot_to_dashboard(self) -> None`：执行 `test full flow reports initial resource snapshot to dashboard` 对应逻辑。 调用：`FullFlowConfig`, `run_full_flow`, `self.assertEqual`。
- `C L203-L212` `FullFlowTests.test_full_flow_reports_initial_resource_snapshot_to_dashboard.FakeDashboardHandle` [CLASS]：封装 `FakeDashboardHandle` 相关数据或行为。
- `N L204-L205` `FullFlowTests.test_full_flow_reports_initial_resource_snapshot_to_dashboard.FakeDashboardHandle.__init__(self, reporter: DashboardReporter) -> None`：初始化实例依赖、配置和运行状态。
- `N L207-L208` `FullFlowTests.test_full_flow_reports_initial_resource_snapshot_to_dashboard.FakeDashboardHandle.__enter__(self)`：执行 `enter` 对应逻辑。
- `N L210-L212` `FullFlowTests.test_full_flow_reports_initial_resource_snapshot_to_dashboard.FakeDashboardHandle.__exit__(self, exc_type, exc, traceback)`：执行 `exit` 对应逻辑。 调用：`self.reporter.close`。
- `N L214-L220` `FullFlowTests.test_full_flow_reports_initial_resource_snapshot_to_dashboard.fake_dashboard_reporter(**kwargs)`：执行 `fake dashboard reporter` 对应逻辑。 调用：`FakeDashboardHandle`。
- `M L258-L303` `FullFlowTests.test_ramp_section_passes_formal_gallery_output_root(self) -> None` [IO-W]：执行 `test ramp section passes formal gallery output root` 对应逻辑。 调用：`FullFlowConfig`, `_FlowRuntime`, `_initial_stage_states`, `_run_ramp_section`, `self.assertEqual`, `utc_now`。

## `src/traning/tests/full_checks/test_full_training_pipeline.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`start.checks`, `traning.conf`, `traning.core.dataset_import`, `traning.core.decision`, `traning.core.spatial`, `traning.core.temporal`

- `C L25-L252` `FullTrainingPipelineTests(unittest.TestCase)` [CLASS]：封装 `FullTrainingPipelineTests` 相关数据或行为。
- `M L26-L252` `FullTrainingPipelineTests.test_pipeline_runs_all_training_steps_and_writes_summary(self) -> None` [IO-R IO-W]：执行 `test pipeline runs all training steps and writes summary` 对应逻辑。 调用：`CandidateCacheBuildResult`, `DataInputReport`, `FullTrainingRunConfig`, `Settings`, `SpatialTrainingResult`, `TemporalDecisionRunResult`。
- `C L255-L264` `_RecordingReporter(NullReporter)` [CLASS]：封装 `RecordingReporter` 相关数据或行为。
- `M L256-L258` `_RecordingReporter.__init__(self) -> None`：初始化实例依赖、配置和运行状态。
- `M L260-L261` `_RecordingReporter.update_metrics(self, **metrics: object) -> None`：执行 `update metrics` 对应逻辑。 调用：`self.metric_updates.append`。
- `M L263-L264` `_RecordingReporter.update_pipeline_stage(self, stage: PipelineStageState) -> None`：执行 `update pipeline stage` 对应逻辑。 调用：`self.stage_updates.append`。

## `src/traning/tests/full_checks/test_gallery_schema.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.state`

- `C L10-L54` `BatchGalleryRequestTests(unittest.TestCase)` [CLASS]：封装 `BatchGalleryRequestTests` 相关数据或行为。
- `M L11-L32` `BatchGalleryRequestTests.test_frame_evaluation_accepts_error_attribution(self) -> None`：执行 `test frame evaluation accepts error attribution` 对应逻辑。 调用：`self.assertEqual`。
- `M L34-L54` `BatchGalleryRequestTests.test_trials_must_share_score_version(self) -> None`：执行 `test trials must share score version` 对应逻辑。 调用：`self.assertRaises`。

## `src/traning/tests/full_checks/test_gated_fusion.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`, `traning.lib.models.local_encoder`

- `C L12-L33` `GatedFusionTests(unittest.TestCase)` [CLASS]：封装 `GatedFusionTests` 相关数据或行为。
- `M L13-L33` `GatedFusionTests.test_forward_and_backward(self) -> None`：执行 `test forward and backward` 对应逻辑。 调用：`GatedSparseFusion`, `LocalFeatures`, `PatchMeta`, `self.assertEqual`, `self.assertIsNotNone`。

## `src/traning/tests/full_checks/test_global_encoder.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.models`

- `C L10-L27` `GlobalEncoderTests(unittest.TestCase)` [CLASS]：封装 `GlobalEncoderTests` 相关数据或行为。
- `M L11-L23` `GlobalEncoderTests.test_lightweight_encoder_and_structure_head(self) -> None`：执行 `test lightweight encoder and structure head` 对应逻辑。 调用：`GlobalStructureHead`, `LightweightGlobalEncoder`, `self.assertEqual`。
- `M L25-L27` `GlobalEncoderTests.test_non_default_backbone_requires_external_setup(self) -> None`：执行 `test non default backbone requires external setup` 对应逻辑。 调用：`LightweightGlobalEncoder`, `self.assertRaises`。

## `src/traning/tests/full_checks/test_global_sampling.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`

- `C L11-L19` `GlobalSamplingTests(unittest.TestCase)` [CLASS]：封装 `GlobalSamplingTests` 相关数据或行为。
- `M L12-L19` `GlobalSamplingTests.test_patch_position_changes_sampled_context(self) -> None`：执行 `test patch position changes sampled context` 对应逻辑。 调用：`PatchMeta`, `sample_global_feature`, `self.assertLess`。

## `src/traning/tests/full_checks/test_local_encoder.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.models`

- `C L10-L23` `LocalEncoderTests(unittest.TestCase)` [CLASS]：封装 `LocalEncoderTests` 相关数据或行为。
- `M L11-L23` `LocalEncoderTests.test_forward_shapes_and_backward(self) -> None`：执行 `test forward shapes and backward` 对应逻辑。 调用：`SmallLocalEncoder`, `self.assertEqual`, `self.assertIn`, `self.assertIsNotNone`。

## `src/traning/tests/full_checks/test_memory_smoke.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`, `traning.lib.runtime`

- `C L23-L93` `MemorySmokeTests(unittest.TestCase)` [CLASS]：封装 `MemorySmokeTests` 相关数据或行为。
- `M L24-L82` `MemorySmokeTests.run_smoke(self, device: torch.device) -> None`：执行 `run smoke` 对应逻辑。 调用：`CudaRuntimeConfig`, `GatedSparseFusion`, `PatchMeta`, `SmallLocalEncoder`, `SpatialPredictionHead`, `autocast_context`。
- `M L84-L85` `MemorySmokeTests.test_cpu_forward_backward_step(self) -> None`：执行 `test cpu forward backward step` 对应逻辑。 调用：`self.run_smoke`。
- `M L87-L93` `MemorySmokeTests.test_cuda_forward_backward_step_when_available(self) -> None`：执行 `test cuda forward backward step when available` 对应逻辑。 调用：`collect_memory_snapshot`, `self.assertIsNotNone`, `self.run_smoke`, `self.skipTest`。

## `src/traning/tests/full_checks/test_model_export.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.model_export`

- `C L15-L43` `ModelExportTests(unittest.TestCase)` [CLASS]：封装 `ModelExportTests` 相关数据或行为。
- `M L16-L43` `ModelExportTests.test_export_model_artifact_copies_files_and_validates_hashes(self) -> None` [IO-W]：执行 `test export model artifact copies files and validates hashes` 对应逻辑。 调用：`ModelArtifactSpec`, `export_model_artifact`, `self.assertEqual`, `self.assertTrue`, `validate_model_artifact`。

## `src/traning/tests/full_checks/test_optimization_module.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.optimization`, `traning.core.optimization.parameter_search`, `traning.lib.metrics`, `traning.state`

- `F L29-L37` `_circle_target(target_id: str='circle-1') -> TargetObject`：执行 `circle target` 对应逻辑。 调用：`TargetObject`。
- `C L40-L255` `OptimizationModuleTests(unittest.TestCase)` [CLASS]：封装 `OptimizationModuleTests` 相关数据或行为。
- `M L41-L57` `OptimizationModuleTests.test_score_trial_aggregates_point_slider_sequence_rules(self) -> None`：执行 `test score trial aggregates point slider sequence rules` 对应逻辑。 调用：`PredictedClick`, `SampleScoringInput`, `_circle_target`, `score_trial`, `self.assertAlmostEqual`, `self.assertEqual`。
- `M L59-L76` `OptimizationModuleTests.test_attribution_groups_temporal_and_decision_errors(self) -> None`：执行 `test attribution groups temporal and decision errors` 对应逻辑。 调用：`PredictedClick`, `SampleScoringInput`, `_circle_target`, `analyze_trial_attribution`, `score_trial`, `self.assertEqual`。
- `M L78-L116` `OptimizationModuleTests.test_parameter_plan_uses_attribution_and_asha_thresholds(self) -> None`：执行 `test parameter plan uses attribution and asha thresholds` 对应逻辑。 调用：`ParameterSearchConfig`, `PredictedClick`, `SampleScoringInput`, `TrialHistoryEntry`, `_circle_target`, `analyze_trial_attribution`。
- `M L118-L134` `OptimizationModuleTests.test_gallery_request_is_built_from_trial_score_report(self) -> None`：执行 `test gallery request is built from trial score report` 对应逻辑。 调用：`PredictedClick`, `SampleScoringInput`, `_circle_target`, `build_batch_gallery_request`, `score_trial`, `self.assertEqual`。
- `M L136-L173` `OptimizationModuleTests.test_curriculum_gate_and_hard_example_sampling(self) -> None`：执行 `test curriculum gate and hard example sampling` 对应逻辑。 调用：`PredictedClick`, `SampleScoringInput`, `_circle_target`, `analyze_trial_attribution`, `build_hard_example_sampling_plan`, `evaluate_curriculum_gate`。
- `M L175-L205` `OptimizationModuleTests.test_execute_optimization_plan_records_trial_and_job(self) -> None` [IO-W]：执行 `test execute optimization plan records trial and job` 对应逻辑。 调用：`OptimizationExecutorConfig`, `PredictedClick`, `SampleScoringInput`, `_circle_target`, `analyze_trial_attribution`, `execute_optimization_plan`。
- `M L207-L236` `OptimizationModuleTests.test_sqlite_trial_store_records_execution(self) -> None`：执行 `test sqlite trial store records execution` 对应逻辑。 调用：`OptimizationExecutorConfig`, `PredictedClick`, `SQLiteTrialStore`, `SampleScoringInput`, `_circle_target`, `analyze_trial_attribution`。
- `M L238-L255` `OptimizationModuleTests.test_multi_objective_score_uses_quality_vram_and_latency(self) -> None`：执行 `test multi objective score uses quality vram and latency` 对应逻辑。 调用：`PredictedClick`, `SampleScoringInput`, `_circle_target`, `score_trial`, `score_trial_objectives`, `self.assertEqual`。

## `src/traning/tests/full_checks/test_patch_stream.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`

- `C L10-L46` `PatchStreamTests(unittest.TestCase)` [CLASS]：封装 `PatchStreamTests` 相关数据或行为。
- `M L11-L23` `PatchStreamTests.assert_full_coverage(self, width: int, height: int) -> None`：执行 `assert full coverage` 对应逻辑。 调用：`PatchStream`, `self.assertEqual`, `self.assertNotIn`, `self.assertTrue`, `stream.iter_patches`。
- `M L25-L27` `PatchStreamTests.test_common_resolutions_are_fully_covered(self) -> None`：执行 `test common resolutions are fully covered` 对应逻辑。 调用：`self.assert_full_coverage`。
- `M L29-L30` `PatchStreamTests.test_odd_dimensions_are_fully_covered(self) -> None`：执行 `test odd dimensions are fully covered` 对应逻辑。 调用：`self.assert_full_coverage`。
- `M L32-L42` `PatchStreamTests.test_small_image_is_padded(self) -> None`：执行 `test small image is padded` 对应逻辑。 调用：`PatchStream`, `self.assertEqual`, `self.assertTrue`, `stream.iter_patches`。
- `M L44-L46` `PatchStreamTests.test_invalid_overlap_raises(self) -> None`：执行 `test invalid overlap raises` 对应逻辑。 调用：`PatchStream`, `self.assertRaises`。

## `src/traning/tests/full_checks/test_plan_gap_closure.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package.coordinates`, `traning.conf`, `traning.core.decision.generator`, `traning.core.model_export`, `traning.core.temporal.trainer`, `traning.state.versioning`

- `C L18-L170` `PlanGapClosureTests(unittest.TestCase)` [CLASS]：封装 `PlanGapClosureTests` 相关数据或行为。
- `M L19-L27` `PlanGapClosureTests.test_explicit_non_centered_playfield_round_trip(self) -> None`：执行 `test explicit non centered playfield round trip` 对应逻辑。 调用：`self.assertAlmostEqual`。
- `M L29-L111` `PlanGapClosureTests.test_action_targets_include_circle_release_slider_repeat_and_spinner(self) -> None`：执行 `test action targets include circle release slider repeat and spinner` 对应逻辑。 调用：`Settings`, `build_candidate_cache_record`, `self.assertEqual`。
- `M L113-L146` `PlanGapClosureTests.test_temporal_loss_weights_change_combined_loss(self) -> None`：执行 `test temporal loss weights change combined loss` 对应逻辑。 调用：`_compute_temporal_loss`, `self.assertGreater`。
- `C L114-L118` `PlanGapClosureTests.test_temporal_loss_weights_change_combined_loss.Weights` [CLASS]：封装 `Weights` 相关数据或行为。
- `C L120-L124` `PlanGapClosureTests.test_temporal_loss_weights_change_combined_loss.TimeHeavy` [CLASS]：封装 `TimeHeavy` 相关数据或行为。
- `M L148-L161` `PlanGapClosureTests.test_version_mismatch_blocks_without_override(self) -> None`：执行 `test version mismatch blocks without override` 对应逻辑。 调用：`ensure_compatible_versions`, `self.assertEqual`, `self.assertFalse`, `self.assertTrue`。
- `M L163-L170` `PlanGapClosureTests.test_settings_migration_adds_schema_and_transform(self) -> None` [IO-W]：执行 `test settings migration adds schema and transform` 对应逻辑。 调用：`migrate_settings_file`, `self.assertIn`, `self.assertTrue`。

## `src/traning/tests/full_checks/test_result_export_gallery.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.visualization.gallery`, `traning.state`

- `C L20-L70` `_FakeSegmentFrameDataset` [CLASS]：封装 `FakeSegmentFrameDataset` 相关数据或行为。
- `M L21-L39` `_FakeSegmentFrameDataset.__init__(self) -> None`：初始化实例依赖、配置和运行状态。
- `M L41-L70` `_FakeSegmentFrameDataset.__getitem__(self, index: int) -> dict[str, object]`：执行 `getitem` 对应逻辑。
- `F L73-L85` `_request(frames: tuple[FrameEvaluation, ...]) -> BatchGalleryRequest`：执行 `request` 对应逻辑。 调用：`BatchGalleryRequest`, `TrialGalleryEvaluation`, `TrialParameters`。
- `C L88-L190` `ResultExportGalleryTests(unittest.TestCase)` [CLASS]：封装 `ResultExportGalleryTests` 相关数据或行为。
- `M L89-L138` `ResultExportGalleryTests.test_outputs_one_folder_per_selected_sample_group(self) -> None` [IO-R]：执行 `test outputs one folder per selected sample group` 对应逻辑。 调用：`FrameEvaluation`, `_FakeSegmentFrameDataset`, `_request`, `self.assertEqual`。
- `M L140-L190` `ResultExportGalleryTests.test_samples_per_group_limits_sample_folders_not_frames(self) -> None` [IO-R]：执行 `test samples per group limits sample folders not frames` 对应逻辑。 调用：`FrameEvaluation`, `_FakeSegmentFrameDataset`, `_request`, `self.assertEqual`。

## `src/traning/tests/full_checks/test_scoring.py`

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

## `src/traning/tests/full_checks/test_sequence_scoring.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.metrics`

- `C L13-L155` `ClickSequenceScoringTests(unittest.TestCase)` [CLASS]：封装 `ClickSequenceScoringTests` 相关数据或行为。
- `M L14-L41` `ClickSequenceScoringTests.test_first_passing_hit_resolves_target_once(self) -> None`：执行 `test first passing hit resolves target once` 对应逻辑。 调用：`PredictedClick`, `TargetObject`, `score_click_sequence`, `self.assertEqual`。
- `M L43-L64` `ClickSequenceScoringTests.test_failed_hit_keeps_target_active_for_later_click(self) -> None`：执行 `test failed hit keeps target active for later click` 对应逻辑。 调用：`PredictedClick`, `TargetObject`, `score_click_sequence`, `self.assertEqual`, `self.assertIn`。
- `M L66-L84` `ClickSequenceScoringTests.test_early_click_is_attributed_to_temporal_parameters(self) -> None`：执行 `test early click is attributed to temporal parameters` 对应逻辑。 调用：`PredictedClick`, `TargetObject`, `score_click_sequence`, `self.assertEqual`。
- `M L86-L117` `ClickSequenceScoringTests.test_overlapping_targets_resolve_by_earliest_active_target(self) -> None`：执行 `test overlapping targets resolve by earliest active target` 对应逻辑。 调用：`PredictedClick`, `TargetObject`, `score_click_sequence`, `self.assertEqual`。
- `M L119-L155` `ClickSequenceScoringTests.test_click_frequency_limit_blocks_high_rate_hits(self) -> None`：执行 `test click frequency limit blocks high rate hits` 对应逻辑。 调用：`PredictedClick`, `SequenceScoreSpec`, `TargetObject`, `score_click_sequence`, `self.assertEqual`, `self.assertTrue`。

## `src/traning/tests/full_checks/test_spatial_decode.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`, `traning.lib.training`

- `F L18-L39` `_prediction(*, height: int=16, width: int=16, embedding_dim: int=4) -> SpatialPrediction`：执行 `prediction` 对应逻辑。 调用：`SpatialPrediction`。
- `C L42-L170` `SpatialDecodeTests(unittest.TestCase)` [CLASS]：封装 `SpatialDecodeTests` 相关数据或行为。
- `M L43-L77` `SpatialDecodeTests.test_canvas_decodes_global_candidate_with_offset_and_type(self) -> None`：执行 `test canvas decodes global candidate with offset and type` 对应逻辑。 调用：`PatchMeta`, `SpatialPredictionCanvas`, `_prediction`, `canvas.to_maps`, `canvas.write_patch`, `decode_spatial_candidates`。
- `M L79-L104` `SpatialDecodeTests.test_padding_region_is_not_written_to_global_canvas(self) -> None`：执行 `test padding region is not written to global canvas` 对应逻辑。 调用：`PatchMeta`, `SpatialPredictionCanvas`, `_prediction`, `canvas.to_maps`, `canvas.write_patch`, `decode_spatial_candidates`。
- `M L106-L130` `SpatialDecodeTests.test_decode_applies_nms(self) -> None`：执行 `test decode applies nms` 对应逻辑。 调用：`PatchMeta`, `SpatialPredictionCanvas`, `_prediction`, `canvas.to_maps`, `canvas.write_patch`, `decode_spatial_candidates`。
- `M L132-L170` `SpatialDecodeTests.test_decode_slider_paths_recovers_ordered_polyline(self) -> None`：执行 `test decode slider paths recovers ordered polyline` 对应逻辑。 调用：`SpatialPredictionMaps`, `decode_slider_paths`, `self.assertAlmostEqual`, `self.assertEqual`, `self.assertFalse`, `self.assertLess`。

## `src/traning/tests/full_checks/test_spatial_inference.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.spatial`

- `F L16-L58` `_tiny_settings() -> Settings`：执行 `tiny settings` 对应逻辑。 调用：`Settings`。
- `C L61-L91` `SpatialInferenceTests(unittest.TestCase)` [CLASS]：封装 `SpatialInferenceTests` 相关数据或行为。
- `M L62-L91` `SpatialInferenceTests.test_cpu_single_frame_inference_reports_cpu_gpu_split(self) -> None`：执行 `test cpu single frame inference reports cpu gpu split` 对应逻辑。 调用：`_tiny_settings`, `result.as_summary`, `run_spatial_frame_inference`, `self.assertEqual`, `self.assertIn`, `self.assertLessEqual`。

## `src/traning/tests/full_checks/test_spatial_model.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.models`

- `C L10-L23` `SpatialModelTests(unittest.TestCase)` [CLASS]：封装 `SpatialModelTests` 相关数据或行为。
- `M L11-L23` `SpatialModelTests.test_prediction_head_outputs_all_required_tasks(self) -> None`：执行 `test prediction head outputs all required tasks` 对应逻辑。 调用：`SpatialPredictionHead`, `self.assertEqual`。

## `src/traning/tests/full_checks/test_spatial_targets.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`, `traning.lib.training`

- `C L10-L86` `SpatialTargetTests(unittest.TestCase)` [CLASS]：封装 `SpatialTargetTests` 相关数据或行为。
- `M L11-L45` `SpatialTargetTests.test_circle_target_contains_center_and_approach_ring(self) -> None`：执行 `test circle target contains center and approach ring` 对应逻辑。 调用：`PatchMeta`, `build_spatial_loss_targets`, `self.assertGreater`, `self.assertIn`。
- `M L47-L72` `SpatialTargetTests.test_slider_target_contains_body_direction_head_and_tail(self) -> None`：执行 `test slider target contains body direction head and tail` 对应逻辑。 调用：`PatchMeta`, `build_spatial_loss_targets`, `self.assertGreater`, `self.assertIn`, `self.assertLess`。
- `M L74-L86` `SpatialTargetTests.test_spinner_target_marks_valid_patch_area(self) -> None`：执行 `test spinner target marks valid patch area` 对应逻辑。 调用：`PatchMeta`, `build_spatial_loss_targets`, `self.assertGreater`, `self.assertIn`, `self.assertLess`。

## `src/traning/tests/full_checks/test_spatial_trainer.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.spatial`

- `C L14-L85` `SpatialTrainerTests(unittest.TestCase)` [CLASS]：封装 `SpatialTrainerTests` 相关数据或行为。
- `M L15-L85` `SpatialTrainerTests.test_cpu_single_step_with_synthetic_sample(self) -> None`：执行 `test cpu single step with synthetic sample` 对应逻辑。 调用：`Settings`, `run_spatial_training`, `self.assertEqual`, `self.assertTrue`。

## `src/traning/tests/full_checks/test_temporal_dataset.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.decision`, `traning.core.temporal`

- `F L18-L37` `_record(sample_key: str, frame_index: int, *, candidates: list[dict] | None=None, temporal_target: dict | None=None) -> dict`：执行 `record` 对应逻辑。
- `F L40-L65` `_candidate(score: float, *, x: float=25.0, y: float=10.0, candidate_id: int=0) -> dict`：执行 `candidate` 对应逻辑。
- `F L68-L82` `_write_cache(path: Path, records: list[dict]) -> None` [IO-W]：写入 `cache` 对应的数据或结果。
- `C L85-L147` `TemporalDatasetTests(unittest.TestCase)` [CLASS]：封装 `TemporalDatasetTests` 相关数据或行为。
- `M L86-L93` `TemporalDatasetTests.test_loads_candidate_cache_records(self) -> None`：执行 `test loads candidate cache records` 对应逻辑。 调用：`_candidate`, `_record`, `_write_cache`, `load_candidate_cache_records`, `self.assertEqual`。
- `M L95-L117` `TemporalDatasetTests.test_encodes_fixed_windows_without_crossing_samples(self) -> None`：执行 `test encodes fixed windows without crossing samples` 对应逻辑。 调用：`TemporalCandidateWindowDataset`, `TemporalFeatureSpec`, `_candidate`, `_record`, `self.assertEqual`, `self.assertFalse`。
- `M L119-L147` `TemporalDatasetTests.test_uses_explicit_temporal_target_when_present(self) -> None`：执行 `test uses explicit temporal target when present` 对应逻辑。 调用：`TemporalCandidateWindowDataset`, `TemporalFeatureSpec`, `_candidate`, `_record`, `self.assertEqual`, `self.assertTrue`。

## `src/traning/tests/full_checks/test_temporal_decision.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.decision`, `traning.core.temporal`

- `F L15-L53` `_record(frame_index: int) -> dict`：执行 `record` 对应逻辑。
- `F L56-L65` `_write_cache(path: Path) -> None` [IO-W]：写入 `cache` 对应的数据或结果。 调用：`_record`。
- `C L68-L103` `TemporalDecisionTests(unittest.TestCase)` [CLASS]：封装 `TemporalDecisionTests` 相关数据或行为。
- `M L69-L103` `TemporalDecisionTests.test_train_then_run_decision(self) -> None` [IO-R]：执行 `test train then run decision` 对应逻辑。 调用：`Settings`, `_write_cache`, `run_temporal_decision`, `run_temporal_training`, `self.assertEqual`, `self.assertIn`。

## `src/traning/tests/full_checks/test_temporal_trainer.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.decision`, `traning.core.temporal`, `traning.core.training_inheritance`

- `F L16-L46` `_record(frame_index: int) -> dict`：执行 `record` 对应逻辑。
- `F L49-L63` `_write_cache(path: Path) -> None` [IO-W]：写入 `cache` 对应的数据或结果。 调用：`_record`。
- `F L66-L89` `_assert_nested_close(case: unittest.TestCase, left, right, *, path: str='root') -> None`：执行 `assert nested close` 对应逻辑。 调用：`_assert_nested_close`。
- `C L92-L183` `TemporalTrainerTests(unittest.TestCase)` [CLASS]：封装 `TemporalTrainerTests` 相关数据或行为。
- `M L93-L117` `TemporalTrainerTests.test_cpu_temporal_training_smoke(self) -> None`：执行 `test cpu temporal training smoke` 对应逻辑。 调用：`Settings`, `_write_cache`, `run_temporal_training`, `self.assertEqual`, `self.assertGreater`, `self.assertTrue`。
- `M L119-L183` `TemporalTrainerTests.test_resume_matches_continuous_temporal_training(self) -> None`：执行 `test resume matches continuous temporal training` 对应逻辑。 调用：`Settings`, `_assert_nested_close`, `_write_cache`, `load_training_checkpoint`, `run_temporal_training`, `self.assertEqual`。

## `src/traning/tests/full_checks/test_training_inheritance.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.training_inheritance`

- `C L16-L74` `TrainingInheritanceTests(unittest.TestCase)` [CLASS]：封装 `TrainingInheritanceTests` 相关数据或行为。
- `M L17-L53` `TrainingInheritanceTests.test_create_and_load_inheritance_package(self) -> None`：执行 `test create and load inheritance package` 对应逻辑。 调用：`create_inheritance_package`, `load_inheritance_package`, `load_settings`, `self.assertEqual`, `self.assertIn`, `self.assertTrue`。
- `M L55-L74` `TrainingInheritanceTests.test_strict_rejects_incompatible_dataset(self) -> None` [IO-R IO-W]：执行 `test strict rejects incompatible dataset` 对应逻辑。 调用：`create_inheritance_package`, `load_inheritance_package`, `load_settings`, `self.assertRaises`。

## `src/traning/tests/full_checks/test_training_ramp.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.training_ramp`

- `C L22-L165` `TrainingRampTests(unittest.TestCase)` [CLASS]：封装 `TrainingRampTests` 相关数据或行为。
- `M L23-L47` `TrainingRampTests.test_build_ramp_levels_clips_and_reaches_target(self) -> None`：执行 `test build ramp levels clips and reaches target` 对应逻辑。 调用：`RampTarget`, `build_ramp_levels`, `self.assertEqual`, `self.assertGreaterEqual`, `self.assertLessEqual`。
- `M L49-L87` `TrainingRampTests.test_ensure_full_target_config_writes_target_and_absolutizes_paths(self) -> None` [IO-R IO-W]：执行 `test ensure full target config writes target and absolutizes paths` 对应逻辑。 调用：`RampTarget`, `ensure_full_target_config`, `self.assertEqual`, `self.assertTrue`。
- `M L89-L165` `TrainingRampTests.test_ramp_reporter_tracks_level_pass_and_failure(self) -> None`：执行 `test ramp reporter tracks level pass and failure` 对应逻辑。 调用：`RampLevelSpec`, `RampTarget`, `_report_level_finished`, `_report_level_started`, `_report_ramp_failed`, `_report_ramp_started`。

## `src/traning/tests/startup_checks/items.py`

职责：traning 启动检测项；检查配置、设备、数据输入和完整训练阶段注册。
工程依赖：`package.checks`, `traning.conf`, `traning.core.dataset_import`, `traning.core.decision`

- `F L13-L38` `check_settings_load(config_path: Path | None=None) -> tuple[StartupCheckResult, Settings]`：执行 `check settings load` 对应逻辑。 调用：`load_settings`, `settings.data_input.validate_tiling`, `settings.tiling.validate_tiling`。
- `F L41-L84` `check_runtime_device(settings: Settings, *, device: torch.device, require_cuda: bool | None=None) -> tuple[StartupCheckResult, None]`：执行 `check runtime device` 对应逻辑。
- `F L87-L116` `check_data_input(settings: Settings, *, split: DataSplit) -> tuple[StartupCheckResult, None]`：执行 `check data input` 对应逻辑。 调用：`inspect_data_input`。
- `F L119-L147` `check_core_entrypoints(_settings: Settings | None=None) -> tuple[StartupCheckResult, None]`：执行 `check core entrypoints` 对应逻辑。

## `src/traning/tests/startup_checks/runner.py`

职责：traning 启动检测统一入口；按顺序运行配置、运行设备、数据输入和 core 入口检测。
工程依赖：`package.checks`, `traning.conf`, `traning.tests.startup_checks.items`

- `F L17-L80` `run_startup_checks(config_path: Path | None=None, *, split: DataSplit='train', device: torch.device | None=None, require_cuda: bool | None=None) -> StartupCheckReport`：执行 `run startup checks` 对应逻辑。 调用：`check_core_entrypoints`, `check_data_input`, `check_runtime_device`, `check_settings_load`, `results.append`。
