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

覆盖 `152` 个 Python 文件、`1015` 个命名函数/方法、`205` 个类。匿名 lambda 不单独列出。

图例：`F` 模块函数，`M` 方法，`N` 嵌套函数，`C` 类；`IO-R/IO-W` 文件读写，`DB` 数据库，`PROCESS` 外部进程。

## `src/traning/conf/settings.py`

职责：训练配置模型与 YAML 加载；解析数据集路径、item 划分、颜色 cue、候选缓存、点击频率上限并校验采样和分块参数。
工程依赖：`package.coordinates`

- `C L29-L32` `SettingsError(Exception)` [CLASS]：表示配置读取、解析或跨字段校验失败。
- `C L35-L37` `RuntimeSettings(BaseModel)` [CLASS]：封装 `RuntimeSettings` 相关数据或行为。
- `C L40-L51` `InputSettings(BaseModel)` [CLASS]：封装 `InputSettings` 相关数据或行为。
- `M L48-L51` `InputSettings._positive_dimension(cls, value: int) -> int` [VALIDATOR]：执行 `positive dimension` 对应逻辑。
- `C L54-L72` `PlayfieldRectSettings(BaseModel)` [CLASS]：封装 `PlayfieldRectSettings` 相关数据或行为。
- `M L62-L65` `PlayfieldRectSettings._finite_number(cls, value: float) -> float` [VALIDATOR]：执行 `finite number` 对应逻辑。
- `M L69-L72` `PlayfieldRectSettings._positive_dimension(cls, value: float) -> float` [VALIDATOR]：执行 `positive dimension` 对应逻辑。
- `C L75-L93` `CropRectSettings(BaseModel)` [CLASS]：封装 `CropRectSettings` 相关数据或行为。
- `M L83-L86` `CropRectSettings._finite_number(cls, value: float) -> float` [VALIDATOR]：执行 `finite number` 对应逻辑。
- `M L90-L93` `CropRectSettings._positive_dimension(cls, value: float) -> float` [VALIDATOR]：执行 `positive dimension` 对应逻辑。
- `C L96-L122` `CoordinateTransformSettings(BaseModel)` [CLASS]：声明 osu! 到训练帧的坐标标定模式及其必需参数。
- `M L108-L122` `CoordinateTransformSettings.validate_transform(self) -> CoordinateTransformSettings` [VALIDATOR]：校验 `transform` 对应的数据或结果。
- `C L125-L155` `TilingConfig(BaseModel)` [CLASS]：封装 `TilingConfig` 相关数据或行为。
- `M L135-L138` `TilingConfig._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `M L142-L145` `TilingConfig._nonnegative_overlap(cls, value: int) -> int` [VALIDATOR]：执行 `nonnegative overlap` 对应逻辑。
- `M L148-L155` `TilingConfig.validate_tiling(self) -> TilingConfig` [VALIDATOR]：校验 `tiling` 对应的数据或结果。
- `C L158-L174` `LocalEncoderConfig(BaseModel)` [CLASS]：封装 `LocalEncoderConfig` 相关数据或行为。
- `M L171-L174` `LocalEncoderConfig._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `C L177-L195` `GlobalEncoderConfig(BaseModel)` [CLASS]：封装 `GlobalEncoderConfig` 相关数据或行为。
- `M L192-L195` `GlobalEncoderConfig._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `C L198-L218` `FusionConfig(BaseModel)` [CLASS]：封装 `FusionConfig` 相关数据或行为。
- `M L209-L212` `FusionConfig._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `M L215-L218` `FusionConfig.validate_attention_shape(self) -> FusionConfig` [VALIDATOR]：校验 `attention shape` 对应的数据或结果。
- `C L221-L233` `TemporalConfig(BaseModel)` [CLASS]：封装 `TemporalConfig` 相关数据或行为。
- `M L230-L233` `TemporalConfig._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `C L236-L247` `TemporalLossWeights(BaseModel)` [CLASS]：封装 `TemporalLossWeights` 相关数据或行为。
- `M L244-L247` `TemporalLossWeights._finite_nonnegative(cls, value: float) -> float` [VALIDATOR]：执行 `finite nonnegative` 对应逻辑。
- `C L250-L260` `SpatialConsistencyLossWeights(BaseModel)` [CLASS]：封装 `SpatialConsistencyLossWeights` 相关数据或行为。
- `M L257-L260` `SpatialConsistencyLossWeights._finite_nonnegative(cls, value: float) -> float` [VALIDATOR]：执行 `finite nonnegative` 对应逻辑。
- `C L263-L269` `TrainingSettings(BaseModel)` [CLASS]：封装 `TrainingSettings` 相关数据或行为。
- `C L272-L308` `MemoryConfig(BaseModel)` [CLASS]：封装 `MemoryConfig` 相关数据或行为。
- `M L291-L294` `MemoryConfig._finite_nonnegative_memory(cls, value: float) -> float` [VALIDATOR]：执行 `finite nonnegative memory` 对应逻辑。
- `M L298-L301` `MemoryConfig._positive_memory(cls, value: float) -> float` [VALIDATOR]：执行 `positive memory` 对应逻辑。
- `M L305-L308` `MemoryConfig._optional_positive_memory(cls, value: float | None) -> float | None` [VALIDATOR]：执行 `optional positive memory` 对应逻辑。
- `C L311-L336` `SMETConfig(BaseModel)` [CLASS]：封装 `SMETConfig` 相关数据或行为。
- `M L319-L322` `SMETConfig._sparsity(cls, value: float) -> float` [VALIDATOR]：执行 `sparsity` 对应逻辑。
- `M L326-L329` `SMETConfig._positive_interval(cls, value: int) -> int` [VALIDATOR]：执行 `positive interval` 对应逻辑。
- `M L333-L336` `SMETConfig._density(cls, value: float) -> float` [VALIDATOR]：执行 `density` 对应逻辑。
- `C L339-L390` `CandidateCacheSettings(BaseModel)` [CLASS]：封装 `CandidateCacheSettings` 相关数据或行为。
- `M L367-L370` `CandidateCacheSettings._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `M L374-L377` `CandidateCacheSettings._probability(cls, value: float) -> float` [VALIDATOR]：执行 `probability` 对应逻辑。
- `M L387-L390` `CandidateCacheSettings._nonnegative_float(cls, value: float) -> float` [VALIDATOR]：执行 `nonnegative float` 对应逻辑。
- `C L393-L431` `OptimizationSettings(BaseModel)` [CLASS]：封装 `OptimizationSettings` 相关数据或行为。
- `M L414-L419` `OptimizationSettings._positive_integer(cls, value: int | None) -> int | None` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `M L423-L431` `OptimizationSettings._finite_objective_weights(cls, value: dict[str, float]) -> dict[str, float]` [VALIDATOR]：执行 `finite objective weights` 对应逻辑。
- `C L434-L470` `LoaderSettings(BaseModel)` [CLASS]：封装 `LoaderSettings` 相关数据或行为。
- `M L445-L448` `LoaderSettings._positive_batch_size(cls, value: int) -> int` [VALIDATOR]：执行 `positive batch size` 对应逻辑。
- `M L452-L455` `LoaderSettings._nonnegative_workers(cls, value: int) -> int` [VALIDATOR]：执行 `nonnegative workers` 对应逻辑。
- `M L459-L462` `LoaderSettings._optional_positive_prefetch(cls, value: int | None) -> int | None` [VALIDATOR]：执行 `optional positive prefetch` 对应逻辑。
- `M L465-L470` `LoaderSettings.validate_worker_options(self) -> LoaderSettings` [VALIDATOR]：校验 `worker options` 对应的数据或结果。
- `C L473-L481` `EvaluationSettings(BaseModel)` [CLASS]：封装 `EvaluationSettings` 相关数据或行为。
- `M L478-L481` `EvaluationSettings._nonnegative_interval(cls, value: float) -> float` [VALIDATOR]：执行 `nonnegative interval` 对应逻辑。
- `C L484-L499` `VisualizationSettings(BaseModel)` [CLASS]：封装 `VisualizationSettings` 相关数据或行为。
- `M L496-L499` `VisualizationSettings._positive_interval(cls, value: int) -> int` [VALIDATOR]：执行 `positive interval` 对应逻辑。
- `C L502-L589` `DataInputSettings(BaseModel)` [CLASS]：封装 `DataInputSettings` 相关数据或行为。
- `M L526-L529` `DataInputSettings._positive_fps(cls, value: float) -> float` [VALIDATOR]：执行 `positive fps` 对应逻辑。
- `M L533-L536` `DataInputSettings._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `M L540-L543` `DataInputSettings._optional_positive_integer(cls, value: int | None) -> int | None` [VALIDATOR]：执行 `optional positive integer` 对应逻辑。
- `M L547-L550` `DataInputSettings._nonnegative_visibility(cls, value: float) -> float` [VALIDATOR]：执行 `nonnegative visibility` 对应逻辑。
- `M L562-L568` `DataInputSettings._unique_nonempty_strings(cls, value: tuple[str, ...]) -> tuple[str, ...]` [VALIDATOR]：执行 `unique nonempty strings` 对应逻辑。
- `M L571-L583` `DataInputSettings.validate_item_splits(self) -> DataInputSettings` [VALIDATOR]：校验 `item splits` 对应的数据或结果。
- `M L585-L589` `DataInputSettings.validate_tiling(self) -> None`：校验 `tiling` 对应的数据或结果。
- `C L592-L638` `Settings(BaseSettings)` [CLASS]：聚合训练各阶段配置，并定义环境变量覆盖优先级。
- `M L625-L638` `Settings.settings_customise_sources(cls, settings_cls: type[BaseSettings], init_settings: PydanticBaseSettingsSource, env_settings: PydanticBaseSettingsSource, dotenv_settings: PydanticBaseSettingsSource, file_secret_settings: PydanticBaseSettingsSource) -> tuple[PydanticBaseSettingsSource, ...]`：执行 `settings customise sources` 对应逻辑。
- `F L641-L650` `_read_config(config_path: Path) -> dict[str, Any]` [IO-R]：读取 `config` 对应的数据或结果。 调用：`SettingsError`。
- `F L653-L697` `_resolve_paths(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]`：相对配置文件目录解析所有持久化输入输出路径。
- `F L700-L711` `load_settings(config_path: Path | None=None) -> Settings`：加载配置并执行 Pydantic 与 tiling 的完整跨字段校验。 调用：`Settings`, `SettingsError`, `_read_config`, `_resolve_paths`, `settings.data_input.validate_tiling`, `settings.tiling.validate_tiling`。

## `src/traning/core/calibration/playfield.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.dataset_import`, `traning.lib.coordinates`, `traning.lib.data.models`

- `C L22-L46` `CalibrationPoint` [CLASS]：封装 `CalibrationPoint` 相关数据或行为。
- `M L34-L46` `CalibrationPoint.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L50-L83` `CalibrationResult` [CLASS]：封装 `CalibrationResult` 相关数据或行为。
- `M L62-L76` `CalibrationResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`point.as_dict`。
- `M L78-L83` `CalibrationResult.write_json(self, path: Path) -> None` [IO-W]：写入 `json` 对应的数据或结果。 调用：`self.as_dict`。
- `F L86-L163` `calibrate_playfield_transform(settings: Settings, *, split: DataSplit='train', max_records: int | None=None, search_radius_px: int=150, max_search_error_px: float=90.0, ransac_threshold_px: float=20.0, min_inliers: int=32, output_path: Path | None=None) -> CalibrationResult`：从多个片段收集观测点，并用 RANSAC 拟合可逆的 2×3 仿射矩阵。 调用：`CalibrationPoint`, `CalibrationResult`, `_detect_record_points`, `_invert_affine`, `_matrix_tuple`, `build_dataset`。
- `F L166-L226` `_detect_record_points(settings: Settings, record: SegmentRecord, *, search_radius_px: int, max_search_error_px: float) -> tuple[CalibrationPoint, ...]` [IO-R]：执行 `detect record points` 对应逻辑。 调用：`CalibrationPoint`, `_detect_circle_near`, `points.append`, `transform_from_settings_or_sample`。
- `F L229-L272` `_detect_circle_near(frame_bgr: np.ndarray, *, search_xy: tuple[float, float], search_radius_px: int, expected_radius_px: float) -> tuple[float, float, float, float] | None`：执行 `detect circle near` 对应逻辑。
- `F L275-L287` `_invert_affine(matrix: np.ndarray) -> np.ndarray`：执行 `invert affine` 对应逻辑。
- `F L290-L296` `_matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]`：执行 `matrix tuple` 对应逻辑。

## `src/traning/core/dataset_import/data_input.py`

职责：训练集导入模块公开门面；提供检查、Dataset 和 DataLoader。
工程依赖：`traning.conf`, `traning.core.dataset_import.loader`, `traning.core.dataset_import.preflight`

- `C L12-L28` `DataInputModule` [CLASS]：封装 `DataInputModule` 相关数据或行为。
- `M L13-L14` `DataInputModule.__init__(self, settings: Settings)`：初始化实例依赖、配置和运行状态。
- `M L16-L17` `DataInputModule.inspect(self, *, split: DataSplit='all') -> DataInputReport`：执行 `inspect` 对应逻辑。 调用：`inspect_data_input`。
- `M L19-L20` `DataInputModule.dataset(self, *, split: DataSplit='train')`：执行 `dataset` 对应逻辑。 调用：`build_dataset`。
- `M L22-L28` `DataInputModule.dataloader(self, *, split: DataSplit='train', shuffle: bool | None=None) -> DataLoader`：执行 `dataloader` 对应逻辑。 调用：`build_dataloader`。
- `F L31-L36` `check_data_input(settings: Settings | None=None, *, split: DataSplit='all') -> DataInputReport`：执行 `check data input` 对应逻辑。 调用：`DataInputModule`, `DataInputModule.inspect`, `load_settings`。

## `src/traning/core/dataset_import/loader.py`

职责：把配置映射为 SegmentFrameDataset 与 PyTorch DataLoader，并为每条记录注入完整坐标变换规格。
工程依赖：`traning.conf`, `traning.core.dataset_import.preflight`, `traning.lib.coordinates`, `traning.lib.data`

- `F L18-L54` `build_dataset(settings: Settings, *, split: DataSplit='train') -> SegmentFrameDataset`：构建 split Dataset，并为每条 record 固化其完整坐标变换规格。 调用：`SegmentFrameDataset`, `discover_data_input`, `spec.as_dict`, `transform_from_settings_or_sample`。
- `F L57-L87` `build_dataloader(settings: Settings, *, split: DataSplit='train', shuffle: bool | None=None) -> DataLoader`：用确定性随机种子和项目配置包装 Dataset。 调用：`build_dataset`。
- `F L90-L94` `_seed_worker(worker_id: int, *, base_seed: int) -> None`：执行 `seed worker` 对应逻辑。

## `src/traning/core/dataset_import/preflight.py`

职责：读取 split manifest 或旧 item 配置，扫描训练片段并生成数量、类别、维度和问题报告。
工程依赖：`package.dataset_split`, `traning.conf`, `traning.lib.data`, `traning.lib.data.models`

- `C L17-L30` `DataInputReport` [CLASS]：封装 `DataInputReport` 相关数据或行为。
- `M L29-L30` `DataInputReport.ok(self) -> bool` [PROPERTY]：执行 `ok` 对应逻辑。
- `F L33-L39` `_combine_item_filters(base_items: tuple[str, ...], split_items: tuple[str, ...]) -> tuple[str, ...]`：执行 `combine item filters` 对应逻辑。
- `F L42-L58` `_split_items(config, split: DataSplit) -> tuple[str, ...]`：执行 `split items` 对应逻辑。
- `F L61-L85` `discover_data_input(settings: Settings, *, split: DataSplit='all') -> DiscoveryResult`：执行 `discover data input` 对应逻辑。 调用：`DatasetIssue`, `DiscoveryResult`, `_combine_item_filters`, `_split_items`, `discover_segments`。
- `F L88-L120` `inspect_data_input(settings: Settings, *, split: DataSplit='all') -> DataInputReport`：执行 `inspect data input` 对应逻辑。 调用：`DataInputReport`, `_distribution_and_topology`, `discover_data_input`。
- `F L123-L169` `_distribution_and_topology(records) -> tuple[dict[str, object], tuple[str, ...]]`：执行 `distribution and topology` 对应逻辑。 调用：`_high_density_windows`, `_kind`, `_slider_topology_issues`, `_summary`, `inter_object_intervals.append`, `slider_durations.append`。
- `F L172-L175` `_summary(values: list[float]) -> dict[str, float | int | None]`：执行 `summary` 对应逻辑。
- `F L178-L184` `_kind(value: str) -> str`：执行 `kind` 对应逻辑。
- `F L187-L193` `_high_density_windows(objects) -> int`：执行 `high density windows` 对应逻辑。
- `F L196-L213` `_slider_topology_issues(record, item) -> tuple[str, ...]`：执行 `slider topology issues` 对应逻辑。 调用：`_self_intersects`, `_touches_branch`, `issues.append`。
- `F L216-L224` `_self_intersects(path: tuple[tuple[float, float], ...]) -> bool`：执行 `self intersects` 对应逻辑。 调用：`_segments_intersect`。
- `F L227-L234` `_touches_branch(path: tuple[tuple[float, float], ...]) -> bool`：执行 `touches branch` 对应逻辑。
- `F L237-L241` `_segments_intersect(a, b, c, d) -> bool`：执行 `segments intersect` 对应逻辑。 调用：`orient`。
- `N L238-L239` `_segments_intersect.orient(p, q, r) -> float`：执行 `orient` 对应逻辑。

## `src/traning/core/decision/generator.py`

职责：离线空间候选缓存生成器；逐帧调用空间推理，将候选逆变换到 osu 空间并按真实 CircleSize 半径匹配 temporal target。
工程依赖：`traning.conf`, `traning.core.dataset_import`, `traning.core.spatial`, `traning.lib.coordinates`, `traning.lib.training`, `traning.lib.training.spatial_decode`, `traning.state.candidate_cache_schema`, `traning.state.versioning`

- `C L36-L64` `CandidateCacheBuildResult` [CLASS]：封装 `CandidateCacheBuildResult` 相关数据或行为。
- `M L49-L64` `CandidateCacheBuildResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `F L67-L296` `generate_candidate_cache(settings: Settings, *, output_dir: Path, device: torch.device, spatial_checkpoint_path: Path | None=None, split: DataSplit='train', max_frames: int | None=None, patch_limit: int | None=None, max_candidates: int | None=None, score_threshold: float | None=None, nms_radius_px: float | None=None, slider_threshold: float | None=None, max_slider_paths: int | None=None, dataset: Sequence[Mapping[str, Any]] | None=None) -> CandidateCacheBuildResult` [IO-W]：执行 `generate candidate cache` 对应逻辑。 调用：`CandidateCacheBuildResult`, `_candidate_cache_indices`, `_mean`, `_percentile`, `_rate`, `_sample_groups_for_indices`。
- `F L299-L356` `_candidate_cache_indices(source: Sequence[Mapping[str, Any]], *, frame_total: int, seed: int, diverse: bool, contiguous_block_frames: int=1) -> tuple[int, ...]`：执行 `candidate cache indices` 对应逻辑。 调用：`_source_group_key`, `groups.setdefault.append`, `selected.append`。
- `F L359-L363` `_sample_groups_for_indices(source: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> tuple[str, ...]`：执行 `sample groups for indices` 对应逻辑。 调用：`_source_group_key`。
- `F L366-L374` `_source_group_key(source: Sequence[Mapping[str, Any]], index: int) -> str`：执行 `source group key` 对应逻辑。
- `F L377-L488` `build_candidate_cache_record(sample: Mapping[str, Any], candidates: Sequence[SpatialCandidate], slider_paths: Sequence[SliderPathCandidate], *, frame_width: int, frame_height: int, device: str, patches_processed: int, frame_channels: int, save_dtype: str, low_confidence_threshold: float, close_score_margin: float, slider_attach_distance_px: float, action_window_ms: float=25.0, settings: Settings | None=None) -> dict[str, Any]`：构建并返回 `candidate cache record` 对应的数据或结果。 调用：`_apply_candidate_reviews`, `_apply_local_refinement`, `_build_temporal_target`, `_candidate_ambiguity_reasons`, `_cast_embedding`, `_circle_radius_osu_pixels`。
- `F L491-L583` `_build_temporal_target(sample: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], *, frame_width: int, frame_height: int, action_window_ms: float, circle_radius_osu_pixels: float, settings: Settings | None=None) -> dict[str, Any]`：构建 `temporal target` 对应的数据或结果。 调用：`_nearest_candidate`, `_optional_float`, `_select_temporal_object`, `transform_from_settings_or_sample`。
- `F L586-L592` `_circle_radius_osu_pixels(sample: Mapping[str, Any]) -> float`：读取谱面真实命中半径；旧样本缺字段时才使用协议默认值。 调用：`_optional_float`。
- `F L595-L627` `_select_temporal_object(objects: object, *, timestamp_ms: float, action_window_ms: float) -> dict[str, Any] | None`：选择 `temporal object` 对应的数据或结果。 调用：`_temporal_target_for_object`。
- `F L630-L694` `_temporal_target_for_object(item: Mapping[str, Any], *, timestamp_ms: float, action_window_ms: float) -> dict[str, Any] | None`：执行 `temporal target for object` 对应逻辑。 调用：`_click_duration_ms`, `_is_release_frame`, `_object_kind`, `_object_osu_point`, `_optional_float`, `_repeat_boundaries`。
- `F L697-L698` `_click_duration_ms(action_window_ms: float) -> float`：执行 `click duration ms` 对应逻辑。
- `F L701-L708` `_is_release_frame(timestamp_ms: float, *, start_ms: float, action_window_ms: float) -> bool`：判断是否 `release frame` 对应的数据或结果。 调用：`_click_duration_ms`。
- `F L711-L732` `_repeat_boundaries(item: Mapping[str, Any], *, start_ms: float, end_ms: float) -> tuple[tuple[float, str, tuple[float, float]], ...]`：执行 `repeat boundaries` 对应逻辑。 调用：`_object_osu_point`, `_optional_float`, `_slider_tail_point`, `boundaries.append`。
- `F L735-L741` `_slider_tail_point(item: Mapping[str, Any]) -> tuple[float, float] | None`：执行 `slider tail point` 对应逻辑。 调用：`_object_osu_point`。
- `F L744-L755` `_object_osu_point(item: Mapping[str, Any]) -> tuple[float, float] | None`：执行 `object osu point` 对应逻辑。 调用：`_object_kind`。
- `F L758-L764` `_object_kind(item: Mapping[str, Any]) -> str`：执行 `object kind` 对应逻辑。
- `F L767-L790` `_nearest_candidate(candidates: Sequence[Mapping[str, Any]], *, target_video_xy: tuple[float, float], target_osu_xy: tuple[float, float], transform: Any, max_distance_osu: float) -> tuple[Mapping[str, Any] | None, list[float], list[float]]`：执行 `nearest candidate` 对应逻辑。 调用：`_optional_float`, `distances_osu.append`, `distances_px.append`。
- `F L793-L812` `_candidate_ambiguity_reasons(index: int, candidates: Sequence[SpatialCandidate], slider_path: SliderPathCandidate | None, *, low_confidence_threshold: float, close_score_margin: float) -> tuple[str, ...]`：执行 `candidate ambiguity reasons` 对应逻辑。 调用：`_has_close_neighbor`, `reasons.append`。
- `F L815-L863` `_apply_candidate_reviews(rows: list[dict[str, Any]], *, slider_rows: Sequence[Mapping[str, Any]], frame_width: int, frame_height: int, enabled: bool, max_candidates: int) -> None`：应用 `candidate reviews` 对应的数据或结果。 调用：`_review_candidate_locally`。
- `F L866-L913` `_apply_local_refinement(rows: list[dict[str, Any]], *, slider_rows: Sequence[Mapping[str, Any]], frame_width: int, frame_height: int, enabled: bool, top_k: int, radius_px: float) -> None`：应用 `local refinement` 对应的数据或结果。 调用：`_refined_candidate_xy`。
- `F L916-L948` `_review_candidate_locally(row: Mapping[str, Any], *, slider_rows: Sequence[Mapping[str, Any]], frame_width: int, frame_height: int) -> dict[str, Any]`：执行 `review candidate locally` 对应逻辑。 调用：`_distance_to_polyline`, `_local_evidence_score`, `_optional_float`, `_polyline_from_row`, `_row_slider_path`。
- `F L951-L968` `_local_evidence_score(row: Mapping[str, Any]) -> float`：执行 `local evidence score` 对应逻辑。 调用：`_optional_float`。
- `F L971-L989` `_refined_candidate_xy(row: Mapping[str, Any], *, slider_rows: Sequence[Mapping[str, Any]], current_xy: tuple[float, float], radius_px: float) -> tuple[float, float]`：执行 `refined candidate xy` 对应逻辑。 调用：`_nearest_polyline_point`, `_point_distance`, `_point_from_row`, `_polyline_from_row`, `_row_slider_path`。
- `F L992-L1002` `_row_slider_path(row: Mapping[str, Any], slider_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None`：执行 `row slider path` 对应逻辑。
- `F L1005-L1012` `_point_from_row(value: Any) -> tuple[float, float] | None`：执行 `point from row` 对应逻辑。
- `F L1015-L1020` `_polyline_from_row(row: Mapping[str, Any]) -> tuple[tuple[float, float], ...]`：执行 `polyline from row` 对应逻辑。 调用：`_point_from_row`。
- `F L1023-L1039` `_nearest_polyline_point(point: tuple[float, float], polyline: Sequence[tuple[float, float]]) -> tuple[float, float] | None`：执行 `nearest polyline point` 对应逻辑。 调用：`_point_distance`, `_project_point_to_segment`。
- `F L1042-L1057` `_project_point_to_segment(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]`：执行 `project point to segment` 对应逻辑。
- `F L1060-L1073` `_has_close_neighbor(index: int, candidates: Sequence[SpatialCandidate], *, margin: float) -> bool`：执行 `has close neighbor` 对应逻辑。
- `F L1076-L1093` `_nearest_slider_path(candidate: SpatialCandidate, paths: Sequence[SliderPathCandidate], *, max_distance: float) -> SliderPathCandidate | None`：执行 `nearest slider path` 对应逻辑。 调用：`_distance_to_polyline`。
- `F L1096-L1107` `_distance_to_polyline(point: tuple[float, float], polyline: Sequence[tuple[float, float]]) -> float`：执行 `distance to polyline` 对应逻辑。 调用：`_point_distance`, `_point_to_segment_distance`。
- `F L1110-L1125` `_point_to_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float`：执行 `point to segment distance` 对应逻辑。 调用：`_point_distance`。
- `F L1128-L1132` `_point_distance(first: tuple[float, float], second: tuple[float, float]) -> float`：执行 `point distance` 对应逻辑。
- `F L1135-L1136` `_mean(values: Sequence[float | int]) -> float | None`：执行 `mean` 对应逻辑。
- `F L1139-L1144` `_percentile(values: Sequence[float | int], percentile: float) -> float | None`：执行 `percentile` 对应逻辑。
- `F L1147-L1148` `_rate(count: int, total: int) -> float`：执行 `rate` 对应逻辑。
- `F L1151-L1157` `_cast_embedding(values: Sequence[float], save_dtype: str) -> list[float]`：执行 `cast embedding` 对应逻辑。
- `F L1160-L1163` `_optional_float(value: Any) -> float | None`：执行 `optional float` 对应逻辑。

## `src/traning/core/decision/pipeline.py`

职责：声明完整训练阶段表；先调用 start.checks 自检，再串接 data-check、空间训练、候选缓存、时序训练和决策导出。
工程依赖：`start.checks`, `traning.conf`, `traning.core.dataset_import`, `traning.core.decision.generator`, `traning.core.decision.runner`, `traning.core.optimization`, `traning.core.result_export`, `traning.core.spatial`, `traning.core.temporal`, `traning.lib.metrics`, `traning.state`, `traning.state.versioning`

- `C L56-L58` `TrainingStage` [CLASS]：封装 `TrainingStage` 相关数据或行为。
- `C L62-L114` `FullTrainingRunConfig` [CLASS]：封装 `FullTrainingRunConfig` 相关数据或行为。
- `M L90-L114` `FullTrainingRunConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L118-L170` `FullTrainingEvaluationResult` [CLASS]：封装 `FullTrainingEvaluationResult` 相关数据或行为。
- `M L144-L170` `FullTrainingEvaluationResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L174-L231` `FullTrainingRunResult` [CLASS]：封装 `FullTrainingRunResult` 相关数据或行为。
- `M L185-L196` `FullTrainingRunResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`_data_input_report_dict`, `self.candidate_cache.as_dict`, `self.decision.as_dict`, `self.evaluation.as_dict`, `self.spatial.as_dict`, `self.startup_checks.as_dict`。
- `M L198-L231` `FullTrainingRunResult.as_summary(self) -> dict[str, Any]`：执行 `as summary` 对应逻辑。
- `F L244-L530` `run_full_training_pipeline(settings: Settings, *, config: FullTrainingRunConfig) -> FullTrainingRunResult` [IO-W]：执行单个参数组的完整可训练、可评分和可追溯闭环。 调用：`FullTrainingRunResult`, `_category_scores_from_report`, `_evaluate_training_outputs`, `_evaluation_stage_message`, `_full_training_parameter_snapshot`, `_json_ready`。
- `F L533-L776` `_evaluate_training_outputs(settings: Settings, *, config: FullTrainingRunConfig, candidate_cache: CandidateCacheBuildResult, spatial: SpatialTrainingResult, temporal: TemporalTrainingResult, decision: TemporalDecisionRunResult) -> FullTrainingEvaluationResult` [IO-W]：执行 `evaluate training outputs` 对应逻辑。 调用：`CurriculumStage`, `OptimizationExecutorConfig`, `ParameterSearchConfig`, `SequenceScoreSpec`, `TrialScoreSpec`, `_evaluation_result_from_score`。
- `F L779-L820` `_evaluation_result_from_score(score_result: DecisionOutputScoreResult, *, report_path: Path, gallery_request_path: Path, gallery_status: str, gallery_output_dir: Path | None, gallery_saved_frame_count: int, attribution_path: Path | None, optimization_plan_path: Path | None, next_job_path: Path | None, gallery_warning: str | None, asha_action: str | None, asha_reasons: tuple[str, ...]) -> FullTrainingEvaluationResult`：执行 `evaluation result from score` 对应逻辑。 调用：`FullTrainingEvaluationResult`。
- `F L823-L833` `run_pipeline(settings: Settings | None=None, *, config: FullTrainingRunConfig | None=None) -> FullTrainingRunResult`：执行 `run pipeline` 对应逻辑。 调用：`FullTrainingRunConfig`, `_device_from_settings`, `load_settings`, `run_full_training_pipeline`。
- `F L836-L848` `_data_input_report_dict(report: DataInputReport) -> dict[str, Any]`：执行 `data input report dict` 对应逻辑。
- `F L851-L854` `_device_from_settings(settings: Settings) -> torch.device`：执行 `device from settings` 对应逻辑。
- `F L857-L869` `_json_ready(value: Any) -> Any`：执行 `json ready` 对应逻辑。 调用：`_json_ready`。
- `F L872-L915` `_full_training_parameter_snapshot(settings: Settings, *, config: FullTrainingRunConfig, spatial: SpatialTrainingResult, candidate_cache: CandidateCacheBuildResult, temporal: TemporalTrainingResult, decision: TemporalDecisionRunResult, evaluation: FullTrainingEvaluationResult) -> dict[str, Any]`：执行 `full training parameter snapshot` 对应逻辑。 调用：`_json_ready`, `_training_parameter_config_snapshot`。
- `F L918-L975` `_training_parameter_config_snapshot(settings: Settings, *, config: FullTrainingRunConfig) -> dict[str, Any]`：执行 `training parameter config snapshot` 对应逻辑。 调用：`_json_ready`。
- `F L978-L1026` `_optimization_base_parameters(settings: Settings, *, config: FullTrainingRunConfig) -> TrialParameters`：执行 `optimization base parameters` 对应逻辑。 调用：`TrialParameters`。
- `F L1029-L1042` `_trial_outcome(evaluation: FullTrainingEvaluationResult) -> tuple[str, str, str, str | None]`：执行 `trial outcome` 对应逻辑。
- `F L1045-L1076` `_report_stage(reporter: TrainingReporter, stage_id: str, name: str, status: str, *, processed: int=0, total: int | None=None, output_path: Path | None=None, warnings: int=0, blocks_training: bool=False, error_reason: str | None=None, score: float | None=None, threshold: float | None=None, message: str | None=None) -> None`：执行 `report stage` 对应逻辑。 调用：`reporter.update_pipeline_stage`。
- `F L1079-L1100` `_evaluation_stage_message(evaluation: FullTrainingEvaluationResult) -> str | None`：执行 `evaluation stage message` 对应逻辑。 调用：`details.append`。
- `F L1103-L1107` `_report_resource(reporter: TrainingReporter) -> None`：执行 `report resource` 对应逻辑。
- `F L1110-L1127` `_category_scores_from_report(report_path: Path) -> dict[str, float]` [IO-R]：执行 `category scores from report` 对应逻辑。 调用：`groups.setdefault.append`。

## `src/traning/core/decision/runner.py`

职责：加载 temporal checkpoint 和候选缓存，导出逐帧动作决策 JSONL。
工程依赖：`traning.conf`, `traning.core.temporal`, `traning.lib.models`, `traning.lib.runtime`

- `C L35-L55` `TemporalDecisionRunResult` [CLASS]：封装 `TemporalDecisionRunResult` 相关数据或行为。
- `M L45-L55` `TemporalDecisionRunResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `F L58-L143` `run_temporal_decision(settings: Settings, *, cache_dir: Path, checkpoint_path: Path, output_dir: Path, device: torch.device) -> TemporalDecisionRunResult` [IO-W]：执行 `run temporal decision` 对应逻辑。 调用：`CausalTemporalModel`, `CudaRuntimeConfig`, `TemporalCandidateWindowDataset.from_cache_dir`, `TemporalDecisionRunResult`, `_decision_diagnostics`, `_decision_row`。
- `F L146-L162` `_load_checkpoint(checkpoint_path: Path, *, device: torch.device) -> Mapping[str, Any]`：加载 `checkpoint` 对应的数据或结果。 调用：`torch.load`。
- `F L165-L232` `_decision_row(window: TemporalWindow, frame_index: int, output) -> dict[str, Any]`：将单帧时序模型输出序列化为坐标空间明确的决策记录。
- `F L235-L259` `_decision_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]`：执行 `decision diagnostics` 对应逻辑。

## `src/traning/core/diagnostics/oracle_ladder.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package.coordinates`, `traning.conf`, `traning.core.dataset_import`, `traning.core.optimization`, `traning.core.temporal`, `traning.lib.coordinates`

- `C L54-L78` `OracleDiagnosticsResult` [CLASS]：封装 `OracleDiagnosticsResult` 相关数据或行为。
- `M L66-L78` `OracleDiagnosticsResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `F L81-L236` `run_oracle_diagnostics(settings: Settings, *, run_dir: Path, output_dir: Path | None=None, fixed_seed: int=2026, max_fixed_frames: int=128, probe_limit: int=12) -> OracleDiagnosticsResult` [IO-W]：逐级注入真值并比较得分，确定最早破坏上限的生产阶段。 调用：`OracleDiagnosticsResult`, `_build_fixed_evaluation_manifest`, `_cache_manifest`, `_candidate_recall`, `_candidate_records_path`, `_decision_diagnostics`。
- `F L239-L241` `_load_candidate_cache(cache_dir: Path) -> tuple[dict[str, Any], ...]`：加载 `candidate cache` 对应的数据或结果。 调用：`_candidate_records_path`, `_read_jsonl`。
- `F L244-L249` `_candidate_records_path(cache_dir: Path) -> Path`：执行 `candidate records path` 对应逻辑。 调用：`_cache_manifest`。
- `F L252-L256` `_cache_manifest(cache_dir: Path) -> dict[str, Any]` [IO-R]：执行 `cache manifest` 对应逻辑。
- `F L259-L318` `_oracle_decisions(records: Sequence[Mapping[str, Any]], *, mode: str, settings: Settings | None=None, candidate_slots: int | None=None) -> tuple[dict[str, Any], ...]`：构造不同阶梯的理想决策，并保持与真实模型相同的输出坐标契约。 调用：`_candidate_id_in_top_slots`, `_osu_to_frame_normalized`, `_point_pair`, `_roundtrip_target_osu`, `_safe_int`, `_target`。
- `F L321-L336` `_roundtrip_target_osu(record: Mapping[str, Any], *, settings: Settings | None) -> tuple[float, float] | None`：将目标视频像素反变换到 osu! 空间，供坐标往返 oracle 使用。 调用：`_point_pair`, `_safe_int`, `_target`, `transform_from_settings_or_sample`。
- `F L339-L364` `_build_fixed_evaluation_manifest(records: Sequence[Mapping[str, Any]], *, seed: int, max_frames: int) -> dict[str, Any]`：构建 `fixed evaluation manifest` 对应的数据或结果。 调用：`_fixed_eval_rows`, `_frame_id`, `_safe_float`, `_safe_int`, `_scene_type`, `_segment_id`。
- `F L367-L403` `_fixed_eval_rows(records: Sequence[Mapping[str, Any]], *, seed: int, max_frames: int) -> tuple[Mapping[str, Any], ...]`：执行 `fixed eval rows` 对应逻辑。 调用：`_coverage_index`, `_scene_type`, `append`, `selected.append`。
- `F L406-L410` `_coverage_index(rows: Sequence[Mapping[str, Any]], offset: int) -> int`：执行 `coverage index` 对应逻辑。
- `F L413-L436` `_candidate_recall(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]`：执行 `candidate recall` 对应逻辑。 调用：`_candidate_distances`, `_empty_recall_bucket`, `_finalize_recall`, `_scene_type`, `_target`, `append`。
- `F L439-L447` `_empty_recall_bucket() -> dict[str, Any]`：执行 `empty recall bucket` 对应逻辑。
- `F L450-L462` `_finalize_recall(bucket: Mapping[str, Any]) -> dict[str, Any]`：执行 `finalize recall` 对应逻辑。 调用：`_mean`, `_percentile`, `_rate`。
- `F L465-L579` `_target_assignment(records: Sequence[Mapping[str, Any]], *, candidate_slots: int | None) -> dict[str, Any]`：执行 `target assignment` 对应逻辑。 调用：`_candidate_distances`, `_candidate_id_in_raw_top_slots`, `_candidate_id_in_top_slots`, `_candidate_rows`, `_frame_id`, `_rate`。
- `F L582-L627` `_temporal_continuity(settings: Settings, cache_dir: Path) -> dict[str, Any]`：执行 `temporal continuity` 对应逻辑。 调用：`TemporalCandidateWindowDataset.from_cache_dir`, `windows.append`。
- `F L630-L723` `_decision_diagnostics(records: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]], *, settings: Settings) -> dict[str, Any]`：执行 `decision diagnostics` 对应逻辑。 调用：`_decision_coordinate_error`, `_frame_key`, `_mean`, `_percentile`, `_rate`, `_safe_float`。
- `F L726-L738` `_decision_coordinate_error(record: Mapping[str, Any], decision: Mapping[str, Any], *, settings: Settings) -> float | None`：执行 `decision coordinate error` 对应逻辑。 调用：`_decision_video_xy`, `_point_pair`, `_target`。
- `F L741-L787` `_write_coordinate_probe_gallery(settings: Settings, records: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]], *, output_dir: Path, probe_limit: int, split: str) -> dict[str, Any]` [IO-W]：写入 `coordinate probe gallery` 对应的数据或结果。 调用：`_coordinate_error_analysis`, `_draw_probe`, `_frame_id`, `_frame_key`, `_probe_points`, `_safe_name`。
- `F L790-L830` `_select_probe_records(records: Sequence[Mapping[str, Any]], *, limit: int) -> tuple[Mapping[str, Any], ...]`：选择 `probe records` 对应的数据或结果。 调用：`_frame_key`, `_osu_distance`, `_point_pair`, `_scene_type`, `_target`, `selected.append`。
- `F L833-L877` `_probe_points(record: Mapping[str, Any], decision: Mapping[str, Any] | None, *, settings: Settings) -> dict[str, Any]`：执行 `probe points` 对应逻辑。 调用：`_decision_video_xy`, `_frame_id`, `_nearest_candidate_point`, `_point_error`, `_point_list`, `_point_pair`。
- `F L880-L894` `_draw_probe(draw: ImageDraw.ImageDraw, probe: Mapping[str, Any]) -> None` [IO-W]：执行 `draw probe` 对应逻辑。
- `F L897-L912` `_coordinate_error_analysis(probes: Sequence[Mapping[str, Any]]) -> dict[str, Any]`：执行 `coordinate error analysis` 对应逻辑。 调用：`_error_pattern`, `_mean`, `_percentile`。
- `F L915-L933` `_error_pattern(dx_values: Sequence[float], dy_values: Sequence[float], distances: Sequence[float]) -> str`：执行 `error pattern` 对应逻辑。 调用：`_mean`。
- `F L936-L952` `_first_error_stage(report: Mapping[str, Any]) -> str`：执行 `first error stage` 对应逻辑。
- `F L955-L977` `_candidate_distances(record: Mapping[str, Any], *, space: str='px') -> list[float]`：执行 `candidate distances` 对应逻辑。 调用：`_candidate_rows`, `_point_pair`, `_target`。
- `F L980-L996` `_nearest_candidate_point(record: Mapping[str, Any], target_video: tuple[float, float] | None) -> tuple[float, float] | None`：执行 `nearest candidate point` 对应逻辑。 调用：`_candidate_rows`。
- `F L999-L1025` `_decision_video_xy(record: Mapping[str, Any], decision: Mapping[str, Any], *, settings: Settings) -> tuple[float, float] | None`：解析诊断决策的视频坐标，候选原始像素优先于模型回归坐标。 调用：`_candidate_rows`, `_point_pair`, `_safe_int`。
- `F L1028-L1057` `_osu_to_frame_normalized(record: Mapping[str, Any], osu_xy: tuple[float, float] | None, *, settings: Settings | None) -> tuple[float, float] | None`：把 osu! 点编码成模型输出使用的整帧归一化坐标。 调用：`_safe_int`, `transform_from_settings_or_sample`。
- `F L1060-L1066` `_candidate_rows(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]`：执行 `candidate rows` 对应逻辑。
- `F L1069-L1076` `_sorted_candidate_rows(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]`：执行 `sorted candidate rows` 对应逻辑。 调用：`_candidate_rows`, `_safe_float`。
- `F L1079-L1087` `_candidate_id_in_top_slots(record: Mapping[str, Any], candidate_id: int, candidate_slots: int | None) -> bool`：执行 `candidate id in top slots` 对应逻辑。 调用：`_candidate_rows`, `_safe_int`。
- `F L1090-L1103` `_candidate_id_in_raw_top_slots(record: Mapping[str, Any], candidate_id: int, candidate_slots: int | None) -> bool`：执行 `candidate id in raw top slots` 对应逻辑。 调用：`_candidate_rows`, `_safe_int`, `_sorted_candidate_rows`。
- `F L1106-L1113` `_temporal_candidate_slots(run_dir: Path, settings: Settings) -> int` [IO-R]：执行 `temporal candidate slots` 对应逻辑。 调用：`_safe_int`。
- `F L1116-L1118` `_target(record: Mapping[str, Any]) -> Mapping[str, Any]`：执行 `target` 对应逻辑。
- `F L1121-L1123` `_target_type(record: Mapping[str, Any]) -> str`：执行 `target type` 对应逻辑。 调用：`_target`。
- `F L1126-L1138` `_scene_type(record: Mapping[str, Any]) -> str`：执行 `scene type` 对应逻辑。 调用：`_candidate_rows`。
- `F L1141-L1143` `_segment_id(record: Mapping[str, Any]) -> str`：执行 `segment id` 对应逻辑。
- `F L1146-L1151` `_record_time_key(record: Mapping[str, Any]) -> tuple[str, int, float]`：执行 `record time key` 对应逻辑。 调用：`_safe_float`, `_safe_int`。
- `F L1154-L1157` `_frame_key(record: Mapping[str, Any]) -> tuple[str, int]`：执行 `frame key` 对应逻辑。 调用：`_safe_int`。
- `F L1160-L1161` `_frame_id(record: Mapping[str, Any]) -> str`：执行 `frame id` 对应逻辑。
- `F L1164-L1172` `_sample_image(image: Any) -> Image.Image`：执行 `sample image` 对应逻辑。
- `F L1175-L1179` `_osu_distance(record: Mapping[str, Any], point: tuple[float, float]) -> float`：执行 `osu distance` 对应逻辑。 调用：`_point_pair`, `_target`。
- `F L1182-L1190` `_point_error(point: tuple[float, float] | None, reference: tuple[float, float] | None) -> dict[str, float] | None`：执行 `point error` 对应逻辑。
- `F L1193-L1202` `_point_pair(value: object) -> tuple[float, float] | None`：执行 `point pair` 对应逻辑。 调用：`_safe_float`。
- `F L1205-L1206` `_point_list(value: tuple[float, float] | None) -> list[float] | None`：执行 `point list` 对应逻辑。
- `F L1209-L1212` `_safe_name(value: str) -> str`：执行 `safe name` 对应逻辑。
- `F L1215-L1220` `_read_jsonl(path: Path) -> tuple[dict[str, Any], ...]` [IO-R]：读取 `jsonl` 对应的数据或结果。 调用：`rows.append`。
- `F L1223-L1228` `_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None` [IO-W]：写入 `jsonl` 对应的数据或结果。
- `F L1231-L1236` `_write_json(path: Path, value: Mapping[str, Any]) -> None` [IO-W]：写入 `json` 对应的数据或结果。
- `F L1239-L1245` `_safe_float(value: object) -> float | None`：执行 `safe float` 对应逻辑。
- `F L1248-L1254` `_safe_int(value: object) -> int | None`：执行 `safe int` 对应逻辑。
- `F L1257-L1258` `_rate(count: int, total: int) -> float`：执行 `rate` 对应逻辑。
- `F L1261-L1262` `_mean(values: Sequence[float]) -> float | None`：执行 `mean` 对应逻辑。
- `F L1265-L1272` `_percentile(values: Sequence[float], percentile: float) -> float | None`：执行 `percentile` 对应逻辑。

## `src/traning/core/full_flow/orchestrator.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`start.flow`, `start.samples`, `traning.conf`, `traning.core.full_flow.result`, `traning.core.full_flow.stages`, `traning.core.model_export`, `traning.core.optimization`, `traning.core.training_inheritance`, `traning.core.training_ramp`, `traning.state.candidate_cache_schema`, `traning.state.versioning`

- `C L51-L85` `FullFlowConfig` [CLASS]：封装 `FullFlowConfig` 相关数据或行为。
- `C L89-L118` `_FlowRuntime` [CLASS]：封装 `FlowRuntime` 相关数据或行为。
- `M L105-L106` `_FlowRuntime.state_path(self) -> Path` [PROPERTY]：执行 `state path` 对应逻辑。
- `M L109-L110` `_FlowRuntime.manifest_path(self) -> Path` [PROPERTY]：执行 `manifest path` 对应逻辑。
- `M L113-L114` `_FlowRuntime.report_json_path(self) -> Path` [PROPERTY]：执行 `report json path` 对应逻辑。
- `M L117-L118` `_FlowRuntime.report_markdown_path(self) -> Path` [PROPERTY]：执行 `report markdown path` 对应逻辑。
- `F L121-L217` `run_full_flow(config: FullFlowConfig) -> FullFlowResult`：按请求模式运行或规划完整流程，并维护可恢复的阶段状态。 调用：`_FlowRuntime`, `_base_manifest`, `_init_layout`, `_initial_stage_states`, `_mark_failed`, `_mark_interrupted`。
- `F L220-L277` `load_full_flow_status(output_root: Path=DEFAULT_FULL_FLOW_ROOT, *, run_id: str | None=None) -> FullFlowResult`：加载 `full flow status` 对应的数据或结果。 调用：`FullFlowResult`, `FullFlowStageState`, `_read_json`。
- `F L280-L358` `_run_startup_section(_runtime: _FlowRuntime, *, reporter) -> None`：执行 `run startup section` 对应逻辑。 调用：`_finish_stage`, `_persist`, `_select_device_name`, `_start_stage`, `_write_json`, `startup.as_dict`。
- `F L361-L412` `_run_resume_section(_runtime: _FlowRuntime, *, reporter)`：执行 `run resume section` 对应逻辑。 调用：`_finish_stage`, `_persist`, `_start_stage`, `_write_json`, `inheritance.as_dict`, `load_inheritance_package`。
- `F L415-L476` `_run_ramp_section(_runtime: _FlowRuntime, *, inheritance, reporter) -> None`：执行 `run ramp section` 对应逻辑。 调用：`_finish_stage`, `_last_training_record`, `_persist`, `_select_device_name`, `_stage_enabled`, `_stage_forced`。
- `F L479-L483` `_run_finalize_section(_runtime: _FlowRuntime) -> None`：执行 `run finalize section` 对应逻辑。 调用：`_finish_export_stage`, `_finish_inheritance_stage`, `_last_training_record`, `load_settings`。
- `F L486-L527` `_finish_export_stage(_runtime: _FlowRuntime, *, settings, record: Mapping[str, Any] | None) -> None`：执行 `finish export stage` 对应逻辑。 调用：`ModelArtifactSpec`, `_record_extra_files`, `_record_path`, `_report_full_flow_stage`, `collect_code_version`, `export_model_artifact`。
- `F L530-L569` `_finish_inheritance_stage(_runtime: _FlowRuntime, *, settings, record: Mapping[str, Any] | None) -> None`：执行 `finish inheritance stage` 对应逻辑。 调用：`_record_extra_files`, `_record_path`, `_report_full_flow_stage`, `create_inheritance_package`, `package.as_dict`, `stage.mark_finished`。
- `F L572-L587` `_mark_plan(_runtime: _FlowRuntime) -> None`：更新状态为 `plan` 对应的数据或结果。 调用：`_report_full_flow_stage`, `_selected_stage_ids`, `_stage_forced`, `state.mark_finished`。
- `F L590-L603` `_mark_training_skipped_for_dry_run(_runtime: _FlowRuntime) -> None`：更新状态为 `training skipped for dry run` 对应的数据或结果。 调用：`_report_full_flow_stage`, `mark_finished`。
- `F L606-L613` `_mark_failed(_runtime: _FlowRuntime, error: Exception) -> None`：更新状态为 `failed` 对应的数据或结果。 调用：`_report_full_flow_stage`, `state.mark_finished`。
- `F L616-L620` `_mark_interrupted(_runtime: _FlowRuntime, reason: str) -> None`：更新状态为 `interrupted` 对应的数据或结果。 调用：`_report_full_flow_stage`, `state.mark_finished`。
- `F L623-L706` `_persist(_runtime: _FlowRuntime, *, status: str, stop_reason: str | None=None) -> None`：在重要阶段边界同步状态、manifest、报告和 latest 指针。 调用：`_report_full_flow_stage`, `_write_json`, `_write_reports`, `report_stage.mark_finished`, `report_stage.mark_started`, `stage.as_dict`。
- `F L709-L730` `_result(_runtime: _FlowRuntime, *, status: str) -> FullFlowResult`：执行 `result` 对应逻辑。 调用：`FullFlowResult`, `utc_now`。
- `F L733-L756` `_base_manifest(config: FullFlowConfig, _runtime: _FlowRuntime) -> dict[str, Any]`：执行 `base manifest` 对应逻辑。 调用：`_dataset_fingerprint`, `_file_sha256`, `_selected_stage_ids`, `collect_code_version`, `collect_code_version.as_dict`, `load_settings`。
- `F L759-L763` `_initial_stage_states() -> dict[str, FullFlowStageState]`：执行 `initial stage states` 对应逻辑。 调用：`FullFlowStageState`。
- `F L766-L783` `_publish_initial_dashboard_stages(_runtime: _FlowRuntime) -> None`：执行 `publish initial dashboard stages` 对应逻辑。 调用：`_selected_stage_ids`, `reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L786-L788` `_report_resource_snapshot(reporter: TrainingReporter) -> None`：执行 `report resource snapshot` 对应逻辑。
- `F L791-L812` `_start_stage(_runtime: _FlowRuntime, stage_id: str, reporter) -> None`：执行 `start stage` 对应逻辑。 调用：`_phase_for_full_flow_stage`, `_report_full_flow_stage`, `_stage_enabled`, `mark_finished`, `reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L815-L835` `_finish_stage(_runtime: _FlowRuntime, stage_id: str, status, *, result: Mapping[str, Any] | None=None, warnings: tuple[str, ...]=(), artifacts: tuple[str, ...]=(), restored: bool=False) -> None`：执行 `finish stage` 对应逻辑。 调用：`_report_full_flow_stage`, `_stage_forced`, `mark_finished`。
- `F L838-L872` `_report_full_flow_stage(_runtime: _FlowRuntime, stage_id: str) -> None`：执行 `report full flow stage` 对应逻辑。 调用：`_dashboard_status`, `_optional_int`, `_phase_for_full_flow_stage`, `reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L875-L888` `_dashboard_status(status: str) -> str`：执行 `dashboard status` 对应逻辑。
- `F L891-L917` `_phase_for_full_flow_stage(stage_id: str) -> PipelinePhase`：执行 `phase for full flow stage` 对应逻辑。
- `F L920-L926` `_optional_int(value: object, *, default: int | None=None) -> int | None`：执行 `optional int` 对应逻辑。
- `F L929-L930` `_stage_enabled(_runtime: _FlowRuntime, stage_id: str) -> bool`：执行 `stage enabled` 对应逻辑。 调用：`_selected_stage_ids`。
- `F L933-L934` `_stage_forced(config: FullFlowConfig, stage_id: str) -> bool`：执行 `stage forced` 对应逻辑。 调用：`validate_stage_id`。
- `F L937-L952` `_selected_stage_ids(config: FullFlowConfig) -> tuple[str, ...]`：执行 `selected stage ids` 对应逻辑。 调用：`validate_stage_id`。
- `F L955-L976` `_validate_config(config: FullFlowConfig) -> None`：校验 `config` 对应的数据或结果。 调用：`_selected_stage_ids`, `validate_stage_id`。
- `F L979-L991` `_init_layout(output_dir: Path) -> None` [IO-W]：执行 `init layout` 对应逻辑。
- `F L994-L996` `_write_resolved_config(config_path: Path, output_dir: Path) -> None`：写入 `resolved config` 对应的数据或结果。
- `F L999-L1025` `_write_reports(_runtime: _FlowRuntime, state: Mapping[str, Any]) -> None` [IO-W]：写入 `reports` 对应的数据或结果。 调用：`_write_json`, `lines.append`。
- `F L1028-L1053` `_last_training_record(ramp_manifest_path: Path | None) -> Mapping[str, Any] | None`：执行 `last training record` 对应逻辑。 调用：`_read_json`。
- `F L1056-L1065` `_record_path(record: Mapping[str, Any], keys: tuple[str, ...]) -> Path | None`：执行 `record path` 对应逻辑。
- `F L1068-L1077` `_record_extra_files(record: Mapping[str, Any]) -> dict[str, Path]`：执行 `record extra files` 对应逻辑。 调用：`_record_path`。
- `F L1080-L1087` `_select_device_name(device: str) -> str`：选择 `device name` 对应的数据或结果。
- `F L1090-L1100` `_dataset_fingerprint(settings) -> dict[str, Any]`：执行 `dataset fingerprint` 对应逻辑。
- `F L1103-L1108` `_file_sha256(path: Path) -> str` [IO-R IO-W]：执行 `file sha256` 对应逻辑。
- `F L1111-L1112` `_new_run_id() -> str`：执行 `new run id` 对应逻辑。
- `F L1115-L1116` `_read_json(path: Path) -> dict[str, Any]` [IO-R]：读取 `json` 对应的数据或结果。
- `F L1119-L1124` `_write_json(path: Path, value: Mapping[str, Any]) -> None` [IO-W]：写入 `json` 对应的数据或结果。 调用：`_json_ready`。
- `F L1127-L1138` `_json_ready(value: Any) -> Any`：执行 `json ready` 对应逻辑。 调用：`_json_ready`。

## `src/traning/core/full_flow/result.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.full_flow.stages`

- `F L13-L14` `utc_now() -> str` [IO-W]：执行 `utc now` 对应逻辑。
- `C L18-L56` `FullFlowStageState` [CLASS]：封装 `FullFlowStageState` 相关数据或行为。
- `M L30-L34` `FullFlowStageState.mark_started(self) -> None`：更新状态为 `started` 对应的数据或结果。 调用：`utc_now`。
- `M L36-L53` `FullFlowStageState.mark_finished(self, status: FullFlowStageStatus, *, result: Mapping[str, Any] | None=None, warnings: tuple[str, ...]=(), error: str | None=None, artifacts: tuple[str, ...]=(), restored: bool=False) -> None`：更新状态为 `finished` 对应的数据或结果。 调用：`utc_now`。
- `M L55-L56` `FullFlowStageState.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L60-L94` `FullFlowResult` [CLASS]：封装 `FullFlowResult` 相关数据或行为。
- `M L77-L94` `FullFlowResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`stage.as_dict`。

## `src/traning/core/full_flow/stages.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L25-L36` `FullFlowStageSpec` [CLASS]：封装 `FullFlowStageSpec` 相关数据或行为。
- `M L35-L36` `FullFlowStageSpec.as_dict(self) -> dict[str, object]`：执行 `as dict` 对应逻辑。
- `F L170-L171` `stage_ids() -> tuple[str, ...]`：执行 `stage ids` 对应逻辑。
- `F L174-L178` `validate_stage_id(stage_id: str) -> str` [IO-W]：校验 `stage id` 对应的数据或结果。

## `src/traning/core/model_export/artifact.py`

职责：导出 inference/resume PyTorch artifact，写 manifest、文件 sha256 和版本信息。
工程依赖：`traning.conf`, `traning.lib.data`, `traning.lib.models`, `traning.state.versioning`

- `C L29-L41` `ArtifactFile` [CLASS]：封装 `ArtifactFile` 相关数据或行为。
- `M L35-L41` `ArtifactFile.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`self.path.as_posix`。
- `C L45-L63` `ModelArtifactSpec` [CLASS]：封装 `ModelArtifactSpec` 相关数据或行为。
- `M L58-L63` `ModelArtifactSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L67-L81` `ModelArtifactResult` [CLASS]：封装 `ModelArtifactResult` 相关数据或行为。
- `M L74-L81` `ModelArtifactResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`item.as_dict`。
- `F L84-L89` `_sha256(path: Path) -> str` [IO-R IO-W]：执行 `sha256` 对应逻辑。
- `F L92-L102` `_copy_file(source: Path, destination: Path, role: str) -> ArtifactFile` [IO-W]：执行 `copy file` 对应逻辑。 调用：`ArtifactFile`, `_sha256`。
- `F L105-L112` `_copy_optional(files: list[ArtifactFile], source: Path | None, destination: Path, role: str) -> None`：执行 `copy optional` 对应逻辑。 调用：`_copy_file`, `files.append`。
- `F L115-L135` `_write_readme(path: Path, spec: ModelArtifactSpec) -> ArtifactFile` [IO-W]：写入 `readme` 对应的数据或结果。 调用：`ArtifactFile`, `_sha256`。
- `F L138-L144` `_manifest_file(item: ArtifactFile, artifact_dir: Path) -> dict[str, Any]`：执行 `manifest file` 对应逻辑。 调用：`item.as_dict`。
- `F L147-L219` `export_model_artifact(spec: ModelArtifactSpec) -> ModelArtifactResult` [IO-W]：复制选定文件并生成包含相对路径、摘要和契约版本的 manifest。 调用：`ArtifactFile`, `ModelArtifactResult`, `_copy_file`, `_copy_optional`, `_manifest_file`, `_sha256`。
- `F L222-L239` `validate_model_artifact(manifest_path: Path | str) -> tuple[str, ...]` [IO-R]：校验 `model artifact` 对应的数据或结果。 调用：`_sha256`, `issues.append`。
- `F L242-L263` `migrate_settings_file(settings_path: Path | str) -> tuple[Path, dict[str, Any]]` [IO-R IO-W]：非破坏性生成新配置，并记录每项兼容迁移动作。 调用：`log.append`。
- `F L266-L279` `import_model_artifact(manifest_path: Path | str) -> dict[str, Any]` [IO-R]：导入 `model artifact` 对应的数据或结果。 调用：`load_settings`, `migrate_settings_file`, `validate_model_artifact`。
- `F L282-L328` `smoke_test_model_artifact(manifest_path: Path | str, *, device: str='cpu') -> dict[str, Any]`：执行 `smoke test model artifact` 对应逻辑。 调用：`CausalTemporalModel`, `append_color_cues`, `build_model_stack`, `import_model_artifact`, `load_settings`, `temporal.initial_state`。

## `src/traning/core/optimization/attribution/analyzer.py`

职责：汇总 trial 错误域、错误 tag 和 hard examples。
工程依赖：`traning.core.optimization.error_attribution`

- `C L25-L47` `HardExample` [CLASS]：封装 `HardExample` 相关数据或行为。
- `M L36-L47` `HardExample.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L51-L65` `AttributionSummary` [CLASS]：封装 `AttributionSummary` 相关数据或行为。
- `M L58-L65` `AttributionSummary.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`example.as_dict`。
- `F L68-L78` `_click_severity(click) -> float`：执行 `click severity` 对应逻辑。
- `F L81-L95` `_unresolved_example(sample: SampleScoreReport, target_id: str) -> HardExample`：执行 `unresolved example` 对应逻辑。 调用：`HardExample`, `classify_unresolved_sample_error`。
- `F L98-L165` `analyze_trial_attribution(report: TrialScoreReport, *, max_hard_examples: int=32) -> AttributionSummary`：执行 `analyze trial attribution` 对应逻辑。 调用：`AttributionSummary`, `HardExample`, `_click_severity`, `_unresolved_example`, `hard_examples.append`。

## `src/traning/core/optimization/error_attribution.py`

职责：为评分图集和优化分析提供共用的 unresolved 错误域分类规则。

- `C L9-L12` `UnresolvedSample(Protocol)` [CLASS]：归因规则所需的最小样本协议，避免反向依赖评分包入口。
- `F L15-L44` `classify_unresolved_sample_error(sample: UnresolvedSample) -> tuple[str, tuple[str, ...], str]`：按最早发生的可观测失败边界归因一个未解析目标。

## `src/traning/core/optimization/parameter_search/curriculum.py`

职责：实现连续通过门槛、子项目 gate 和课程晋级检查结果。
工程依赖：`traning.core.optimization.scoring`

- `C L14-L25` `SubprojectPassRule` [CLASS]：封装 `SubprojectPassRule` 相关数据或行为。
- `M L19-L25` `SubprojectPassRule.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L37-L57` `SubprojectGateResult` [CLASS]：封装 `SubprojectGateResult` 相关数据或行为。
- `M L45-L57` `SubprojectGateResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L61-L72` `CurriculumGateResult` [CLASS]：封装 `CurriculumGateResult` 相关数据或行为。
- `M L65-L72` `CurriculumGateResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`result.as_dict`, `self.subprojects.items`。
- `F L75-L102` `_gate_subproject(subproject: str, samples: Sequence[SampleScoreReport], rule: SubprojectPassRule) -> SubprojectGateResult`：执行 `gate subproject` 对应逻辑。 调用：`SubprojectGateResult`。
- `F L105-L126` `evaluate_curriculum_gate(samples: Sequence[SampleScoreReport], *, rules: Mapping[str, SubprojectPassRule]=DEFAULT_CURRICULUM_RULES) -> CurriculumGateResult`：执行 `evaluate curriculum gate` 对应逻辑。 调用：`CurriculumGateResult`, `_gate_subproject`, `append`。

## `src/traning/core/optimization/parameter_search/executor.py`

职责：根据优化计划生成有界绝对参数 job，统一校验 checkpoint 继承、短 trial ID 和 JSONL/SQLite 记录。
工程依赖：`traning.core.optimization.attribution`, `traning.core.optimization.parameter_search.curriculum`, `traning.core.optimization.parameter_search.hard_examples`, `traning.core.optimization.parameter_search.planner`, `traning.core.optimization.scoring`, `traning.state`

- `C L42-L57` `_ParameterSpec` [CLASS]：封装 `ParameterSpec` 相关数据或行为。
- `M L48-L57` `_ParameterSpec.normalize(self, value: object) -> int | float`：执行 `normalize` 对应逻辑。
- `C L84-L119` `TrainingJobSpec` [CLASS]：封装 `TrainingJobSpec` 相关数据或行为。
- `M L93-L104` `TrainingJobSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L106-L119` `TrainingJobSpec.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`self.parameters.model_dump`。
- `C L123-L147` `OptimizationExecution` [CLASS]：封装 `OptimizationExecution` 相关数据或行为。
- `M L135-L147` `OptimizationExecution.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`self.job.as_dict`, `self.trial.model_dump`。
- `C L151-L166` `OptimizationExecutorConfig` [CLASS]：封装 `OptimizationExecutorConfig` 相关数据或行为。
- `M L160-L166` `OptimizationExecutorConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L169-L192` `JsonlTrialStore` [CLASS]：封装 `JsonlTrialStore` 相关数据或行为。
- `M L170-L171` `JsonlTrialStore.__init__(self, path: Path) -> None`：初始化实例依赖、配置和运行状态。
- `M L173-L183` `JsonlTrialStore.append(self, execution: OptimizationExecution) -> None` [IO-W]：执行 `append` 对应逻辑。 调用：`execution.as_dict`, `self.path.parent.mkdir`。
- `M L185-L192` `JsonlTrialStore.load(self) -> tuple[dict[str, Any], ...]` [IO-R]：执行 `load` 对应逻辑。 调用：`records.append`, `self.path.exists`, `self.path.read_text`, `self.path.read_text.splitlines`。
- `C L195-L270` `SQLiteTrialStore` [CLASS]：封装 `SQLiteTrialStore` 相关数据或行为。
- `M L196-L197` `SQLiteTrialStore.__init__(self, path: Path) -> None`：初始化实例依赖、配置和运行状态。
- `M L199-L223` `SQLiteTrialStore._connect(self) -> sqlite3.Connection` [IO-W]：执行 `connect` 对应逻辑。 调用：`self.path.parent.mkdir`。
- `M L225-L257` `SQLiteTrialStore.append(self, execution: OptimizationExecution) -> None`：执行 `append` 对应逻辑。 调用：`execution.as_dict`, `self._connect`。
- `M L259-L270` `SQLiteTrialStore.load(self) -> tuple[dict[str, Any], ...]`：执行 `load` 对应逻辑。 调用：`self._connect`, `self.path.exists`。
- `F L273-L283` `create_trial_store(*, backend: str, jsonl_path: Path, sqlite_path: Path) -> JsonlTrialStore | SQLiteTrialStore`：执行 `create trial store` 对应逻辑。 调用：`JsonlTrialStore`, `SQLiteTrialStore`。
- `F L286-L331` `trial_history_from_records(records: Sequence[Mapping[str, Any]], *, score_version: str | None=None) -> tuple[TrialHistoryEntry, ...]`：把已评估 execution 记录还原为 ASHA 可比较的源 trial 历史。 调用：`CurriculumStage`, `TrialHistoryEntry`。
- `F L334-L380` `_apply_section_updates(section: str, base: Mapping[str, object], updates: Mapping[str, Any]) -> dict[str, object]`：应用 `section updates` 对应的数据或结果。 调用：`spec.normalize`。
- `F L383-L403` `_apply_parameter_updates(parameters: TrialParameters, updates: Mapping[str, Mapping[str, Any]]) -> TrialParameters`：应用 `parameter updates` 对应的数据或结果。 调用：`TrialParameters`, `_apply_section_updates`。
- `F L406-L409` `normalize_trial_parameters(parameters: TrialParameters) -> TrialParameters`：把 job 中已有参数规范成 runner 可直接消费的绝对值。 调用：`_apply_parameter_updates`。
- `F L412-L443` `training_job_from_dict(raw: Mapping[str, Any]) -> TrainingJobSpec`：解析并校验持久化 job；dry-run 和真实执行必须共用此入口。 调用：`CurriculumStage`, `TrainingJobSpec`, `normalize_trial_parameters`。
- `F L446-L450` `_budget_steps(config: OptimizationExecutorConfig, rung: int) -> int`：执行 `budget steps` 对应逻辑。
- `F L453-L457` `_next_trial_id(source_trial_id: str, rung: int, stage: CurriculumStage) -> str`：执行 `next trial id` 对应逻辑。
- `F L460-L525` `execute_optimization_plan(report: TrialScoreReport, attribution: AttributionSummary, plan: OptimizationPlan, *, base_parameters: TrialParameters | None=None, parent_checkpoint_path: Path | None=None, config: OptimizationExecutorConfig=OptimizationExecutorConfig(), store: JsonlTrialStore | SQLiteTrialStore | None=None) -> OptimizationExecution`：执行 `execute optimization plan` 对应逻辑。 调用：`JsonlTrialStore`, `OptimizationExecution`, `TrainingJobSpec`, `TrialMetadata`, `_apply_parameter_updates`, `_budget_steps`。

## `src/traning/core/optimization/parameter_search/hard_examples.py`

职责：把归因 hard examples 转换为样本采样权重计划。
工程依赖：`traning.core.optimization.attribution`

- `C L14-L25` `HardExampleSamplingPlan` [CLASS]：封装 `HardExampleSamplingPlan` 相关数据或行为。
- `M L18-L25` `HardExampleSamplingPlan.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`self.reasons.items`。
- `F L28-L58` `build_hard_example_sampling_plan(attribution: AttributionSummary, *, base_weight: float=1.0, severity_multiplier: float=1.5, max_examples: int=128) -> HardExampleSamplingPlan`：构建并返回 `hard example sampling plan` 对应的数据或结果。 调用：`HardExampleSamplingPlan`, `append`。

## `src/traning/core/optimization/parameter_search/objectives.py`

职责：计算 quality/VRAM/latency 多目标排序值和可复现 sort key。
工程依赖：`traning.core.optimization.scoring`

- `C L21-L46` `ObjectiveScore` [CLASS]：封装 `ObjectiveScore` 相关数据或行为。
- `M L26-L30` `ObjectiveScore.composite_score(self) -> float` [PROPERTY]：执行 `composite score` 对应逻辑。 调用：`self.values.get`, `self.weights.items`。
- `M L32-L39` `ObjectiveScore.as_dict(self) -> dict[str, object]`：执行 `as dict` 对应逻辑。 调用：`self.sort_key`。
- `M L41-L46` `ObjectiveScore.sort_key(self) -> tuple[float, float, float]`：执行 `sort key` 对应逻辑。 调用：`self.values.get`。
- `F L49-L55` `objective_values_from_report(report: TrialScoreReport) -> dict[str, float]`：执行 `objective values from report` 对应逻辑。
- `F L58-L66` `score_trial_objectives(report: TrialScoreReport, *, weights: Mapping[str, float] | None=None) -> ObjectiveScore`：执行 `score trial objectives` 对应逻辑。 调用：`ObjectiveScore`, `objective_values_from_report`。

## `src/traning/core/optimization/parameter_search/planner.py`

职责：根据评分、归因、历史 trial、资源指标和多目标分数生成下一轮参数调整计划。
工程依赖：`traning.core.optimization.attribution`, `traning.core.optimization.parameter_search.objectives`, `traning.core.optimization.scoring`, `traning.state`

- `C L26-L56` `ASHAConfig` [CLASS]：封装 `ASHAConfig` 相关数据或行为。
- `M L40-L56` `ASHAConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L60-L82` `ParameterSearchConfig` [CLASS]：封装 `ParameterSearchConfig` 相关数据或行为。
- `M L70-L82` `ParameterSearchConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`self.objective_weights.values`。
- `C L86-L101` `TrialHistoryEntry` [CLASS]：封装 `TrialHistoryEntry` 相关数据或行为。
- `M L94-L101` `TrialHistoryEntry.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L105-L148` `OptimizationPlan` [CLASS]：封装 `OptimizationPlan` 相关数据或行为。
- `M L125-L148` `OptimizationPlan.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`self.parameter_updates.items`。
- `F L159-L161` `_next_stage(stage: CurriculumStage) -> CurriculumStage`：执行 `next stage` 对应逻辑。
- `F L164-L169` `_quantile(values: Sequence[float], quantile: float) -> float`：执行 `quantile` 对应逻辑。
- `F L172-L206` `_asha_action(report: TrialScoreReport, history: Sequence[TrialHistoryEntry], *, current_stage: CurriculumStage, rung: int, config: ASHAConfig) -> tuple[ASHAAction, tuple[str, ...]]`：执行 `asha action` 对应逻辑。 调用：`_quantile`, `reasons.append`。
- `F L209-L219` `_priority_domains(attribution: AttributionSummary) -> tuple[str, ...]`：执行 `priority domains` 对应逻辑。
- `F L222-L237` `_hard_example_keys(attribution: AttributionSummary, *, limit: int) -> tuple[str, ...]`：执行 `hard example keys` 对应逻辑。 调用：`keys.append`。
- `F L240-L255` `_set_update(updates: dict[str, dict[str, Any]], section: str, name: str, value: Any) -> None`：执行 `set update` 对应逻辑。
- `F L258-L285` `_apply_domain_updates(updates: dict[str, dict[str, Any]], attribution: AttributionSummary, reasons: list[str]) -> None`：应用 `domain updates` 对应的数据或结果。 调用：`_set_update`, `reasons.append`。
- `F L288-L308` `_apply_overall_updates(updates: dict[str, dict[str, Any]], report: TrialScoreReport, config: ParameterSearchConfig, reasons: list[str]) -> None`：应用 `overall updates` 对应的数据或结果。 调用：`_set_update`, `reasons.append`。
- `F L311-L388` `plan_next_trial(report: TrialScoreReport, attribution: AttributionSummary, *, history: Sequence[TrialHistoryEntry]=(), current_stage: CurriculumStage=CurriculumStage.BASIC, rung: int=0, config: ParameterSearchConfig=ParameterSearchConfig()) -> OptimizationPlan`：执行 `plan next trial` 对应逻辑。 调用：`OptimizationPlan`, `_apply_domain_updates`, `_apply_overall_updates`, `_asha_action`, `_hard_example_keys`, `_next_stage`。

## `src/traning/core/optimization/scoring/evaluator.py`

职责：按 point-slider-v2 和 click-sequence-v1 聚合 sample/trial 级质量分；目标帧按目标数加权，防止大量空帧 no-op 虚抬试验分。
工程依赖：`traning.lib.metrics`, `traning.state`

- `C L24-L47` `TrialScoreSpec` [CLASS]：封装 `TrialScoreSpec` 相关数据或行为。
- `M L33-L47` `TrialScoreSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L51-L71` `SampleScoringInput` [CLASS]：封装 `SampleScoringInput` 相关数据或行为。
- `M L60-L71` `SampleScoringInput.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L75-L106` `SampleScoreReport` [CLASS]：封装 `SampleScoreReport` 相关数据或行为。
- `M L91-L106` `SampleScoreReport.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L110-L164` `TrialScoreReport` [CLASS]：封装 `TrialScoreReport` 相关数据或行为。
- `M L120-L121` `TrialScoreReport.target_count(self) -> int` [PROPERTY]：执行 `target count` 对应逻辑。
- `M L124-L125` `TrialScoreReport.hit_count(self) -> int` [PROPERTY]：执行 `hit count` 对应逻辑。
- `M L128-L129` `TrialScoreReport.miss_count(self) -> int` [PROPERTY]：执行 `miss count` 对应逻辑。
- `M L132-L133` `TrialScoreReport.unresolved_count(self) -> int` [PROPERTY]：执行 `unresolved count` 对应逻辑。
- `M L136-L137` `TrialScoreReport.frequency_limited_count(self) -> int` [PROPERTY]：执行 `frequency limited count` 对应逻辑。
- `M L140-L148` `TrialScoreReport.passed(self) -> bool` [PROPERTY]：执行 `passed` 对应逻辑。
- `M L150-L164` `TrialScoreReport.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`sample.as_dict`, `self.parameters.model_dump`。
- `F L167-L168` `_clamp01(value: float) -> float`：执行 `clamp01` 对应逻辑。
- `F L171-L172` `_safe_rate(count: int, total: int) -> float`：执行 `safe rate` 对应逻辑。
- `F L175-L181` `_resolved_object_score(sequence: SequenceScore, target_count: int) -> float`：执行 `resolved object score` 对应逻辑。 调用：`_clamp01`。
- `F L184-L224` `score_sample(sample: SampleScoringInput, *, spec: TrialScoreSpec=TrialScoreSpec()) -> SampleScoreReport`：执行 `score sample` 对应逻辑。 调用：`SampleScoreReport`, `_clamp01`, `_resolved_object_score`, `_safe_rate`, `score_click_sequence`。
- `F L227-L265` `score_trial(trial_id: str, samples: Sequence[SampleScoringInput], *, parameters: TrialParameters | None=None, metrics: Mapping[str, float] | None=None, spec: TrialScoreSpec=TrialScoreSpec()) -> TrialScoreReport`：执行 `score trial` 对应逻辑。 调用：`TrialParameters`, `TrialScoreReport`, `_clamp01`, `sample_weight`, `score_sample`。
- `N L243-L248` `score_trial.sample_weight(report: SampleScoreReport) -> float`：执行 `sample weight` 对应逻辑。

## `src/traning/core/optimization/scoring/gallery.py`

职责：把 TrialScoreReport 转换为结果导出使用的 BatchGalleryRequest。
工程依赖：`traning.core.optimization.error_attribution`, `traning.core.optimization.scoring.evaluator`, `traning.state`

- `F L21-L29` `_metadata_point(value: object) -> tuple[float, float] | None`：执行 `metadata point` 对应逻辑。
- `F L32-L37` `_metadata_probability(value: object) -> float | None`：执行 `metadata probability` 对应逻辑。
- `F L40-L44` `_metadata_action(value: object) -> str | None`：执行 `metadata action` 对应逻辑。
- `F L47-L62` `_representative_click(sample: SampleScoreReport)`：执行 `representative click` 对应逻辑。
- `F L65-L72` `_unresolved_source_index(sample: SampleScoreReport) -> int | None`：执行 `unresolved source index` 对应逻辑。
- `F L75-L118` `_frame_evaluation(sample: SampleScoreReport) -> FrameEvaluation`：执行 `frame evaluation` 对应逻辑。 调用：`FrameEvaluation`, `_metadata_action`, `_metadata_point`, `_metadata_probability`, `_representative_click`, `_unresolved_source_index`。
- `F L121-L143` `build_batch_gallery_request(report: TrialScoreReport, *, batch_id: str | None=None, random_seed: int=2026, metadata: dict[str, object] | None=None) -> BatchGalleryRequest`：Build the result-export request directly from optimization scoring。 调用：`BatchGalleryRequest`, `TrialGalleryEvaluation`, `_frame_evaluation`。

## `src/traning/core/optimization/scoring/run_outputs.py`

职责：把候选缓存与决策输出统一换算到 osu 坐标评分；模型归一化坐标先还原为训练帧像素。
工程依赖：`package.coordinates`, `traning.core.optimization.scoring.evaluator`, `traning.lib.coordinates`, `traning.lib.metrics`, `traning.lib.training`, `traning.state`

- `C L29-L52` `DecisionOutputScoreResult` [CLASS]：封装 `DecisionOutputScoreResult` 相关数据或行为。
- `M L37-L52` `DecisionOutputScoreResult.as_summary(self) -> dict[str, Any]`：执行 `as summary` 对应逻辑。
- `F L55-L125` `score_decision_outputs(*, parameter_group_id: str, candidate_cache_path: Path, decisions_path: Path, metrics: Mapping[str, float] | None=None, circle_radius: float | None=None, spec: TrialScoreSpec=TrialScoreSpec(), settings: Any | None=None) -> DecisionOutputScoreResult`：执行 `score decision outputs` 对应逻辑。 调用：`DecisionOutputScoreResult`, `TrialParameters`, `_frame_key`, `_index_unique_frames`, `_read_jsonl`, `_sample_from_rows`。
- `F L128-L183` `_sample_from_rows(cache_row: Mapping[str, Any], decision: Mapping[str, Any], *, parameter_group_id: str, circle_radius_override: float | None, settings: Any | None=None) -> SampleScoringInput`：执行 `sample from rows` 对应逻辑。 调用：`SampleScoringInput`, `_circle_radius_from_row`, `_predicted_clicks`, `_prediction_video_xy`, `_safe_int`, `_subproject_from_sample_key`。
- `F L186-L203` `_circle_radius_from_row(cache_row: Mapping[str, Any], target_metadata: Mapping[str, Any], *, override: float | None) -> float`：按显式覆盖、新缓存字段、目标字段、旧协议默认值依次解析半径。 调用：`_safe_float`。
- `F L206-L244` `_target_objects(row: Mapping[str, Any], *, settings: Any | None=None) -> tuple[TargetObject, ...]`：执行 `target objects` 对应逻辑。 调用：`TargetObject`, `_point_pair`, `_safe_float`, `_safe_int`, `_video_to_osu_pair`。
- `F L247-L281` `_predicted_clicks(cache_row: Mapping[str, Any], decision: Mapping[str, Any], *, predicted_video_xy: tuple[float, float] | None=None, settings: Any | None=None) -> tuple[PredictedClick, ...]`：把决策位置统一转换为 osu! 坐标后构造评分点击事件。 调用：`PredictedClick`, `_normalized_frame_to_osu`, `_safe_float`, `_video_to_osu`。
- `F L284-L324` `_prediction_video_xy(cache_row: Mapping[str, Any], decision: Mapping[str, Any]) -> tuple[float, float] | None`：解析决策对应的视频像素，优先使用被选候选的原始像素位置。 调用：`_point_pair`, `_safe_float`, `_safe_int`。
- `F L327-L337` `_video_to_osu_pair(value: object, row: Mapping[str, Any], *, settings: Any | None=None) -> tuple[float, float] | None`：校验一个视频像素坐标对，并使用样本变换映射到 osu! 空间。 调用：`_point_pair`, `_video_to_osu`。
- `F L340-L358` `_video_to_osu(x: float, y: float, row: Mapping[str, Any], *, settings: Any | None=None) -> tuple[float, float] | None`：使用与当前缓存样本匹配的变换执行 frame pixel -> osu。 调用：`_safe_int`, `transform_from_settings_or_sample`。
- `F L361-L383` `_normalized_frame_to_osu(value: object, row: Mapping[str, Any], *, settings: Any | None=None) -> tuple[float, float] | None`：按整帧尺寸还原模型坐标，再由样本变换转换到 osu! 空间。 调用：`_point_pair`, `_safe_int`, `_video_to_osu`。
- `F L386-L395` `_point_pair(value: object) -> tuple[float, float] | None`：执行 `point pair` 对应逻辑。 调用：`_safe_float`。
- `F L398-L409` `_read_jsonl(path: Path) -> list[dict[str, Any]]` [IO-W]：读取 `jsonl` 对应的数据或结果。 调用：`rows.append`。
- `F L412-L415` `_frame_key(row: Mapping[str, Any]) -> tuple[str, int]`：执行 `frame key` 对应逻辑。 调用：`_safe_int`。
- `F L418-L433` `_index_unique_frames(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[tuple[str, int], Mapping[str, Any]]`：校验帧身份并构建唯一索引，避免字典覆盖重复帧。 调用：`_frame_key`。
- `F L436-L441` `_subproject_from_sample_key(sample_key: str) -> str`：执行 `subproject from sample key` 对应逻辑。
- `F L444-L450` `_safe_float(value: object) -> float | None`：执行 `safe float` 对应逻辑。
- `F L453-L459` `_safe_int(value: object) -> int | None`：执行 `safe int` 对应逻辑。

## `src/traning/core/result_export/preview.py`

职责：组装 Dataset、单帧点击标注和批次最佳参数图集。
工程依赖：`traning.conf`, `traning.core.dataset_import`, `traning.core.result_export.service`, `traning.lib.visualization`, `traning.state`

- `F L18-L40` `visualize_click_label(settings: Settings, *, segment_index: int=0, object_index: int=0, output_path: Path | None=None, show_window: bool | None=None) -> VisualizationResult`：执行 `visualize click label` 对应逻辑。 调用：`OptionalTrainingVisualizer`, `build_dataset`, `select_click_frame`, `visualizer.visualize`。
- `F L43-L57` `save_annotation_gallery(settings: Settings, request: BatchGalleryRequest, *, output_root: Path | None=None, samples_per_group: int | None=None) -> GalleryResult`：执行 `save annotation gallery` 对应逻辑。 调用：`OptionalTrainingVisualizer`, `build_dataset`, `visualizer.save_gallery`。

## `src/traning/core/result_export/service.py`

职责：可选可视化故障隔离、一次性告警和训练步频率控制。
工程依赖：`traning.conf`, `traning.lib.data`, `traning.lib.visualization`, `traning.state`

- `C L25-L193` `OptionalTrainingVisualizer` [CLASS]：尽力完成可视化，但绝不把旁路故障抛回训练主流程。
- `M L28-L33` `OptionalTrainingVisualizer.__init__(self, settings: VisualizationSettings)`：初始化实例依赖、配置和运行状态。
- `M L35-L39` `OptionalTrainingVisualizer._warning_once(self, message: str) -> str | None`：执行 `warning once` 对应逻辑。
- `M L41-L112` `OptionalTrainingVisualizer.visualize(self, sample: dict[str, Any], *, target_source_index: int | None=None, output_path: Path | None=None, force: bool=False, show_window: bool | None=None) -> VisualizationResult`：执行 `visualize` 对应逻辑。 调用：`VisualizationResult`, `allocate_output_identity`, `launch_image_window`, `render_annotated_frame`, `save_annotated_frame`, `self._default_output_path`。
- `M L114-L133` `OptionalTrainingVisualizer.maybe_visualize_step(self, sample: dict[str, Any], *, global_step: int, target_source_index: int | None=None) -> VisualizationResult`：执行 `maybe visualize step` 对应逻辑。 调用：`VisualizationResult`, `self._warning_once`, `self.visualize`。
- `M L135-L181` `OptionalTrainingVisualizer.save_gallery(self, dataset: SegmentFrameDataset, request: BatchGalleryRequest, *, output_root: Path | None=None, samples_per_group: int | None=None) -> GalleryResult`：执行 `save gallery` 对应逻辑。 调用：`GalleryResult`, `self._warning_once`。
- `M L183-L193` `OptionalTrainingVisualizer._default_output_path(self, sample: dict[str, Any], output_identity: OutputIdentity) -> Path` [IO-W]：执行 `default output path` 对应逻辑。

## `src/traning/core/spatial/spatial_inference.py`

职责：单帧空间推理处理器；显式分离 GPU 前向与 CPU 画布融合、候选解码和输出缓存。
工程依赖：`traning.conf`, `traning.core.training_inheritance`, `traning.lib.data`, `traning.lib.models`, `traning.lib.runtime`, `traning.lib.training.spatial_decode`

- `C L59-L92` `SpatialFrameInferenceResult` [CLASS]：封装 `SpatialFrameInferenceResult` 相关数据或行为。
- `M L71-L92` `SpatialFrameInferenceResult.as_summary(self) -> dict[str, Any]`：执行 `as summary` 对应逻辑。 调用：`self.diagnostics.as_dict`。
- `C L96-L133` `SpatialFrameInferenceRunner` [CLASS]：封装 `SpatialFrameInferenceRunner` 相关数据或行为。
- `M L104-L133` `SpatialFrameInferenceRunner.infer_frame(self, sample: Mapping[str, Any], *, max_candidates: int=16, score_threshold: float=0.0, nms_radius_px: float=32.0, slider_threshold: float=0.5, max_slider_paths: int=16, slider_min_cells: int=4, slider_path_points: int=32, patch_limit: int | None=None) -> SpatialFrameInferenceResult`：执行 `infer frame` 对应逻辑。 调用：`_run_spatial_frame_inference_prepared`。
- `F L136-L187` `prepare_spatial_frame_inference(settings: Settings, *, device: torch.device, checkpoint_path: Path | None=None) -> SpatialFrameInferenceRunner`：一次性准备空间模型和运行时，供多帧推理复用。 调用：`CudaRuntimeConfig`, `PatchStream`, `SpatialFrameInferenceRunner`, `_load_spatial_checkpoint`, `build_model_stack`, `configure_torch_runtime`。
- `F L190-L222` `run_spatial_frame_inference(settings: Settings, sample: Mapping[str, Any], *, device: torch.device, checkpoint_path: Path | None=None, max_candidates: int=16, score_threshold: float=0.0, nms_radius_px: float=32.0, slider_threshold: float=0.5, max_slider_paths: int=16, slider_min_cells: int=4, slider_path_points: int=32, patch_limit: int | None=None) -> SpatialFrameInferenceResult`：执行单帧空间推理，并显式分离 GPU 前向与 CPU 画布融合。 调用：`prepare_spatial_frame_inference`, `runner.infer_frame`。
- `F L225-L316` `_run_spatial_frame_inference_prepared(settings: Settings, sample: Mapping[str, Any], *, device: torch.device, memory_budget: RuntimeMemoryBudget, stream: PatchStream, runtime_state: Any, modules: Mapping[str, torch.nn.Module], max_candidates: int=16, score_threshold: float=0.0, nms_radius_px: float=32.0, slider_threshold: float=0.5, max_slider_paths: int=16, slider_min_cells: int=4, slider_path_points: int=32, patch_limit: int | None=None) -> SpatialFrameInferenceResult`：执行 `run spatial frame inference prepared` 对应逻辑。 调用：`SpatialFrameInferenceResult`, `SpatialPredictionCanvas`, `_model_frame`, `autocast_context`, `canvas.to_maps`, `canvas.write_patch`。
- `F L319-L333` `spatial_candidate_to_dict(candidate: SpatialCandidate) -> dict[str, Any]`：执行 `spatial candidate to dict` 对应逻辑。
- `F L336-L350` `slider_path_to_dict(path: SliderPathCandidate) -> dict[str, Any]`：执行 `slider path to dict` 对应逻辑。
- `F L353-L359` `_model_frame(image: torch.Tensor, *, settings: Settings) -> torch.Tensor`：执行 `model frame` 对应逻辑。 调用：`append_color_cues`。
- `F L362-L388` `_load_spatial_checkpoint(modules: Mapping[str, torch.nn.Module], checkpoint_path: Path) -> None`：加载 `spatial checkpoint` 对应的数据或结果。 调用：`load_training_checkpoint`, `restore_module_state`, `restored.append`。

## `src/traning/core/spatial/spatial_trainer.py`

职责：首版单帧空间训练循环；冻结 global、串行 patch 前向和逐 patch backward。
工程依赖：`traning.conf`, `traning.core.dataset_import`, `traning.core.training_inheritance`, `traning.lib.data`, `traning.lib.models`, `traning.lib.reporting`, `traning.lib.runtime`, `traning.lib.training.losses`, `traning.lib.training.spatial_targets`

- `C L51-L89` `SpatialTrainingResult` [CLASS]：封装 `SpatialTrainingResult` 相关数据或行为。
- `M L68-L70` `SpatialTrainingResult.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L72-L89` `SpatialTrainingResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `F L92-L347` `run_spatial_training(settings: Settings, *, device: torch.device, run_dir: Path, split: DataSplit='train', max_steps: int=1, learning_rate: float=0.0001, patch_limit: int | None=None, dataset: Sequence[dict[str, Any]] | None=None, reporter: TrainingReporter=NullReporter(), resume_checkpoint_path: Path | None=None, resume_policy: str='none') -> SpatialTrainingResult` [IO-W]：运行逐帧、逐 patch 梯度累积的空间训练循环。 调用：`CudaRuntimeConfig`, `PatchStream`, `SpatialTrainingResult`, `TrainingPosition`, `_add_spatial_consistency_losses`, `_normalize_frame`。
- `F L350-L381` `_add_spatial_consistency_losses(loss_dict: dict[str, torch.Tensor], *, prediction, target, weights) -> None`：执行 `add spatial consistency losses` 对应逻辑。 调用：`temporal_consistency_loss`。
- `F L384-L389` `_normalize_frame(frame: torch.Tensor) -> torch.Tensor`：规范化 `frame` 对应的数据或结果。
- `F L392-L398` `_write_summary(result: SpatialTrainingResult) -> None` [IO-W]：写入 `summary` 对应的数据或结果。 调用：`result.as_dict`。
- `F L401-L461` `_write_checkpoint(result: SpatialTrainingResult, *, modules: dict[str, torch.nn.Module], settings: Settings, optimizer: torch.optim.Optimizer, scaler, position: TrainingPosition, checkpoint_kind: str) -> None` [IO-W]：写入 `checkpoint` 对应的数据或结果。 调用：`atomic_torch_save_checkpoint`, `build_training_checkpoint`。
- `F L464-L531` `_restore_spatial_training_state(*, modules: dict[str, torch.nn.Module], optimizer: torch.optim.Optimizer, scaler, checkpoint_path: Path | None, policy: str, reporter: TrainingReporter) -> TrainingPosition`：执行 `restore spatial training state` 对应逻辑。 调用：`TrainingPosition`, `TrainingPosition.from_mapping`, `_optimizer_state_to_device`, `load_training_checkpoint`, `reporter.emit_event`, `restore_module_state`。
- `F L534-L541` `_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None`：执行 `optimizer state to device` 对应逻辑。
- `F L544-L589` `_report_spatial_step(reporter: TrainingReporter, *, step: int, target: int, loss: float, sample: dict[str, Any], total_samples: int, generated_patches: int, device: torch.device) -> None`：执行 `report spatial step` 对应逻辑。 调用：`reporter.update_metrics`, `reporter.update_pipeline_stage`。

## `src/traning/core/temporal/dataset.py`

职责：读取当前候选缓存 JSONL，拒绝将历史固定半径标签用于训练，并按 sample_key 生成时序窗口、mask 和动作监督。
工程依赖：`traning.lib.models`, `traning.state.candidate_cache_schema`

- `C L30-L49` `TemporalFeatureSpec` [CLASS]：封装 `TemporalFeatureSpec` 相关数据或行为。
- `M L35-L41` `TemporalFeatureSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L44-L45` `TemporalFeatureSpec.candidate_feature_dim(self) -> int` [PROPERTY]：执行 `candidate feature dim` 对应逻辑。
- `M L48-L49` `TemporalFeatureSpec.frame_feature_dim(self) -> int` [PROPERTY]：执行 `frame feature dim` 对应逻辑。
- `C L53-L66` `TemporalWindow` [CLASS]：封装 `TemporalWindow` 相关数据或行为。
- `C L69-L151` `TemporalCandidateWindowDataset(Dataset[TemporalWindow])` [CLASS]：封装 `TemporalCandidateWindowDataset` 相关数据或行为。
- `M L70-L92` `TemporalCandidateWindowDataset.__init__(self, records: Sequence[Mapping[str, Any]], *, sequence_length: int, feature_spec: TemporalFeatureSpec, stride: int | None=None, drop_short: bool=False) -> None`：初始化实例依赖、配置和运行状态。 调用：`self._build_windows`。
- `M L95-L119` `TemporalCandidateWindowDataset.from_cache_dir(cls, cache_dir: Path, *, sequence_length: int, candidate_slots: int, embedding_dim: int | None=None, stride: int | None=None, drop_short: bool=False) -> TemporalCandidateWindowDataset`：执行 `from cache dir` 对应逻辑。 调用：`TemporalFeatureSpec`, `_infer_embedding_dim`, `load_candidate_cache_records`。
- `M L121-L122` `TemporalCandidateWindowDataset.__len__(self) -> int`：执行 `len` 对应逻辑。
- `M L124-L125` `TemporalCandidateWindowDataset.__getitem__(self, index: int) -> TemporalWindow`：执行 `getitem` 对应逻辑。
- `M L127-L151` `TemporalCandidateWindowDataset._build_windows(self, *, records: Sequence[Mapping[str, Any]], drop_short: bool) -> list[TemporalWindow]`：构建 `windows` 对应的数据或结果。 调用：`_encode_window`, `_group_records_by_sample`, `windows.append`。
- `F L154-L204` `load_candidate_cache_records(cache_dir: Path, *, allow_legacy: bool=False) -> tuple[dict[str, Any], ...]` [IO-R]：读取候选缓存记录。 调用：`records.append`。
- `F L207-L222` `_group_records_by_sample(records: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]`：执行 `group records by sample` 对应逻辑。 调用：`_optional_string`, `current.append`, `groups.append`。
- `F L225-L233` `_record_sort_key(record: Mapping[str, Any]) -> tuple[str, int, float]`：执行 `record sort key` 对应逻辑。 调用：`_optional_float`, `_optional_int`, `_optional_string`。
- `F L236-L342` `_encode_window(records: Sequence[Mapping[str, Any]], *, sequence_length: int, spec: TemporalFeatureSpec) -> TemporalWindow`：执行 `encode window` 对应逻辑。 调用：`TemporalWindow`, `_action_id_from_target`, `_encode_candidate`, `_optional_float`, `_optional_int`, `_optional_string`。
- `F L345-L350` `_action_id_from_target(target: Mapping[str, Any]) -> int`：执行 `action id from target` 对应逻辑。
- `F L353-L363` `_selected_candidate_slot(target: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> int`：执行 `selected candidate slot` 对应逻辑。 调用：`_optional_int`。
- `F L366-L394` `_encode_candidate(candidate: Mapping[str, Any], *, record: Mapping[str, Any], spec: TemporalFeatureSpec) -> torch.Tensor`：执行 `encode candidate` 对应逻辑。 调用：`_candidate_embedding`, `_float_field`, `_optional_float`, `_optional_string`。
- `F L397-L407` `_sorted_candidates(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]`：执行 `sorted candidates` 对应逻辑。 调用：`_float_field`。
- `F L410-L444` `_temporal_slot_candidates(record: Mapping[str, Any], *, temporal_target: object, candidate_slots: int) -> tuple[Mapping[str, Any], ...]`：执行 `temporal slot candidates` 对应逻辑。 调用：`_optional_int`, `_sorted_candidates`, `top.append`。
- `F L447-L455` `_candidate_embedding(candidate: Mapping[str, Any], embedding_dim: int) -> list[float]`：执行 `candidate embedding` 对应逻辑。
- `F L458-L464` `_infer_embedding_dim(records: Sequence[Mapping[str, Any]]) -> int`：执行 `infer embedding dim` 对应逻辑。 调用：`_sorted_candidates`。
- `F L467-L471` `_float_field(candidate: Mapping[str, Any], key: str) -> float`：执行 `float field` 对应逻辑。
- `F L474-L475` `_optional_string(value: Any) -> str | None`：执行 `optional string` 对应逻辑。
- `F L478-L479` `_optional_int(value: Any) -> int | None`：执行 `optional int` 对应逻辑。
- `F L482-L485` `_optional_float(value: Any) -> float | None`：执行 `optional float` 对应逻辑。

## `src/traning/core/temporal/trainer.py`

职责：因果 GRU 时序训练入口；消费候选窗口并写 summary/checkpoint。
工程依赖：`traning.conf`, `traning.core.temporal.dataset`, `traning.core.training_inheritance`, `traning.lib.models`, `traning.lib.reporting`, `traning.lib.runtime`

- `C L55-L106` `TemporalTrainingResult` [CLASS]：封装 `TemporalTrainingResult` 相关数据或行为。
- `M L84-L106` `TemporalTrainingResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `F L109-L347` `run_temporal_training(settings: Settings, *, cache_dir: Path, device: torch.device, run_dir: Path, max_steps: int=1, learning_rate: float=0.0001, sequence_length: int | None=None, candidate_slots: int | None=None, dataset: Sequence[TemporalWindow] | None=None, reporter: TrainingReporter=NullReporter(), resume_checkpoint_path: Path | None=None, resume_policy: str='none') -> TemporalTrainingResult`：执行 `run temporal training` 对应逻辑。 调用：`CausalTemporalModel`, `CudaRuntimeConfig`, `TemporalCandidateWindowDataset.from_cache_dir`, `TemporalTrainingResult`, `TrainingPosition`, `_compute_temporal_loss`。
- `F L350-L369` `_window_to_device(window: TemporalWindow, *, device: torch.device) -> dict[str, torch.Tensor]`：执行 `window to device` 对应逻辑。 调用：`tensor_to_device`。
- `F L372-L391` `_temporal_target_diagnostics(source: Sequence[TemporalWindow]) -> dict[str, int]`：执行 `temporal target diagnostics` 对应逻辑。
- `F L394-L450` `_compute_temporal_loss(outputs, *, action_target: torch.Tensor, selected_candidate_target: torch.Tensor, xy_target: torch.Tensor, time_offset_target: torch.Tensor, frame_mask: torch.Tensor, weights) -> tuple[torch.Tensor, dict[str, torch.Tensor]]`：按有效帧、候选真值和动作帧三种 mask 组合多任务损失。
- `F L453-L458` `_write_summary(result: TemporalTrainingResult) -> None` [IO-W]：写入 `summary` 对应的数据或结果。 调用：`result.as_dict`。
- `F L461-L536` `_write_checkpoint(result: TemporalTrainingResult, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler, hidden_size: int, layers: int, position: TrainingPosition, checkpoint_kind: str) -> None` [IO-W]：写入 `checkpoint` 对应的数据或结果。 调用：`atomic_torch_save_checkpoint`, `build_training_checkpoint`。
- `F L539-L594` `_restore_temporal_training_state(*, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler, checkpoint_path: Path | None, policy: str, reporter: TrainingReporter) -> TrainingPosition`：执行 `restore temporal training state` 对应逻辑。 调用：`TrainingPosition`, `TrainingPosition.from_mapping`, `_optimizer_state_to_device`, `load_training_checkpoint`, `reporter.emit_event`, `restore_module_state`。
- `F L597-L604` `_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None`：执行 `optimizer state to device` 对应逻辑。
- `F L607-L651` `_report_temporal_step(reporter: TrainingReporter, *, step: int, target: int, loss: float, window: TemporalWindow, total_windows: int, device: torch.device) -> None`：执行 `report temporal step` 对应逻辑。 调用：`reporter.update_metrics`, `reporter.update_pipeline_stage`。

## `src/traning/core/training_inheritance/checkpoint.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L20-L51` `TrainingPosition` [CLASS]：描述最近一次已提交 optimizer step 之后可安全恢复的位置。
- `M L32-L33` `TrainingPosition.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `M L36-L51` `TrainingPosition.from_mapping(cls, raw: Mapping[str, Any] | None) -> TrainingPosition`：执行 `from mapping` 对应逻辑。
- `C L55-L80` `CheckpointRestorePlan` [CLASS]：封装 `CheckpointRestorePlan` 相关数据或行为。
- `M L67-L68` `CheckpointRestorePlan.enabled(self) -> bool` [PROPERTY]：执行 `enabled` 对应逻辑。
- `M L70-L80` `CheckpointRestorePlan.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。 调用：`self.position.as_dict`。
- `F L83-L92` `capture_rng_state() -> dict[str, Any]`：执行 `capture rng state` 对应逻辑。
- `F L95-L112` `restore_rng_state(raw: Mapping[str, Any] | None) -> bool`：执行 `restore rng state` 对应逻辑。
- `F L115-L158` `build_training_checkpoint(*, checkpoint_kind: str, run_id: str, trial_id: str, models: Mapping[str, Any], optimizer: torch.optim.Optimizer | None, scheduler: Any | None, scaler: Any | None, position: TrainingPosition, score_state: Mapping[str, Any] | None=None, grade_state: Mapping[str, Any] | None=None, promotion_state: Mapping[str, Any] | None=None, dataset_state: Mapping[str, Any] | None=None, sampler_state: Mapping[str, Any] | None=None, resolved_config: Mapping[str, Any] | None=None, dataset_fingerprint: Mapping[str, Any] | None=None, extra: Mapping[str, Any] | None=None) -> dict[str, Any]`：构建并返回 `training checkpoint` 对应的数据或结果。 调用：`capture_rng_state`, `position.as_dict`。
- `F L161-L178` `atomic_torch_save_checkpoint(payload: Mapping[str, Any], path: Path, *, expected_kind: str) -> None` [IO-W]：写入临时文件并回读校验后，再替换正式检查点。 调用：`torch.load`, `validate_training_checkpoint`。
- `F L181-L185` `load_training_checkpoint(path: Path) -> dict[str, Any]`：加载 `training checkpoint` 对应的数据或结果。 调用：`torch.load`。
- `F L188-L210` `validate_training_checkpoint(payload: Mapping[str, Any], *, expected_kind: str | None=None) -> None`：校验 `training checkpoint` 对应的数据或结果。 调用：`TrainingPosition.from_mapping`。
- `F L213-L232` `restore_module_state(module: torch.nn.Module, state_dict: Mapping[str, Any], *, strict: bool) -> tuple[tuple[str, ...], tuple[str, ...]]`：执行 `restore module state` 对应逻辑。

## `src/traning/core/training_inheritance/manager.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.training_inheritance.checkpoint`, `traning.state.versioning`

- `C L28-L42` `InheritancePackage` [CLASS]：封装 `InheritancePackage` 相关数据或行为。
- `M L35-L42` `InheritancePackage.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L46-L72` `InheritanceLoadResult` [CLASS]：封装 `InheritanceLoadResult` 相关数据或行为。
- `M L59-L72` `InheritanceLoadResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `F L75-L136` `create_inheritance_package(*, output_dir: Path, settings: Settings, resolved_config_path: Path | None=None, latest_checkpoint_path: Path | None=None, best_checkpoint_path: Path | None=None, training_state: dict[str, Any] | None=None, score_state: dict[str, Any] | None=None, promotion_state: dict[str, Any] | None=None, artifacts: dict[str, Any] | None=None, stage_checkpoints: Mapping[str, Path | None] | None=None) -> InheritancePackage` [IO-W]：复制恢复所需文件并在最后发布 manifest 与 latest 指针。 调用：`InheritancePackage`, `_copy_checkpoint`, `_dataset_fingerprint`, `_rng_state`, `_write_json`, `collect_code_version`。
- `F L139-L238` `load_inheritance_package(*, inherit_from: Path | str | None, current_settings: Settings, policy: ResumePolicy) -> InheritanceLoadResult` [IO-R]：加载 `inheritance package` 对应的数据或结果。 调用：`InheritanceLoadResult`, `_compatibility_reasons`, `_stage_checkpoint_paths`, `load_training_checkpoint`, `missing_fields.append`, `reasons.append`。
- `F L241-L254` `resolve_inheritance_path(value: Path | str | None) -> Path | None` [IO-R]：解析并定位 `inheritance path` 对应的数据或结果。
- `F L257-L279` `_compatibility_reasons(manifest: dict[str, Any], settings: Settings) -> list[str]`：列出继承包与当前训练设置之间所有会影响安全恢复的差异。 调用：`_comparable`, `_dataset_fingerprint`, `reasons.append`, `version_manifest`。
- `F L282-L293` `_stage_checkpoint_paths(root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]`：执行 `stage checkpoint paths` 对应逻辑。
- `F L296-L306` `_dataset_fingerprint(settings: Settings) -> dict[str, Any]`：执行 `dataset fingerprint` 对应逻辑。
- `F L309-L315` `_rng_state() -> dict[str, Any]`：执行 `rng state` 对应逻辑。
- `F L318-L323` `_comparable(value: Any) -> Any`：执行 `comparable` 对应逻辑。 调用：`_comparable`。
- `F L326-L331` `_copy_checkpoint(source: Path | None, destination: Path) -> Path | None` [IO-W]：执行 `copy checkpoint` 对应逻辑。
- `F L334-L339` `_write_json(path: Path, value: Any) -> None` [IO-W]：写入 `json` 对应的数据或结果。 调用：`_json_ready`。
- `F L342-L355` `_json_ready(value: Any) -> Any`：执行 `json ready` 对应逻辑。 调用：`_json_ready`, `value.as_dict`。

## `src/traning/core/training_ramp.py`

职责：执行受控渐进训练；严格样本门禁未通过时按配置消费下一个优化 job，并持久化待执行/耗尽状态。
工程依赖：`traning.conf`, `traning.core.dataset_import`, `traning.core.decision`, `traning.core.model_export`, `traning.core.optimization`, `traning.core.training_inheritance`, `traning.state`, `traning.state.versioning`

- `C L54-L76` `RampLevelSpec` [CLASS]：封装 `RampLevelSpec` 相关数据或行为。
- `M L65-L76` `RampLevelSpec.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L80-L100` `RampTarget` [CLASS]：封装 `RampTarget` 相关数据或行为。
- `M L89-L100` `RampTarget.as_level(self) -> RampLevelSpec`：执行 `as level` 对应逻辑。 调用：`RampLevelSpec`。
- `C L104-L122` `RampRunResult` [CLASS]：封装 `RampRunResult` 相关数据或行为。
- `M L113-L122` `RampRunResult.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `C L125-L126` `RampGateError(RuntimeError)` [CLASS]：封装 `RampGateError` 相关数据或行为。
- `C L129-L130` `RampEvaluationGateError(RampGateError)` [CLASS]：表示训练产物有效，但当前参数尚未通过严格评估。
- `C L133-L134` `RampSearchExhausted(RampGateError)` [CLASS]：表示仍有下一轮参数，但自动执行被关闭或试验预算已耗尽。
- `F L137-L198` `run_training_ramp(*, config_path: Path, device: str, output_root: Path=DEFAULT_OUTPUT_ROOT, target_config_path: Path | None=None, run_id: str | None=None, auto_launch_full: bool=False, force_level: bool=False, max_levels: int | None=None, run_full_checks: bool=True, progress_ui: str='auto', progress_language: str='zh-CN', resume_policy: str='none', resume_stage_checkpoints: Mapping[str, Path] | None=None, full_gallery_output_root: Path | None=None, full_gallery_samples_per_group: int | None=None, reporter: TrainingReporter | None=None) -> RampRunResult`：执行 `run training ramp` 对应逻辑。 调用：`_init_layout`, `_run_training_ramp_with_reporter`。
- `F L201-L431` `_run_training_ramp_with_reporter(*, config_path: Path, device: str, target_config_path: Path | None, run_id: str, output_dir: Path, auto_launch_full: bool, force_level: bool, max_levels: int | None, run_full_checks: bool, reporter: TrainingReporter, resume_policy: str, resume_stage_checkpoints: Mapping[str, Path], full_gallery_output_root: Path | None, full_gallery_samples_per_group: int | None) -> RampRunResult`：执行 `run training ramp with reporter` 对应逻辑。 调用：`RampRunResult`, `_launch_full_training`, `_read_json`, `_record_ramp_interrupted`, `_report_full_training_finished`, `_report_full_training_started`。
- `F L434-L450` `ensure_full_target_config(*, source_config: Path, target_config: Path, output_dir: Path) -> tuple[Path, RampTarget]` [IO-W]：确保 `full target config` 对应的数据或结果。 调用：`_absolutize_config`, `_build_default_full_config`, `_read_yaml`, `_target_from_raw`, `_write_yaml`。
- `F L453-L512` `build_ramp_levels(target: RampTarget) -> list[RampLevelSpec]`：生成单调递增且每个维度都不超过最终目标的预算序列。 调用：`RampLevelSpec`, `_clip_level`, `_level_reaches_target`, `as_dict`, `clipped.as_dict`, `levels.append`。
- `F L515-L660` `_run_preflight(*, config_path: Path, device: str, output_dir: Path, run_full_checks: bool, reporter: TrainingReporter) -> dict[str, Any]` [IO-W PROCESS]：执行 `run preflight` 对应逻辑。 调用：`RampGateError`, `_check_output_disk_space`, `_write_json`, `inspect_data_input`, `load_settings`, `reporter.update_metrics`。
- `F L663-L665` `_free_disk_bytes(path: Path) -> int`：执行 `free disk bytes` 对应逻辑。
- `F L668-L671` `_minimum_ramp_output_free_bytes() -> int`：执行 `minimum ramp output free bytes` 对应逻辑。
- `F L674-L706` `_check_output_disk_space(output_dir: Path, *, reporter: TrainingReporter) -> int`：执行 `check output disk space` 对应逻辑。 调用：`RampGateError`, `_free_disk_bytes`, `_minimum_ramp_output_free_bytes`, `reporter.update_pipeline_stage`。
- `F L709-L758` `_report_ramp_started(reporter: TrainingReporter, *, levels: list[RampLevelSpec], target: RampTarget, auto_launch_full: bool) -> None`：执行 `report ramp started` 对应逻辑。 调用：`reporter.emit_event`, `reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L761-L809` `_report_level_started(reporter: TrainingReporter, *, level: RampLevelSpec, index: int, total_levels: int) -> None`：执行 `report level started` 对应逻辑。 调用：`_level_stage_id`, `_level_title`, `level.as_dict`, `reporter.emit_event`, `reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L812-L897` `_report_level_finished(reporter: TrainingReporter, *, level: RampLevelSpec, index: int, total_levels: int, record: Mapping[str, Any], restored: bool=False) -> None`：执行 `report level finished` 对应逻辑。 调用：`_level_stage_id`, `_level_title`, `_record_gallery_path`, `_record_pass_threshold`, `_record_quality_score`, `reporter.emit_event`。
- `F L900-L942` `_report_ramp_finished(reporter: TrainingReporter, *, levels: list[RampLevelSpec], readiness_path: Path, auto_launch_full: bool) -> None`：执行 `report ramp finished` 对应逻辑。 调用：`reporter.emit_event`, `reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L945-L975` `_report_full_training_started(reporter: TrainingReporter, *, level: RampLevelSpec) -> None`：执行 `report full training started` 对应逻辑。 调用：`reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L978-L1008` `_report_full_training_finished(reporter: TrainingReporter, *, record: Mapping[str, Any]) -> None`：执行 `report full training finished` 对应逻辑。 调用：`_summary_quality_score`, `reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L1011-L1073` `_report_ramp_failed(reporter: TrainingReporter, *, error: Exception, active_level: RampLevelSpec | None, active_index: int, completed_levels: int, total_levels: int) -> None`：执行 `report ramp failed` 对应逻辑。 调用：`_level_stage_id`, `_level_title`, `reporter.update_metrics`, `reporter.update_pipeline_stage`。
- `F L1076-L1114` `_record_ramp_interrupted(*, manifest: dict[str, Any], output_dir: Path, target: RampTarget, levels: list[RampLevelSpec], auto_launch_full: bool, reporter: TrainingReporter, active_index: int) -> None`：在用户中断时落盘可恢复状态，而不是把 manifest 留在 running。 调用：`_write_final_readiness`, `_write_json`。
- `F L1117-L1118` `_level_stage_id(level: RampLevelSpec) -> str`：执行 `level stage id` 对应逻辑。
- `F L1121-L1124` `_level_title(level: RampLevelSpec | None) -> str`：执行 `level title` 对应逻辑。
- `F L1127-L1132` `_record_quality_score(record: Mapping[str, Any]) -> float | None`：执行 `record quality score` 对应逻辑。
- `F L1135-L1140` `_record_pass_threshold(record: Mapping[str, Any]) -> float | None`：执行 `record pass threshold` 对应逻辑。
- `F L1143-L1145` `_summary_quality_score(summary: Mapping[str, Any]) -> float | None`：执行 `summary quality score` 对应逻辑。
- `F L1148-L1153` `_record_gallery_path(record: Mapping[str, Any]) -> str | None`：执行 `record gallery path` 对应逻辑。
- `F L1156-L1367` `_run_level(*, level: RampLevelSpec, base_config: Path, level_dir: Path, device: str, reporter: TrainingReporter, resume_policy: str, resume_stage_checkpoints: Mapping[str, Path], gallery_output_root: Path | None, gallery_samples_per_group: int | None) -> dict[str, Any]` [IO-W]：执行 `run level` 对应逻辑。 调用：`RampGateError`, `RampSearchExhausted`, `_level_title`, `_load_next_job`, `_load_pending_search`, `_ramp_parameter_snapshot`。
- `F L1370-L1495` `_run_level_trial(*, level: RampLevelSpec, settings, config_path: Path, level_dir: Path, device: str, reporter: TrainingReporter, resume_policy: str, resume_stage_checkpoints: Mapping[str, Path], gallery_output_root: Path | None, gallery_samples_per_group: int | None, started: float, trial_index: int, trial_job: Mapping[str, Any] | None) -> dict[str, Any]`：执行 `run level trial` 对应逻辑。 调用：`FullTrainingRunConfig`, `ModelArtifactSpec`, `_gate_level`, `_run_job_dry_run`, `_trial_curriculum_stage`, `_trial_rung`。
- `F L1498-L1502` `_trial_next_job_path(level_dir: Path, trial_index: int) -> Path`：执行 `trial next job path` 对应逻辑。
- `F L1505-L1528` `_load_pending_search(level_dir: Path) -> tuple[dict[str, Any], Path] | None`：读取可恢复的搜索边界，避免同一 run_id 重跑初始 trial。 调用：`RampGateError`, `_read_json`。
- `F L1531-L1538` `_load_next_job(path: Path) -> dict[str, Any] | None`：加载 `next job` 对应的数据或结果。 调用：`RampGateError`, `_read_json`, `training_job_from_dict`, `training_job_from_dict.as_dict`。
- `F L1541-L1548` `_trial_curriculum_stage(trial_job: Mapping[str, Any] | None) -> CurriculumStage`：执行 `trial curriculum stage` 对应逻辑。 调用：`CurriculumStage`, `RampGateError`。
- `F L1551-L1559` `_trial_rung(trial_job: Mapping[str, Any] | None) -> int`：执行 `trial rung` 对应逻辑。 调用：`RampGateError`。
- `F L1562-L1637` `_trial_runtime_overrides(*, settings, level: RampLevelSpec, trial_index: int, budget_steps: int, trial_job: Mapping[str, Any] | None, parent_checkpoint_path: Path | None) -> dict[str, Any]`：执行 `trial runtime overrides` 对应逻辑。 调用：`_candidate_cache_default`, `_checkpoint_temporal_step`, `_float_override`, `_mapping`, `_max_candidates_override`, `_optional_limit_override`。
- `F L1640-L1641` `_mapping(value: Any) -> Mapping[str, Any]`：执行 `mapping` 对应逻辑。
- `F L1644-L1646` `_candidate_cache_default(settings: Any, field_name: str, fallback: Any) -> Any`：执行 `candidate cache default` 对应逻辑。
- `F L1649-L1660` `_score_threshold_override(default: float, explicit: Any, delta: Any) -> float`：执行 `score threshold override` 对应逻辑。 调用：`_clamp`。
- `F L1663-L1672` `_max_candidates_override(default: int, explicit: Any, delta: Any) -> int`：执行 `max candidates override` 对应逻辑。
- `F L1675-L1685` `_scaled_positive_int(default: int, *, explicit: Any, multiplier: Any) -> int`：执行 `scaled positive int` 对应逻辑。
- `F L1688-L1689` `_float_override(default: float, explicit: Any) -> float`：执行 `float override` 对应逻辑。
- `F L1692-L1695` `_positive_float_override(default: float, explicit: Any) -> float`：执行 `positive float override` 对应逻辑。
- `F L1698-L1701` `_positive_int_override(default: int, explicit: Any) -> int`：执行 `positive int override` 对应逻辑。
- `F L1704-L1710` `_optional_limit_override(default: int, explicit: Any) -> int | None`：解析 patch/帧上限；0 与训练 CLI 一样表示不限制。
- `F L1713-L1714` `_clamp(value: float, lower: float, upper: float) -> float`：执行 `clamp` 对应逻辑。
- `F L1717-L1726` `_checkpoint_temporal_step(path: Path | None) -> int`：执行 `checkpoint temporal step` 对应逻辑。 调用：`TrainingPosition.from_mapping`, `torch.load`。
- `F L1729-L1850` `_gate_level(*, level: RampLevelSpec, result, elapsed: float, artifact_path: Path, artifact_issues: tuple[str, ...], artifact_smoke: dict[str, Any], dry_run: dict[str, Any], expected_spatial_steps: int | None=None, expected_temporal_steps: int | None=None, runtime: Mapping[str, Any] | None=None) -> dict[str, Any]`：执行 `gate level` 对应逻辑。 调用：`RampEvaluationGateError`, `RampGateError`, `_json_ready`, `_read_json`, `failures.append`, `level.as_dict`。
- `F L1853-L1892` `_ramp_parameter_snapshot(*, level: RampLevelSpec, record: Mapping[str, Any], config_path: Path, device: str, resume_policy: str, resume_stage_checkpoints: Mapping[str, Path]) -> dict[str, Any]`：执行 `ramp parameter snapshot` 对应逻辑。 调用：`level.as_dict`。
- `F L1895-L1937` `_launch_full_training(*, level: RampLevelSpec, config_path: Path, run_dir: Path, device: str, reporter: TrainingReporter, resume_policy: str, resume_stage_checkpoints: Mapping[str, Path], gallery_output_root: Path | None, gallery_samples_per_group: int | None) -> dict[str, Any]`：执行 `launch full training` 对应逻辑。 调用：`FullTrainingRunConfig`, `load_settings`, `result.as_summary`, `run_full_training_pipeline`。
- `F L1940-L2013` `_write_final_readiness(*, output_dir: Path, manifest: dict[str, Any], target: RampTarget, levels: list[RampLevelSpec], auto_launch_full: bool, failure: str | None=None, status: str | None=None, updated_at_utc: str | None=None) -> Path` [IO-W]：写入 `final readiness` 对应的数据或结果。 调用：`_full_command_text`, `_write_json`, `lines.append`。
- `F L2016-L2052` `_run_job_dry_run(*, job_path: Path | None, config_path: Path, level_dir: Path, device: str) -> dict[str, Any]` [IO-W PROCESS]：执行 `run job dry run` 对应逻辑。 调用：`_pythonpath_with_src`, `subprocess.run`。
- `F L2055-L2060` `_pythonpath_with_src() -> str`：执行 `pythonpath with src` 对应逻辑。 调用：`entries.append`。
- `F L2063-L2080` `_write_level_config(base_config: Path, level_dir: Path, level: RampLevelSpec) -> Path`：写入 `level config` 对应的数据或结果。 调用：`_absolutize_config`, `_read_yaml`, `_write_yaml`, `level.as_dict`。
- `F L2083-L2095` `_build_default_full_config(source: dict[str, Any]) -> dict[str, Any]`：构建 `default full config` 对应的数据或结果。 调用：`RampTarget`。
- `F L2098-L2113` `_target_from_raw(raw: dict[str, Any]) -> RampTarget`：执行 `target from raw` 对应逻辑。 调用：`RampTarget`。
- `F L2116-L2130` `_clip_level(level: RampLevelSpec, target: RampTarget) -> RampLevelSpec`：执行 `clip level` 对应逻辑。 调用：`RampLevelSpec`。
- `F L2133-L2141` `_level_reaches_target(level: RampLevelSpec, target: RampTarget) -> bool`：执行 `level reaches target` 对应逻辑。
- `F L2144-L2146` `_init_layout(output_dir: Path) -> None` [IO-W]：执行 `init layout` 对应逻辑。
- `F L2149-L2161` `_full_command_text(config_path: Path, target: RampTarget) -> str`：执行 `full command text` 对应逻辑。
- `F L2164-L2165` `_read_json(path: Path) -> dict[str, Any]` [IO-R]：读取 `json` 对应的数据或结果。
- `F L2168-L2173` `_write_json(path: Path, value: dict[str, Any]) -> None` [IO-W]：写入 `json` 对应的数据或结果。 调用：`_json_ready`。
- `F L2176-L2180` `_read_yaml(path: Path) -> dict[str, Any]` [IO-R]：读取 `yaml` 对应的数据或结果。
- `F L2183-L2188` `_write_yaml(path: Path, value: dict[str, Any]) -> None` [IO-W]：写入 `yaml` 对应的数据或结果。 调用：`_json_ready`。
- `F L2191-L2206` `_absolutize_config(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]`：执行 `absolutize config` 对应逻辑。
- `F L2209-L2218` `_json_ready(value: Any) -> Any`：执行 `json ready` 对应逻辑。 调用：`_json_ready`。

## `src/traning/lib/coordinates.py`

职责：统一解析配置、样本与预处理元数据中的 osu 到训练帧坐标变换。
工程依赖：`package.coordinates`

- `F L20-L69` `transform_from_settings_or_sample(settings: Any | None, sample: Mapping[str, Any] | None=None, *, frame_width: int | None=None, frame_height: int | None=None) -> tuple[OsuVideoCoordinateTransform, CoordinateTransformSpec]`：解析以完整训练帧像素为目标空间的 playfield 变换。 调用：`_sample_transform_spec`, `coordinate_chain_from_settings_or_sample`。
- `F L72-L129` `coordinate_chain_from_settings_or_sample(settings: Any | None, sample: Mapping[str, Any] | None=None, *, frame_width: int | None=None, frame_height: int | None=None) -> CoordinateTransformChain | None`：构建 osu -> 原视频 -> 裁剪 -> 完整训练帧的可追踪变换链。 调用：`_config_crop_rect`, `_metadata_crop_rect`, `_metadata_source_size`, `_preprocessing_metadata`, `_rect_from_object`。
- `F L132-L164` `_sample_transform_spec(sample: Mapping[str, Any] | None) -> CoordinateTransformSpec | None`：从单个样本恢复可持久化规格，包括 affine matrix。 调用：`PlayfieldRect.from_mapping`, `_matrix_from_value`, `_rect_from_mapping`, `_size_from_mapping`。
- `F L167-L173` `_rect_from_object(value: Any) -> PlayfieldRect`：执行 `rect from object` 对应逻辑。
- `F L176-L182` `_rect_from_mapping(value: object) -> PlayfieldRect | None`：执行 `rect from mapping` 对应逻辑。 调用：`PlayfieldRect.from_mapping`。
- `F L185-L191` `_size_from_mapping(value: object) -> ImageSize | None`：执行 `size from mapping` 对应逻辑。 调用：`ImageSize.from_mapping`。
- `F L194-L205` `_matrix_from_value(value: object) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None`：执行 `matrix from value` 对应逻辑。
- `F L208-L210` `_config_crop_rect(config: Any) -> PlayfieldRect | None`：执行 `config crop rect` 对应逻辑。 调用：`_rect_from_object`。
- `F L213-L217` `_preprocessing_metadata(sample: Mapping[str, Any] | None) -> Mapping[str, Any] | None`：执行 `preprocessing metadata` 对应逻辑。
- `F L220-L223` `_metadata_crop_rect(metadata: Mapping[str, Any] | None) -> PlayfieldRect | None`：执行 `metadata crop rect` 对应逻辑。 调用：`_rect_from_mapping`。
- `F L226-L229` `_metadata_source_size(metadata: Mapping[str, Any] | None) -> ImageSize | None`：执行 `metadata source size` 对应逻辑。 调用：`_size_from_mapping`。

## `src/traning/lib/data/annotation.py`

职责：beatmap.json 的 Pydantic 契约和按帧可见 HitObject 筛选。

- `C L12-L32` `HitObjectAnnotation(BaseModel)` [CLASS]：封装 `HitObjectAnnotation` 相关数据或行为。
- `M L28-L32` `HitObjectAnnotation._valid_end(cls, value: int, info: Any) -> int` [VALIDATOR]：执行 `valid end` 对应逻辑。
- `C L35-L39` `DifficultyAnnotation(BaseModel)` [CLASS]：封装 `DifficultyAnnotation` 相关数据或行为。
- `C L42-L56` `SourceAnnotation(BaseModel)` [CLASS]：封装 `SourceAnnotation` 相关数据或行为。
- `M L52-L56` `SourceAnnotation._valid_clip_end(cls, value: int, info: Any) -> int` [VALIDATOR]：执行 `valid clip end` 对应逻辑。
- `C L59-L72` `SegmentAnnotation(BaseModel)` [CLASS]：封装 `SegmentAnnotation` 相关数据或行为。
- `M L71-L72` `SegmentAnnotation.duration_ms(self) -> int` [PROPERTY]：执行 `duration ms` 对应逻辑。
- `F L75-L80` `load_annotation(path: Path) -> SegmentAnnotation` [IO-R]：加载 `annotation` 对应的数据或结果。
- `F L83-L99` `visible_hit_objects(annotation: SegmentAnnotation, timestamp_ms: float, *, visibility_post_ms: float) -> tuple[HitObjectAnnotation, ...]`：返回当前帧应可见的物件，时间均为 segment 内相对毫秒。

## `src/traning/lib/data/collate.py`

职责：组装图像批次并保留可变长度样本元数据。

- `F L10-L24` `collate_frame_samples(samples: list[dict[str, Any]]) -> dict[str, Any]`：执行 `collate frame samples` 对应逻辑。

## `src/traning/lib/data/color_cues.py`

职责：从 RGB 帧派生 osu 色号、白色数字/内纹和目标相关边缘输入 cue。

- `F L25-L30` `color_cue_channel_count(mode: ColorCueMode) -> int`：执行 `color cue channel count` 对应逻辑。
- `F L33-L39` `append_color_cues(frame: torch.Tensor, *, mode: ColorCueMode) -> torch.Tensor`：在归一化 CHW RGB 帧后追加确定性的 osu! 视觉提示通道。 调用：`extract_osu_basic_color_cues`。
- `F L42-L60` `extract_osu_basic_color_cues(frame: torch.Tensor) -> torch.Tensor`：返回 ``3xHxW`` 的配色、白色字形和物件边缘响应。 调用：`_object_edge_response`, `_palette_response`, `_white_glyph_response`。
- `F L63-L82` `_palette_response(rgb: torch.Tensor, *, saturation: torch.Tensor, value: torch.Tensor) -> torch.Tensor`：执行 `palette response` 对应逻辑。
- `F L85-L92` `_white_glyph_response(*, saturation: torch.Tensor, value: torch.Tensor) -> torch.Tensor`：执行 `white glyph response` 对应逻辑。
- `F L95-L120` `_object_edge_response(rgb: torch.Tensor, *, object_prior: torch.Tensor) -> torch.Tensor`：执行 `object edge response` 对应逻辑。

## `src/traning/lib/data/coordinates.py`

职责：Patch local/global 与 image/feature-grid 坐标转换辅助函数。
工程依赖：`traning.lib.data.patch_stream`

- `F L8-L11` `local_to_global(meta: PatchMeta, x: float, y: float) -> tuple[float, float]`：把 patch 局部图像像素平移到完整帧像素。
- `F L14-L17` `global_to_local(meta: PatchMeta, x: float, y: float) -> tuple[float, float]`：把完整帧像素平移到 patch 局部图像像素。
- `F L20-L32` `global_to_patch_indices(metas: tuple[PatchMeta, ...], x: float, y: float) -> tuple[int, ...]`：返回有效图像区包含该完整帧点的所有 patch 索引。
- `F L35-L45` `image_to_feature_grid(x: float, y: float, *, stride: int) -> tuple[float, float]`：按 stride 把图像像素映射到连续特征网格坐标。
- `F L48-L58` `feature_grid_to_image(gx: float, gy: float, *, stride: int) -> tuple[float, float]`：按 stride 把连续特征网格坐标还原为图像像素。

## `src/traning/lib/data/dataset.py`

职责：按片段帧索引解码原分辨率 RGB Tensor、可变长度标签、difficulty 参数及记录级坐标变换。
工程依赖：`traning.lib.data.annotation`, `traning.lib.data.models`, `traning.lib.data.sampling`, `traning.lib.data.video_reader`

- `C L17-L106` `SegmentFrameDataset(Dataset[dict[str, Any]])` [CLASS]：视频帧 Dataset；坐标规格按 ``record.key`` 随样本一起传播。
- `M L20-L49` `SegmentFrameDataset.__init__(self, records: tuple[SegmentRecord, ...], *, sample_fps: float, frame_step: int=1, max_frames_per_segment: int | None=None, visibility_post_ms: float=100.0, normalize_images: bool=True, coordinate_transform: dict[str, Any] | None=None, coordinate_transforms: Mapping[str, Mapping[str, Any]] | None=None)`：初始化实例依赖、配置和运行状态。 调用：`build_frame_references`。
- `M L51-L52` `SegmentFrameDataset.__len__(self) -> int`：执行 `len` 对应逻辑。
- `M L54-L57` `SegmentFrameDataset._video_reader(self) -> VideoReader`：执行 `video reader` 对应逻辑。 调用：`VideoReader`。
- `M L59-L101` `SegmentFrameDataset.__getitem__(self, index: int) -> dict[str, Any]`：执行 `getitem` 对应逻辑。 调用：`self._video_reader`, `self._video_reader.read_frame_at`, `self.coordinate_transforms.get`, `visible_hit_objects`。
- `M L103-L106` `SegmentFrameDataset.__getstate__(self) -> dict[str, Any]`：执行 `getstate` 对应逻辑。

## `src/traning/lib/data/discovery.py`

职责：发现 video.mp4 与 beatmap.json 配对并构建稳定片段记录。
工程依赖：`traning.lib.data.annotation`, `traning.lib.data.models`, `traning.lib.data.preprocessing_metadata`

- `F L12-L78` `discover_segments(dataset_root: Path, *, dimensions: tuple[str, ...]=(), categories: tuple[str, ...]=(), include_items: tuple[str, ...]=(), exclude_items: tuple[str, ...]=(), max_segments: int | None=None) -> DiscoveryResult`：执行 `discover segments` 对应逻辑。 调用：`DatasetIssue`, `DiscoveryResult`, `SegmentRecord`, `issues.append`, `load_annotation`, `load_preprocessing_metadata`。

## `src/traning/lib/data/models.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data.annotation`

- `C L13-L22` `SegmentRecord` [CLASS]：封装 `SegmentRecord` 相关数据或行为。
- `C L26-L28` `DatasetIssue` [CLASS]：封装 `DatasetIssue` 相关数据或行为。
- `C L32-L34` `DiscoveryResult` [CLASS]：封装 `DiscoveryResult` 相关数据或行为。
- `C L38-L41` `FrameReference` [CLASS]：封装 `FrameReference` 相关数据或行为。

## `src/traning/lib/data/patch_stream.py`

职责：基于现有 tiling 窗口生成固定尺寸 CHW patch、padding 和含 padded 尺寸的 PatchMeta 元数据。
工程依赖：`traning.lib.data.tiling`

- `C L15-L48` `PatchMeta` [CLASS]：一个 CHW patch 在完整帧中的像素位置。
- `M L35-L36` `PatchMeta.width(self) -> int` [PROPERTY]：执行 `width` 对应逻辑。
- `M L39-L40` `PatchMeta.height(self) -> int` [PROPERTY]：执行 `height` 对应逻辑。
- `M L43-L44` `PatchMeta.padded_width(self) -> int` [PROPERTY]：执行 `padded width` 对应逻辑。
- `M L47-L48` `PatchMeta.padded_height(self) -> int` [PROPERTY]：执行 `padded height` 对应逻辑。
- `C L51-L179` `PatchStream` [CLASS]：在 CPU 上生成固定尺寸 CHW patch，不耦合任何模型执行。
- `M L54-L75` `PatchStream.__init__(self, *, patch_width: int=512, patch_height: int=512, overlap_x: int=128, overlap_y: int=128, pin_memory: bool=False, padding_value: float=0.0) -> None`：初始化实例依赖、配置和运行状态。
- `M L77-L107` `PatchStream.metas(self, *, frame_width: int, frame_height: int) -> tuple[PatchMeta, ...]`：返回按行优先排列、完整覆盖全帧的确定性 patch 元数据。 调用：`PatchMeta`, `build_patch_windows`, `self._validate_coverage`。
- `M L109-L113` `PatchStream.count(self, frame: torch.Tensor) -> int`：Return the number of patches that ``iter_patches`` would emit。 调用：`self._shape`, `self.metas`。
- `M L115-L145` `PatchStream.iter_patches(self, frame: torch.Tensor) -> Iterator[tuple[torch.Tensor, PatchMeta]]`：从 CHW 图像产生 ``(patch, meta)``。 调用：`self._shape`, `self.metas`。
- `M L147-L152` `PatchStream.to_device(self, patch: torch.Tensor, device: torch.device | str) -> torch.Tensor`：Move a patch to a device using non-blocking transfer when possible。
- `M L155-L161` `PatchStream._shape(frame: torch.Tensor) -> tuple[int, int, int]`：执行 `shape` 对应逻辑。
- `M L164-L179` `PatchStream._validate_coverage(metas: tuple[PatchMeta, ...], *, frame_width: int, frame_height: int) -> None`：校验 `coverage` 对应的数据或结果。

## `src/traning/lib/data/preprocessing_metadata.py`

职责：Python 模块；具体职责见下方符号及调用。

- `F L11-L57` `load_preprocessing_metadata(dataset_root: Path, item_name: str) -> dict[str, Any] | None`：读取最近一次成功视频预处理记录；缺失或损坏时返回 ``None``。 调用：`_status_db_for_dataset_root`。
- `F L60-L68` `_status_db_for_dataset_root(dataset_root: Path) -> Path | None`：执行 `status db for dataset root` 对应逻辑。

## `src/traning/lib/data/sampling.py`

职责：根据片段时长、FPS 和步长建立帧引用表。
工程依赖：`traning.lib.data.models`

- `F L10-L36` `build_frame_references(records: tuple[SegmentRecord, ...], *, sample_fps: float, frame_step: int, max_frames_per_segment: int | None) -> tuple[FrameReference, ...]`：按固定采样频率生成 segment 相对时间引用，不读取视频内容。 调用：`FrameReference`。

## `src/traning/lib/data/synthetic_structures.py`

职责：生成跨 patch 圆环、边界圆、slider、spinner 和噪声合成测试图像。

- `C L11-L18` `SyntheticStructure` [CLASS]：模型与融合冒烟测试使用的小型合成图像及其几何真值。
- `F L21-L26` `_coordinate_grid(width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]`：执行 `coordinate grid` 对应逻辑。
- `F L29-L30` `_image_from_mask(mask: torch.Tensor, *, channels: int=3) -> torch.Tensor`：执行 `image from mask` 对应逻辑。
- `F L33-L51` `make_cross_patch_ring(*, width: int=768, height: int=768, center: tuple[float, float]=(384.0, 384.0), radius: float=210.0, thickness: float=8.0) -> SyntheticStructure`：Create a ring whose circumference crosses four 512px patches。 调用：`SyntheticStructure`, `_coordinate_grid`, `_image_from_mask`。
- `F L54-L70` `make_boundary_circle(*, width: int=768, height: int=512, center: tuple[float, float]=(512.0, 256.0), radius: float=48.0) -> SyntheticStructure`：Create a filled circle centered on a typical patch boundary。 调用：`SyntheticStructure`, `_coordinate_grid`, `_image_from_mask`。
- `F L73-L99` `make_cross_patch_slider(*, width: int=1152, height: int=512, start: tuple[float, float]=(120.0, 256.0), end: tuple[float, float]=(1032.0, 256.0), thickness: float=12.0) -> SyntheticStructure`：Create a long straight slider spanning multiple patch windows。 调用：`SyntheticStructure`, `_coordinate_grid`, `_image_from_mask`。
- `F L102-L116` `make_spinner(*, width: int=768, height: int=768, center: tuple[float, float]=(384.0, 384.0), radius: float=260.0) -> SyntheticStructure`：Create a large spinner-like disk with a bright rim。 调用：`SyntheticStructure`, `_coordinate_grid`。
- `F L119-L130` `make_noise_background(*, width: int=512, height: int=512, seed: int=2026) -> SyntheticStructure`：Create deterministic noise for background robustness smoke tests。 调用：`SyntheticStructure`。

## `src/traning/lib/data/tiling.py`

职责：构建覆盖完整画面的重叠 patch 窗口并返回 Tensor 视图。

- `C L12-L24` `PatchWindow` [CLASS]：封装 `PatchWindow` 相关数据或行为。
- `M L19-L20` `PatchWindow.right(self) -> int` [PROPERTY]：执行 `right` 对应逻辑。
- `M L23-L24` `PatchWindow.bottom(self) -> int` [PROPERTY]：执行 `bottom` 对应逻辑。
- `F L27-L41` `_axis_starts(size: int, patch_size: int, overlap: int) -> tuple[int, ...]`：执行 `axis starts` 对应逻辑。 调用：`starts.append`。
- `F L44-L66` `build_patch_windows(image_width: int, image_height: int, *, patch_width: int, patch_height: int, overlap_x: int, overlap_y: int) -> tuple[PatchWindow, ...]`：构建行优先窗口；窗口坐标属于完整帧，right/bottom 为半开上界。 调用：`PatchWindow`, `_axis_starts`。
- `F L69-L83` `iter_patches(image: Tensor, windows: tuple[PatchWindow, ...]) -> Iterator[tuple[PatchWindow, Tensor]]`：执行 `iter patches` 对应逻辑。

## `src/traning/lib/data/video_reader.py`

职责：带有限打开文件缓存的 OpenCV 视频帧读取器。

- `C L12-L61` `VideoReader` [CLASS]：带 LRU 句柄缓存的随机访问视频读取器。
- `M L15-L19` `VideoReader.__init__(self, max_open_videos: int=4)`：初始化实例依赖、配置和运行状态。
- `M L21-L33` `VideoReader._capture(self, path: Path) -> cv2.VideoCapture`：执行 `capture` 对应逻辑。 调用：`self._captures.pop`, `self._captures.popitem`。
- `M L35-L41` `VideoReader.read_frame(self, path: Path, frame_index: int) -> np.ndarray` [IO-R]：读取 `frame` 对应的数据或结果。 调用：`self._capture`。
- `M L43-L53` `VideoReader.read_frame_at(self, path: Path, timestamp_ms: float) -> np.ndarray` [IO-R]：读取 `frame at` 对应的数据或结果。 调用：`self._capture`。
- `M L55-L58` `VideoReader.close(self) -> None`：执行 `close` 对应逻辑。 调用：`self._captures.clear`, `self._captures.values`。
- `M L60-L61` `VideoReader.__del__(self) -> None`：执行 `del` 对应逻辑。 调用：`self.close`。

## `src/traning/lib/metrics/scoring.py`

职责：实现点与 slider 的空间、时间、1.5x 膨胀路径覆盖和组合评分。

- `C L15-L69` `ScoreSpec` [CLASS]：连续评分分段阈值；空间量均以 circle radius 为单位，时间量为毫秒。
- `M L33-L60` `ScoreSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L63-L64` `ScoreSpec.maximum_coefficient(self) -> float` [PROPERTY]：执行 `maximum coefficient` 对应逻辑。
- `M L67-L69` `ScoreSpec.maximum_raw_score(self) -> float` [PROPERTY]：执行 `maximum raw score` 对应逻辑。
- `C L73-L77` `CombinedScore` [CLASS]：封装 `CombinedScore` 相关数据或行为。
- `C L81-L86` `PointScore` [CLASS]：封装 `PointScore` 相关数据或行为。
- `C L90-L97` `PathScore` [CLASS]：封装 `PathScore` 相关数据或行为。
- `C L101-L105` `SliderScore` [CLASS]：封装 `SliderScore` 相关数据或行为。
- `F L108-L116` `_interpolate(value: float, start: float, end: float, start_score: float, end_score: float) -> float`：执行 `interpolate` 对应逻辑。
- `F L119-L145` `spatial_coefficient(distance_ratio: float, *, spec: ScoreSpec=ScoreSpec()) -> float`：把非负距离半径比映射为连续空间系数。
- `F L148-L192` `temporal_coefficient(time_error_ms: float, *, spec: ScoreSpec=ScoreSpec()) -> float`：按绝对时间误差分段插值为连续时间系数。 调用：`_interpolate`。
- `F L195-L214` `combine_coefficients(spatial: float, temporal: float, *, spec: ScoreSpec=ScoreSpec()) -> CombinedScore`：组合空间与时间系数，并按理论最大值归一化。 调用：`CombinedScore`。
- `F L217-L250` `score_point(reference_xy: Point, predicted_xy: Point, *, circle_radius: float, reference_time_ms: float, predicted_time_ms: float, spec: ScoreSpec=ScoreSpec()) -> PointScore`：在同一坐标空间内评分点位置，并结合毫秒级打击时间。 调用：`PointScore`, `combine_coefficients`, `spatial_coefficient`, `temporal_coefficient`。
- `F L253-L272` `_point_to_segment_distance(point: Point, start: Point, end: Point) -> float`：执行 `point to segment distance` 对应逻辑。
- `F L275-L281` `_minimum_distance(point: Point, path: PathPoints) -> float`：执行 `minimum distance` 对应逻辑。 调用：`_point_to_segment_distance`。
- `F L284-L298` `_densify_path(path: PathPoints, *, maximum_step: float) -> PathPoints`：执行 `densify path` 对应逻辑。
- `F L301-L312` `_directed_path_statistics(source: PathPoints, target: PathPoints, *, distance_limit: float) -> tuple[float, float]`：统计 source 中落入 target 膨胀走廊的中心线采样点。 调用：`_minimum_distance`。
- `F L315-L384` `score_slider_path(reference_path: PathPoints, predicted_path: PathPoints, *, circle_radius: float, spec: ScoreSpec=ScoreSpec()) -> PathScore`：执行 `score slider path` 对应逻辑。 调用：`PathScore`, `_densify_path`, `_directed_path_statistics`。
- `F L387-L435` `score_slider(reference_head_xy: Point | None, predicted_head_xy: Point | None, reference_path: PathPoints, predicted_path: PathPoints, *, circle_radius: float, reference_start_ms: float, predicted_start_ms: float, spec: ScoreSpec=ScoreSpec()) -> SliderScore`：执行 `score slider` 对应逻辑。 调用：`SliderScore`, `combine_coefficients`, `score_point`, `score_slider_path`。

## `src/traning/lib/metrics/sequence.py`

职责：按点击时间模拟目标一次性命中、重叠目标递进、最小点击间隔限制和错误归因。
工程依赖：`traning.lib.metrics.scoring`

- `C L36-L44` `SequenceScoreSpec` [CLASS]：序列级点击频率限制与单物件评分规格。
- `M L42-L44` `SequenceScoreSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L48-L68` `TargetObject` [CLASS]：封装 `TargetObject` 相关数据或行为。
- `M L58-L68` `TargetObject.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L72-L80` `PredictedClick` [CLASS]：封装 `PredictedClick` 相关数据或行为。
- `M L78-L80` `PredictedClick.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L84-L89` `TargetResolution` [CLASS]：封装 `TargetResolution` 相关数据或行为。
- `C L93-L107` `ClickEvaluation` [CLASS]：封装 `ClickEvaluation` 相关数据或行为。
- `M L106-L107` `ClickEvaluation.frequency_limited(self) -> bool` [PROPERTY]：执行 `frequency limited` 对应逻辑。
- `C L111-L126` `SequenceScore` [CLASS]：封装 `SequenceScore` 相关数据或行为。
- `M L117-L118` `SequenceScore.hit_count(self) -> int` [PROPERTY]：执行 `hit count` 对应逻辑。
- `M L121-L122` `SequenceScore.miss_count(self) -> int` [PROPERTY]：执行 `miss count` 对应逻辑。
- `M L125-L126` `SequenceScore.frequency_limited_count(self) -> int` [PROPERTY]：执行 `frequency limited count` 对应逻辑。
- `F L129-L135` `_target_sort_key(target: TargetObject) -> tuple[float, int, str]`：执行 `target sort key` 对应逻辑。
- `F L138-L165` `_score_target(target: TargetObject, click: PredictedClick, *, circle_radius: float, spec: ScoreSpec) -> PointScore | SliderScore`：执行 `score target` 对应逻辑。 调用：`score_point`, `score_slider`。
- `F L168-L169` `_score_value(score: PointScore | SliderScore) -> float`：执行 `score value` 对应逻辑。
- `F L172-L175` `_spatial_passed(score: PointScore | SliderScore, spec: ScoreSpec) -> bool`：执行 `spatial passed` 对应逻辑。
- `F L178-L180` `_temporal_passed(score: PointScore | SliderScore, spec: ScoreSpec) -> bool`：执行 `temporal passed` 对应逻辑。
- `F L183-L186` `_spatial_error(score: PointScore | SliderScore) -> float`：执行 `spatial error` 对应逻辑。
- `F L189-L193` `_temporal_error_ms(target: TargetObject, click: PredictedClick) -> float`：执行 `temporal error ms` 对应逻辑。
- `F L196-L203` `_spatial_excess(score: PointScore | SliderScore, spec: ScoreSpec) -> float`：执行 `spatial excess` 对应逻辑。
- `F L206-L213` `_temporal_excess(score: PointScore | SliderScore, spec: ScoreSpec) -> float`：执行 `temporal excess` 对应逻辑。
- `F L216-L254` `_error_attribution(target: TargetObject, click: PredictedClick, score: PointScore | SliderScore, *, spec: ScoreSpec) -> tuple[ErrorDomain, tuple[ErrorTag, ...], float, float]`：执行 `error attribution` 对应逻辑。 调用：`_spatial_error`, `_spatial_excess`, `_spatial_passed`, `_temporal_error_ms`, `_temporal_excess`, `_temporal_passed`。
- `F L257-L278` `_best_scored_target(targets: tuple[TargetObject, ...], click: PredictedClick, *, circle_radius: float, spec: ScoreSpec) -> tuple[TargetObject, PointScore | SliderScore] | None`：执行 `best scored target` 对应逻辑。 调用：`_score_target`, `_score_value`。
- `F L281-L442` `score_click_sequence(targets: tuple[TargetObject, ...], clicks: tuple[PredictedClick, ...], *, circle_radius: float, spec: SequenceScoreSpec=SequenceScoreSpec()) -> SequenceScore`：按时间稳定排序点击，每个目标最多解析一次，并记录未解析目标。 调用：`ClickEvaluation`, `SequenceScore`, `TargetResolution`, `_best_scored_target`, `_error_attribution`, `_score_target`。

## `src/traning/lib/models/gated_sparse_fusion.py`

职责：纯 PyTorch grid_sample 全局门控注入与稀疏跨区域采样融合。
工程依赖：`traning.lib.data`, `traning.lib.models.local_encoder`

- `C L16-L21` `FusedPatchFeatures` [CLASS]：融合结果；dense/global_context 均与当前 patch 的 BCHW 网格对齐。
- `F L24-L55` `_base_grid(meta: PatchMeta, *, height: int, width: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor`：执行 `base grid` 对应逻辑。
- `F L58-L82` `sample_global_feature(global_feature: torch.Tensor, patch_meta: PatchMeta, local_feature_shape: tuple[int, int]) -> torch.Tensor`：在当前 patch 特征网格对应的全帧位置双线性采样全局特征。 调用：`_base_grid`。
- `C L85-L230` `GatedSparseFusion(nn.Module)` [CLASS]：用稀疏低分辨率全局上下文门控融合局部 patch 特征。
- `M L88-L133` `GatedSparseFusion.__init__(self, *, local_channels: int, global_channels: int, hidden_dim: int=96, heads: int=4, sampling_points: int=4, layers: int=2, enabled: bool=True) -> None`：初始化实例依赖、配置和运行状态。 调用：`super.__init__`。
- `M L135-L173` `GatedSparseFusion.forward(self, *, local_features: LocalFeatures, global_features: torch.Tensor, patch_meta: PatchMeta) -> FusedPatchFeatures`：执行 `forward` 对应逻辑。 调用：`FusedPatchFeatures`, `sample_global_feature`, `self._sparse_context`, `self.context_project`, `self.gate_project`, `self.refinement`。
- `M L175-L230` `GatedSparseFusion._sparse_context(self, *, local: torch.Tensor, global_features: torch.Tensor, patch_meta: PatchMeta) -> torch.Tensor`：执行 `sparse context` 对应逻辑。 调用：`_base_grid`, `self.global_project`, `self.offset_predictor`, `self.weight_predictor`, `self.weight_predictor.view`。

## `src/traning/lib/models/global_encoder.py`

职责：无网络依赖的低分辨率完整画面全局 CNN encoder。

- `C L13-L19` `GlobalFeatures` [CLASS]：BCHW 低分辨率完整帧上下文、多尺度金字塔及 BNC token。
- `F L22-L26` `_group_count(channels: int) -> int`：执行 `group count` 对应逻辑。
- `C L29-L55` `_ConvBlock(nn.Module)` [CLASS]：封装 `ConvBlock` 相关数据或行为。
- `M L30-L52` `_ConvBlock.__init__(self, in_channels: int, out_channels: int, *, stride: int) -> None`：初始化实例依赖、配置和运行状态。 调用：`_group_count`, `super.__init__`。
- `M L54-L55` `_ConvBlock.forward(self, x: torch.Tensor) -> torch.Tensor`：执行 `forward` 对应逻辑。 调用：`self.block`。
- `C L58-L120` `LightweightGlobalEncoder(nn.Module)` [CLASS]：离线编码缩小后的完整帧，以较低显存提供物件全局上下文。
- `M L61-L92` `LightweightGlobalEncoder.__init__(self, *, in_channels: int=3, input_height: int=360, input_width: int=640, feature_channels: int=64, backbone: str='lightweight_cnn', pretrained: bool=False, frozen: bool=False) -> None`：初始化实例依赖、配置和运行状态。 调用：`_ConvBlock`, `self.parameters`, `super.__init__`。
- `M L94-L120` `LightweightGlobalEncoder.forward(self, frame: torch.Tensor) -> GlobalFeatures`：执行 `forward` 对应逻辑。 调用：`GlobalFeatures`, `self.stage16`, `self.stage2`, `self.stage4`, `self.stage8`。

## `src/traning/lib/models/global_structure_head.py`

职责：全局对象性、圆心、圆环、slider、spinner、粗半径和 context token 预测头。

- `C L13-L22` `GlobalStructurePrediction` [CLASS]：全局粗预测；除 coarse_radius 外各空间分支保留 logits。
- `C L25-L70` `GlobalStructureHead(nn.Module)` [CLASS]：从低分辨率全局特征预测完整帧的粗粒度物件结构。
- `M L28-L54` `GlobalStructureHead.__init__(self, in_channels: int, *, hidden_channels: int | None=None, context_dim: int | None=None) -> None`：初始化实例依赖、配置和运行状态。 调用：`super.__init__`。
- `M L56-L70` `GlobalStructureHead.forward(self, features: torch.Tensor) -> GlobalStructurePrediction`：执行 `forward` 对应逻辑。 调用：`GlobalStructurePrediction`, `self.center_heatmap`, `self.coarse_radius`, `self.context_projection`, `self.objectness`, `self.ring_likelihood`。

## `src/traning/lib/models/local_encoder.py`

职责：小显存高分辨率局部 CNN；GroupNorm、depthwise separable residual block 和 stride-8 pyramid。

- `C L14-L23` `LocalFeatures` [CLASS]：高分辨率 patch 特征。
- `F L26-L30` `_group_count(channels: int) -> int`：执行 `group count` 对应逻辑。
- `C L33-L50` `DepthwiseSeparableConv(nn.Module)` [CLASS]：封装 `DepthwiseSeparableConv` 相关数据或行为。
- `M L34-L47` `DepthwiseSeparableConv.__init__(self, in_channels: int, out_channels: int, *, stride: int=1) -> None`：初始化实例依赖、配置和运行状态。 调用：`_group_count`, `super.__init__`。
- `M L49-L50` `DepthwiseSeparableConv.forward(self, x: torch.Tensor) -> torch.Tensor`：执行 `forward` 对应逻辑。 调用：`self.act`, `self.depthwise`, `self.norm`, `self.pointwise`。
- `C L53-L78` `SeparableResidualBlock(nn.Module)` [CLASS]：封装 `SeparableResidualBlock` 相关数据或行为。
- `M L54-L75` `SeparableResidualBlock.__init__(self, in_channels: int, out_channels: int, *, stride: int=1) -> None`：初始化实例依赖、配置和运行状态。 调用：`DepthwiseSeparableConv`, `_group_count`, `super.__init__`。
- `M L77-L78` `SeparableResidualBlock.forward(self, x: torch.Tensor) -> torch.Tensor`：执行 `forward` 对应逻辑。 调用：`self.act`, `self.conv1`, `self.conv2`, `self.skip`。
- `C L81-L143` `SmallLocalEncoder(nn.Module)` [CLASS]：面向串行高分辨率 patch 训练的小通道局部 CNN。
- `M L84-L119` `SmallLocalEncoder.__init__(self, *, in_channels: int=3, stem_channels: int=8, feature_channels: int=48, output_stride: int=8, gradient_checkpointing: bool=False) -> None`：初始化实例依赖、配置和运行状态。 调用：`SeparableResidualBlock`, `_group_count`, `super.__init__`。
- `M L121-L129` `SmallLocalEncoder._maybe_checkpoint(self, module: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor`：执行 `maybe checkpoint` 对应逻辑。
- `M L131-L143` `SmallLocalEncoder.forward(self, patch: torch.Tensor) -> LocalFeatures`：执行 `forward` 对应逻辑。 调用：`LocalFeatures`, `self._maybe_checkpoint`, `self.p2_project`, `self.p4_project`, `self.p8_project`, `self.stem`。

## `src/traning/lib/models/object_heads.py`

职责：空间多任务 dense prediction head 和对象类型表。
工程依赖：`traning.lib.models.outputs`

- `F L24-L28` `_group_count(channels: int) -> int`：执行 `group count` 对应逻辑。
- `C L31-L92` `SpatialPredictionHead(nn.Module)` [CLASS]：为一个融合后的高分辨率 patch 特征图生成多任务稠密预测。
- `M L34-L72` `SpatialPredictionHead.__init__(self, in_channels: int, *, hidden_channels: int | None=None, embedding_dim: int=96, object_type_count: int=len(OBJECT_TYPE_NAMES)) -> None`：初始化实例依赖、配置和运行状态。 调用：`_group_count`, `super.__init__`。
- `M L74-L92` `SpatialPredictionHead.forward(self, features: torch.Tensor) -> SpatialPrediction`：执行 `forward` 对应逻辑。 调用：`F.normalize`, `SpatialPrediction`, `self.trunk`。

## `src/traning/lib/models/outputs.py`

职责：空间预测与因果动作预测 dataclass 契约。

- `C L11-L23` `SpatialPrediction` [CLASS]：patch 特征网格上的 BCHW 稠密空间预测。
- `C L27-L35` `ActionPrediction` [CLASS]：单个因果帧步骤的 BF 动作、候选、坐标与时间预测。

## `src/traning/lib/models/smet.py`

职责：动态 top-k 稀疏线性层和普通/稀疏 Linear 工厂。

- `C L10-L69` `DynamicSparseLinear(nn.Module)` [CLASS]：使用确定性权重绝对值 top-k 动态掩码的线性层。
- `M L13-L40` `DynamicSparseLinear.__init__(self, in_features: int, out_features: int, *, bias: bool=True, sparsity: float=0.5, update_interval: int=16, min_density: float=0.05) -> None`：初始化实例依赖、配置和运行状态。 调用：`self.refresh_mask`, `self.register_buffer`, `self.reset_parameters`, `super.__init__`。
- `M L43-L44` `DynamicSparseLinear.density(self) -> float` [PROPERTY]：执行 `density` 对应逻辑。
- `M L46-L51` `DynamicSparseLinear.reset_parameters(self) -> None`：执行 `reset parameters` 对应逻辑。
- `M L53-L54` `DynamicSparseLinear.refresh_mask(self) -> None`：执行 `refresh mask` 对应逻辑。 调用：`self._mask_from_weight`, `self._mask_from_weight.to`, `self.mask.copy_`。
- `M L56-L59` `DynamicSparseLinear.forward(self, input: torch.Tensor) -> torch.Tensor`：执行 `forward` 对应逻辑。 调用：`self._mask_from_weight`。
- `M L61-L69` `DynamicSparseLinear._mask_from_weight(self) -> torch.Tensor`：执行 `mask from weight` 对应逻辑。 调用：`self.weight.detach`, `self.weight.numel`。
- `F L72-L91` `maybe_sparse_linear(in_features: int, out_features: int, *, enabled: bool, bias: bool=True, sparsity: float=0.5, update_interval: int=16, min_density: float=0.05) -> nn.Module`：执行 `maybe sparse linear` 对应逻辑。 调用：`DynamicSparseLinear`。

## `src/traning/lib/models/stack.py`

职责：从 Settings 统一构建 local/global/structure/fusion/spatial head 模型栈。
工程依赖：`traning.conf`, `traning.lib.data`, `traning.lib.models.gated_sparse_fusion`, `traning.lib.models.global_encoder`, `traning.lib.models.global_structure_head`, `traning.lib.models.local_encoder`, `traning.lib.models.object_heads`

- `F L16-L58` `build_model_stack(settings: Settings) -> dict[str, torch.nn.Module]`：从配置构建共享的局部、全局、融合和空间预测模型栈。 调用：`GatedSparseFusion`, `GlobalStructureHead`, `LightweightGlobalEncoder`, `SmallLocalEncoder`, `SpatialPredictionHead`, `color_cue_channel_count`。

## `src/traning/lib/models/temporal_model.py`

职责：因果 GRU 时序模型；提供 initial_state、step 流式接口和可选 SMET 稀疏 heads。
工程依赖：`traning.lib.models.outputs`, `traning.lib.models.smet`

- `C L12-L115` `CausalTemporalModel(nn.Module)` [CLASS]：支持逐帧流式推理的因果 GRU 动作头。
- `M L15-L47` `CausalTemporalModel.__init__(self, *, input_size: int, hidden_size: int=256, layers: int=2, candidate_slots: int=64, action_classes: int=4, smet_enabled: bool=False, smet_sparsity: float=0.5, smet_update_interval: int=16, smet_min_density: float=0.05) -> None`：初始化实例依赖、配置和运行状态。 调用：`maybe_sparse_linear`, `super.__init__`。
- `M L49-L68` `CausalTemporalModel.initial_state(self, batch_size: int, device: torch.device | str, *, dtype: torch.dtype | None=None) -> torch.Tensor`：创建 ``layers x batch x hidden`` 隐状态，与模型设备/精度一致。 调用：`self.parameters`。
- `M L70-L100` `CausalTemporalModel.step(self, current_features: torch.Tensor, previous_state: torch.Tensor) -> tuple[ActionPrediction, torch.Tensor]`：消费一个 BF 帧特征和上一状态，不读取任何未来帧。 调用：`ActionPrediction`, `next_states.append`, `self.action_head`, `self.candidate_head`, `self.time_head`, `self.xy_head`。
- `M L102-L115` `CausalTemporalModel.forward(self, sequence: torch.Tensor) -> tuple[list[ActionPrediction], torch.Tensor]`：按 ``T x B x F`` 时间顺序重复调用因果 step。 调用：`outputs.append`, `self.initial_state`, `self.step`。

## `src/traning/lib/reporting.py`

职责：Python 模块；具体职责见下方符号及调用。

- `F L6-L10` `should_report_training_step(step: int, target: int) -> bool`：执行 `should report training step` 对应逻辑。

## `src/traning/lib/runtime/memory.py`

职责：统一 CUDA/runtime memory policy；管理 AMP、GradScaler、channels-last、TF32、显存/RAM预算、显存快照和 OOM 建议。

- `C L17-L24` `MemorySnapshot` [CLASS]：PyTorch CUDA allocator 的峰值与当前 allocated/reserved 快照。
- `C L28-L58` `RuntimeMemoryBudget` [CLASS]：预算检查后得到的主存、显存和 CUDA allocator 上限记录。
- `M L44-L58` `RuntimeMemoryBudget.as_dict(self) -> dict[str, float | str | None]`：执行 `as dict` 对应逻辑。
- `C L62-L68` `CudaRuntimeConfig` [CLASS]：CUDA 数值性能开关；CPU 设备会忽略 CUDA 专属项。
- `C L72-L79` `CudaRuntimeState` [CLASS]：封装 `CudaRuntimeState` 相关数据或行为。
- `F L82-L186` `enforce_runtime_memory_budget(*, device: torch.device, max_vram_gib: float, reserve_vram_gib: float, max_ram_gib: float | None, reserve_ram_gib: float, set_cuda_fraction: bool=True) -> RuntimeMemoryBudget`：验证 CPU/CUDA 预算，并为宿主系统保留不可占用的余量。 调用：`RuntimeMemoryBudget`, `_finite`。
- `F L189-L210` `resolve_amp_dtype(device: torch.device, amp_dtype: AmpDType) -> torch.dtype | None`：解析 AMP 精度；非 CUDA 或 float32 返回 ``None`` 表示禁用 autocast。
- `F L214-L223` `autocast_context(device: torch.device, amp_dtype: AmpDType) -> Iterator[None]`：提供统一上下文；AMP 关闭时退化为无操作上下文。 调用：`resolve_amp_dtype`。
- `F L226-L259` `configure_torch_runtime(*, device: torch.device, amp_dtype: AmpDType, runtime: CudaRuntimeConfig=CudaRuntimeConfig()) -> CudaRuntimeState`：应用训练与冒烟测试共用的 CUDA 数值和卷积运行时设置。 调用：`CudaRuntimeState`, `amp_uses_grad_scaler`, `resolve_amp_dtype`。
- `F L262-L267` `amp_uses_grad_scaler(device: torch.device, amp_dtype: AmpDType) -> bool`：仅 float16 CUDA 路径需要 GradScaler；bfloat16 通常无需缩放。 调用：`resolve_amp_dtype`。
- `F L270-L283` `create_grad_scaler(*, device: torch.device, amp_dtype: AmpDType, mode: str='auto') -> torch.amp.GradScaler`：执行 `create grad scaler` 对应逻辑。 调用：`amp_uses_grad_scaler`。
- `F L286-L297` `module_to_device(module: nn.Module, device: torch.device, *, channels_last: bool) -> nn.Module`：搬运模块，并仅在 CUDA 请求时切为 channels-last 内存格式。
- `F L300-L310` `maybe_compile_module(module: nn.Module, *, enabled: bool, mode: str='default') -> nn.Module`：执行 `maybe compile module` 对应逻辑。
- `F L313-L328` `tensor_to_device(tensor: torch.Tensor, device: torch.device, *, channels_last: bool, non_blocking: bool=True) -> torch.Tensor`：搬运张量；只有四维 CUDA 图像张量应用 channels-last。
- `F L331-L348` `collect_memory_snapshot() -> MemorySnapshot`：采集当前默认 CUDA 设备的 PyTorch allocator 统计。 调用：`MemorySnapshot`。
- `F L351-L384` `format_oom_guidance(*, patch_size: tuple[int, int], global_size: tuple[int, int], batch_size: int, amp_dtype: str, config_path: str | None) -> str`：执行 `format oom guidance` 对应逻辑。 调用：`collect_memory_snapshot`。
- `F L387-L388` `_finite(value: float) -> bool`：执行 `finite` 对应逻辑。

## `src/traning/lib/training/feature_canvas.py`

职责：detached CPU feature canvas；按 patch 元数据累计融合特征。
工程依赖：`traning.lib.data`

- `C L15-L85` `FeatureCanvas` [CLASS]：在 CPU 累加已 detach 的重叠 patch 特征。
- `M L24-L30` `FeatureCanvas.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L32-L76` `FeatureCanvas.write_patch(self, features: torch.Tensor, meta: PatchMeta, *, weight: torch.Tensor | None=None) -> None`：按 PatchMeta 的完整帧位置累加一个 CHW 或 1CHW 特征张量。
- `M L78-L81` `FeatureCanvas.to_tensor(self) -> torch.Tensor`：返回 detach 的 CPU 加权平均画布，形状为 CHW。 调用：`self._weights.clamp_min`。
- `M L84-L85` `FeatureCanvas.weights(self) -> torch.Tensor` [PROPERTY]：执行 `weights` 对应逻辑。

## `src/traning/lib/training/losses.py`

职责：空间多任务损失、全局局部一致性、跨 patch embedding 和时序一致性损失。
工程依赖：`traning.lib.models.outputs`

- `C L14-L26` `LossWeights` [CLASS]：封装 `LossWeights` 相关数据或行为。
- `C L30-L41` `SpatialLossTargets` [CLASS]：与 SpatialPrediction 各稠密分支逐网格对齐的监督张量。
- `F L44-L58` `_masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor`：执行 `masked smooth l1` 对应逻辑。
- `F L61-L123` `compute_spatial_loss(prediction: SpatialPrediction, target: SpatialLossTargets, *, weights: LossWeights=LossWeights()) -> dict[str, torch.Tensor]`：计算首版稠密多任务空间损失，并返回各分项及加权 total。 调用：`_masked_smooth_l1`。
- `F L126-L153` `cosine_embedding_consistency_loss(embeddings: torch.Tensor, object_ids: torch.Tensor, *, margin: float=0.4) -> torch.Tensor`：拉近同一物件 embedding，并把不同物件推到 margin 之外。 调用：`F.normalize`, `pieces.append`。
- `F L156-L165` `global_local_consistency_loss(local_logits: torch.Tensor, sampled_global_logits: torch.Tensor) -> torch.Tensor`：让局部稠密预测与同位置采样的全局上下文一致。
- `F L168-L181` `temporal_consistency_loss(current: torch.Tensor, previous: torch.Tensor, *, mask: torch.Tensor | None=None) -> torch.Tensor`：惩罚相邻帧稠密预测的突变；previous 被 detach 作为稳定目标。

## `src/traning/lib/training/spatial_decode.py`

职责：把 patch dense 空间预测融合为全图概率画布，并解码 Top-K 空间候选和首版 slider 连通域路径。
工程依赖：`traning.lib.data`, `traning.lib.models`

- `C L17-L33` `SpatialPredictionMaps` [CLASS]：融合后的完整帧 CPU 特征网格；x/y 候选最终以帧像素表示。
- `C L37-L52` `SpatialCandidate` [CLASS]：由一个局部极大网格单元解码出的完整帧像素候选。
- `C L56-L70` `SliderPathCandidate` [CLASS]：slider 连通分量恢复出的等弧长采样折线及歧义诊断。
- `C L74-L112` `SpatialDecodeDiagnostics` [CLASS]：封装 `SpatialDecodeDiagnostics` 相关数据或行为。
- `M L93-L112` `SpatialDecodeDiagnostics.as_dict(self) -> dict[str, float | int]`：执行 `as dict` 对应逻辑。
- `C L115-L222` `SpatialPredictionCanvas` [CLASS]：在 CPU 上融合多个 patch 已 detach 的稠密空间预测。
- `M L118-L151` `SpatialPredictionCanvas.__init__(self, *, frame_width: int, frame_height: int, stride: int, object_type_count: int=len(OBJECT_TYPE_NAMES), embedding_dim: int, dtype: torch.dtype=torch.float32, feather_edges: bool=True) -> None`：初始化实例依赖、配置和运行状态。
- `M L153-L194` `SpatialPredictionCanvas.write_patch(self, prediction: SpatialPrediction, meta: PatchMeta) -> None`：裁掉 padding 后，按完整帧网格位置加权写入一个 patch 预测。 调用：`_patch_weight`, `_prediction_to_payload`, `_write_region`。
- `M L196-L222` `SpatialPredictionCanvas.to_maps(self) -> SpatialPredictionMaps`：完成重叠区加权平均，并重新单位化向量型输出。 调用：`F.normalize`, `SpatialPredictionMaps`, `self._values.items`, `self._weights.clamp_min`, `self._weights.clone`。
- `F L225-L291` `decode_spatial_candidates(maps: SpatialPredictionMaps, *, max_candidates: int=32, score_threshold: float=0.05, nms_radius_px: float=32.0) -> tuple[SpatialCandidate, ...]`：从完整帧网格做局部极大值筛选、阈值过滤和像素空间 NMS。 调用：`SpatialCandidate`, `_is_suppressed`, `selected.append`。
- `F L294-L343` `diagnose_spatial_candidate_decode(maps: SpatialPredictionMaps, *, selected_count: int, max_candidates: int, score_threshold: float, nms_radius_px: float) -> SpatialDecodeDiagnostics`：执行 `diagnose spatial candidate decode` 对应逻辑。 调用：`SpatialDecodeDiagnostics`。
- `F L346-L385` `decode_slider_paths(maps: SpatialPredictionMaps, *, threshold: float=0.5, min_cells: int=4, max_paths: int=16, sample_points: int=32, continuity_threshold: float=0.75) -> tuple[SliderPathCandidate, ...]`：从融合后的 CPU 画布恢复首版 slider 路径候选。 调用：`_connected_components`, `_decode_slider_component`, `paths.append`。
- `F L388-L411` `_prediction_to_payload(prediction: SpatialPrediction, *, dtype: torch.dtype) -> dict[str, torch.Tensor]`：执行 `prediction to payload` 对应逻辑。
- `F L414-L444` `_write_region(meta: PatchMeta, *, feature_height: int, feature_width: int, frame_height: int, frame_width: int, stride: int) -> tuple[slice, slice, slice, slice] | None`：同时计算 patch 有效特征裁片和完整帧 canvas 目标切片。
- `F L447-L459` `_patch_weight(height: int, width: int, *, dtype: torch.dtype, feather_edges: bool) -> torch.Tensor`：执行 `patch weight` 对应逻辑。 调用：`_hann_axis`。
- `F L462-L465` `_hann_axis(size: int, *, dtype: torch.dtype) -> torch.Tensor`：执行 `hann axis` 对应逻辑。
- `F L468-L481` `_is_suppressed(selected: list[SpatialCandidate], *, x: float, y: float, radius: float) -> bool`：判断是否 `suppressed` 对应的数据或结果。
- `F L484-L508` `_connected_components(mask: torch.Tensor) -> tuple[tuple[tuple[int, int], ...], ...]`：用八邻域 BFS 提取二维布尔 mask 的确定性连通分量。 调用：`_neighbor_cells`, `component.append`, `components.append`, `queue.append`。
- `F L511-L558` `_decode_slider_component(maps: SpatialPredictionMaps, *, component_id: int, component: tuple[tuple[int, int], ...], sample_points: int, continuity_threshold: float) -> SliderPathCandidate`：执行 `decode slider component` 对应逻辑。 调用：`SliderPathCandidate`, `_cell_to_xy`, `_component_bbox`, `_component_degree`, `_orient_slider_cells`, `_sample_polyline`。
- `F L561-L577` `_neighbor_cells(cell: tuple[int, int], *, height: int, width: int) -> tuple[tuple[int, int], ...]`：执行 `neighbor cells` 对应逻辑。 调用：`neighbors.append`。
- `F L580-L598` `_component_neighbors(cell: tuple[int, int], component: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]`：执行 `component neighbors` 对应逻辑。
- `F L601-L605` `_component_degree(cell: tuple[int, int], component: set[tuple[int, int]]) -> int`：执行 `component degree` 对应逻辑。 调用：`_component_neighbors`。
- `F L608-L613` `_select_component_endpoints(component: tuple[tuple[int, int], ...], endpoints: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], tuple[int, int]]`：选择 `component endpoints` 对应的数据或结果。 调用：`_farthest_pair`。
- `F L616-L623` `_farthest_pair(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], tuple[int, int]]`：执行 `farthest pair` 对应逻辑。 调用：`_cell_distance_squared`。
- `F L626-L630` `_cell_distance_squared(first: tuple[int, int], second: tuple[int, int]) -> int`：执行 `cell distance squared` 对应逻辑。
- `F L633-L658` `_shortest_component_path(component: set[tuple[int, int]], *, start: tuple[int, int], end: tuple[int, int]) -> tuple[tuple[int, int], ...]`：执行 `shortest component path` 对应逻辑。 调用：`_component_neighbors`, `path.append`, `queue.append`。
- `F L661-L680` `_orient_slider_cells(cells: tuple[tuple[int, int], ...], maps: SpatialPredictionMaps) -> tuple[tuple[int, int], ...]`：执行 `orient slider cells` 对应逻辑。
- `F L683-L690` `_cell_to_xy(cell: tuple[int, int], maps: SpatialPredictionMaps) -> tuple[float, float]`：执行 `cell to xy` 对应逻辑。
- `F L693-L734` `_sample_polyline(points: tuple[tuple[float, float], ...], *, sample_points: int) -> tuple[tuple[float, float], ...]`：按累计弧长等距重采样折线，使不同 cell 数的候选具有统一长度。 调用：`distances.append`, `sampled.append`。
- `F L737-L747` `_component_bbox(component: tuple[tuple[int, int], ...], maps: SpatialPredictionMaps) -> tuple[float, float, float, float]`：执行 `component bbox` 对应逻辑。
- `F L750-L767` `_slider_ambiguity_reasons(*, endpoint_count: int, branch_points: int, continuity: float, continuity_threshold: float, polyline: tuple[tuple[float, float], ...]) -> tuple[str, ...]`：执行 `slider ambiguity reasons` 对应逻辑。 调用：`reasons.append`。

## `src/traning/lib/training/spatial_targets.py`

职责：用统一轴对齐/仿射坐标契约把单帧 osu 标注按 PatchMeta 光栅化为空间多任务 dense loss target。
工程依赖：`package.coordinates`, `traning.lib.coordinates`, `traning.lib.data`, `traning.lib.models`, `traning.lib.training.losses`

- `F L25-L106` `build_spatial_loss_targets(sample: Mapping[str, Any], patch_meta: PatchMeta, feature_size: Sequence[int], *, settings: Any | None=None, device: torch.device | str | None=None, dtype: torch.dtype=torch.float32) -> SpatialLossTargets`：把一帧样本栅格化为单个 patch 特征网格的稠密监督目标。 调用：`SpatialLossTargets`, `_empty_targets`, `_finite_float`, `_normalize_feature_size`, `_object_kind`, `_paint_circle`。
- `F L109-L115` `_normalize_feature_size(feature_size: Sequence[int]) -> tuple[int, int]`：规范化 `feature size` 对应的数据或结果。
- `F L118-L171` `_empty_targets(*, feature_height: int, feature_width: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]`：执行 `empty targets` 对应逻辑。
- `F L174-L204` `_patch_grid(patch_meta: PatchMeta, *, feature_height: int, feature_width: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor | float]`：执行 `patch grid` 对应逻辑。
- `F L207-L214` `_finite_float(value: Any, default: float) -> float`：执行 `finite float` 对应逻辑。
- `F L217-L225` `_object_kind(item: Mapping[str, Any]) -> str` [IO-W]：执行 `object kind` 对应逻辑。
- `F L228-L235` `_set_type(target: dict[str, torch.Tensor], mask: torch.Tensor, object_type: str) -> None`：执行 `set type` 对应逻辑。
- `F L238-L244` `_set_heatmap_max(tensor: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> None`：执行 `set heatmap max` 对应逻辑。
- `F L247-L255` `_point_to_local(point: tuple[float, float], transform: OsuVideoCoordinateTransform, grid: Mapping[str, torch.Tensor | float]) -> tuple[float, float]`：先从 osu 转到完整帧像素，再减去 patch 原点得到局部坐标。
- `F L258-L275` `_object_points(item: Mapping[str, Any]) -> tuple[tuple[float, float], ...]`：执行 `object points` 对应逻辑。 调用：`_distance`, `_finite_float`, `points.append`。
- `F L278-L279` `_distance(first: tuple[float, float], second: tuple[float, float]) -> float`：执行 `distance` 对应逻辑。
- `F L282-L292` `_distance_to_point(grid: Mapping[str, torch.Tensor | float], *, local_x: float, local_y: float) -> torch.Tensor`：执行 `distance to point` 对应逻辑。
- `F L295-L318` `_paint_center(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], *, local_x: float, local_y: float, radius: float, object_type: str) -> None`：执行 `paint center` 对应逻辑。 调用：`_distance_to_point`, `_set_heatmap_max`, `_set_type`, `_write_offset`。
- `F L321-L348` `_write_offset(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], *, local_x: float, local_y: float) -> None`：写入 `offset` 对应的数据或结果。
- `F L351-L390` `_paint_circle(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], item: Mapping[str, Any], *, transform: OsuVideoCoordinateTransform, hit_radius: float, timestamp_ms: float, preempt_ms: float) -> None`：执行 `paint circle` 对应逻辑。 调用：`_distance_to_point`, `_finite_float`, `_object_points`, `_paint_center`, `_point_to_local`, `_set_heatmap_max`。
- `F L393-L441` `_paint_slider(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], item: Mapping[str, Any], *, transform: OsuVideoCoordinateTransform, hit_radius: float) -> None`：执行 `paint slider` 对应逻辑。 调用：`_finite_float`, `_object_points`, `_paint_center`, `_paint_repeat_points`, `_paint_slider_body`, `_point_to_local`。
- `F L444-L487` `_paint_slider_body(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], points: tuple[tuple[float, float], ...], *, tube_radius: float) -> None`：执行 `paint slider body` 对应逻辑。 调用：`_set_heatmap_max`, `_set_type`, `_unoriented_direction`。
- `F L490-L493` `_unoriented_direction(vx: float, vy: float) -> tuple[float, float]`：执行 `unoriented direction` 对应逻辑。
- `F L496-L513` `_paint_repeat_points(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], points: tuple[tuple[float, float], ...], *, repeats: int, radius: float) -> None`：执行 `paint repeat points` 对应逻辑。 调用：`_distance_to_point`, `_set_type`, `repeat_points.append`。
- `F L516-L544` `_paint_spinner(target: dict[str, torch.Tensor], grid: Mapping[str, torch.Tensor | float], *, transform: OsuVideoCoordinateTransform) -> None`：执行 `paint spinner` 对应逻辑。 调用：`_paint_center`, `_set_heatmap_max`, `_set_type`。

## `src/traning/lib/visualization/display.py`

职责：通过独立 ffplay 子进程把标注图片显示到主机 X11。

- `F L11-L56` `launch_image_window(image_path: Path, *, title: str, ffplay_binary: str='ffplay', display: str | None=None, previous_process: subprocess.Popen[bytes] | None=None) -> subprocess.Popen[bytes]`：启动 ffplay 循环显示单张图片，并确认进程没有立即失败。

## `src/traning/lib/visualization/models.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L20-L29` `VisualizationResult` [CLASS]：单帧保存/显示结果；warning 用于可恢复的可视化失败。
- `M L28-L29` `VisualizationResult.succeeded(self) -> bool` [PROPERTY]：执行 `succeeded` 对应逻辑。
- `C L33-L44` `GalleryResult` [CLASS]：gallery 导出结果及实际保存帧数。
- `M L43-L44` `GalleryResult.succeeded(self) -> bool` [PROPERTY]：执行 `succeeded` 对应逻辑。
- `C L48-L55` `SelectedFrame` [CLASS]：用户物件索引解析到的 Dataset 帧引用。

## `src/traning/lib/visualization/output_identity.py`

职责：为 traning_example 输出分配进程安全的递增次数和 UTC 时间标识。

- `C L17-L24` `OutputIdentity` [CLASS]：封装 `OutputIdentity` 相关数据或行为。
- `M L23-L24` `OutputIdentity.prefix(self) -> str` [PROPERTY]：执行 `prefix` 对应逻辑。
- `C L27-L60` `OutputIdentityReservation` [CLASS]：持锁的两阶段编号预留；只有 commit 后才推进持久化计数器。
- `M L30-L34` `OutputIdentityReservation.__init__(self, *, identity: OutputIdentity, counter_path: Path, lock_file)`：初始化实例依赖、配置和运行状态。
- `M L36-L37` `OutputIdentityReservation.__enter__(self) -> OutputIdentityReservation`：执行 `enter` 对应逻辑。
- `M L39-L45` `OutputIdentityReservation.__exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None) -> None`：执行 `exit` 对应逻辑。 调用：`self.close`。
- `M L47-L50` `OutputIdentityReservation.commit(self) -> None` [IO-W]：执行 `commit` 对应逻辑。 调用：`self._counter_path.write_text`。
- `M L52-L56` `OutputIdentityReservation.close(self) -> None`：执行 `close` 对应逻辑。 调用：`self._lock_file.close`, `self._lock_file.fileno`。
- `M L59-L60` `OutputIdentityReservation.committed(self) -> bool` [PROPERTY]：执行 `committed` 对应逻辑。
- `F L63-L67` `_read_counter(path: Path) -> int` [IO-R]：读取 `counter` 对应的数据或结果。
- `F L70-L76` `_existing_max_sequence(output_root: Path) -> int`：执行 `existing max sequence` 对应逻辑。
- `F L79-L105` `allocate_output_identity(output_root: Path) -> OutputIdentity` [IO-W]：原子分配并立即提交一个新编号。 调用：`OutputIdentity`, `_existing_max_sequence`, `_read_counter`。
- `F L108-L133` `reserve_output_identity_for_commit(output_root: Path) -> OutputIdentityReservation` [IO-W]：持有排他锁并预留编号，由调用方完成输出后显式 commit。 调用：`OutputIdentity`, `OutputIdentityReservation`, `_existing_max_sequence`, `_read_counter`。

## `src/traning/lib/visualization/render.py`

职责：把帧 Tensor、osu 标签和样本携带的共享坐标变换渲染为标注图片。
工程依赖：`package`, `traning.lib.coordinates`

- `F L28-L37` `_image_from_tensor(image: torch.Tensor) -> Image.Image`：执行 `image from tensor` 对应逻辑。
- `F L40-L51` `_annotation_font(image_height: int) -> ImageFont.ImageFont`：优先加载容器保证安装的中文字库，小图则自动缩小字号。
- `F L54-L62` `_point(transform: OsuVideoCoordinateTransform, x: float, y: float) -> tuple[int, int]`：通过公共变换把 osu 坐标投影到完整帧像素并在绘制前取整。
- `F L65-L74` `_draw_cross(draw: ImageDraw.ImageDraw, point: tuple[int, int], color: tuple[int, int, int], size: int=12, width: int=3) -> None`：执行 `draw cross` 对应逻辑。
- `F L77-L84` `_is_target(hit_object: Mapping[str, Any], target_source_index: int | None) -> bool`：判断是否 `target` 对应的数据或结果。
- `F L87-L115` `_draw_circle(draw: ImageDraw.ImageDraw, hit_object: Mapping[str, Any], transform: OsuVideoCoordinateTransform, radius: int, target_source_index: int | None) -> tuple[int, int] | None`：执行 `draw circle` 对应逻辑。 调用：`_draw_cross`, `_is_target`, `_point`。
- `F L118-L170` `_draw_slider(draw: ImageDraw.ImageDraw, hit_object: Mapping[str, Any], transform: OsuVideoCoordinateTransform, radius: int, target_source_index: int | None) -> tuple[int, int] | None`：执行 `draw slider` 对应逻辑。 调用：`_draw_cross`, `_is_target`, `_point`。
- `F L173-L291` `render_annotated_frame(sample: Mapping[str, Any], *, target_source_index: int | None=None, include_all_objects: bool=False, predicted_osu_xy: tuple[float, float] | None=None, predicted_video_xy: tuple[float, float] | None=None, metadata_lines: Sequence[str]=()) -> Image.Image`：渲染单帧标注；优先复用样本随 Dataset 传播的坐标规格。 调用：`_annotation_font`, `_draw_circle`, `_draw_cross`, `_draw_slider`, `_image_from_tensor`, `_is_target`。
- `F L294-L297` `save_annotated_frame(image: Image.Image, output_path: Path) -> Path` [IO-W]：执行 `save annotated frame` 对应逻辑。

## `src/traning/lib/visualization/selection.py`

职责：根据 HitObject 起始时间反推最接近的采样帧。
工程依赖：`traning.lib.data.dataset`, `traning.lib.visualization.models`

- `F L9-L52` `select_click_frame(dataset: SegmentFrameDataset, *, segment_index: int, object_index: int=0) -> SelectedFrame`：选择指定物件开始时间附近、实际存在于 Dataset 的最近抽样帧。 调用：`SelectedFrame`。

## `src/traning/main.py`

职责：Typer CLI；执行环境/数据检查、端到端 run、模型 smoke、空间训练、候选缓存、时序训练、决策和结果导出。
工程依赖：`traning.conf`, `traning.core.dataset_import`, `traning.core.decision`, `traning.core.diagnostics`, `traning.core.full_flow`, `traning.core.optimization`, `traning.core.result_export`, `traning.core.spatial`, `traning.core.temporal`, `traning.core.training_inheritance`, `traning.core.training_ramp`, `traning.lib.data`, `traning.lib.models`, `traning.lib.runtime`, `traning.state`

- `C L83-L84` `CliParameterError(ValueError)` [CLASS]：Raised when a plain business entry receives an invalid CLI-like value。
- `F L87-L99` `_render_report(report) -> None`：执行 `render report` 对应逻辑。
- `F L102-L105` `_format_bool(value: bool | None) -> str`：执行 `format bool` 对应逻辑。
- `F L108-L111` `_format_gib(value: float | None) -> str`：执行 `format gib` 对应逻辑。
- `F L114-L151` `_render_env_report(report) -> None`：执行 `render env report` 对应逻辑。 调用：`_format_bool`, `_format_gib`。
- `F L154-L158` `_run_dir(kind: str, *, root: Path | None=None) -> Path` [IO-W]：执行 `run dir` 对应逻辑。
- `F L161-L172` `_select_device(device: str) -> torch.device`：解析设备并在显式请求 CUDA 但不可用时立即失败。 调用：`CliParameterError`。
- `F L175-L178` `_load_image_tensor(path: Path) -> torch.Tensor` [IO-W]：加载 `image tensor` 对应的数据或结果。
- `F L181-L182` `_build_model_stack(settings) -> dict[str, torch.nn.Module]`：构建 `model stack` 对应的数据或结果。 调用：`build_model_stack`。
- `F L185-L329` `_execute_model_smoke(*, config: Path | None, device: torch.device, backward: bool) -> dict[str, Any]`：执行 `execute model smoke` 对应逻辑。 调用：`CudaRuntimeConfig`, `PatchStream`, `_build_model_stack`, `append_color_cues`, `autocast_context`, `collect_memory_snapshot`。
- `F L332-L339` `_render_dict_table(title: str, values: dict[str, Any]) -> None`：执行 `render dict table` 对应逻辑。
- `F L342-L374` `_render_parameter_group_score(evaluation) -> None`：执行 `render parameter group score` 对应逻辑。
- `F L377-L390` `_compact_slider_path(path: dict[str, Any]) -> dict[str, Any]`：执行 `compact slider path` 对应逻辑。
- `F L393-L397` `_write_summary_txt(output_dir: Path, summary: dict[str, Any]) -> None` [IO-W]：写入 `summary txt` 对应的数据或结果。
- `F L400-L405` `_write_json_report(path: Path, payload: dict[str, Any]) -> None` [IO-W]：写入 `json report` 对应的数据或结果。
- `F L408-L415` `inspect_training_data(*, config: Path | None=None, split: DataSplit='all')`：执行 `inspect training data` 对应逻辑。 调用：`inspect_data_input`, `load_settings`。
- `F L418-L419` `collect_training_environment()`：执行 `collect training environment` 对应逻辑。
- `F L422-L453` `preview_training_sample(*, index: int=0, split: DataSplit='train', config: Path | None=None) -> dict[str, Any]`：执行 `preview training sample` 对应逻辑。 调用：`CliParameterError`, `build_dataset`, `build_patch_windows`, `load_settings`。
- `F L456-L577` `run_training(*, config: Path=DEFAULT_TRAINING_CONFIG, split: DataSplit='train', device: str='auto', spatial_max_steps: int=1, temporal_max_steps: int=1, spatial_learning_rate: float=0.0001, temporal_learning_rate: float=0.0001, patch_limit: int=1, cache_max_frames: int=1, sequence_length: int | None=None, candidate_slots: int | None=None, max_candidates: int | None=None, score_threshold: float | None=None, nms_radius_px: float | None=None, slider_threshold: float | None=None, max_slider_paths: int | None=None, parameter_group_id: str='pg-0001', optimization_stage: CurriculumStage=CurriculumStage.BASIC, optimization_rung: int=0, render_gallery: bool=True, gallery_output_root: Path | None=None, gallery_samples_per_group: int | None=None, progress_ui: str='auto', progress_language: str='zh-CN', inherit_from: Path | str | None=None, resume_policy: str='none', direct_stage_checkpoints: Mapping[str, Path] | None=None)`：执行 `run training` 对应逻辑。 调用：`CliParameterError`, `FullTrainingRunConfig`, `_run_dir`, `_safe_create_inheritance_package`, `_select_device`, `_write_json_report`。
- `F L580-L616` `_safe_create_inheritance_package(*, run_dir: Path, settings, config: Path, result, reporter)`：执行 `safe create inheritance package` 对应逻辑。 调用：`create_inheritance_package`, `reporter.emit_event`, `result.as_summary`, `result.evaluation.as_dict`。
- `F L619-L673` `run_training_job_spec(*, job: Path, config: Path=DEFAULT_TRAINING_CONFIG, device: str='auto', execute: bool=True)` [IO-R]：执行 `run training job spec` 对应逻辑。 调用：`CliParameterError`, `result.as_summary`, `run_training`, `training_job_from_dict`。
- `F L676-L719` `run_training_ramp_job(*, config: Path=DEFAULT_TRAINING_CONFIG, device: str='auto', output_root: Path=Path('artifacts') / 'training_ramp', target_config: Path | None=None, run_id: str | None=None, auto_launch_full: bool=False, force_level: bool=False, max_levels: int | None=None, run_full_checks: bool=True, progress_ui: str='auto', progress_language: str='zh-CN', inherit_from: Path | str | None=None, resume_policy: str='none')`：执行 `run training ramp job` 对应逻辑。 调用：`CliParameterError`, `_select_device`, `load_inheritance_package`, `load_settings`, `run_training_ramp`。
- `F L722-L780` `run_full_flow_job(*, config: Path=DEFAULT_TRAINING_CONFIG, device: str='auto', mode: str='execute', output_root: Path=DEFAULT_FULL_FLOW_ROOT, target_config: Path | None=None, run_id: str | None=None, auto_launch_full: bool=False, force_level: bool=False, max_levels: int | None=None, run_full_checks: bool=True, progress_ui: str='auto', progress_language: str='zh-CN', inherit_from: Path | str | None=None, resume_policy: str='none', resume: bool=False, from_stage: str | None=None, until_stage: str | None=None, force_stages: tuple[str, ...]=(), skip_stages: tuple[str, ...]=())`：执行 `run full flow job` 对应逻辑。 调用：`CliParameterError`, `FullFlowConfig`, `load_full_flow_status`, `run_full_flow`。
- `F L783-L798` `run_model_smoke(*, config: Path=DEFAULT_TRAINING_CONFIG, device: str='cpu', backward: bool=True) -> dict[str, Any]`：执行 `run model smoke` 对应逻辑。 调用：`_execute_model_smoke`, `_run_dir`, `_select_device`, `_write_summary_txt`。
- `F L801-L857` `run_spatial_decode_smoke(*, config: Path=DEFAULT_TRAINING_CONFIG, split: DataSplit='train', index: int=0, device: str='cpu', max_candidates: int=16, score_threshold: float=0.0, nms_radius_px: float=32.0, slider_threshold: float=0.5, max_slider_paths: int=16, patch_limit: int | None=None) -> dict[str, Any]` [IO-W]：执行 `run spatial decode smoke` 对应逻辑。 调用：`CliParameterError`, `_compact_slider_path`, `_run_dir`, `_select_device`, `_write_summary_txt`, `build_dataset`。
- `F L860-L891` `run_candidate_cache_build(*, config: Path=DEFAULT_TRAINING_CONFIG, split: DataSplit='train', device: str='cpu', max_frames: int | None=None, patch_limit: int | None=None, max_candidates: int | None=None, score_threshold: float | None=None, nms_radius_px: float | None=None, slider_threshold: float | None=None, max_slider_paths: int | None=None, output: Path | None=None)`：执行 `run candidate cache build` 对应逻辑。 调用：`_run_dir`, `_select_device`, `generate_candidate_cache`, `load_settings`。
- `F L894-L908` `run_memory_profile(*, config: Path=DEFAULT_TRAINING_CONFIG, device: str='cuda') -> dict[str, Any]`：执行 `run memory profile` 对应逻辑。 调用：`_execute_model_smoke`, `_run_dir`, `_select_device`, `_write_summary_txt`。
- `F L911-L933` `visualize_patch_windows(*, input_image: Path, output: Path | None=None, config: Path=DEFAULT_TRAINING_CONFIG) -> Path` [IO-W]：执行 `visualize patch windows` 对应逻辑。 调用：`PatchStream`, `_run_dir`, `load_settings`, `stream.metas`。
- `F L936-L1017` `visualize_fusion_context(*, input_image: Path, output: Path | None=None, config: Path=DEFAULT_TRAINING_CONFIG, device: str='cpu') -> Path` [IO-W]：执行 `visualize fusion context` 对应逻辑。 调用：`CudaRuntimeConfig`, `PatchStream`, `_build_model_stack`, `_load_image_tensor`, `_run_dir`, `_select_device`。
- `F L1020-L1040` `run_spatial_training_job(*, config: Path=DEFAULT_TRAINING_CONFIG, split: DataSplit='train', device: str='auto', max_steps: int=1, learning_rate: float=0.0001, patch_limit: int | None=None)`：执行 `run spatial training job` 对应逻辑。 调用：`_run_dir`, `_select_device`, `load_settings`, `run_spatial_training`。
- `F L1043-L1057` `spatial_training_oom_guidance(config: Path) -> str`：执行 `spatial training oom guidance` 对应逻辑。 调用：`format_oom_guidance`, `load_settings`。
- `F L1060-L1082` `run_temporal_training_job(*, config: Path=DEFAULT_TRAINING_CONFIG, cache: Path, device: str='auto', max_steps: int=1, learning_rate: float=0.0001, sequence_length: int | None=None, candidate_slots: int | None=None)`：执行 `run temporal training job` 对应逻辑。 调用：`_run_dir`, `_select_device`, `load_settings`, `run_temporal_training`。
- `F L1085-L1102` `run_decision_job(*, config: Path=DEFAULT_TRAINING_CONFIG, cache: Path, checkpoint: Path, output: Path | None=None, device: str='auto')`：执行 `run decision job` 对应逻辑。 调用：`_run_dir`, `_select_device`, `load_settings`, `run_temporal_decision`。
- `F L1105-L1122` `run_oracle_diagnostics_job(*, config: Path=DEFAULT_TRAINING_CONFIG, run_dir: Path, output: Path | None=None, fixed_seed: int=2026, max_fixed_frames: int=128, probe_limit: int=12)`：执行 `run oracle diagnostics job` 对应逻辑。 调用：`load_settings`, `run_oracle_diagnostics`。
- `F L1125-L1139` `run_label_visualization(*, segment_index: int=0, object_index: int=0, output: Path | None=None, show: bool=False, config: Path | None=None)`：执行 `run label visualization` 对应逻辑。 调用：`load_settings`, `visualize_click_label`。
- `F L1142-L1154` `run_gallery_export(*, results: Path, output_root: Path | None=None, samples_per_group: int | None=None, config: Path | None=None)`：执行 `run gallery export` 对应逻辑。 调用：`load_batch_gallery_request`, `load_settings`, `save_annotation_gallery`。
- `F L1157-L1158` `_raise_cli_parameter(error: CliParameterError) -> NoReturn`：执行 `raise cli parameter` 对应逻辑。
- `F L1162-L1169` `data_check(config: Path | None=typer.Option(None, '--config'), split: DataSplit=typer.Option('all', '--split')) -> None` [CLI]：执行 `data check` 对应逻辑。 调用：`_render_report`, `inspect_training_data`。
- `F L1173-L1192` `env_check(strict: bool=typer.Option(False, '--strict/--no-strict', help='Exit non-zero when required runtime dependencies are missing.'), require_cuda: bool=typer.Option(False, '--require-cuda/--no-require-cuda', help='Treat CUDA unavailability as a failure in strict mode.')) -> None` [CLI]：执行 `env check` 对应逻辑。 调用：`_render_env_report`, `collect_training_environment`。
- `F L1196-L1209` `data_preview(index: int=typer.Option(0, '--index', min=0), split: DataSplit=typer.Option('train', '--split'), config: Path | None=typer.Option(None, '--config')) -> None` [CLI]：执行 `data preview` 对应逻辑。 调用：`_raise_cli_parameter`, `preview_training_sample`。
- `F L1213-L1296` `run(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), split: DataSplit=typer.Option('train', '--split'), device: str=typer.Option('auto', '--device', help='cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.'), spatial_max_steps: int=typer.Option(1, '--spatial-max-steps', min=1), temporal_max_steps: int=typer.Option(1, '--temporal-max-steps', min=1), spatial_learning_rate: float=typer.Option(0.0001, '--spatial-lr', min=1e-08), temporal_learning_rate: float=typer.Option(0.0001, '--temporal-lr', min=1e-08), patch_limit: int=typer.Option(1, '--patch-limit', min=0, help='0 means process all patches in each frame.'), cache_max_frames: int=typer.Option(1, '--cache-max-frames', min=0, help='0 means no frame limit for candidate cache generation.'), sequence_length: int | None=typer.Option(None, '--sequence-length', min=1), candidate_slots: int | None=typer.Option(None, '--candidate-slots', min=1), parameter_group_id: str=typer.Option('pg-0001', '--parameter-group-id'), render_gallery: bool=typer.Option(True, '--render-gallery/--no-render-gallery', help='Render the best parameter group gallery after the training round.'), gallery_output_root: Path | None=typer.Option(None, '--gallery-output-root'), gallery_samples_per_group: int | None=typer.Option(None, '--gallery-samples-per-group', min=1), progress_ui: str=typer.Option('auto', '--progress-ui'), progress_language: str=typer.Option('zh-CN', '--progress-language'), inherit_from: str | None=typer.Option(None, '--inherit-from'), resume_policy: str=typer.Option('none', '--resume-policy'), resume: bool=typer.Option(False, '--resume')) -> None` [CLI]：执行该处理器的完整工作流。 调用：`_raise_cli_parameter`, `_render_dict_table`, `_render_parameter_group_score`, `result.as_summary`, `run_training`。
- `F L1300-L1322` `run_job(job: Path=typer.Option(..., '--job', exists=True, file_okay=True, dir_okay=False, readable=True), config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), device: str=typer.Option('auto', '--device'), execute: bool=typer.Option(True, '--execute/--dry-run')) -> None` [CLI]：执行 `run job` 对应逻辑。 调用：`_raise_cli_parameter`, `_render_dict_table`, `run_training_job_spec`。
- `F L1326-L1389` `full_flow(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), device: str=typer.Option('auto', '--device', help='cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.'), mode: str=typer.Option('execute', '--mode', help='execute, plan, dry-run, or status.'), output_root: Path=typer.Option(DEFAULT_FULL_FLOW_ROOT, '--output-root'), target_config: Path | None=typer.Option(None, '--target-config'), run_id: str | None=typer.Option(None, '--run-id'), auto_launch_full: bool=typer.Option(False, '--auto-launch-full/--no-auto-launch-full', help='Launch finite full training after ramp gates pass.'), force_level: bool=typer.Option(False, '--force-level/--resume-passed-levels'), max_levels: int | None=typer.Option(None, '--max-levels', min=1), run_full_checks: bool=typer.Option(True, '--run-full-checks/--skip-full-checks'), progress_ui: str=typer.Option('auto', '--progress-ui'), progress_language: str=typer.Option('zh-CN', '--progress-language'), inherit_from: str | None=typer.Option(None, '--inherit-from'), resume_policy: str=typer.Option('none', '--resume-policy'), resume: bool=typer.Option(False, '--resume'), from_stage: str | None=typer.Option(None, '--from-stage'), until_stage: str | None=typer.Option(None, '--until-stage'), force_stage: list[str] | None=typer.Option(None, '--force-stage'), skip_stage: list[str] | None=typer.Option(None, '--skip-stage')) -> None` [CLI]：执行 `full flow` 对应逻辑。 调用：`CliParameterError`, `_raise_cli_parameter`, `_render_dict_table`, `result.as_dict`, `run_full_flow_job`。
- `F L1393-L1401` `full_flow_status(output_root: Path=typer.Option(DEFAULT_FULL_FLOW_ROOT, '--output-root'), run_id: str | None=typer.Option(None, '--run-id')) -> None` [CLI]：执行 `full flow status` 对应逻辑。 调用：`CliParameterError`, `_raise_cli_parameter`, `_render_dict_table`, `load_full_flow_status`, `result.as_dict`。
- `F L1405-L1473` `ramp_to_full(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), device: str=typer.Option('auto', '--device', help='cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.'), output_root: Path=typer.Option(Path('artifacts') / 'training_ramp', '--output-root'), target_config: Path | None=typer.Option(None, '--target-config'), run_id: str | None=typer.Option(None, '--run-id', help='Resume or extend an existing ramp output run id.'), auto_launch_full: bool=typer.Option(False, '--auto-launch-full/--no-auto-launch-full', help='Launch the finite full training run after all ramp gates pass.'), force_level: bool=typer.Option(False, '--force-level/--resume-passed-levels', help='Re-run levels even when their level_state.json is already passed.'), max_levels: int | None=typer.Option(None, '--max-levels', min=1, help='Limit levels for controlled validation; omit for target ramp.'), run_full_checks: bool=typer.Option(True, '--run-full-checks/--skip-full-checks', help='Run full pytest checks during preflight.'), progress_ui: str=typer.Option('auto', '--progress-ui'), progress_language: str=typer.Option('zh-CN', '--progress-language'), inherit_from: str | None=typer.Option(None, '--inherit-from'), resume_policy: str=typer.Option('none', '--resume-policy'), resume: bool=typer.Option(False, '--resume')) -> None` [CLI]：执行 `ramp to full` 对应逻辑。 调用：`_raise_cli_parameter`, `_render_dict_table`, `result.as_dict`, `run_training_ramp_job`。
- `F L1477-L1498` `model_smoke(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), device: str=typer.Option('cpu', '--device', help='cpu, cuda, or auto. CPU is the default smoke path.'), backward: bool=typer.Option(True, '--backward/--no-backward', help='Run backward and optimizer step in addition to forward.')) -> None` [CLI]：执行 `model smoke` 对应逻辑。 调用：`_raise_cli_parameter`, `_render_dict_table`, `run_model_smoke`。
- `F L1502-L1529` `spatial_decode_smoke(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), split: DataSplit=typer.Option('train', '--split'), index: int=typer.Option(0, '--index', min=0), device: str=typer.Option('cpu', '--device'), max_candidates: int=typer.Option(16, '--max-candidates', min=1), score_threshold: float=typer.Option(0.0, '--score-threshold', min=0.0), nms_radius_px: float=typer.Option(32.0, '--nms-radius-px', min=0.0), slider_threshold: float=typer.Option(0.5, '--slider-threshold', min=0.0, max=1.0), max_slider_paths: int=typer.Option(16, '--max-slider-paths', min=1), patch_limit: int | None=typer.Option(None, '--patch-limit', min=1)) -> None` [CLI]：执行 `spatial decode smoke` 对应逻辑。 调用：`_raise_cli_parameter`, `_render_dict_table`, `run_spatial_decode_smoke`。
- `F L1533-L1572` `build_candidate_cache(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), split: DataSplit=typer.Option('train', '--split'), device: str=typer.Option('cpu', '--device'), max_frames: int | None=typer.Option(None, '--max-frames', min=1), patch_limit: int | None=typer.Option(None, '--patch-limit', min=1), max_candidates: int | None=typer.Option(None, '--max-candidates', min=1), score_threshold: float | None=typer.Option(None, '--score-threshold', min=0.0, max=1.0), nms_radius_px: float | None=typer.Option(None, '--nms-radius-px', min=0.0), slider_threshold: float | None=typer.Option(None, '--slider-threshold', min=0.0, max=1.0), max_slider_paths: int | None=typer.Option(None, '--max-slider-paths', min=1), output: Path | None=typer.Option(None, '--output')) -> None` [CLI]：构建并返回 `candidate cache` 对应的数据或结果。 调用：`_raise_cli_parameter`, `_render_dict_table`, `result.as_dict`, `run_candidate_cache_build`。
- `F L1576-L1593` `memory_profile(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), device: str=typer.Option('cuda', '--device', help='cuda, cpu, or auto. CUDA is the default for memory profiling.')) -> None` [CLI]：执行 `memory profile` 对应逻辑。 调用：`_raise_cli_parameter`, `_render_dict_table`, `_select_device`, `run_memory_profile`。
- `F L1597-L1614` `visualize_patches(input_image: Path=typer.Option(..., '--input', exists=True, file_okay=True, dir_okay=False, readable=True), output: Path | None=typer.Option(None, '--output'), config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config')) -> None` [CLI]：执行 `visualize patches` 对应逻辑。 调用：`visualize_patch_windows`。
- `F L1618-L1640` `visualize_fusion(input_image: Path=typer.Option(..., '--input', exists=True, file_okay=True, dir_okay=False, readable=True), output: Path | None=typer.Option(None, '--output'), config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), device: str=typer.Option('cpu', '--device')) -> None` [CLI]：执行 `visualize fusion` 对应逻辑。 调用：`_raise_cli_parameter`, `visualize_fusion_context`。
- `F L1644-L1671` `train_spatial(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), split: DataSplit=typer.Option('train', '--split'), device: str=typer.Option('auto', '--device', help='cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.'), max_steps: int=typer.Option(1, '--max-steps', min=1), learning_rate: float=typer.Option(0.0001, '--lr', min=1e-08), patch_limit: int | None=typer.Option(None, '--patch-limit', min=1)) -> None` [CLI]：执行 `train spatial` 对应逻辑。 调用：`_raise_cli_parameter`, `_render_dict_table`, `result.as_dict`, `run_spatial_training_job`, `spatial_training_oom_guidance`。
- `F L1675-L1708` `train_temporal(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), cache: Path=typer.Option(..., '--cache', exists=True, file_okay=False, dir_okay=True, readable=True, help='Candidate cache directory containing manifest.json and frames.jsonl.'), device: str=typer.Option('auto', '--device', help='cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.'), max_steps: int=typer.Option(1, '--max-steps', min=1), learning_rate: float=typer.Option(0.0001, '--lr', min=1e-08), sequence_length: int | None=typer.Option(None, '--sequence-length', min=1), candidate_slots: int | None=typer.Option(None, '--candidate-slots', min=1)) -> None` [CLI]：执行 `train temporal` 对应逻辑。 调用：`_raise_cli_parameter`, `_render_dict_table`, `result.as_dict`, `run_temporal_training_job`。
- `F L1712-L1749` `run_decision(config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), cache: Path=typer.Option(..., '--cache', exists=True, file_okay=False, dir_okay=True, readable=True, help='Candidate cache directory containing manifest.json and frames.jsonl.'), checkpoint: Path=typer.Option(..., '--checkpoint', exists=True, file_okay=True, dir_okay=False, readable=True, help='Temporal checkpoint produced by train-temporal.'), output: Path | None=typer.Option(None, '--output'), device: str=typer.Option('auto', '--device', help='cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.')) -> None` [CLI]：执行 `run decision` 对应逻辑。 调用：`_raise_cli_parameter`, `_render_dict_table`, `result.as_dict`, `run_decision_job`。
- `F L1753-L1776` `diagnose_oracle(run_dir: Path=typer.Option(..., '--run-dir', exists=True, file_okay=False, dir_okay=True, readable=True), config: Path=typer.Option(DEFAULT_TRAINING_CONFIG, '--config'), output: Path | None=typer.Option(None, '--output'), fixed_seed: int=typer.Option(2026, '--fixed-seed'), max_fixed_frames: int=typer.Option(128, '--max-fixed-frames', min=1), probe_limit: int=typer.Option(12, '--probe-limit', min=1)) -> None` [CLI]：执行 `diagnose oracle` 对应逻辑。 调用：`_render_dict_table`, `result.as_dict`, `run_oracle_diagnostics_job`。
- `F L1780-L1799` `visualize_label(segment_index: int=typer.Option(0, '--segment-index', min=0), object_index: int=typer.Option(0, '--object-index', min=0), output: Path | None=typer.Option(None, '--output'), show: bool=typer.Option(False, '--show/--no-show'), config: Path | None=typer.Option(None, '--config')) -> None` [CLI]：执行 `visualize label` 对应逻辑。 调用：`run_label_visualization`。
- `F L1803-L1837` `save_gallery(results: Path=typer.Option(..., '--results', exists=True, file_okay=True, dir_okay=False, readable=True), output_root: Path | None=typer.Option(None, '--output-root'), samples_per_group: int | None=typer.Option(None, '--samples-per-group', min=1), config: Path | None=typer.Option(None, '--config')) -> None` [CLI]：执行 `save gallery` 对应逻辑。 调用：`run_gallery_export`。

## `src/traning/state/checkpoint_schema.py`

职责：检查点 lineage 与模型/优化器/scheduler/AMP 恢复契约。
工程依赖：`traning.state.experiment_schema`

- `C L12-L31` `CheckpointMetadata(BaseModel)` [CLASS]：记录检查点位置、训练进度、父节点及可恢复状态范围。
- `M L28-L31` `CheckpointMetadata._nonnegative_integer(cls, value: int) -> int` [VALIDATOR]：执行 `nonnegative integer` 对应逻辑。

## `src/traning/state/experiment_schema.py`

职责：三层参数、规则式搜索、历史搜索枚举、ASHA trial、课程和独立评估契约。

- `C L10-L13` `SearchMethod(StrEnum)` [CLASS]：封装 `SearchMethod` 相关数据或行为。
- `C L16-L22` `TrialStatus(StrEnum)` [CLASS]：封装 `TrialStatus` 相关数据或行为。
- `C L25-L29` `CurriculumStage(StrEnum)` [CLASS]：封装 `CurriculumStage` 相关数据或行为。
- `C L32-L35` `TrialParameters(BaseModel)` [CLASS]：封装 `TrialParameters` 相关数据或行为。
- `C L38-L61` `TrialMetadata(BaseModel)` [CLASS]：记录单个 trial 的参数、预算、进度和课程位置。
- `M L58-L61` `TrialMetadata._nonnegative_integer(cls, value: int) -> int` [VALIDATOR]：执行 `nonnegative integer` 对应逻辑。
- `C L64-L70` `EvaluationRunMetadata(BaseModel)` [CLASS]：封装 `EvaluationRunMetadata` 相关数据或行为。
- `C L73-L83` `ExperimentMetadata(BaseModel)` [CLASS]：封装 `ExperimentMetadata` 相关数据或行为。

## `src/traning/state/gallery_schema.py`

职责：批次 trial 分数、稳定帧引用、错误归因和最佳参数图集输入契约。
工程依赖：`traning.state.experiment_schema`

- `C L26-L74` `FrameEvaluation(BaseModel)` [CLASS]：保存一帧的通过状态、错误归因和显式 osu/video 预测坐标。
- `M L47-L50` `FrameEvaluation._nonnegative_frame_index(cls, value: int) -> int` [VALIDATOR]：执行 `nonnegative frame index` 对应逻辑。
- `M L54-L60` `FrameEvaluation._finite_optional_point(cls, value: tuple[float, float] | None) -> tuple[float, float] | None` [VALIDATOR]：执行 `finite optional point` 对应逻辑。
- `M L64-L67` `FrameEvaluation._finite_optional_metric(cls, value: float | None) -> float | None` [VALIDATOR]：执行 `finite optional metric` 对应逻辑。
- `M L71-L74` `FrameEvaluation._optional_probability(cls, value: float | None) -> float | None` [VALIDATOR]：执行 `optional probability` 对应逻辑。
- `C L77-L90` `TrialGalleryEvaluation(BaseModel)` [CLASS]：封装 `TrialGalleryEvaluation` 相关数据或行为。
- `M L87-L90` `TrialGalleryEvaluation._finite_score(cls, value: float) -> float` [VALIDATOR]：执行 `finite score` 对应逻辑。
- `C L93-L121` `BatchGalleryRequest(BaseModel)` [CLASS]：聚合同一批次可比较的 trial，并提供确定性的最佳 trial 选择。
- `M L103-L109` `BatchGalleryRequest._require_trials(cls, value: tuple[TrialGalleryEvaluation, ...]) -> tuple[TrialGalleryEvaluation, ...]` [VALIDATOR]：执行 `require trials` 对应逻辑。
- `M L112-L117` `BatchGalleryRequest._require_one_score_version(self) -> BatchGalleryRequest` [VALIDATOR]：执行 `require one score version` 对应逻辑。
- `M L120-L121` `BatchGalleryRequest.best_trial(self) -> TrialGalleryEvaluation` [PROPERTY]：执行 `best trial` 对应逻辑。
- `F L124-L129` `load_batch_gallery_request(path: Path) -> BatchGalleryRequest` [IO-R]：加载 `batch gallery request` 对应的数据或结果。

## `src/traning/state/run_state.py`

职责：保存 trial、课程阶段、rung、预算和全局步数的运行状态。
工程依赖：`traning.state.experiment_schema`

- `C L11-L20` `RunState` [CLASS]：进程内训练状态；持久化边界由 checkpoint/experiment 契约负责。

## `src/traning/state/versioning.py`

职责：生成数据、配置与完整坐标方程版本指纹，并检查缓存/继承兼容性。
工程依赖：`package.coordinates`

- `C L23-L29` `CodeVersion` [CLASS]：封装 `CodeVersion` 相关数据或行为。
- `M L28-L29` `CodeVersion.as_dict(self) -> dict[str, Any]`：执行 `as dict` 对应逻辑。
- `F L32-L51` `collect_code_version(repo_root: Path | None=None) -> CodeVersion`：读取 Git 提交和 dirty 状态；Git 不可用时返回显式降级标记。 调用：`CodeVersion`。
- `F L54-L69` `dataset_version(settings: Any) -> str`：摘要会改变训练样本成员关系的数据输入选择字段。
- `F L72-L83` `version_manifest(settings: Any) -> dict[str, Any]`：生成可写入缓存、检查点和图集的统一版本清单。 调用：`_transform_fingerprint`, `collect_code_version`, `collect_code_version.as_dict`, `dataset_version`。
- `F L86-L124` `ensure_compatible_versions(left: Mapping[str, Any], right: Mapping[str, Any], *, override: bool=False) -> tuple[bool, tuple[str, ...]]`：比较两个版本清单，并返回是否允许复用及所有不兼容字段。
- `F L127-L150` `_transform_fingerprint(settings: Any) -> str`：为完整坐标方程和训练帧尺寸生成稳定指纹。

## `src/traning/tests/full_checks/runner.py`

职责：traning 全面检测统一入口；运行 full_checks 下的 pytest。
工程依赖：`package.checks`

- `F L16-L39` `run_full_checks() -> StartupCheckReport`：执行 `run full checks` 对应逻辑。 调用：`_run_pytest`, `_tail`。
- `F L42-L56` `_run_pytest(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]` [PROCESS]：执行 `run pytest` 对应逻辑。 调用：`subprocess.run`。
- `F L59-L60` `_tail(text: str, *, max_lines: int=80) -> str`：执行 `tail` 对应逻辑。

## `src/traning/tests/full_checks/test_candidate_cache.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package.coordinates`, `traning.conf`, `traning.core.decision`, `traning.lib.training`, `traning.lib.training.spatial_decode`

- `C L26-L56` `_GroupedSampleDataset` [CLASS]：封装 `GroupedSampleDataset` 相关数据或行为。
- `M L27-L43` `_GroupedSampleDataset.__init__(self, group_count: int=6, frames_per_group: int=3) -> None`：初始化实例依赖、配置和运行状态。 调用：`references.append`。
- `M L45-L46` `_GroupedSampleDataset.__len__(self) -> int`：执行 `len` 对应逻辑。
- `M L48-L56` `_GroupedSampleDataset.__getitem__(self, index: int) -> dict[str, object]`：执行 `getitem` 对应逻辑。
- `F L59-L80` `_candidate(*, score: float=0.55, object_type: str='slider_head', x: float=16.0, y: float=20.0) -> SpatialCandidate`：执行 `candidate` 对应逻辑。 调用：`SpatialCandidate`。
- `F L83-L97` `_slider_path(*, ambiguous: bool=False) -> SliderPathCandidate`：执行 `slider path` 对应逻辑。 调用：`SliderPathCandidate`。
- `C L100-L336` `CandidateCacheTests(unittest.TestCase)` [CLASS]：封装 `CandidateCacheTests` 相关数据或行为。
- `M L101-L161` `CandidateCacheTests.test_target_matching_uses_actual_circle_radius_after_affine_mapping(self) -> None`：CS3 的真实半径应接纳指定回归样本中距中心 74px 的候选。 调用：`_candidate`, `build_candidate_cache_record`, `self.assertAlmostEqual`, `self.assertEqual`, `self.assertGreater`, `self.assertLess`。
- `M L163-L207` `CandidateCacheTests.test_record_keeps_embedding_and_candidate_ambiguity(self) -> None`：执行 `test record keeps embedding and candidate ambiguity` 对应逻辑。 调用：`_candidate`, `_slider_path`, `build_candidate_cache_record`, `self.assertEqual`, `self.assertIn`。
- `M L209-L251` `CandidateCacheTests.test_generate_candidate_cache_writes_manifest_and_jsonl(self) -> None` [IO-R]：执行 `test generate candidate cache writes manifest and jsonl` 对应逻辑。 调用：`Settings`, `_candidate`, `_slider_path`, `generate_candidate_cache`, `self.assertEqual`。
- `M L253-L303` `CandidateCacheTests.test_candidate_cache_max_frames_samples_across_groups(self) -> None` [IO-R]：执行 `test candidate cache max frames samples across groups` 对应逻辑。 调用：`Settings`, `_GroupedSampleDataset`, `_candidate`, `generate_candidate_cache`, `self.assertEqual`。
- `M L305-L336` `CandidateCacheTests.test_local_consistency_review_resolves_supported_ambiguity(self) -> None`：执行 `test local consistency review resolves supported ambiguity` 对应逻辑。 调用：`Settings`, `_candidate`, `_slider_path`, `build_candidate_cache_record`, `self.assertEqual`, `self.assertFalse`。

## `src/traning/tests/full_checks/test_causal_temporal.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.models`

- `C L12-L154` `CausalTemporalTests(unittest.TestCase)` [CLASS]：封装 `CausalTemporalTests` 相关数据或行为。
- `M L13-L25` `CausalTemporalTests.test_future_frames_do_not_change_past_outputs(self) -> None`：执行 `test future frames do not change past outputs` 对应逻辑。 调用：`CausalTemporalModel`, `self.assertTrue`。
- `M L27-L35` `CausalTemporalTests.test_reset_state_repeats_output(self) -> None`：执行 `test reset state repeats output` 对应逻辑。 调用：`CausalTemporalModel`, `model.initial_state`, `model.step`, `self.assertTrue`。
- `M L37-L41` `CausalTemporalTests.test_batch_size_one_runs(self) -> None`：执行 `test batch size one runs` 对应逻辑。 调用：`CausalTemporalModel`, `model.initial_state`, `model.step`, `self.assertEqual`。
- `M L43-L57` `CausalTemporalTests.test_smet_sparse_heads_run(self) -> None`：执行 `test smet sparse heads run` 对应逻辑。 调用：`CausalTemporalModel`, `self.assertEqual`, `self.assertTrue`。
- `M L59-L85` `CausalTemporalTests.test_smet_sparse_heads_backward_after_dynamic_updates(self) -> None`：执行 `test smet sparse heads backward after dynamic updates` 对应逻辑。 调用：`CausalTemporalModel`, `self.assertIsNotNone`, `self.assertTrue`。
- `M L87-L113` `CausalTemporalTests.test_smet_training_forward_does_not_mutate_mask_buffers(self) -> None`：执行 `test smet training forward does not mutate mask buffers` 对应逻辑。 调用：`CausalTemporalModel`, `self.assertEqual`。
- `M L115-L128` `CausalTemporalTests.test_mutating_future_window_does_not_change_prefix(self) -> None`：执行 `test mutating future window does not change prefix` 对应逻辑。 调用：`CausalTemporalModel`, `self.assertTrue`。
- `M L130-L154` `CausalTemporalTests.test_segmented_execution_matches_continuous_and_batch_isolated(self) -> None`：执行 `test segmented execution matches continuous and batch isolated` 对应逻辑。 调用：`CausalTemporalModel`, `model.initial_state`, `model.step`, `segmented.append`, `self.assertTrue`。

## `src/traning/tests/full_checks/test_cli_adapters.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning`, `traning.core.decision`

- `C L17-L102` `TrainingCliAdapterTests(unittest.TestCase)` [CLASS]：封装 `TrainingCliAdapterTests` 相关数据或行为。
- `M L18-L47` `TrainingCliAdapterTests.test_business_run_training_calls_pipeline_without_typer(self) -> None`：执行 `test business run training calls pipeline without typer` 对应逻辑。 调用：`self.assertEqual`, `self.assertIs`, `self.assertIsInstance`, `self.assertIsNone`, `training_main.run_training`。
- `M L49-L91` `TrainingCliAdapterTests.test_run_cli_passes_arguments_to_business_function(self) -> None`：执行 `test run cli passes arguments to business function` 对应逻辑。 调用：`self.assertEqual`。
- `M L93-L102` `TrainingCliAdapterTests.test_cli_parameter_error_maps_to_typer_exit(self) -> None`：执行 `test cli parameter error maps to typer exit` 对应逻辑。 调用：`self.assertEqual`, `self.assertIn`, `training_main.CliParameterError`。

## `src/traning/tests/full_checks/test_color_cues.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.lib.data`, `traning.lib.models`

- `C L18-L56` `ColorCueTests(unittest.TestCase)` [CLASS]：封装 `ColorCueTests` 相关数据或行为。
- `M L19-L32` `ColorCueTests.test_osu_basic_cues_highlight_colored_target_and_white_number(self) -> None`：执行 `test osu basic cues highlight colored target and white number` 对应逻辑。 调用：`extract_osu_basic_color_cues`, `self.assertEqual`, `self.assertGreater`, `self.assertLess`。
- `M L34-L41` `ColorCueTests.test_append_color_cues_is_configurable(self) -> None`：执行 `test append color cues is configurable` 对应逻辑。 调用：`append_color_cues`, `color_cue_channel_count`, `self.assertEqual`, `self.assertIs`。
- `M L43-L56` `ColorCueTests.test_model_stack_accepts_augmented_input_channels(self) -> None`：执行 `test model stack accepts augmented input channels` 对应逻辑。 调用：`Settings`, `build_model_stack`, `self.assertEqual`。

## `src/traning/tests/full_checks/test_coordinates.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package.coordinates`, `traning.lib.data`

- `C L25-L121` `CoordinateTests(unittest.TestCase)` [CLASS]：封装 `CoordinateTests` 相关数据或行为。
- `M L26-L40` `CoordinateTests.test_local_global_round_trip(self) -> None`：执行 `test local global round trip` 对应逻辑。 调用：`PatchMeta`, `global_to_local`, `local_to_global`, `self.assertEqual`。
- `M L42-L47` `CoordinateTests.test_global_to_patch_indices_returns_all_overlaps(self) -> None`：执行 `test global to patch indices returns all overlaps` 对应逻辑。 调用：`PatchMeta`, `global_to_patch_indices`, `self.assertEqual`。
- `M L49-L52` `CoordinateTests.test_feature_grid_round_trip(self) -> None`：执行 `test feature grid round trip` 对应逻辑。 调用：`feature_grid_to_image`, `image_to_feature_grid`, `self.assertEqual`。
- `M L54-L79` `CoordinateTests.test_beatmap_video_input_round_trip_boundaries_and_center(self) -> None`：执行 `test beatmap video input round trip boundaries and center` 对应逻辑。 调用：`self.assertAlmostEqual`, `self.subTest`。
- `M L81-L105` `CoordinateTests.test_coordinate_chain_random_property_round_trip(self) -> None`：执行 `test coordinate chain random property round trip` 对应逻辑。 调用：`self.assertLess`。
- `M L107-L121` `CoordinateTests.test_screen_mapping_uses_beatmap_as_authority(self) -> None`：执行 `test screen mapping uses beatmap as authority` 对应逻辑。 调用：`self.assertEqual`。

## `src/traning/tests/full_checks/test_cross_patch_ring.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`

- `C L13-L41` `CrossPatchRingTests(unittest.TestCase)` [CLASS]：封装 `CrossPatchRingTests` 相关数据或行为。
- `M L14-L41` `CrossPatchRingTests.test_ring_is_visible_from_multiple_patches_with_global_context(self) -> None`：执行 `test ring is visible from multiple patches with global context` 对应逻辑。 调用：`PatchStream`, `make_cross_patch_ring`, `sample_global_feature`, `self.assertGreaterEqual`, `self.assertTrue`, `stream.metas`。

## `src/traning/tests/full_checks/test_cross_patch_slider.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`

- `C L13-L41` `CrossPatchSliderTests(unittest.TestCase)` [CLASS]：封装 `CrossPatchSliderTests` 相关数据或行为。
- `M L14-L41` `CrossPatchSliderTests.test_slider_spans_multiple_patches_with_shared_global_context(self) -> None`：执行 `test slider spans multiple patches with shared global context` 对应逻辑。 调用：`PatchStream`, `make_cross_patch_slider`, `sample_global_feature`, `self.assertGreater`, `self.assertGreaterEqual`, `stream.metas`。

## `src/traning/tests/full_checks/test_cuda_config.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`

- `C L12-L35` `CudaConfigTests(unittest.TestCase)` [CLASS]：封装 `CudaConfigTests` 相关数据或行为。
- `M L13-L24` `CudaConfigTests.test_memory_defaults_enable_cuda_optimized_runtime(self) -> None`：执行 `test memory defaults enable cuda optimized runtime` 对应逻辑。 调用：`MemoryConfig`, `self.assertEqual`, `self.assertFalse`, `self.assertTrue`。
- `M L26-L35` `CudaConfigTests.test_loader_worker_options_require_workers(self) -> None`：执行 `test loader worker options require workers` 对应逻辑。 调用：`LoaderSettings`, `self.assertEqual`, `self.assertRaises`, `self.assertTrue`。

## `src/traning/tests/full_checks/test_cuda_optimization.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.runtime`

- `C L23-L104` `CudaOptimizationTests(unittest.TestCase)` [CLASS]：封装 `CudaOptimizationTests` 相关数据或行为。
- `M L24-L33` `CudaOptimizationTests.test_cpu_runtime_keeps_cuda_only_options_inactive(self) -> None`：执行 `test cpu runtime keeps cuda only options inactive` 对应逻辑。 调用：`CudaRuntimeConfig`, `configure_torch_runtime`, `self.assertEqual`, `self.assertFalse`。
- `M L35-L42` `CudaOptimizationTests.test_grad_scaler_auto_is_disabled_without_fp16_cuda(self) -> None`：执行 `test grad scaler auto is disabled without fp16 cuda` 对应逻辑。 调用：`amp_uses_grad_scaler`, `create_grad_scaler`, `self.assertFalse`。
- `M L44-L51` `CudaOptimizationTests.test_resolved_amp_dtype_values_can_be_reused(self) -> None`：执行 `test resolved amp dtype values can be reused` 对应逻辑。 调用：`amp_uses_grad_scaler`, `resolve_amp_dtype`, `self.assertEqual`, `self.assertIsNone`, `self.assertTrue`。
- `M L53-L60` `CudaOptimizationTests.test_tensor_to_device_preserves_cpu_contiguous_layout(self) -> None`：执行 `test tensor to device preserves cpu contiguous layout` 对应逻辑。 调用：`self.assertTrue`, `tensor_to_device`。
- `M L62-L73` `CudaOptimizationTests.test_cpu_memory_budget_reports_system_reserve(self) -> None`：执行 `test cpu memory budget reports system reserve` 对应逻辑。 调用：`enforce_runtime_memory_budget`, `self.assertEqual`, `self.assertGreater`, `self.assertIsNone`。
- `M L75-L86` `CudaOptimizationTests.test_cpu_memory_budget_rejects_unavailable_reserve(self) -> None`：执行 `test cpu memory budget rejects unavailable reserve` 对应逻辑。 调用：`enforce_runtime_memory_budget`, `self.assertRaises`。
- `M L88-L104` `CudaOptimizationTests.test_cuda_channels_last_when_available(self) -> None`：执行 `test cuda channels last when available` 对应逻辑。 调用：`module_to_device`, `self.assertTrue`, `self.skipTest`, `tensor_to_device`。

## `src/traning/tests/full_checks/test_data_sampling.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.dataset_import`

- `C L15-L24` `_IdDataset(Dataset[dict[str, object]])` [CLASS]：封装 `IdDataset` 相关数据或行为。
- `M L16-L17` `_IdDataset.__init__(self, size: int=12) -> None`：初始化实例依赖、配置和运行状态。
- `M L19-L20` `_IdDataset.__len__(self) -> int`：执行 `len` 对应逻辑。
- `M L22-L24` `_IdDataset.__getitem__(self, index: int) -> dict[str, object]`：执行 `getitem` 对应逻辑。
- `F L27-L39` `_settings(*, seed: int, shuffle: bool) -> SimpleNamespace`：执行 `settings` 对应逻辑。
- `F L42-L46` `_sample_order(settings: SimpleNamespace) -> tuple[int, ...]`：执行 `sample order` 对应逻辑。 调用：`_IdDataset`, `loader_module.build_dataloader`。
- `C L49-L64` `DataSamplingTests(unittest.TestCase)` [CLASS]：封装 `DataSamplingTests` 相关数据或行为。
- `M L50-L57` `DataSamplingTests.test_training_shuffle_is_seeded_and_not_sequential(self) -> None`：执行 `test training shuffle is seeded and not sequential` 对应逻辑。 调用：`_sample_order`, `_settings`, `self.assertEqual`, `self.assertNotEqual`。
- `M L59-L64` `DataSamplingTests.test_evaluation_order_is_deterministic_when_shuffle_is_disabled(self) -> None`：执行 `test evaluation order is deterministic when shuffle is disabled` 对应逻辑。 调用：`_sample_order`, `_settings`, `self.assertEqual`。

## `src/traning/tests/full_checks/test_dataset_split_manifest.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package.dataset_split`, `traning.conf`, `traning.core.dataset_import.preflight`, `traning.lib.data.models`

- `F L16-L19` `_segment(root: Path, item_name: str, segment_id: str) -> None` [IO-W]：执行 `segment` 对应逻辑。
- `C L22-L60` `DatasetSplitManifestTests(unittest.TestCase)` [CLASS]：封装 `DatasetSplitManifestTests` 相关数据或行为。
- `M L23-L60` `DatasetSplitManifestTests.test_discovery_uses_split_manifest_when_present(self) -> None`：执行 `test discovery uses split manifest when present` 对应逻辑。 调用：`DiscoveryResult`, `Settings`, `_segment`, `discover_data_input`, `self.assertEqual`。

## `src/traning/tests/full_checks/test_decision_output_scoring.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package.coordinates`, `traning.core.optimization`

- `C L17-L351` `DecisionOutputScoringTests(unittest.TestCase)` [CLASS]：封装 `DecisionOutputScoringTests` 相关数据或行为。
- `M L18-L74` `DecisionOutputScoringTests.test_scoring_rejects_missing_or_orphan_decision_frames(self) -> None` [IO-W]：执行 `test scoring rejects missing or orphan decision frames` 对应逻辑。 调用：`score_decision_outputs`, `self.assertRaisesRegex`。
- `M L76-L137` `DecisionOutputScoringTests.test_scoring_uses_circle_radius_persisted_for_each_sample(self) -> None` [IO-W]：执行 `test scoring uses circle radius persisted for each sample` 对应逻辑。 调用：`score_decision_outputs`, `self.assertEqual`, `transform.spec.as_dict`。
- `M L139-L215` `DecisionOutputScoringTests.test_scores_parameter_group_from_cache_and_decisions(self) -> None` [IO-W]：执行 `test scores parameter group from cache and decisions` 对应逻辑。 调用：`build_batch_gallery_request`, `result.as_summary`, `score_decision_outputs`, `self.assertEqual`, `self.assertGreater`。
- `M L217-L277` `DecisionOutputScoringTests.test_scores_time_offset_as_frame_minus_action_boundary(self) -> None` [IO-W]：执行 `test scores time offset as frame minus action boundary` 对应逻辑。 调用：`score_decision_outputs`, `self.assertEqual`, `self.assertGreater`。
- `M L279-L351` `DecisionOutputScoringTests.test_normalized_model_output_round_trips_through_frame_and_affine_space(self) -> None` [IO-W]：验证模型归一化坐标先还原训练帧像素，再经逆仿射变换回 osu。 调用：`score_decision_outputs`, `self.assertEqual`, `transform.spec.as_dict`。

## `src/traning/tests/full_checks/test_discovery.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`

- `F L13-L40` `_write_segment(root: Path, item_name: str, segment_id: str) -> None` [IO-W]：写入 `segment` 对应的数据或结果。
- `C L43-L64` `DiscoverySplitTests(unittest.TestCase)` [CLASS]：封装 `DiscoverySplitTests` 相关数据或行为。
- `M L44-L53` `DiscoverySplitTests.test_include_items_filters_records_before_loading(self) -> None`：执行 `test include items filters records before loading` 对应逻辑。 调用：`_write_segment`, `discover_segments`, `self.assertEqual`。
- `M L55-L64` `DiscoverySplitTests.test_exclude_items_removes_records(self) -> None`：执行 `test exclude items removes records` 对应逻辑。 调用：`_write_segment`, `discover_segments`, `self.assertEqual`。

## `src/traning/tests/full_checks/test_env_check.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L13-L26` `EnvironmentCheckTests(unittest.TestCase)` [CLASS]：封装 `EnvironmentCheckTests` 相关数据或行为。
- `M L14-L19` `EnvironmentCheckTests.test_collect_environment_report_is_non_destructive(self) -> None`：执行 `test collect environment report is non destructive` 对应逻辑。 调用：`self.assertIsNotNone`, `self.assertTrue`。
- `M L21-L26` `EnvironmentCheckTests.test_required_package_specs_are_reported(self) -> None`：执行 `test required package specs are reported` 对应逻辑。 调用：`self.assertTrue`。

## `src/traning/tests/full_checks/test_full_flow.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.full_flow`, `traning.core.full_flow.orchestrator`, `traning.core.full_flow.result`, `traning.core.training_ramp`

- `C L30-L349` `FullFlowTests(unittest.TestCase)` [CLASS]：封装 `FullFlowTests` 相关数据或行为。
- `M L31-L36` `FullFlowTests.test_stage_ids_are_unique_and_ordered(self) -> None`：执行 `test stage ids are unique and ordered` 对应逻辑。 调用：`self.assertEqual`, `self.assertTrue`, `stage_ids`。
- `M L38-L62` `FullFlowTests.test_plan_mode_writes_manifest_state_and_reports(self) -> None`：执行 `test plan mode writes manifest state and reports` 对应逻辑。 调用：`FullFlowConfig`, `load_full_flow_status`, `run_full_flow`, `self.assertEqual`, `self.assertTrue`。
- `M L64-L76` `FullFlowTests.test_critical_stages_cannot_be_skipped(self) -> None`：执行 `test critical stages cannot be skipped` 对应逻辑。 调用：`FullFlowConfig`, `run_full_flow`, `self.assertRaises`。
- `M L78-L100` `FullFlowTests.test_force_stage_is_reported_in_plan_manifest(self) -> None` [IO-R]：执行 `test force stage is reported in plan manifest` 对应逻辑。 调用：`FullFlowConfig`, `run_full_flow`, `self.assertEqual`, `self.assertIn`, `self.assertTrue`。
- `M L102-L115` `FullFlowTests.test_force_stage_cannot_conflict_with_skip(self) -> None`：执行 `test force stage cannot conflict with skip` 对应逻辑。 调用：`FullFlowConfig`, `run_full_flow`, `self.assertRaises`。
- `M L117-L130` `FullFlowTests.test_force_stage_must_be_inside_selected_range(self) -> None`：执行 `test force stage must be inside selected range` 对应逻辑。 调用：`FullFlowConfig`, `run_full_flow`, `self.assertRaises`。
- `M L132-L167` `FullFlowTests.test_ramp_gate_error_returns_failed_result_without_raising(self) -> None`：执行 `test ramp gate error returns failed result without raising` 对应逻辑。 调用：`FullFlowConfig`, `RampGateError`, `run_full_flow`, `self.assertEqual`, `self.assertIn`, `self.assertTrue`。
- `M L169-L205` `FullFlowTests.test_finish_stage_updates_dashboard_reporter(self) -> None`：执行 `test finish stage updates dashboard reporter` 对应逻辑。 调用：`FullFlowConfig`, `_FlowRuntime`, `_finish_stage`, `_initial_stage_states`, `self.assertEqual`, `utc_now`。
- `M L207-L238` `FullFlowTests.test_initial_dashboard_stages_are_published_as_pending(self) -> None`：执行 `test initial dashboard stages are published as pending` 对应逻辑。 调用：`FullFlowConfig`, `_FlowRuntime`, `_initial_stage_states`, `_publish_initial_dashboard_stages`, `self.assertEqual`, `self.assertTrue`。
- `M L240-L300` `FullFlowTests.test_full_flow_reports_initial_resource_snapshot_to_dashboard(self) -> None`：执行 `test full flow reports initial resource snapshot to dashboard` 对应逻辑。 调用：`FullFlowConfig`, `run_full_flow`, `self.assertEqual`。
- `C L247-L256` `FullFlowTests.test_full_flow_reports_initial_resource_snapshot_to_dashboard.FakeDashboardHandle` [CLASS]：封装 `FakeDashboardHandle` 相关数据或行为。
- `N L248-L249` `FullFlowTests.test_full_flow_reports_initial_resource_snapshot_to_dashboard.FakeDashboardHandle.__init__(self, reporter: DashboardReporter) -> None`：初始化实例依赖、配置和运行状态。
- `N L251-L252` `FullFlowTests.test_full_flow_reports_initial_resource_snapshot_to_dashboard.FakeDashboardHandle.__enter__(self)`：执行 `enter` 对应逻辑。
- `N L254-L256` `FullFlowTests.test_full_flow_reports_initial_resource_snapshot_to_dashboard.FakeDashboardHandle.__exit__(self, exc_type, exc, traceback)`：执行 `exit` 对应逻辑。 调用：`self.reporter.close`。
- `N L258-L264` `FullFlowTests.test_full_flow_reports_initial_resource_snapshot_to_dashboard.fake_dashboard_reporter(**kwargs)`：执行 `fake dashboard reporter` 对应逻辑。 调用：`FakeDashboardHandle`。
- `M L302-L349` `FullFlowTests.test_ramp_section_passes_formal_gallery_output_root(self) -> None` [IO-W]：执行 `test ramp section passes formal gallery output root` 对应逻辑。 调用：`FullFlowConfig`, `_FlowRuntime`, `_initial_stage_states`, `_run_ramp_section`, `self.assertEqual`, `utc_now`。

## `src/traning/tests/full_checks/test_full_training_pipeline.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`start.checks`, `traning.conf`, `traning.core.dataset_import`, `traning.core.decision`, `traning.core.decision.pipeline`, `traning.core.spatial`, `traning.core.temporal`, `traning.lib.visualization`

- `C L33-L448` `FullTrainingPipelineTests(unittest.TestCase)` [CLASS]：封装 `FullTrainingPipelineTests` 相关数据或行为。
- `M L34-L58` `FullTrainingPipelineTests.test_optimization_base_parameters_include_candidate_cache_defaults(self) -> None`：执行 `test optimization base parameters include candidate cache defaults` 对应逻辑。 调用：`FullTrainingRunConfig`, `Settings`, `_optimization_base_parameters`, `self.assertEqual`。
- `M L60-L76` `FullTrainingPipelineTests.test_evaluation_message_distinguishes_strict_gate_failure(self) -> None`：执行 `test evaluation message distinguishes strict gate failure` 对应逻辑。 调用：`_evaluation_stage_message`, `self.assertIn`。
- `M L78-L309` `FullTrainingPipelineTests.test_pipeline_runs_all_training_steps_and_writes_summary(self) -> None` [IO-R IO-W]：执行 `test pipeline runs all training steps and writes summary` 对应逻辑。 调用：`CandidateCacheBuildResult`, `DataInputReport`, `FullTrainingRunConfig`, `Settings`, `SpatialTrainingResult`, `TemporalDecisionRunResult`。
- `M L311-L448` `FullTrainingPipelineTests.test_gallery_export_failure_emits_fail_event(self) -> None` [IO-W]：执行 `test gallery export failure emits fail event` 对应逻辑。 调用：`CandidateCacheBuildResult`, `DataInputReport`, `FullTrainingRunConfig`, `GalleryResult`, `Settings`, `SpatialTrainingResult`。
- `C L451-L464` `_RecordingReporter(NullReporter)` [CLASS]：封装 `RecordingReporter` 相关数据或行为。
- `M L452-L455` `_RecordingReporter.__init__(self) -> None`：初始化实例依赖、配置和运行状态。
- `M L457-L458` `_RecordingReporter.update_metrics(self, **metrics: object) -> None`：执行 `update metrics` 对应逻辑。 调用：`self.metric_updates.append`。
- `M L460-L461` `_RecordingReporter.update_pipeline_stage(self, stage: PipelineStageState) -> None`：执行 `update pipeline stage` 对应逻辑。 调用：`self.stage_updates.append`。
- `M L463-L464` `_RecordingReporter.emit_event(self, event) -> None`：执行 `emit event` 对应逻辑。 调用：`self.events.append`。

## `src/traning/tests/full_checks/test_gallery_schema.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.state`

- `C L12-L58` `BatchGalleryRequestTests(unittest.TestCase)` [CLASS]：封装 `BatchGalleryRequestTests` 相关数据或行为。
- `M L13-L34` `BatchGalleryRequestTests.test_frame_evaluation_accepts_error_attribution(self) -> None`：执行 `test frame evaluation accepts error attribution` 对应逻辑。 调用：`self.assertEqual`。
- `M L36-L58` `BatchGalleryRequestTests.test_trials_must_share_score_version(self) -> None`：执行 `test trials must share score version` 对应逻辑。 调用：`self.assertRaises`。

## `src/traning/tests/full_checks/test_gated_fusion.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`, `traning.lib.models.local_encoder`

- `C L14-L36` `GatedFusionTests(unittest.TestCase)` [CLASS]：封装 `GatedFusionTests` 相关数据或行为。
- `M L15-L36` `GatedFusionTests.test_forward_and_backward(self) -> None`：执行 `test forward and backward` 对应逻辑。 调用：`GatedSparseFusion`, `LocalFeatures`, `PatchMeta`, `self.assertEqual`, `self.assertIsNotNone`。

## `src/traning/tests/full_checks/test_global_encoder.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.models`

- `C L12-L29` `GlobalEncoderTests(unittest.TestCase)` [CLASS]：封装 `GlobalEncoderTests` 相关数据或行为。
- `M L13-L25` `GlobalEncoderTests.test_lightweight_encoder_and_structure_head(self) -> None`：执行 `test lightweight encoder and structure head` 对应逻辑。 调用：`GlobalStructureHead`, `LightweightGlobalEncoder`, `self.assertEqual`。
- `M L27-L29` `GlobalEncoderTests.test_non_default_backbone_requires_external_setup(self) -> None`：执行 `test non default backbone requires external setup` 对应逻辑。 调用：`LightweightGlobalEncoder`, `self.assertRaises`。

## `src/traning/tests/full_checks/test_global_sampling.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`

- `C L13-L23` `GlobalSamplingTests(unittest.TestCase)` [CLASS]：封装 `GlobalSamplingTests` 相关数据或行为。
- `M L14-L23` `GlobalSamplingTests.test_patch_position_changes_sampled_context(self) -> None`：执行 `test patch position changes sampled context` 对应逻辑。 调用：`PatchMeta`, `sample_global_feature`, `self.assertLess`。

## `src/traning/tests/full_checks/test_local_encoder.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.models`

- `C L12-L25` `LocalEncoderTests(unittest.TestCase)` [CLASS]：封装 `LocalEncoderTests` 相关数据或行为。
- `M L13-L25` `LocalEncoderTests.test_forward_shapes_and_backward(self) -> None`：执行 `test forward shapes and backward` 对应逻辑。 调用：`SmallLocalEncoder`, `self.assertEqual`, `self.assertIn`, `self.assertIsNotNone`。

## `src/traning/tests/full_checks/test_memory_smoke.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`, `traning.lib.runtime`

- `C L25-L97` `MemorySmokeTests(unittest.TestCase)` [CLASS]：封装 `MemorySmokeTests` 相关数据或行为。
- `M L26-L86` `MemorySmokeTests.run_smoke(self, device: torch.device) -> None`：执行 `run smoke` 对应逻辑。 调用：`CudaRuntimeConfig`, `GatedSparseFusion`, `PatchMeta`, `SmallLocalEncoder`, `SpatialPredictionHead`, `autocast_context`。
- `M L88-L89` `MemorySmokeTests.test_cpu_forward_backward_step(self) -> None`：执行 `test cpu forward backward step` 对应逻辑。 调用：`self.run_smoke`。
- `M L91-L97` `MemorySmokeTests.test_cuda_forward_backward_step_when_available(self) -> None`：执行 `test cuda forward backward step when available` 对应逻辑。 调用：`collect_memory_snapshot`, `self.assertIsNotNone`, `self.run_smoke`, `self.skipTest`。

## `src/traning/tests/full_checks/test_model_export.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.model_export`

- `C L17-L47` `ModelExportTests(unittest.TestCase)` [CLASS]：封装 `ModelExportTests` 相关数据或行为。
- `M L18-L47` `ModelExportTests.test_export_model_artifact_copies_files_and_validates_hashes(self) -> None` [IO-W]：执行 `test export model artifact copies files and validates hashes` 对应逻辑。 调用：`ModelArtifactSpec`, `export_model_artifact`, `self.assertEqual`, `self.assertTrue`, `validate_model_artifact`。

## `src/traning/tests/full_checks/test_optimization_module.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.optimization`, `traning.core.optimization.parameter_search`, `traning.lib.metrics`, `traning.state`

- `F L33-L41` `_circle_target(target_id: str='circle-1') -> TargetObject`：执行 `circle target` 对应逻辑。 调用：`TargetObject`。
- `C L44-L472` `OptimizationModuleTests(unittest.TestCase)` [CLASS]：封装 `OptimizationModuleTests` 相关数据或行为。
- `M L45-L51` `OptimizationModuleTests.test_parameter_search_rejects_unimplemented_method(self) -> None`：执行 `test parameter search rejects unimplemented method` 对应逻辑。 调用：`ParameterSearchConfig`, `self.assertEqual`, `self.assertRaisesRegex`。
- `M L53-L64` `OptimizationModuleTests.test_training_job_rejects_unwired_hard_example_weights(self) -> None`：执行 `test training job rejects unwired hard example weights` 对应逻辑。 调用：`self.assertRaisesRegex`, `training_job_from_dict`。
- `M L66-L82` `OptimizationModuleTests.test_score_trial_aggregates_point_slider_sequence_rules(self) -> None`：执行 `test score trial aggregates point slider sequence rules` 对应逻辑。 调用：`PredictedClick`, `SampleScoringInput`, `_circle_target`, `score_trial`, `self.assertAlmostEqual`, `self.assertEqual`。
- `M L84-L101` `OptimizationModuleTests.test_attribution_groups_temporal_and_decision_errors(self) -> None`：执行 `test attribution groups temporal and decision errors` 对应逻辑。 调用：`PredictedClick`, `SampleScoringInput`, `_circle_target`, `analyze_trial_attribution`, `score_trial`, `self.assertEqual`。
- `M L103-L143` `OptimizationModuleTests.test_parameter_plan_uses_attribution_and_asha_thresholds(self) -> None`：执行 `test parameter plan uses attribution and asha thresholds` 对应逻辑。 调用：`ParameterSearchConfig`, `PredictedClick`, `SampleScoringInput`, `TrialHistoryEntry`, `_circle_target`, `analyze_trial_attribution`。
- `M L145-L171` `OptimizationModuleTests.test_asha_does_not_promote_until_every_sample_passes(self) -> None`：执行 `test asha does not promote until every sample passes` 对应逻辑。 调用：`PredictedClick`, `SampleScoringInput`, `_circle_target`, `analyze_trial_attribution`, `plan_next_trial`, `score_trial`。
- `M L173-L195` `OptimizationModuleTests.test_promotion_preserves_stage_cap_and_advances_rung(self) -> None`：执行 `test promotion preserves stage cap and advances rung` 对应逻辑。 调用：`ParameterSearchConfig`, `PredictedClick`, `SampleScoringInput`, `_circle_target`, `analyze_trial_attribution`, `plan_next_trial`。
- `M L197-L243` `OptimizationModuleTests.test_executor_clamps_resolved_absolute_parameters(self) -> None`：执行 `test executor clamps resolved absolute parameters` 对应逻辑。 调用：`OptimizationExecutorConfig`, `SampleScoringInput`, `TrialParameters`, `_circle_target`, `analyze_trial_attribution`, `execute_optimization_plan`。
- `M L245-L261` `OptimizationModuleTests.test_gallery_request_is_built_from_trial_score_report(self) -> None`：执行 `test gallery request is built from trial score report` 对应逻辑。 调用：`PredictedClick`, `SampleScoringInput`, `_circle_target`, `build_batch_gallery_request`, `score_trial`, `self.assertEqual`。
- `M L263-L311` `OptimizationModuleTests.test_gallery_reuses_spatial_attribution_for_unresolved_target(self) -> None`：执行 `test gallery reuses spatial attribution for unresolved target` 对应逻辑。 调用：`SampleScoringInput`, `TargetObject`, `analyze_trial_attribution`, `build_batch_gallery_request`, `score_trial`, `self.assertAlmostEqual`。
- `M L313-L350` `OptimizationModuleTests.test_curriculum_gate_and_hard_example_sampling(self) -> None`：执行 `test curriculum gate and hard example sampling` 对应逻辑。 调用：`PredictedClick`, `SampleScoringInput`, `_circle_target`, `analyze_trial_attribution`, `build_hard_example_sampling_plan`, `evaluate_curriculum_gate`。
- `M L352-L382` `OptimizationModuleTests.test_execute_optimization_plan_records_trial_and_job(self) -> None` [IO-W]：执行 `test execute optimization plan records trial and job` 对应逻辑。 调用：`OptimizationExecutorConfig`, `PredictedClick`, `SampleScoringInput`, `_circle_target`, `analyze_trial_attribution`, `execute_optimization_plan`。
- `M L384-L413` `OptimizationModuleTests.test_sqlite_trial_store_records_execution(self) -> None`：执行 `test sqlite trial store records execution` 对应逻辑。 调用：`OptimizationExecutorConfig`, `PredictedClick`, `SQLiteTrialStore`, `SampleScoringInput`, `_circle_target`, `analyze_trial_attribution`。
- `M L415-L451` `OptimizationModuleTests.test_trial_history_uses_source_stage_rung_and_score_version(self) -> None`：执行 `test trial history uses source stage rung and score version` 对应逻辑。 调用：`self.assertEqual`, `trial_history_from_records`。
- `M L453-L472` `OptimizationModuleTests.test_multi_objective_score_uses_quality_vram_and_latency(self) -> None`：执行 `test multi objective score uses quality vram and latency` 对应逻辑。 调用：`PredictedClick`, `SampleScoringInput`, `_circle_target`, `score_trial`, `score_trial_objectives`, `self.assertEqual`。

## `src/traning/tests/full_checks/test_patch_stream.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`

- `C L12-L50` `PatchStreamTests(unittest.TestCase)` [CLASS]：封装 `PatchStreamTests` 相关数据或行为。
- `M L13-L27` `PatchStreamTests.assert_full_coverage(self, width: int, height: int) -> None`：执行 `assert full coverage` 对应逻辑。 调用：`PatchStream`, `self.assertEqual`, `self.assertNotIn`, `self.assertTrue`, `stream.iter_patches`。
- `M L29-L31` `PatchStreamTests.test_common_resolutions_are_fully_covered(self) -> None`：执行 `test common resolutions are fully covered` 对应逻辑。 调用：`self.assert_full_coverage`。
- `M L33-L34` `PatchStreamTests.test_odd_dimensions_are_fully_covered(self) -> None`：执行 `test odd dimensions are fully covered` 对应逻辑。 调用：`self.assert_full_coverage`。
- `M L36-L46` `PatchStreamTests.test_small_image_is_padded(self) -> None`：执行 `test small image is padded` 对应逻辑。 调用：`PatchStream`, `self.assertEqual`, `self.assertTrue`, `stream.iter_patches`。
- `M L48-L50` `PatchStreamTests.test_invalid_overlap_raises(self) -> None`：执行 `test invalid overlap raises` 对应逻辑。 调用：`PatchStream`, `self.assertRaises`。

## `src/traning/tests/full_checks/test_plan_gap_closure.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package.coordinates`, `traning.conf`, `traning.core.decision.generator`, `traning.core.diagnostics.oracle_ladder`, `traning.core.model_export`, `traning.core.temporal.trainer`, `traning.lib.coordinates`, `traning.state.versioning`

- `C L44-L361` `PlanGapClosureTests(unittest.TestCase)` [CLASS]：封装 `PlanGapClosureTests` 相关数据或行为。
- `M L45-L75` `PlanGapClosureTests.test_oracle_target_assignment_uses_v2_osu_match_distance(self) -> None`：执行 `test oracle target assignment uses v2 osu match distance` 对应逻辑。 调用：`_target_assignment`, `self.assertEqual`。
- `M L77-L100` `PlanGapClosureTests.test_training_configs_use_calibrated_affine_matrix(self) -> None`：确保两份正式训练配置引用同一组经过校准的仿射系数。 调用：`load_settings`, `self.assertAlmostEqual`, `self.assertEqual`, `self.assertIsNotNone`, `self.subTest`, `transform_from_settings_or_sample`。
- `M L102-L118` `PlanGapClosureTests.test_final_affine_matches_independent_observed_control_points(self) -> None`：用独立观测点约束拟合结果，防止只对某一张示例图片过拟合。 调用：`self.assertLessEqual`, `self.subTest`。
- `M L120-L128` `PlanGapClosureTests.test_explicit_non_centered_playfield_round_trip(self) -> None`：执行 `test explicit non centered playfield round trip` 对应逻辑。 调用：`self.assertAlmostEqual`。
- `M L130-L138` `PlanGapClosureTests.test_affine_playfield_round_trip(self) -> None`：执行 `test affine playfield round trip` 对应逻辑。 调用：`self.assertAlmostEqual`。
- `M L140-L165` `PlanGapClosureTests.test_source_rect_applies_crop_offset_before_video_mapping(self) -> None`：执行 `test source rect applies crop offset before video mapping` 对应逻辑。 调用：`Settings`, `self.assertEqual`, `transform_from_settings_or_sample`。
- `M L167-L280` `PlanGapClosureTests.test_action_targets_include_circle_release_slider_repeat_and_spinner(self) -> None`：执行 `test action targets include circle release slider repeat and spinner` 对应逻辑。 调用：`Settings`, `build_candidate_cache_record`, `self.assertEqual`。
- `M L282-L317` `PlanGapClosureTests.test_temporal_loss_weights_change_combined_loss(self) -> None`：执行 `test temporal loss weights change combined loss` 对应逻辑。 调用：`_compute_temporal_loss`, `self.assertGreater`。
- `C L283-L287` `PlanGapClosureTests.test_temporal_loss_weights_change_combined_loss.Weights` [CLASS]：封装 `Weights` 相关数据或行为。
- `C L289-L293` `PlanGapClosureTests.test_temporal_loss_weights_change_combined_loss.TimeHeavy` [CLASS]：封装 `TimeHeavy` 相关数据或行为。
- `M L319-L332` `PlanGapClosureTests.test_version_mismatch_blocks_without_override(self) -> None`：执行 `test version mismatch blocks without override` 对应逻辑。 调用：`ensure_compatible_versions`, `self.assertEqual`, `self.assertFalse`, `self.assertTrue`。
- `M L334-L352` `PlanGapClosureTests.test_transform_fingerprint_mismatch_blocks_without_override(self) -> None`：协议版本相同但方程摘要不同，也必须阻止缓存或检查点复用。 调用：`ensure_compatible_versions`, `self.assertEqual`, `self.assertFalse`, `self.assertTrue`。
- `M L354-L361` `PlanGapClosureTests.test_settings_migration_adds_schema_and_transform(self) -> None` [IO-W]：执行 `test settings migration adds schema and transform` 对应逻辑。 调用：`migrate_settings_file`, `self.assertIn`, `self.assertTrue`。

## `src/traning/tests/full_checks/test_result_export_gallery.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package.coordinates`, `traning.lib.visualization`, `traning.lib.visualization.gallery`, `traning.lib.visualization.render`, `traning.state`

- `C L38-L90` `_FakeSegmentFrameDataset` [CLASS]：封装 `FakeSegmentFrameDataset` 相关数据或行为。
- `M L39-L59` `_FakeSegmentFrameDataset.__init__(self) -> None`：初始化实例依赖、配置和运行状态。
- `M L61-L90` `_FakeSegmentFrameDataset.__getitem__(self, index: int) -> dict[str, object]`：执行 `getitem` 对应逻辑。
- `C L93-L106` `_DiverseFakeSegmentFrameDataset(_FakeSegmentFrameDataset)` [CLASS]：封装 `DiverseFakeSegmentFrameDataset` 相关数据或行为。
- `M L94-L106` `_DiverseFakeSegmentFrameDataset.__init__(self, size: int=6) -> None`：初始化实例依赖、配置和运行状态。
- `F L109-L121` `_request(frames: tuple[FrameEvaluation, ...]) -> BatchGalleryRequest`：执行 `request` 对应逻辑。 调用：`BatchGalleryRequest`, `TrialGalleryEvaluation`, `TrialParameters`。
- `F L124-L140` `_multi_trial_request(trials: tuple[tuple[str, float, tuple[FrameEvaluation, ...]], ...]) -> BatchGalleryRequest`：执行 `multi trial request` 对应逻辑。 调用：`BatchGalleryRequest`, `TrialGalleryEvaluation`, `TrialParameters`。
- `C L143-L565` `ResultExportGalleryTests(unittest.TestCase)` [CLASS]：封装 `ResultExportGalleryTests` 相关数据或行为。
- `M L144-L155` `ResultExportGalleryTests.test_safe_name_caps_long_tokens_with_stable_collision_resistant_hash(self) -> None`：执行 `test safe name caps long tokens with stable collision resistant hash` 对应逻辑。 调用：`_safe_name`, `self.assertEqual`, `self.assertLessEqual`, `self.assertNotEqual`。
- `M L157-L162` `ResultExportGalleryTests.test_annotation_font_distinguishes_chinese_diagnostic_text(self) -> None`：执行 `test annotation font distinguishes chinese diagnostic text` 对应逻辑。 调用：`_annotation_font`, `self.assertNotEqual`。
- `M L164-L211` `ResultExportGalleryTests.test_failed_gallery_persists_no_op_reason_and_action(self) -> None` [IO-R IO-W]：执行 `test failed gallery persists no op reason and action` 对应逻辑。 调用：`FrameEvaluation`, `_FakeSegmentFrameDataset`, `_request`, `self.assertEqual`, `self.assertIn`, `self.assertIsNone`。
- `M L213-L281` `ResultExportGalleryTests.test_affine_circle_slider_and_spinner_render_at_affine_marker(self) -> None`：验证三类对象都消费样本携带的仿射规格，而非仅修正某一渲染分支。 调用：`render_annotated_frame`, `self.assertEqual`, `self.assertNotEqual`, `self.subTest`, `transform_spec.as_dict`。
- `M L283-L330` `ResultExportGalleryTests.test_outputs_one_folder_per_selected_sample_group(self) -> None` [IO-R]：执行 `test outputs one folder per selected sample group` 对应逻辑。 调用：`FrameEvaluation`, `_FakeSegmentFrameDataset`, `_request`, `self.assertEqual`。
- `M L332-L380` `ResultExportGalleryTests.test_samples_per_group_limits_sample_folders_not_frames(self) -> None` [IO-R]：执行 `test samples per group limits sample folders not frames` 对应逻辑。 调用：`FrameEvaluation`, `_FakeSegmentFrameDataset`, `_request`, `self.assertEqual`。
- `M L382-L425` `ResultExportGalleryTests.test_gallery_samples_diverse_groups_by_seed_not_first_n(self) -> None` [IO-R]：执行 `test gallery samples diverse groups by seed not first n` 对应逻辑。 调用：`BatchGalleryRequest`, `FrameEvaluation`, `TrialGalleryEvaluation`, `TrialParameters`, `_DiverseFakeSegmentFrameDataset`, `self.assertEqual`。
- `M L427-L463` `ResultExportGalleryTests.test_best_trial_exports_even_below_promotion_threshold(self) -> None` [IO-R]：执行 `test best trial exports even below promotion threshold` 对应逻辑。 调用：`FrameEvaluation`, `_FakeSegmentFrameDataset`, `_multi_trial_request`, `self.assertEqual`。
- `M L465-L500` `ResultExportGalleryTests.test_failed_samples_export_without_any_passed_sample(self) -> None`：执行 `test failed samples export without any passed sample` 对应逻辑。 调用：`FrameEvaluation`, `_FakeSegmentFrameDataset`, `_multi_trial_request`, `self.assertEqual`。
- `M L502-L532` `ResultExportGalleryTests.test_score_tie_selects_lexicographically_first_trial_id(self) -> None` [IO-R]：执行 `test score tie selects lexicographically first trial id` 对应逻辑。 调用：`FrameEvaluation`, `_FakeSegmentFrameDataset`, `_multi_trial_request`, `self.assertEqual`。
- `M L534-L565` `ResultExportGalleryTests.test_export_failure_does_not_commit_counter_or_formal_artifact(self) -> None`：执行 `test export failure does not commit counter or formal artifact` 对应逻辑。 调用：`FrameEvaluation`, `_FakeSegmentFrameDataset`, `_request`, `self.assertEqual`, `self.assertFalse`, `self.assertRaisesRegex`。

## `src/traning/tests/full_checks/test_scoring.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.metrics`

- `C L16-L84` `PointScoringTests(unittest.TestCase)` [CLASS]：封装 `PointScoringTests` 相关数据或行为。
- `M L17-L18` `PointScoringTests.setUp(self) -> None`：执行 `setUp` 对应逻辑。 调用：`ScoreSpec`。
- `M L20-L26` `PointScoringTests.test_spatial_bonus_clamps_inside_sixty_percent(self) -> None`：执行 `test spatial bonus clamps inside sixty percent` 对应逻辑。 调用：`self.assertEqual`, `self.assertGreater`, `spatial_coefficient`。
- `M L28-L34` `PointScoringTests.test_spatial_comfort_and_zero_boundaries(self) -> None`：执行 `test spatial comfort and zero boundaries` 对应逻辑。 调用：`self.assertEqual`, `self.assertGreater`, `self.assertLessEqual`, `spatial_coefficient`。
- `M L36-L48` `PointScoringTests.test_temporal_boundaries_follow_v2_bands(self) -> None`：执行 `test temporal boundaries follow v2 bands` 对应逻辑。 调用：`self.assertEqual`, `self.assertGreater`, `self.assertLess`, `temporal_coefficient`。
- `M L50-L84` `PointScoringTests.test_point_pass_requires_space_and_time(self) -> None`：执行 `test point pass requires space and time` 对应逻辑。 调用：`score_point`, `self.assertAlmostEqual`, `self.assertFalse`, `self.assertTrue`。
- `C L87-L170` `SliderScoringTests(unittest.TestCase)` [CLASS]：封装 `SliderScoringTests` 相关数据或行为。
- `M L88-L100` `SliderScoringTests.test_slider_uses_first_path_point_as_missing_head(self) -> None`：执行 `test slider uses first path point as missing head` 对应逻辑。 调用：`score_slider`, `self.assertEqual`, `self.assertTrue`。
- `M L102-L146` `SliderScoringTests.test_slider_requires_bidirectional_path_match(self) -> None`：执行 `test slider requires bidirectional path match` 对应逻辑。 调用：`ScoreSpec`, `score_slider`, `self.assertEqual`, `self.assertFalse`, `self.assertLessEqual`, `self.assertTrue`。
- `M L148-L170` `SliderScoringTests.test_slider_corridor_uses_one_point_five_radius(self) -> None`：执行 `test slider corridor uses one point five radius` 对应逻辑。 调用：`score_slider`, `self.assertFalse`, `self.assertTrue`。

## `src/traning/tests/full_checks/test_sequence_scoring.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.metrics`

- `C L15-L162` `ClickSequenceScoringTests(unittest.TestCase)` [CLASS]：封装 `ClickSequenceScoringTests` 相关数据或行为。
- `M L16-L45` `ClickSequenceScoringTests.test_first_passing_hit_resolves_target_once(self) -> None`：执行 `test first passing hit resolves target once` 对应逻辑。 调用：`PredictedClick`, `TargetObject`, `score_click_sequence`, `self.assertEqual`。
- `M L47-L68` `ClickSequenceScoringTests.test_failed_hit_keeps_target_active_for_later_click(self) -> None`：执行 `test failed hit keeps target active for later click` 对应逻辑。 调用：`PredictedClick`, `TargetObject`, `score_click_sequence`, `self.assertEqual`, `self.assertIn`。
- `M L70-L88` `ClickSequenceScoringTests.test_early_click_is_attributed_to_temporal_parameters(self) -> None`：执行 `test early click is attributed to temporal parameters` 对应逻辑。 调用：`PredictedClick`, `TargetObject`, `score_click_sequence`, `self.assertEqual`。
- `M L90-L122` `ClickSequenceScoringTests.test_overlapping_targets_resolve_by_earliest_active_target(self) -> None`：执行 `test overlapping targets resolve by earliest active target` 对应逻辑。 调用：`PredictedClick`, `TargetObject`, `score_click_sequence`, `self.assertEqual`。
- `M L124-L162` `ClickSequenceScoringTests.test_click_frequency_limit_blocks_high_rate_hits(self) -> None`：执行 `test click frequency limit blocks high rate hits` 对应逻辑。 调用：`PredictedClick`, `SequenceScoreSpec`, `TargetObject`, `score_click_sequence`, `self.assertEqual`, `self.assertTrue`。

## `src/traning/tests/full_checks/test_spatial_decode.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data`, `traning.lib.models`, `traning.lib.training`

- `F L20-L43` `_prediction(*, height: int=16, width: int=16, embedding_dim: int=4) -> SpatialPrediction`：执行 `prediction` 对应逻辑。 调用：`F.normalize`, `SpatialPrediction`。
- `C L46-L180` `SpatialDecodeTests(unittest.TestCase)` [CLASS]：封装 `SpatialDecodeTests` 相关数据或行为。
- `M L47-L83` `SpatialDecodeTests.test_canvas_decodes_global_candidate_with_offset_and_type(self) -> None`：执行 `test canvas decodes global candidate with offset and type` 对应逻辑。 调用：`PatchMeta`, `SpatialPredictionCanvas`, `_prediction`, `canvas.to_maps`, `canvas.write_patch`, `decode_spatial_candidates`。
- `M L85-L112` `SpatialDecodeTests.test_padding_region_is_not_written_to_global_canvas(self) -> None`：执行 `test padding region is not written to global canvas` 对应逻辑。 调用：`PatchMeta`, `SpatialPredictionCanvas`, `_prediction`, `canvas.to_maps`, `canvas.write_patch`, `decode_spatial_candidates`。
- `M L114-L138` `SpatialDecodeTests.test_decode_applies_nms(self) -> None`：执行 `test decode applies nms` 对应逻辑。 调用：`PatchMeta`, `SpatialPredictionCanvas`, `_prediction`, `canvas.to_maps`, `canvas.write_patch`, `decode_spatial_candidates`。
- `M L140-L180` `SpatialDecodeTests.test_decode_slider_paths_recovers_ordered_polyline(self) -> None`：执行 `test decode slider paths recovers ordered polyline` 对应逻辑。 调用：`SpatialPredictionMaps`, `decode_slider_paths`, `self.assertAlmostEqual`, `self.assertEqual`, `self.assertFalse`, `self.assertLess`。

## `src/traning/tests/full_checks/test_spatial_inference.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.spatial`, `traning.lib.models`

- `F L21-L64` `_tiny_settings() -> Settings`：执行 `tiny settings` 对应逻辑。 调用：`Settings`。
- `C L67-L124` `SpatialInferenceTests(unittest.TestCase)` [CLASS]：封装 `SpatialInferenceTests` 相关数据或行为。
- `M L68-L97` `SpatialInferenceTests.test_cpu_single_frame_inference_reports_cpu_gpu_split(self) -> None`：执行 `test cpu single frame inference reports cpu gpu split` 对应逻辑。 调用：`_tiny_settings`, `result.as_summary`, `run_spatial_frame_inference`, `self.assertEqual`, `self.assertIn`, `self.assertLessEqual`。
- `M L99-L124` `SpatialInferenceTests.test_inference_loads_spatial_checkpoint(self) -> None`：执行 `test inference loads spatial checkpoint` 对应逻辑。 调用：`_tiny_settings`, `build_model_stack`, `run_spatial_frame_inference`, `self.assertEqual`。

## `src/traning/tests/full_checks/test_spatial_model.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.models`

- `C L12-L25` `SpatialModelTests(unittest.TestCase)` [CLASS]：封装 `SpatialModelTests` 相关数据或行为。
- `M L13-L25` `SpatialModelTests.test_prediction_head_outputs_all_required_tasks(self) -> None`：执行 `test prediction head outputs all required tasks` 对应逻辑。 调用：`SpatialPredictionHead`, `self.assertEqual`。

## `src/traning/tests/full_checks/test_spatial_targets.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package.coordinates`, `traning.lib.data`, `traning.lib.models`, `traning.lib.training`

- `C L13-L151` `SpatialTargetTests(unittest.TestCase)` [CLASS]：封装 `SpatialTargetTests` 相关数据或行为。
- `M L14-L50` `SpatialTargetTests.test_circle_target_contains_center_and_approach_ring(self) -> None`：执行 `test circle target contains center and approach ring` 对应逻辑。 调用：`PatchMeta`, `build_spatial_loss_targets`, `self.assertGreater`, `self.assertIn`。
- `M L52-L78` `SpatialTargetTests.test_slider_target_contains_body_direction_head_and_tail(self) -> None`：执行 `test slider target contains body direction head and tail` 对应逻辑。 调用：`PatchMeta`, `build_spatial_loss_targets`, `self.assertGreater`, `self.assertIn`, `self.assertLess`。
- `M L80-L93` `SpatialTargetTests.test_spinner_target_marks_valid_patch_area(self) -> None`：执行 `test spinner target marks valid patch area` 对应逻辑。 调用：`PatchMeta`, `build_spatial_loss_targets`, `self.assertGreater`, `self.assertIn`, `self.assertLess`。
- `M L95-L151` `SpatialTargetTests.test_affine_spinner_target_uses_transformed_playfield_center(self) -> None`：验证 spinner 的 dense target 使用仿射后的 osu 中心，而非轴对齐字段。 调用：`PatchMeta`, `build_spatial_loss_targets`, `self.assertAlmostEqual`, `transform.spec.as_dict`。

## `src/traning/tests/full_checks/test_spatial_trainer.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.spatial`

- `C L16-L91` `SpatialTrainerTests(unittest.TestCase)` [CLASS]：封装 `SpatialTrainerTests` 相关数据或行为。
- `M L17-L91` `SpatialTrainerTests.test_cpu_single_step_with_synthetic_sample(self) -> None`：执行 `test cpu single step with synthetic sample` 对应逻辑。 调用：`Settings`, `run_spatial_training`, `self.assertEqual`, `self.assertTrue`。

## `src/traning/tests/full_checks/test_temporal_dataset.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.decision`, `traning.core.temporal`

- `F L23-L44` `_record(sample_key: str, frame_index: int, *, candidates: list[dict] | None=None, temporal_target: dict | None=None) -> dict`：执行 `record` 对应逻辑。
- `F L47-L72` `_candidate(score: float, *, x: float=25.0, y: float=10.0, candidate_id: int=0) -> dict`：执行 `candidate` 对应逻辑。
- `F L75-L95` `_write_cache(path: Path, records: list[dict], *, version: str=CANDIDATE_CACHE_VERSION) -> None` [IO-W]：写入 `cache` 对应的数据或结果。
- `C L98-L228` `TemporalDatasetTests(unittest.TestCase)` [CLASS]：封装 `TemporalDatasetTests` 相关数据或行为。
- `M L99-L112` `TemporalDatasetTests.test_temporal_package_imports_in_fresh_interpreter(self) -> None` [PROCESS]：执行 `test temporal package imports in fresh interpreter` 对应逻辑。 调用：`self.assertEqual`, `subprocess.run`。
- `M L114-L127` `TemporalDatasetTests.test_legacy_v1_cache_requires_explicit_diagnostic_mode(self) -> None`：执行 `test legacy v1 cache requires explicit diagnostic mode` 对应逻辑。 调用：`_candidate`, `_record`, `_write_cache`, `load_candidate_cache_records`, `self.assertEqual`, `self.assertRaisesRegex`。
- `M L129-L136` `TemporalDatasetTests.test_loads_candidate_cache_records(self) -> None`：执行 `test loads candidate cache records` 对应逻辑。 调用：`_candidate`, `_record`, `_write_cache`, `load_candidate_cache_records`, `self.assertEqual`。
- `M L138-L162` `TemporalDatasetTests.test_encodes_fixed_windows_without_crossing_samples(self) -> None`：执行 `test encodes fixed windows without crossing samples` 对应逻辑。 调用：`TemporalCandidateWindowDataset`, `TemporalFeatureSpec`, `_candidate`, `_record`, `self.assertEqual`, `self.assertFalse`。
- `M L164-L195` `TemporalDatasetTests.test_uses_explicit_temporal_target_when_present(self) -> None`：执行 `test uses explicit temporal target when present` 对应逻辑。 调用：`TemporalCandidateWindowDataset`, `TemporalFeatureSpec`, `_candidate`, `_record`, `self.assertEqual`, `self.assertTrue`。
- `M L197-L228` `TemporalDatasetTests.test_preserves_selected_target_candidate_when_outside_top_scores(self) -> None`：执行 `test preserves selected target candidate when outside top scores` 对应逻辑。 调用：`TemporalCandidateWindowDataset`, `TemporalFeatureSpec`, `_candidate`, `_record`, `self.assertEqual`, `self.assertTrue`。

## `src/traning/tests/full_checks/test_temporal_decision.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.decision`, `traning.core.temporal`

- `F L17-L55` `_record(frame_index: int) -> dict`：执行 `record` 对应逻辑。
- `F L58-L67` `_write_cache(path: Path) -> None` [IO-W]：写入 `cache` 对应的数据或结果。 调用：`_record`。
- `C L70-L107` `TemporalDecisionTests(unittest.TestCase)` [CLASS]：封装 `TemporalDecisionTests` 相关数据或行为。
- `M L71-L107` `TemporalDecisionTests.test_train_then_run_decision(self) -> None` [IO-R]：执行 `test train then run decision` 对应逻辑。 调用：`Settings`, `_write_cache`, `run_temporal_decision`, `run_temporal_training`, `self.assertEqual`, `self.assertIn`。

## `src/traning/tests/full_checks/test_temporal_trainer.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.decision`, `traning.core.temporal`, `traning.core.training_inheritance`

- `F L18-L48` `_record(frame_index: int) -> dict`：执行 `record` 对应逻辑。
- `F L51-L65` `_write_cache(path: Path) -> None` [IO-W]：写入 `cache` 对应的数据或结果。 调用：`_record`。
- `F L68-L93` `_assert_nested_close(case: unittest.TestCase, left, right, *, path: str='root') -> None`：执行 `assert nested close` 对应逻辑。 调用：`_assert_nested_close`。
- `C L96-L189` `TemporalTrainerTests(unittest.TestCase)` [CLASS]：封装 `TemporalTrainerTests` 相关数据或行为。
- `M L97-L121` `TemporalTrainerTests.test_cpu_temporal_training_smoke(self) -> None`：执行 `test cpu temporal training smoke` 对应逻辑。 调用：`Settings`, `_write_cache`, `run_temporal_training`, `self.assertEqual`, `self.assertGreater`, `self.assertTrue`。
- `M L123-L189` `TemporalTrainerTests.test_resume_matches_continuous_temporal_training(self) -> None`：执行 `test resume matches continuous temporal training` 对应逻辑。 调用：`Settings`, `_assert_nested_close`, `_write_cache`, `load_training_checkpoint`, `run_temporal_training`, `self.assertEqual`。

## `src/traning/tests/full_checks/test_training_inheritance.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.conf`, `traning.core.training_inheritance`

- `C L18-L111` `TrainingInheritanceTests(unittest.TestCase)` [CLASS]：封装 `TrainingInheritanceTests` 相关数据或行为。
- `M L19-L55` `TrainingInheritanceTests.test_create_and_load_inheritance_package(self) -> None`：执行 `test create and load inheritance package` 对应逻辑。 调用：`create_inheritance_package`, `load_inheritance_package`, `load_settings`, `self.assertEqual`, `self.assertIn`, `self.assertTrue`。
- `M L57-L78` `TrainingInheritanceTests.test_strict_rejects_incompatible_dataset(self) -> None` [IO-R IO-W]：执行 `test strict rejects incompatible dataset` 对应逻辑。 调用：`create_inheritance_package`, `load_inheritance_package`, `load_settings`, `self.assertRaises`。
- `M L80-L111` `TrainingInheritanceTests.test_auto_downgrades_when_transform_equation_changes(self) -> None`：确认只改仿射偏移量也会使 auto 继承降级为仅加载权重。 调用：`create_inheritance_package`, `load_inheritance_package`, `load_settings`, `self.assertEqual`, `self.assertFalse`, `self.assertIn`。

## `src/traning/tests/full_checks/test_training_ramp.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.training_ramp`, `traning.main`

- `C L38-L1131` `TrainingRampTests(unittest.TestCase)` [CLASS]：封装 `TrainingRampTests` 相关数据或行为。
- `M L39-L65` `TrainingRampTests.test_build_ramp_levels_clips_and_reaches_target(self) -> None`：执行 `test build ramp levels clips and reaches target` 对应逻辑。 调用：`RampTarget`, `build_ramp_levels`, `self.assertEqual`, `self.assertGreaterEqual`, `self.assertLessEqual`。
- `M L67-L109` `TrainingRampTests.test_ensure_full_target_config_writes_target_and_absolutizes_paths(self) -> None` [IO-R IO-W]：执行 `test ensure full target config writes target and absolutizes paths` 对应逻辑。 调用：`RampTarget`, `ensure_full_target_config`, `self.assertEqual`, `self.assertTrue`。
- `M L111-L140` `TrainingRampTests.test_level_config_isolates_jsonl_and_sqlite_trial_history(self) -> None` [IO-R IO-W]：执行 `test level config isolates jsonl and sqlite trial history` 对应逻辑。 调用：`RampLevelSpec`, `_write_level_config`, `self.assertEqual`。
- `M L142-L218` `TrainingRampTests.test_ramp_reporter_tracks_level_pass_and_failure(self) -> None`：执行 `test ramp reporter tracks level pass and failure` 对应逻辑。 调用：`RampLevelSpec`, `RampTarget`, `_report_level_finished`, `_report_level_started`, `_report_ramp_failed`, `_report_ramp_started`。
- `M L220-L284` `TrainingRampTests.test_preflight_marks_gpu_bridge_passed_when_cuda_is_visible(self) -> None`：执行 `test preflight marks gpu bridge passed when cuda is visible` 对应逻辑。 调用：`_run_preflight`, `self.assertEqual`。
- `M L286-L343` `TrainingRampTests.test_preflight_reports_disk_space_gate_failure(self) -> None`：执行 `test preflight reports disk space gate failure` 对应逻辑。 调用：`_run_preflight`, `self.assertEqual`, `self.assertRaisesRegex`, `self.assertTrue`。
- `M L345-L405` `TrainingRampTests.test_gate_rejects_quality_score_below_threshold(self) -> None` [IO-W]：执行 `test gate rejects quality score below threshold` 对应逻辑。 调用：`RampLevelSpec`, `_gate_level`, `self.assertRaisesRegex`。
- `M L407-L475` `TrainingRampTests.test_gate_reports_unresolved_evaluation_when_score_is_above_threshold(self) -> None` [IO-W]：执行 `test gate reports unresolved evaluation when score is above threshold` 对应逻辑。 调用：`RampLevelSpec`, `_gate_level`, `self.assertRaisesRegex`。
- `M L477-L535` `TrainingRampTests.test_gate_rejects_background_only_evaluation(self) -> None` [IO-W]：执行 `test gate rejects background only evaluation` 对应逻辑。 调用：`RampLevelSpec`, `_gate_level`, `self.assertRaisesRegex`。
- `M L537-L645` `TrainingRampTests.test_level_training_uses_configured_gallery_output_root(self) -> None` [IO-W]：执行 `test level training uses configured gallery output root` 对应逻辑。 调用：`RampLevelSpec`, `_run_level`, `self.assertEqual`。
- `M L647-L704` `TrainingRampTests.test_trial_runtime_consumes_optimizer_parameters_and_resume_budget(self) -> None`：执行 `test trial runtime consumes optimizer parameters and resume budget` 对应逻辑。 调用：`RampLevelSpec`, `_trial_runtime_overrides`, `self.assertAlmostEqual`, `self.assertEqual`。
- `M L706-L741` `TrainingRampTests.test_trial_runtime_clamps_legacy_negative_absolute_threshold(self) -> None`：执行 `test trial runtime clamps legacy negative absolute threshold` 对应逻辑。 调用：`RampLevelSpec`, `_trial_runtime_overrides`, `self.assertEqual`, `self.assertIsNone`。
- `M L743-L804` `TrainingRampTests.test_unbounded_level_consumes_jobs_until_strict_pass(self) -> None` [IO-R IO-W]：执行 `test unbounded level consumes jobs until strict pass` 对应逻辑。 调用：`RampEvaluationGateError`, `RampLevelSpec`, `_run_level`, `self._passing_level_record`, `self.assertEqual`。
- `M L806-L869` `TrainingRampTests.test_finite_trial_budget_preserves_pending_job(self) -> None` [IO-R IO-W]：执行 `test finite trial budget preserves pending job` 对应逻辑。 调用：`RampEvaluationGateError`, `RampLevelSpec`, `_run_level`, `self.assertEqual`, `self.assertRaisesRegex`。
- `M L871-L947` `TrainingRampTests.test_level_resumes_persisted_pending_job_for_same_run(self) -> None` [IO-W]：执行 `test level resumes persisted pending job for same run` 对应逻辑。 调用：`RampLevelSpec`, `_run_level`, `self._passing_level_record`, `self.assertEqual`。
- `M L949-L1004` `TrainingRampTests.test_disabled_generated_job_execution_keeps_first_pending_job(self) -> None` [IO-R IO-W]：执行 `test disabled generated job execution keeps first pending job` 对应逻辑。 调用：`RampEvaluationGateError`, `RampLevelSpec`, `_run_level`, `self.assertEqual`, `self.assertRaisesRegex`。
- `M L1006-L1058` `TrainingRampTests.test_run_job_validates_and_consumes_resolved_parameters(self) -> None` [IO-W]：执行 `test run job validates and consumes resolved parameters` 对应逻辑。 调用：`run_training_job_spec`, `self.assertEqual`。
- `M L1060-L1078` `TrainingRampTests.test_run_job_dry_run_rejects_missing_parent_checkpoint(self) -> None` [IO-W]：执行 `test run job dry run rejects missing parent checkpoint` 对应逻辑。 调用：`run_training_job_spec`, `self.assertRaisesRegex`。
- `M L1080-L1110` `TrainingRampTests.test_user_interrupt_persists_manifest_and_readiness(self) -> None` [IO-R]：执行 `test user interrupt persists manifest and readiness` 对应逻辑。 调用：`RampLevelSpec`, `RampTarget`, `_record_ramp_interrupted`, `self.assertEqual`, `self.assertIn`。
- `M L1113-L1131` `TrainingRampTests._passing_level_record(trial_id: str) -> dict[str, object]`：执行 `passing level record` 对应逻辑。

## `src/traning/tests/full_checks/test_training_reporting.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib`

- `C L10-L19` `TrainingReportingTests(unittest.TestCase)` [CLASS]：封装 `TrainingReportingTests` 相关数据或行为。
- `M L11-L19` `TrainingReportingTests.test_step_reporting_is_throttled_and_keeps_boundaries(self) -> None`：执行 `test step reporting is throttled and keeps boundaries` 对应逻辑。 调用：`self.assertFalse`, `self.assertTrue`, `should_report_training_step`。

## `src/traning/tests/full_checks/test_trial_aggregate_scoring.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.optimization.scoring.evaluator`, `traning.lib.metrics`

- `C L15-L79` `TrialAggregateScoringTests(unittest.TestCase)` [CLASS]：封装 `TrialAggregateScoringTests` 相关数据或行为。
- `M L16-L34` `TrialAggregateScoringTests.test_background_only_trial_cannot_pass_without_target_coverage(self) -> None`：执行 `test background only trial cannot pass without target coverage` 对应逻辑。 调用：`SampleScoringInput`, `score_trial`, `self.assertEqual`, `self.assertFalse`, `self.assertTrue`。
- `M L36-L79` `TrialAggregateScoringTests.test_no_op_background_frames_cannot_hide_all_unresolved_targets(self) -> None`：执行 `test no op background frames cannot hide all unresolved targets` 对应逻辑。 调用：`SampleScoringInput`, `TargetObject`, `score_trial`, `self.assertAlmostEqual`, `self.assertEqual`, `self.assertFalse`。

## `src/traning/tests/startup_checks/items.py`

职责：traning 启动检测项；检查配置、设备、数据输入和完整训练阶段注册。
工程依赖：`package.checks`, `traning.conf`, `traning.core.dataset_import`, `traning.core.decision`

- `F L15-L40` `check_settings_load(config_path: Path | None=None) -> tuple[StartupCheckResult, Settings]`：执行 `check settings load` 对应逻辑。 调用：`load_settings`, `settings.data_input.validate_tiling`, `settings.tiling.validate_tiling`。
- `F L43-L86` `check_runtime_device(settings: Settings, *, device: torch.device, require_cuda: bool | None=None) -> tuple[StartupCheckResult, None]`：执行 `check runtime device` 对应逻辑。
- `F L89-L118` `check_data_input(settings: Settings, *, split: DataSplit) -> tuple[StartupCheckResult, None]`：执行 `check data input` 对应逻辑。 调用：`inspect_data_input`。
- `F L121-L149` `check_core_entrypoints(_settings: Settings | None=None) -> tuple[StartupCheckResult, None]`：执行 `check core entrypoints` 对应逻辑。

## `src/traning/tests/startup_checks/runner.py`

职责：traning 启动检测统一入口；按顺序运行配置、运行设备、数据输入和 core 入口检测。
工程依赖：`package.checks`, `traning.conf`, `traning.tests.startup_checks.items`

- `F L19-L84` `run_startup_checks(config_path: Path | None=None, *, split: DataSplit='train', device: torch.device | None=None, require_cuda: bool | None=None) -> StartupCheckReport`：执行 `run startup checks` 对应逻辑。 调用：`check_core_entrypoints`, `check_data_input`, `check_runtime_device`, `check_settings_load`, `results.append`。
