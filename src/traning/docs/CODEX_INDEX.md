# traning Codex Index

> 自动生成文件，请勿手工修改。运行 `python project_index/build_index.py` 重建。

面向 Codex 的低 token 工程导航；先按阶段定位，再读取命中的源码。

## 调用分层

```text
main.py -> core/pipeline.py:TRAINING_STAGES
        -> core/data_input (配置映射、preflight、组装)
        -> Lib/data (发现、标签、采样、解码、分块、collate)
        -> state (run / experiment / checkpoint metadata)
```

## 六阶段入口

| key | Core 入口 | 当前状态 |
|---|---|---|
| `data_input` | `core/data_input/data_input.py` | 已实现并登记 |
| `spatial` | `core/spatial` | 目录边界已建立 |
| `candidate_cache` | `core/candidate_cache` | 目录边界已建立 |
| `temporal` | `core/temporal` | 目录边界已建立 |
| `evaluation` | `core/evaluation` | 目录边界已建立 |
| `export` | `core/export` | 目录边界已建立 |

快速查询：`python project_index/build_index.py --lookup 符号名`。

## 符号索引

覆盖 `57` 个 Python 文件、`159` 个命名函数/方法、`59` 个类。匿名 lambda 不单独列出。

图例：`F` 模块函数，`M` 方法，`N` 嵌套函数，`C` 类；`IO-R/IO-W` 文件读写，`DB` 数据库，`PROCESS` 外部进程。

## `src/traning/Lib/data/annotation.py`

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

## `src/traning/Lib/data/collate.py`

职责：组装图像批次并保留可变长度样本元数据。

- `F L8-L21` `collate_frame_samples(samples: list[dict[str, Any]]) -> dict[str, Any]`：执行 `collate frame samples` 对应逻辑。

## `src/traning/Lib/data/dataset.py`

职责：按片段帧索引解码原分辨率 RGB Tensor 和可变长度标签。
工程依赖：`traning.Lib.data.annotation`, `traning.Lib.data.models`, `traning.Lib.data.sampling`, `traning.Lib.data.video_reader`

- `C L14-L86` `SegmentFrameDataset(Dataset[dict[str, Any]])` [CLASS]：封装 `SegmentFrameDataset` 相关数据或行为。
- `M L15-L36` `SegmentFrameDataset.__init__(self, records: tuple[SegmentRecord, ...], *, sample_fps: float, frame_step: int=1, max_frames_per_segment: int | None=None, visibility_post_ms: float=100.0, normalize_images: bool=True)`：初始化实例依赖、配置和运行状态。 调用：`build_frame_references`。
- `M L38-L39` `SegmentFrameDataset.__len__(self) -> int`：执行 `len` 对应逻辑。
- `M L41-L44` `SegmentFrameDataset._video_reader(self) -> VideoReader`：执行 `video reader` 对应逻辑。 调用：`VideoReader`。
- `M L46-L81` `SegmentFrameDataset.__getitem__(self, index: int) -> dict[str, Any]`：执行 `getitem` 对应逻辑。 调用：`self._video_reader`, `self._video_reader.read_frame_at`, `visible_hit_objects`。
- `M L83-L86` `SegmentFrameDataset.__getstate__(self) -> dict[str, Any]`：执行 `getstate` 对应逻辑。

## `src/traning/Lib/data/discovery.py`

职责：发现 video.mp4 与 beatmap.json 配对并构建稳定片段记录。
工程依赖：`traning.Lib.data.annotation`, `traning.Lib.data.models`

- `F L9-L71` `discover_segments(dataset_root: Path, *, dimensions: tuple[str, ...]=(), categories: tuple[str, ...]=(), include_items: tuple[str, ...]=(), exclude_items: tuple[str, ...]=(), max_segments: int | None=None) -> DiscoveryResult`：执行 `discover segments` 对应逻辑。 调用：`DatasetIssue`, `DiscoveryResult`, `SegmentRecord`, `load_annotation`。

## `src/traning/Lib/data/models.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.Lib.data.annotation`

- `C L10-L18` `SegmentRecord` [CLASS]：封装 `SegmentRecord` 相关数据或行为。
- `C L22-L24` `DatasetIssue` [CLASS]：封装 `DatasetIssue` 相关数据或行为。
- `C L28-L30` `DiscoveryResult` [CLASS]：封装 `DiscoveryResult` 相关数据或行为。
- `C L34-L37` `FrameReference` [CLASS]：封装 `FrameReference` 相关数据或行为。

## `src/traning/Lib/data/sampling.py`

职责：根据片段时长、FPS 和步长建立帧引用表。
工程依赖：`traning.Lib.data.models`

- `F L8-L31` `build_frame_references(records: tuple[SegmentRecord, ...], *, sample_fps: float, frame_step: int, max_frames_per_segment: int | None) -> tuple[FrameReference, ...]`：构建并返回 `frame references` 对应的数据或结果。 调用：`FrameReference`。

## `src/traning/Lib/data/tiling.py`

职责：构建覆盖完整画面的重叠 patch 窗口并返回 Tensor 视图。

- `C L10-L22` `PatchWindow` [CLASS]：封装 `PatchWindow` 相关数据或行为。
- `M L17-L18` `PatchWindow.right(self) -> int` [PROPERTY]：执行 `right` 对应逻辑。
- `M L21-L22` `PatchWindow.bottom(self) -> int` [PROPERTY]：执行 `bottom` 对应逻辑。
- `F L25-L38` `_axis_starts(size: int, patch_size: int, overlap: int) -> tuple[int, ...]`：执行 `axis starts` 对应逻辑。
- `F L41-L61` `build_patch_windows(image_width: int, image_height: int, *, patch_width: int, patch_height: int, overlap_x: int, overlap_y: int) -> tuple[PatchWindow, ...]`：构建并返回 `patch windows` 对应的数据或结果。 调用：`PatchWindow`, `_axis_starts`。
- `F L64-L78` `iter_patches(image: Tensor, windows: tuple[PatchWindow, ...]) -> Iterator[tuple[PatchWindow, Tensor]]`：执行 `iter patches` 对应逻辑。

## `src/traning/Lib/data/video_reader.py`

职责：带有限打开文件缓存的 OpenCV 视频帧读取器。

- `C L10-L56` `VideoReader` [CLASS]：封装 `VideoReader` 相关数据或行为。
- `M L11-L15` `VideoReader.__init__(self, max_open_videos: int=4)`：初始化实例依赖、配置和运行状态。
- `M L17-L28` `VideoReader._capture(self, path: Path) -> cv2.VideoCapture`：执行 `capture` 对应逻辑。 调用：`self._captures.pop`, `self._captures.popitem`。
- `M L30-L36` `VideoReader.read_frame(self, path: Path, frame_index: int) -> np.ndarray` [IO-R]：读取 `frame` 对应的数据或结果。 调用：`self._capture`。
- `M L38-L48` `VideoReader.read_frame_at(self, path: Path, timestamp_ms: float) -> np.ndarray` [IO-R]：读取 `frame at` 对应的数据或结果。 调用：`self._capture`。
- `M L50-L53` `VideoReader.close(self) -> None`：执行 `close` 对应逻辑。 调用：`self._captures.clear`, `self._captures.values`。
- `M L55-L56` `VideoReader.__del__(self) -> None`：执行 `del` 对应逻辑。 调用：`self.close`。

## `src/traning/Lib/metrics/scoring.py`

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

## `src/traning/Lib/metrics/sequence.py`

职责：按点击时间模拟目标一次性命中、重叠目标递进、最小点击间隔限制和错误归因。
工程依赖：`traning.Lib.metrics.scoring`

- `C L35-L44` `SequenceScoreSpec` [CLASS]：封装 `SequenceScoreSpec` 相关数据或行为。
- `M L39-L44` `SequenceScoreSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
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
- `F L172-L178` `_spatial_passed(score: PointScore | SliderScore, spec: ScoreSpec) -> bool`：执行 `spatial passed` 对应逻辑。
- `F L181-L183` `_temporal_passed(score: PointScore | SliderScore, spec: ScoreSpec) -> bool`：执行 `temporal passed` 对应逻辑。
- `F L186-L189` `_spatial_error(score: PointScore | SliderScore) -> float`：执行 `spatial error` 对应逻辑。
- `F L192-L196` `_temporal_error_ms(target: TargetObject, click: PredictedClick) -> float`：执行 `temporal error ms` 对应逻辑。
- `F L199-L206` `_spatial_excess(score: PointScore | SliderScore, spec: ScoreSpec) -> float`：执行 `spatial excess` 对应逻辑。
- `F L209-L216` `_temporal_excess(score: PointScore | SliderScore, spec: ScoreSpec) -> float`：执行 `temporal excess` 对应逻辑。
- `F L219-L256` `_error_attribution(target: TargetObject, click: PredictedClick, score: PointScore | SliderScore, *, spec: ScoreSpec) -> tuple[ErrorDomain, tuple[ErrorTag, ...], float, float]`：执行 `error attribution` 对应逻辑。 调用：`_spatial_error`, `_spatial_excess`, `_spatial_passed`, `_temporal_error_ms`, `_temporal_excess`, `_temporal_passed`。
- `F L259-L280` `_best_scored_target(targets: tuple[TargetObject, ...], click: PredictedClick, *, circle_radius: float, spec: ScoreSpec) -> tuple[TargetObject, PointScore | SliderScore] | None`：执行 `best scored target` 对应逻辑。 调用：`_score_target`, `_score_value`。
- `F L283-L441` `score_click_sequence(targets: tuple[TargetObject, ...], clicks: tuple[PredictedClick, ...], *, circle_radius: float, spec: SequenceScoreSpec=SequenceScoreSpec()) -> SequenceScore`：执行 `score click sequence` 对应逻辑。 调用：`ClickEvaluation`, `SequenceScore`, `TargetResolution`, `_best_scored_target`, `_error_attribution`, `_score_target`。

## `src/traning/Lib/visualization/display.py`

职责：通过独立 ffplay 子进程把标注图片显示到主机 X11。

- `F L9-L50` `launch_image_window(image_path: Path, *, title: str, ffplay_binary: str='ffplay', display: str | None=None, previous_process: subprocess.Popen[bytes] | None=None) -> subprocess.Popen[bytes]`：执行 `launch image window` 对应逻辑。

## `src/traning/Lib/visualization/gallery.py`

职责：选择批次最高分 trial，并按通过状态和六个子项目随机保存标注帧图集。
工程依赖：`traning.Lib.data`, `traning.Lib.visualization.output_identity`, `traning.Lib.visualization.render`, `traning.state.gallery_schema`

- `F L34-L36` `_safe_name(value: str) -> str`：执行 `safe name` 对应逻辑。
- `F L39-L43` `_subproject_for_record(record: SegmentRecord) -> str`：执行 `subproject for record` 对应逻辑。
- `F L46-L58` `_frame_lookup(dataset: SegmentFrameDataset) -> dict[tuple[str, int], tuple[int, str]]`：执行 `frame lookup` 对应逻辑。 调用：`_subproject_for_record`。
- `F L61-L65` `_metric_lines(metrics: Mapping[str, float]) -> tuple[str, ...]`：执行 `metric lines` 对应逻辑。
- `F L68-L223` `save_best_trial_gallery(dataset: SegmentFrameDataset, request: BatchGalleryRequest, *, output_root: Path, samples_per_group: int=10) -> tuple[Path, int, tuple[str, ...]]` [IO-W]：执行 `save best trial gallery` 对应逻辑。 调用：`_frame_lookup`, `_metric_lines`, `_safe_name`, `allocate_output_identity`, `render_annotated_frame`, `save_annotated_frame`。

## `src/traning/Lib/visualization/models.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L18-L25` `VisualizationResult` [CLASS]：封装 `VisualizationResult` 相关数据或行为。
- `M L24-L25` `VisualizationResult.succeeded(self) -> bool` [PROPERTY]：执行 `succeeded` 对应逻辑。
- `C L29-L38` `GalleryResult` [CLASS]：封装 `GalleryResult` 相关数据或行为。
- `M L37-L38` `GalleryResult.succeeded(self) -> bool` [PROPERTY]：执行 `succeeded` 对应逻辑。
- `C L42-L47` `SelectedFrame` [CLASS]：封装 `SelectedFrame` 相关数据或行为。

## `src/traning/Lib/visualization/output_identity.py`

职责：为 traning_example 输出分配进程安全的递增次数和 UTC 时间标识。

- `C L14-L21` `OutputIdentity` [CLASS]：封装 `OutputIdentity` 相关数据或行为。
- `M L20-L21` `OutputIdentity.prefix(self) -> str` [PROPERTY]：执行 `prefix` 对应逻辑。
- `F L24-L28` `_read_counter(path: Path) -> int` [IO-R]：读取 `counter` 对应的数据或结果。
- `F L31-L37` `_existing_max_sequence(output_root: Path) -> int`：执行 `existing max sequence` 对应逻辑。
- `F L40-L63` `allocate_output_identity(output_root: Path) -> OutputIdentity` [IO-W]：执行 `allocate output identity` 对应逻辑。 调用：`OutputIdentity`, `_existing_max_sequence`, `_read_counter`。

## `src/traning/Lib/visualization/render.py`

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

## `src/traning/Lib/visualization/selection.py`

职责：根据 HitObject 起始时间反推最接近的采样帧。
工程依赖：`traning.Lib.data.dataset`, `traning.Lib.visualization.models`

- `F L7-L47` `select_click_frame(dataset: SegmentFrameDataset, *, segment_index: int, object_index: int=0) -> SelectedFrame`：选择 `click frame` 对应的数据或结果。 调用：`SelectedFrame`。

## `src/traning/conf/settings.py`

职责：训练配置模型与 YAML 加载；解析数据集路径、item 划分、点击频率上限并校验采样和分块参数。

- `C L26-L27` `SettingsError(Exception)` [CLASS]：封装 `SettingsError` 相关数据或行为。
- `C L30-L32` `RuntimeSettings(BaseModel)` [CLASS]：封装 `RuntimeSettings` 相关数据或行为。
- `C L35-L53` `LoaderSettings(BaseModel)` [CLASS]：封装 `LoaderSettings` 相关数据或行为。
- `M L43-L46` `LoaderSettings._positive_batch_size(cls, value: int) -> int` [VALIDATOR]：执行 `positive batch size` 对应逻辑。
- `M L50-L53` `LoaderSettings._nonnegative_workers(cls, value: int) -> int` [VALIDATOR]：执行 `nonnegative workers` 对应逻辑。
- `C L56-L64` `EvaluationSettings(BaseModel)` [CLASS]：封装 `EvaluationSettings` 相关数据或行为。
- `M L61-L64` `EvaluationSettings._nonnegative_interval(cls, value: float) -> float` [VALIDATOR]：执行 `nonnegative interval` 对应逻辑。
- `C L67-L82` `VisualizationSettings(BaseModel)` [CLASS]：封装 `VisualizationSettings` 相关数据或行为。
- `M L79-L82` `VisualizationSettings._positive_interval(cls, value: int) -> int` [VALIDATOR]：执行 `positive interval` 对应逻辑。
- `C L85-L162` `DataInputSettings(BaseModel)` [CLASS]：封装 `DataInputSettings` 相关数据或行为。
- `M L107-L110` `DataInputSettings._positive_fps(cls, value: float) -> float` [VALIDATOR]：执行 `positive fps` 对应逻辑。
- `M L114-L117` `DataInputSettings._positive_integer(cls, value: int) -> int` [VALIDATOR]：执行 `positive integer` 对应逻辑。
- `M L121-L124` `DataInputSettings._optional_positive_integer(cls, value: int | None) -> int | None` [VALIDATOR]：执行 `optional positive integer` 对应逻辑。
- `M L128-L131` `DataInputSettings._nonnegative_visibility(cls, value: float) -> float` [VALIDATOR]：执行 `nonnegative visibility` 对应逻辑。
- `M L142-L148` `DataInputSettings._unique_nonempty_strings(cls, value: tuple[str, ...]) -> tuple[str, ...]` [VALIDATOR]：执行 `unique nonempty strings` 对应逻辑。
- `M L151-L156` `DataInputSettings.validate_item_splits(self) -> DataInputSettings` [VALIDATOR]：校验 `item splits` 对应的数据或结果。
- `M L158-L162` `DataInputSettings.validate_tiling(self) -> None`：校验 `tiling` 对应的数据或结果。
- `C L165-L195` `Settings(BaseSettings)` [CLASS]：封装 `Settings` 相关数据或行为。
- `M L182-L195` `Settings.settings_customise_sources(cls, settings_cls: type[BaseSettings], init_settings: PydanticBaseSettingsSource, env_settings: PydanticBaseSettingsSource, dotenv_settings: PydanticBaseSettingsSource, file_secret_settings: PydanticBaseSettingsSource) -> tuple[PydanticBaseSettingsSource, ...]`：执行 `settings customise sources` 对应逻辑。
- `F L198-L207` `_read_config(config_path: Path) -> dict[str, Any]` [IO-R]：读取 `config` 对应的数据或结果。 调用：`SettingsError`。
- `F L210-L228` `_resolve_paths(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]`：解析并定位 `paths` 对应的数据或结果。
- `F L231-L239` `load_settings(config_path: Path | None=None) -> Settings`：加载 `settings` 对应的数据或结果。 调用：`Settings`, `SettingsError`, `_read_config`, `_resolve_paths`, `settings.data_input.validate_tiling`。

## `src/traning/core/data_input/data_input.py`

职责：数据输入模块公开门面；提供检查、Dataset 和 DataLoader。
工程依赖：`traning.conf`, `traning.core.data_input.loader`, `traning.core.data_input.preflight`

- `C L10-L26` `DataInputModule` [CLASS]：封装 `DataInputModule` 相关数据或行为。
- `M L11-L12` `DataInputModule.__init__(self, settings: Settings)`：初始化实例依赖、配置和运行状态。
- `M L14-L15` `DataInputModule.inspect(self, *, split: DataSplit='all') -> DataInputReport`：执行 `inspect` 对应逻辑。 调用：`inspect_data_input`。
- `M L17-L18` `DataInputModule.dataset(self, *, split: DataSplit='train')`：执行 `dataset` 对应逻辑。 调用：`build_dataset`。
- `M L20-L26` `DataInputModule.dataloader(self, *, split: DataSplit='train', shuffle: bool | None=None) -> DataLoader`：执行 `dataloader` 对应逻辑。 调用：`build_dataloader`。
- `F L29-L34` `check_data_input(settings: Settings | None=None, *, split: DataSplit='all') -> DataInputReport`：执行 `check data input` 对应逻辑。 调用：`DataInputModule`, `DataInputModule.inspect`, `load_settings`。

## `src/traning/core/data_input/loader.py`

职责：把配置映射为 SegmentFrameDataset 与 PyTorch DataLoader。
工程依赖：`traning.Lib.data`, `traning.conf`, `traning.core.data_input.preflight`

- `F L10-L32` `build_dataset(settings: Settings, *, split: DataSplit='train') -> SegmentFrameDataset`：构建并返回 `dataset` 对应的数据或结果。 调用：`SegmentFrameDataset`, `discover_data_input`。
- `F L35-L49` `build_dataloader(settings: Settings, *, split: DataSplit='train', shuffle: bool | None=None) -> DataLoader`：构建并返回 `dataloader` 对应的数据或结果。 调用：`build_dataset`。

## `src/traning/core/data_input/preflight.py`

职责：扫描训练片段并生成数量、类别、维度和问题报告。
工程依赖：`traning.Lib.data`, `traning.Lib.data.models`, `traning.conf`

- `C L13-L25` `DataInputReport` [CLASS]：封装 `DataInputReport` 相关数据或行为。
- `M L24-L25` `DataInputReport.ok(self) -> bool` [PROPERTY]：执行 `ok` 对应逻辑。
- `F L28-L34` `_combine_item_filters(base_items: tuple[str, ...], split_items: tuple[str, ...]) -> tuple[str, ...]`：执行 `combine item filters` 对应逻辑。
- `F L37-L42` `_split_items(config, split: DataSplit) -> tuple[str, ...]`：执行 `split items` 对应逻辑。
- `F L45-L69` `discover_data_input(settings: Settings, *, split: DataSplit='all') -> DiscoveryResult`：执行 `discover data input` 对应逻辑。 调用：`DatasetIssue`, `DiscoveryResult`, `_combine_item_filters`, `_split_items`, `discover_segments`。
- `F L72-L101` `inspect_data_input(settings: Settings, *, split: DataSplit='all') -> DataInputReport`：执行 `inspect data input` 对应逻辑。 调用：`DataInputReport`, `discover_data_input`。

## `src/traning/core/env_check.py`

职责：收集 Python、PyTorch/CUDA、GPU、FFmpeg 和关键依赖状态，供 CLI 环境检查使用。

- `C L13-L17` `PackageSpec` [CLASS]：封装 `PackageSpec` 相关数据或行为。
- `C L21-L24` `PackageCheck` [CLASS]：封装 `PackageCheck` 相关数据或行为。
- `C L28-L40` `TorchCheck` [CLASS]：封装 `TorchCheck` 相关数据或行为。
- `C L44-L68` `EnvironmentReport` [CLASS]：封装 `EnvironmentReport` 相关数据或行为。
- `M L54-L59` `EnvironmentReport.missing_required_packages(self) -> tuple[str, ...]` [PROPERTY]：执行 `missing required packages` 对应逻辑。
- `M L61-L68` `EnvironmentReport.ready(self, *, require_cuda: bool=False) -> bool`：执行 `ready` 对应逻辑。
- `F L106-L112` `_metadata_version(distributions: Iterable[str]) -> str | None`：执行 `metadata version` 对应逻辑。
- `F L115-L125` `check_package(spec: PackageSpec) -> PackageCheck`：执行 `check package` 对应逻辑。 调用：`PackageCheck`, `_metadata_version`。
- `F L128-L204` `collect_torch_check() -> TorchCheck`：执行 `collect torch check` 对应逻辑。 调用：`TorchCheck`, `_metadata_version`。
- `F L207-L219` `collect_environment_report() -> EnvironmentReport` [IO-W]：执行 `collect environment report` 对应逻辑。 调用：`EnvironmentReport`, `check_package`, `collect_torch_check`。

## `src/traning/core/pipeline.py`

职责：声明训练阶段注册表；当前登记 data_input，后续扩展空间、时序和导出阶段。
工程依赖：`traning.conf`, `traning.core.data_input`

- `C L11-L13` `TrainingStage` [CLASS]：封装 `TrainingStage` 相关数据或行为。
- `F L21-L23` `run_pipeline(settings: Settings | None=None) -> dict[str, object]`：执行 `run pipeline` 对应逻辑。 调用：`load_settings`, `stage.run`。

## `src/traning/core/visualization/preview.py`

职责：组装 Dataset、单帧点击标注和批次最佳参数图集。
工程依赖：`traning.Lib.visualization`, `traning.conf`, `traning.core.data_input`, `traning.core.visualization.service`, `traning.state`

- `F L16-L38` `visualize_click_label(settings: Settings, *, segment_index: int=0, object_index: int=0, output_path: Path | None=None, show_window: bool | None=None) -> VisualizationResult`：执行 `visualize click label` 对应逻辑。 调用：`OptionalTrainingVisualizer`, `build_dataset`, `select_click_frame`, `visualizer.visualize`。
- `F L41-L55` `save_annotation_gallery(settings: Settings, request: BatchGalleryRequest, *, output_root: Path | None=None, samples_per_group: int | None=None) -> GalleryResult`：执行 `save annotation gallery` 对应逻辑。 调用：`OptionalTrainingVisualizer`, `build_dataset`, `visualizer.save_gallery`。

## `src/traning/core/visualization/service.py`

职责：可选可视化故障隔离、一次性告警和训练步频率控制。
工程依赖：`traning.Lib.data`, `traning.Lib.visualization`, `traning.conf`, `traning.state`

- `C L22-L187` `OptionalTrainingVisualizer` [CLASS]：Best-effort visualization that never raises into training code。
- `M L25-L30` `OptionalTrainingVisualizer.__init__(self, settings: VisualizationSettings)`：初始化实例依赖、配置和运行状态。
- `M L32-L36` `OptionalTrainingVisualizer._warning_once(self, message: str) -> str | None`：执行 `warning once` 对应逻辑。
- `M L38-L107` `OptionalTrainingVisualizer.visualize(self, sample: dict[str, Any], *, target_source_index: int | None=None, output_path: Path | None=None, force: bool=False, show_window: bool | None=None) -> VisualizationResult`：执行 `visualize` 对应逻辑。 调用：`VisualizationResult`, `allocate_output_identity`, `launch_image_window`, `render_annotated_frame`, `save_annotated_frame`, `self._default_output_path`。
- `M L109-L128` `OptionalTrainingVisualizer.maybe_visualize_step(self, sample: dict[str, Any], *, global_step: int, target_source_index: int | None=None) -> VisualizationResult`：执行 `maybe visualize step` 对应逻辑。 调用：`VisualizationResult`, `self._warning_once`, `self.visualize`。
- `M L130-L175` `OptionalTrainingVisualizer.save_gallery(self, dataset: SegmentFrameDataset, request: BatchGalleryRequest, *, output_root: Path | None=None, samples_per_group: int | None=None) -> GalleryResult`：执行 `save gallery` 对应逻辑。 调用：`GalleryResult`, `save_best_trial_gallery`, `self._warning_once`。
- `M L177-L187` `OptionalTrainingVisualizer._default_output_path(self, sample: dict[str, Any], output_identity: OutputIdentity) -> Path` [IO-W]：执行 `default output path` 对应逻辑。

## `src/traning/main.py`

职责：Typer CLI；执行数据检查、样本预览和训练阶段注册表。
工程依赖：`traning.Lib.data`, `traning.conf`, `traning.core.data_input`, `traning.core.env_check`, `traning.core.pipeline`, `traning.core.visualization`, `traning.state`

- `F L29-L41` `_render_report(report) -> None`：执行 `render report` 对应逻辑。
- `F L44-L47` `_format_bool(value: bool | None) -> str`：执行 `format bool` 对应逻辑。
- `F L50-L53` `_format_gib(value: float | None) -> str`：执行 `format gib` 对应逻辑。
- `F L56-L91` `_render_env_report(report) -> None`：执行 `render env report` 对应逻辑。 调用：`_format_bool`, `_format_gib`。
- `F L95-L103` `data_check(config: Path | None=typer.Option(None, '--config'), split: DataSplit=typer.Option('all', '--split')) -> None` [CLI]：执行 `data check` 对应逻辑。 调用：`_render_report`, `inspect_data_input`, `load_settings`。
- `F L107-L126` `env_check(strict: bool=typer.Option(False, '--strict/--no-strict', help='Exit non-zero when required runtime dependencies are missing.'), require_cuda: bool=typer.Option(False, '--require-cuda/--no-require-cuda', help='Treat CUDA unavailability as a failure in strict mode.')) -> None` [CLI]：执行 `env check` 对应逻辑。 调用：`_render_env_report`, `collect_environment_report`, `report.ready`。
- `F L130-L162` `data_preview(index: int=typer.Option(0, '--index', min=0), split: DataSplit=typer.Option('train', '--split'), config: Path | None=typer.Option(None, '--config')) -> None` [CLI]：执行 `data preview` 对应逻辑。 调用：`build_dataset`, `build_patch_windows`, `load_settings`。
- `F L166-L170` `run(config: Path | None=typer.Option(None, '--config')) -> None` [CLI]：执行该处理器的完整工作流。 调用：`_render_report`, `load_settings`, `run_pipeline`。
- `F L174-L193` `visualize_label(segment_index: int=typer.Option(0, '--segment-index', min=0), object_index: int=typer.Option(0, '--object-index', min=0), output: Path | None=typer.Option(None, '--output'), show: bool=typer.Option(False, '--show/--no-show'), config: Path | None=typer.Option(None, '--config')) -> None` [CLI]：执行 `visualize label` 对应逻辑。 调用：`load_settings`, `visualize_click_label`。
- `F L197-L231` `save_gallery(results: Path=typer.Option(..., '--results', exists=True, file_okay=True, dir_okay=False, readable=True), output_root: Path | None=typer.Option(None, '--output-root'), samples_per_group: int | None=typer.Option(None, '--samples-per-group', min=1), config: Path | None=typer.Option(None, '--config')) -> None` [CLI]：执行 `save gallery` 对应逻辑。 调用：`load_batch_gallery_request`, `load_settings`, `save_annotation_gallery`。

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

## `src/traning/tests/test_discovery.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.Lib.data`

- `F L11-L36` `_write_segment(root: Path, item_name: str, segment_id: str) -> None` [IO-W]：写入 `segment` 对应的数据或结果。
- `C L39-L60` `DiscoverySplitTests(unittest.TestCase)` [CLASS]：封装 `DiscoverySplitTests` 相关数据或行为。
- `M L40-L49` `DiscoverySplitTests.test_include_items_filters_records_before_loading(self) -> None`：执行 `test include items filters records before loading` 对应逻辑。 调用：`_write_segment`, `discover_segments`, `self.assertEqual`。
- `M L51-L60` `DiscoverySplitTests.test_exclude_items_removes_records(self) -> None`：执行 `test exclude items removes records` 对应逻辑。 调用：`_write_segment`, `discover_segments`, `self.assertEqual`。

## `src/traning/tests/test_env_check.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.core.env_check`

- `C L11-L24` `EnvironmentCheckTests(unittest.TestCase)` [CLASS]：封装 `EnvironmentCheckTests` 相关数据或行为。
- `M L12-L17` `EnvironmentCheckTests.test_collect_environment_report_is_non_destructive(self) -> None`：执行 `test collect environment report is non destructive` 对应逻辑。 调用：`collect_environment_report`, `self.assertIsNotNone`, `self.assertTrue`。
- `M L19-L24` `EnvironmentCheckTests.test_required_package_specs_are_reported(self) -> None`：执行 `test required package specs are reported` 对应逻辑。 调用：`collect_environment_report`, `self.assertTrue`。

## `src/traning/tests/test_gallery_schema.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.state`

- `C L10-L54` `BatchGalleryRequestTests(unittest.TestCase)` [CLASS]：封装 `BatchGalleryRequestTests` 相关数据或行为。
- `M L11-L32` `BatchGalleryRequestTests.test_frame_evaluation_accepts_error_attribution(self) -> None`：执行 `test frame evaluation accepts error attribution` 对应逻辑。 调用：`self.assertEqual`。
- `M L34-L54` `BatchGalleryRequestTests.test_trials_must_share_score_version(self) -> None`：执行 `test trials must share score version` 对应逻辑。 调用：`self.assertRaises`。

## `src/traning/tests/test_scoring.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.Lib.metrics`

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
工程依赖：`traning.Lib.metrics`

- `C L13-L155` `ClickSequenceScoringTests(unittest.TestCase)` [CLASS]：封装 `ClickSequenceScoringTests` 相关数据或行为。
- `M L14-L41` `ClickSequenceScoringTests.test_first_passing_hit_resolves_target_once(self) -> None`：执行 `test first passing hit resolves target once` 对应逻辑。 调用：`PredictedClick`, `TargetObject`, `score_click_sequence`, `self.assertEqual`。
- `M L43-L64` `ClickSequenceScoringTests.test_failed_hit_keeps_target_active_for_later_click(self) -> None`：执行 `test failed hit keeps target active for later click` 对应逻辑。 调用：`PredictedClick`, `TargetObject`, `score_click_sequence`, `self.assertEqual`, `self.assertIn`。
- `M L66-L84` `ClickSequenceScoringTests.test_early_click_is_attributed_to_temporal_parameters(self) -> None`：执行 `test early click is attributed to temporal parameters` 对应逻辑。 调用：`PredictedClick`, `TargetObject`, `score_click_sequence`, `self.assertEqual`。
- `M L86-L117` `ClickSequenceScoringTests.test_overlapping_targets_resolve_by_earliest_active_target(self) -> None`：执行 `test overlapping targets resolve by earliest active target` 对应逻辑。 调用：`PredictedClick`, `TargetObject`, `score_click_sequence`, `self.assertEqual`。
- `M L119-L155` `ClickSequenceScoringTests.test_click_frequency_limit_blocks_high_rate_hits(self) -> None`：执行 `test click frequency limit blocks high rate hits` 对应逻辑。 调用：`PredictedClick`, `SequenceScoreSpec`, `TargetObject`, `score_click_sequence`, `self.assertEqual`, `self.assertTrue`。
