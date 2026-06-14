# Function Locations

> 自动生成文件，请勿手工修改。运行 `python project_index/build_index.py` 重建。

快速位置表：`380` 个命名函数/方法，附带 `72` 个类定义。
格式为 `起止行  类型  限定名`；路径按模块分组。

快速搜索：`rg -n "符号名" project_index/FUNCTION_LOCATIONS.md`。

## `src/before_traning/Lib/beatmap/folder_store.py`

- `18-242` `C` `BeatmapFolderStore`
- `26-40` `M` `BeatmapFolderStore.__init__`
- `42-50` `M` `BeatmapFolderStore._normalize_folder_name`
- `52-53` `M` `BeatmapFolderStore._registered_names`
- `55-57` `M` `BeatmapFolderStore.is_registered`
- `59-63` `M` `BeatmapFolderStore._assert_registered`
- `65-68` `M` `BeatmapFolderStore.get_folder_path`
- `70-72` `M` `BeatmapFolderStore.folder_exists`
- `74-80` `M` `BeatmapFolderStore._require_existing_folder`
- `82-85` `M` `BeatmapFolderStore.find_files`
- `87-88` `M` `BeatmapFolderStore.find_osu_files`
- `90-95` `M` `BeatmapFolderStore.get_file_path`
- `97-98` `M` `BeatmapFolderStore.file_exists`
- `100-136` `M` `BeatmapFolderStore.write_text`
- `138-155` `M` `BeatmapFolderStore.write_lines`
- `157-168` `M` `BeatmapFolderStore.append_line`
- `170-174` `M` `BeatmapFolderStore.read_text`
- `176-187` `M` `BeatmapFolderStore.create_output_directory`
- `189-207` `M` `BeatmapFolderStore.recover_atomic_outputs`
- `210-242` `M` `BeatmapFolderStore.atomic_output_folder`

## `src/before_traning/Lib/beatmap/hit_objects.py`

- `8-11` `C` `HitObject`
- `13-17` `C` `Circle`
- `16-17` `M` `Circle.__post_init__`
- `19-25` `C` `Slider`
- `24-25` `M` `Slider.__post_init__`
- `27-29` `C` `Spinner`
- `28-29` `M` `Spinner.__post_init__`

## `src/before_traning/Lib/beatmap/manifest.py`

- `28-32` `C` `ManifestEntry`
- `35-393` `C` `PackageManifest`
- `38-56` `M` `PackageManifest.__init__`
- `58-68` `M` `PackageManifest._ensure_schema`
- `70-73` `M` `PackageManifest._all_items`
- `75-81` `M` `PackageManifest._normalize_source_name`
- `83-91` `M` `PackageManifest._next_folder_number`
- `93-94` `M` `PackageManifest._folder_name`
- `96-101` `M` `PackageManifest._legacy_osu_filename`
- `103-152` `M` `PackageManifest._rename_legacy_folders`
- `154-167` `M` `PackageManifest._restore_legacy_folders`
- `169-211` `M` `PackageManifest._migrate_legacy_order`
- `213-234` `M` `PackageManifest._migrate_legacy_difficulty_files`
- `236-248` `M` `PackageManifest.export_table`
- `250-285` `M` `PackageManifest.replace`
- `287-294` `M` `PackageManifest.read_folder_names`
- `296-297` `M` `PackageManifest.read_all_folder_names`
- `299-305` `M` `PackageManifest.source_name_for`
- `307-320` `M` `PackageManifest.set_difficulty`
- `322-329` `M` `PackageManifest.difficulty_for`
- `331-367` `M` `PackageManifest.save_beatmap_data`
- `369-390` `M` `PackageManifest.beatmap_data_for`
- `392-393` `M` `PackageManifest.is_active`
- `396-414` `C` `ManifestFolderWalker`
- `397-408` `M` `ManifestFolderWalker.__init__`
- `410-411` `M` `ManifestFolderWalker.read_folder_names`
- `413-414` `M` `ManifestFolderWalker.source_name_for`

## `src/before_traning/Lib/beatmap/osu_metadata.py`

- `6-31` `F` `read_section_key`
- `34-35` `F` `read_audio_filename`
- `38-39` `F` `read_overall_difficulty`

## `src/before_traning/Lib/beatmap/osu_parser.py`

- `10-212` `C` `VerifyOsuParser`
- `11-35` `M` `VerifyOsuParser.parse_sections`
- `37-44` `M` `VerifyOsuParser.parse_key_value_section`
- `46-68` `M` `VerifyOsuParser.parse_timing_points`
- `70-97` `M` `VerifyOsuParser.get_effective_timing`
- `99-170` `M` `VerifyOsuParser.parse_hitobjects`
- `172-188` `M` `VerifyOsuParser.objects_to_lines`
- `190-212` `M` `VerifyOsuParser.hit_object_to_dict`

## `src/before_traning/Lib/beatmap/osz.py`

- `12-20` `C` `OsuEntry`
- `23-66` `F` `read_osz_entry`

## `src/before_traning/Lib/beatmap/package.py`

- `12-98` `C` `PackageUpdater`
- `20-39` `M` `PackageUpdater.__init__`
- `41-42` `M` `PackageUpdater.load_manifest_folder_names`
- `44-45` `M` `PackageUpdater.load_registered_names`
- `47-48` `M` `PackageUpdater.replace_manifest`
- `50-51` `M` `PackageUpdater.is_registered`
- `53-68` `M` `PackageUpdater.create_folder_if_registered`
- `70-80` `M` `PackageUpdater.sync_folders_from_manifest`
- `82-98` `M` `PackageUpdater.find_unregistered_existing_folders`

## `src/before_traning/Lib/beatmap/standard.py`

- `21-33` `C` `ParsedStandardBeatmap`
- `32-33` `M` `ParsedStandardBeatmap.approach_preempt_ms`
- `36-43` `F` `approach_preempt_ms`
- `46-90` `F` `parse_standard_beatmap`
- `93-97` `F` `parse_standard_hit_objects`
- `100-116` `F` `_beatmap_to_payload`
- `119-151` `F` `_hit_object_from_payload`
- `154-173` `F` `_beatmap_from_payload`
- `176-207` `F` `load_standard_beatmap`

## `src/before_traning/Lib/beatmap/timing_points.py`

- `7-15` `C` `OsuOriginalTimingPoint`

## `src/before_traning/Lib/common/batch.py`

- `36-39` `C` `ConfigValueSpec`
- `46-54` `F` `_normalize_config_path`
- `57-70` `F` `_normalize_config_path_group`
- `73-81` `F` `_config_values`
- `84-85` `F` `config_resolved_paths`
- `88-89` `F` `config_filenames`
- `92-93` `F` `config_nonempty_strs`
- `96-97` `F` `config_bools`
- `100-101` `F` `config_string_tuples`
- `104-105` `F` `config_suffix_tuples`
- `108-109` `F` `config_positive_ints`
- `112-113` `F` `config_nonnegative_ints`
- `116-117` `F` `config_positive_floats`
- `120-121` `F` `config_floats`
- `124-133` `F` `merge_config_specs`
- `136-147` `F` `prefix_config_keys`
- `150-165` `F` `read_config_values`
- `168-170` `C` `FolderWalkerLike`
- `169-170` `M` `FolderWalkerLike.read_folder_names`
- `173-244` `C` `FolderBatchProcessor`
- `178-181` `M` `FolderBatchProcessor.__init__`
- `183-189` `M` `FolderBatchProcessor.progress_message`
- `191-192` `M` `FolderBatchProcessor.iter_folder_names`
- `194-195` `M` `FolderBatchProcessor.handle_failure`
- `198-203` `M` `FolderBatchProcessor.process_one`
- `205-216` `M` `FolderBatchProcessor._record_result`
- `218-222` `M` `FolderBatchProcessor._print_summary`
- `224-244` `M` `FolderBatchProcessor.run`

## `src/before_traning/Lib/common/failures.py`

- `11-21` `F` `_error_traceback`
- `24-29` `F` `callable_location`
- `32-42` `F` `exception_location`
- `45-59` `F` `failure_detail`
- `62-70` `F` `exception_detail`
- `73-77` `F` `format_failure`
- `80-81` `F` `format_exception`

## `src/before_traning/Lib/common/pathspec.py`

- `9-15` `F` `suffix_pattern`
- `18-19` `F` `suffix_patterns`
- `22-23` `F` `gitwildmatch_spec`
- `26-27` `F` `suffix_spec`
- `30-31` `F` `matches_name`
- `34-35` `F` `filter_files`

## `src/before_traning/Lib/common/processing.py`

- `16-24` `C` `FolderStoreLike`
- `17-18` `M` `FolderStoreLike.folder_exists`
- `20-21` `M` `FolderStoreLike.file_exists`
- `23-24` `M` `FolderStoreLike.find_files`
- `27-50` `C` `StatusManagerLike`
- `30-31` `M` `StatusManagerLike.ensure_status_file`
- `33-34` `M` `StatusManagerLike.is_step_done`
- `36-42` `M` `StatusManagerLike.mark_step_done`
- `44-50` `M` `StatusManagerLike.mark_step_pending`
- `53-62` `F` `matching_files`
- `66-174` `C` `ProcessingGuard`
- `72-79` `M` `ProcessingGuard.__post_init__`
- `81-98` `M` `ProcessingGuard.prepare_folder`
- `100-109` `M` `ProcessingGuard.ensure_required_steps`
- `111-119` `M` `ProcessingGuard.output_files_exist`
- `121-137` `M` `ProcessingGuard.is_complete`
- `139-143` `M` `ProcessingGuard.step_done`
- `145-157` `M` `ProcessingGuard.reconcile_existing`
- `159-164` `M` `ProcessingGuard.mark_done`
- `166-174` `M` `ProcessingGuard.record_failure`

## `src/before_traning/Lib/common/sequence.py`

- `4-17` `F` `format_sequence_name`

## `src/before_traning/Lib/tasks/flows.py`

- `18-125` `C` `TaskPipeline`
- `19-34` `M` `TaskPipeline.__init__`
- `36-50` `M` `TaskPipeline._call_stage`
- `52-74` `M` `TaskPipeline._run`
- `76-87` `M` `TaskPipeline._run_prefect`
- `89-100` `M` `TaskPipeline.run_prefect`
- `102-114` `M` `TaskPipeline.run_direct`
- `116-125` `M` `TaskPipeline.__call__`
- `128-140` `F` `build_task_pipeline`

## `src/before_traning/Lib/tasks/tasks.py`

- `13-16` `F` `require_success`
- `20-32` `C` `TaskSpec`
- `28-32` `M` `TaskSpec.default_enabled`
- `36-38` `C` `RegisteredTask`
- `41-50` `F` `_build_prefect_task`
- `42-43` `N` `_build_prefect_task.run_registered_task`
- `53-113` `C` `TaskRegistry`
- `54-76` `M` `TaskRegistry.__init__`
- `79-80` `M` `TaskRegistry.registered`
- `82-113` `M` `TaskRegistry.select`

## `src/before_traning/Lib/tools/ffmpeg.py`

- `42-46` `F` `_command_error_text`
- `49-55` `F` `_run_command`
- `58-61` `F` `run_ffmpeg`
- `64-86` `F` `build_extract_wav_args`
- `89-104` `F` `extract_wav`
- `107-126` `F` `build_trim_video_args`
- `129-148` `F` `trim_video`
- `151-170` `F` `build_segment_video_args`
- `173-192` `F` `segment_video`
- `195-214` `F` `build_crop_video_args`
- `217-236` `F` `crop_video`
- `239-256` `F` `run_ffprobe_json`
- `259-281` `F` `get_audio_stream_start_time`
- `284-304` `F` `get_media_duration_seconds`
- `307-328` `F` `get_video_size`

## `src/before_traning/Lib/video/av_processing/steps.py`

- `13-295` `C` `AVCoreStepsMixin`
- `14-20` `M` `AVCoreStepsMixin._extract_audio_to_wav`
- `22-42` `M` `AVCoreStepsMixin._load_wav_samples`
- `44-50` `M` `AVCoreStepsMixin._normalize_series`
- `52-77` `M` `AVCoreStepsMixin._build_feature_series`
- `79-91` `M` `AVCoreStepsMixin._lowpass_samples`
- `93-99` `M` `AVCoreStepsMixin._build_music_refine_series`
- `101-128` `M` `AVCoreStepsMixin._estimate_best_start_frame`
- `130-176` `M` `AVCoreStepsMixin._estimate_offset_seconds`
- `178-191` `M` `AVCoreStepsMixin._parse_verify_hit_times_ms`
- `193-206` `M` `AVCoreStepsMixin._build_verify_click_train`
- `208-257` `M` `AVCoreStepsMixin._estimate_verify_adjustment_seconds`
- `259-281` `M` `AVCoreStepsMixin._validate_trim_window`
- `283-295` `M` `AVCoreStepsMixin._trim_video`

## `src/before_traning/Lib/video/clipping/geometry.py`

- `8-95` `C` `ClipGeometryMixin`
- `9-10` `M` `ClipGeometryMixin.get_video_size`
- `12-18` `M` `ClipGeometryMixin._scale_crop_coordinate`
- `20-75` `M` `ClipGeometryMixin._resolve_scaled_crop`
- `77-92` `M` `ClipGeometryMixin._validate_crop_bounds`
- `94-95` `M` `ClipGeometryMixin.describe_crop_for_video`

## `src/before_traning/Lib/video/segment_dataset.py`

- `67-192` `C` `SegmentDatasetManifest`
- `68-80` `M` `SegmentDatasetManifest.__init__`
- `82-89` `M` `SegmentDatasetManifest._records`
- `91-95` `M` `SegmentDatasetManifest.read_rows`
- `97-129` `M` `SegmentDatasetManifest.replace_folder`
- `131-145` `M` `SegmentDatasetManifest.write_table`
- `147-152` `M` `SegmentDatasetManifest.export_table`
- `154-174` `M` `SegmentDatasetManifest.import_existing_table`
- `176-192` `M` `SegmentDatasetManifest.output_complete`
- `195-199` `F` `write_json_file`

## `src/before_traning/Lib/video/segmentation/planner.py`

- `40-74` `C` `SegmentPlan`
- `57-58` `M` `SegmentPlan.duration_seconds`
- `61-62` `M` `SegmentPlan.pre_context_seconds`
- `65-66` `M` `SegmentPlan.post_context_seconds`
- `69-70` `M` `SegmentPlan.clip_start_ms`
- `73-74` `M` `SegmentPlan.clip_end_ms`
- `77-89` `F` `circle_radius_from_size`
- `92-103` `F` `circle_overlap_ratio`
- `106-125` `F` `_slider_polyline`
- `128-135` `F` `_object_polyline`
- `138-156` `F` `_point_to_segment_distance`
- `159-167` `F` `_orientation`
- `170-192` `F` `_segments_intersect`
- `195-247` `F` `_polyline_distance`
- `250-260` `F` `hit_objects_overlap_ratio`
- `263-332` `F` `group_hit_objects`
- `335-355` `F` `classify_hit_group`
- `358-411` `F` `_build_plan`
- `414-473` `F` `build_segment_plans`
- `476-589` `F` `build_long_sequence_plans`
- `506-527` `N` `build_long_sequence_plans.combined_plan`
- `529-548` `N` `build_long_sequence_plans.flush`

## `src/before_traning/Lib/video/segmentation/segmentation.py`

- `14-20` `C` `SegmentPlanCollection`
- `19-20` `M` `SegmentPlanCollection.all`
- `23-65` `F` `plan_video_segments`

## `src/before_traning/conf/field_groups.py`

- `72-73` `F` `group_values`
- `76-78` `F` `assign_group`
- `81-82` `F` `forward_kwargs`

## `src/before_traning/conf/legacy_config.py`

- `20-21` `C` `CheckDataConfigError`
- `24-37` `C` `ConfigReader`
- `27-31` `M` `ConfigReader.__init__`
- `33-34` `M` `ConfigReader.get`
- `36-37` `M` `ConfigReader.read`
- `52-53` `F` `load_config`
- `56-57` `F` `load_process_steps_config`
- `60-69` `F` `load_process_steps_config_or_default`
- `72-119` `F` `settings_kwargs`
- `122-139` `F` `_coerce_like`
- `142-153` `F` `settings_namespace`
- `156-157` `F` `_settings_kwargs`
- `160-174` `F` `_filter_builder_kwargs`
- `177-183` `F` `build_from_config`
- `186-200` `F` `build_from_config_or_default`

## `src/before_traning/conf/runtime.py`

- `9-12` `F` `ensure_prefect_home`

## `src/before_traning/conf/settings.py`

- `16-17` `C` `SettingsError`
- `20-22` `C` `RuntimeSettings`
- `25-28` `C` `CheckDataSettings`
- `31-44` `C` `VideoClipSettings`
- `41-44` `M` `VideoClipSettings._finite_offset`
- `47-54` `C` `FileManagementSettings`
- `57-59` `C` `FileFormatSettings`
- `62-68` `C` `AVSettings`
- `71-73` `C` `AudioMatchSettings`
- `76-84` `C` `PackageSettings`
- `87-95` `C` `ClipSettings`
- `98-158` `C` `SegmentSettings`
- `112-117` `M` `SegmentSettings._nonnegative_interval`
- `121-124` `M` `SegmentSettings._long_sequence_object_limit`
- `128-131` `M` `SegmentSettings._approach_ratio`
- `135-138` `M` `SegmentSettings._overlap_ratio`
- `144-149` `M` `SegmentSettings._nonnegative_context`
- `153-158` `M` `SegmentSettings._positive_duration`
- `161-173` `C` `ProgressSettings`
- `176-209` `C` `Settings`
- `196-197` `M` `Settings.target_root`
- `200-201` `M` `Settings.overwrite`
- `204-205` `M` `Settings.continue_on_error`
- `208-209` `M` `Settings.global_offset_ms`
- `212-222` `F` `_resolve_paths`
- `225-260` `F` `_extract_nested`
- `263-280` `F` `_read_config`
- `283-289` `F` `load_settings`

## `src/before_traning/core/audio/matching/matching.py`

- `30-31` `F` `_load_audio_match_experiment_config`
- `34-42` `F` `build_audio_match_experiment_from_config_or_default`
- `45-85` `C` `AudioMatchExperiment`
- `50-85` `M` `AudioMatchExperiment.__init__`
- `88-89` `C` `AudioMatchProcessor`
- `92-94` `F` `main`

## `src/before_traning/core/audio/matching/preflight.py`

- `9-97` `C` `AudioMatchPreflightMixin`
- `10-15` `M` `AudioMatchPreflightMixin._folder_has_video`
- `17-45` `M` `AudioMatchPreflightMixin._sync_video_matched_status`
- `47-62` `M` `AudioMatchPreflightMixin._pending_folder_names`
- `64-74` `M` `AudioMatchPreflightMixin._candidate_folder_names`
- `76-97` `M` `AudioMatchPreflightMixin._candidate_videos`

## `src/before_traning/core/audio/matching/steps.py`

- `12-194` `C` `AudioMatchStepsMixin`
- `13-17` `M` `AudioMatchStepsMixin._extract_samples`
- `19-32` `M` `AudioMatchStepsMixin._build_alignment_features`
- `34-69` `M` `AudioMatchStepsMixin._estimate_offset_from_features`
- `71-79` `M` `AudioMatchStepsMixin._result_sort_key`
- `81-173` `M` `AudioMatchStepsMixin._score_pairs`
- `175-194` `M` `AudioMatchStepsMixin._select_greedy_matches`

## `src/before_traning/core/audio/matching/wrapup.py`

- `9-160` `C` `AudioMatchWrapUpMixin`
- `10-26` `M` `AudioMatchWrapUpMixin._print_greedy_matches`
- `28-133` `M` `AudioMatchWrapUpMixin._apply_matches`
- `135-160` `M` `AudioMatchWrapUpMixin.run`

## `src/before_traning/core/beatmap/beatmap.py`

- `21-22` `F` `run_beatmap`

## `src/before_traning/core/beatmap/difficulty.py`

- `18-20` `C` `DifficultyEntry`
- `23-115` `C` `BeatmapDifficultyProcessor`
- `24-43` `M` `BeatmapDifficultyProcessor.__init__`
- `45-50` `M` `BeatmapDifficultyProcessor.write_difficulty`
- `52-56` `M` `BeatmapDifficultyProcessor.read_difficulty`
- `58-90` `M` `BeatmapDifficultyProcessor.process_one`
- `92-93` `M` `BeatmapDifficultyProcessor.handle_failure`
- `95-115` `M` `BeatmapDifficultyProcessor.list_difficulties`
- `121-131` `F` `export_difficulty`

## `src/before_traning/core/beatmap/importer.py`

- `21-183` `C` `BeatmapImportProcessor`
- `22-52` `M` `BeatmapImportProcessor.__init__`
- `54-59` `M` `BeatmapImportProcessor._scan_single_osz`
- `61-111` `M` `BeatmapImportProcessor._scan_entries`
- `113-127` `M` `BeatmapImportProcessor._rebuild_manifest`
- `129-161` `M` `BeatmapImportProcessor._write_entries`
- `163-183` `M` `BeatmapImportProcessor.run`
- `189-192` `F` `build_beatmap_import_processor_from_config_or_default`
- `195-198` `F` `build_osu_osz_processor_from_config_or_default`
- `201-206` `F` `import_beatmaps`

## `src/before_traning/core/beatmap/pipeline.py`

- `108-112` `F` `prepare_beatmaps`
- `115-163` `C` `TemporaryTrainingRunner`
- `116-117` `M` `TemporaryTrainingRunner.__init__`
- `119-163` `M` `TemporaryTrainingRunner.run`

## `src/before_traning/core/beatmap/verify.py`

- `17-21` `F` `build_store`
- `24-87` `C` `BeatmapVerifyExporter`
- `25-45` `M` `BeatmapVerifyExporter.__init__`
- `47-84` `M` `BeatmapVerifyExporter.process_one`
- `86-87` `M` `BeatmapVerifyExporter.handle_failure`
- `93-97` `F` `build_verify_exporter_from_config_or_default`
- `100-103` `F` `build_beatmap_verify_exporter_from_config_or_default`
- `106-113` `F` `export_verify`

## `src/before_traning/core/video/__init__.py`

- `8-11` `F` `prepare_videos`

## `src/before_traning/core/video/av.py`

- `13-32` `F` `av_correspondence`

## `src/before_traning/core/video/av_processing/av_processing.py`

- `28-29` `F` `_load_av_correspondence_processor_config`
- `32-40` `F` `build_av_correspondence_processor_from_config_or_default`
- `43-86` `C` `AVCorrespondenceProcessor`
- `50-86` `M` `AVCorrespondenceProcessor.__init__`
- `89-90` `C` `VideoAVProcessor`
- `93-95` `F` `main`

## `src/before_traning/core/video/av_processing/preflight.py`

- `11-109` `C` `AVPreflightMixin`
- `12-28` `M` `AVPreflightMixin._validate_config`
- `30-39` `M` `AVPreflightMixin._ensure_status_steps_registered`
- `41-59` `M` `AVPreflightMixin._resolve_source_video_path`
- `61-65` `M` `AVPreflightMixin._resolve_song_audio_path`
- `67-68` `M` `AVPreflightMixin._resolve_verify_path`
- `70-97` `M` `AVPreflightMixin._sync_output_status`
- `99-109` `M` `AVPreflightMixin._ensure_required_steps_done`

## `src/before_traning/core/video/av_processing/steps.py`

- `9-137` `C` `AVProcessStepsMixin`
- `10-137` `M` `AVProcessStepsMixin.process_one`

## `src/before_traning/core/video/av_processing/wrapup.py`

- `6-74` `C` `AVWrapUpMixin`
- `7-11` `M` `AVWrapUpMixin._update_progress`
- `13-14` `M` `AVWrapUpMixin.progress_message`
- `16-58` `M` `AVWrapUpMixin._build_done_detail`
- `60-65` `M` `AVWrapUpMixin._mark_done`
- `67-74` `M` `AVWrapUpMixin.handle_failure`

## `src/before_traning/core/video/clip.py`

- `11-16` `F` `crop_video`

## `src/before_traning/core/video/clipping/clipping.py`

- `24-36` `F` `build_fixed_region_video_crop_processor_from_config_or_default`
- `39-112` `C` `FixedRegionVideoCropProcessor`
- `47-52` `M` `FixedRegionVideoCropProcessor.from_settings`
- `54-112` `M` `FixedRegionVideoCropProcessor.__init__`
- `115-116` `C` `VideoClipProcessor`
- `119-121` `F` `main`

## `src/before_traning/core/video/clipping/preflight.py`

- `4-32` `C` `ClipPreflightMixin`
- `5-14` `M` `ClipPreflightMixin._ensure_status_steps_registered`
- `16-32` `M` `ClipPreflightMixin._ensure_folder_ready`

## `src/before_traning/core/video/clipping/steps.py`

- `9-45` `C` `ClipStepsMixin`
- `10-27` `M` `ClipStepsMixin._crop_video_in_place`
- `29-45` `M` `ClipStepsMixin.process_one`

## `src/before_traning/core/video/clipping/wrapup.py`

- `8-71` `C` `ClipWrapUpMixin`
- `9-10` `M` `ClipWrapUpMixin.progress_message`
- `12-20` `M` `ClipWrapUpMixin._reference_detail`
- `22-37` `M` `ClipWrapUpMixin._mark_cropping`
- `39-58` `M` `ClipWrapUpMixin._mark_done`
- `60-71` `M` `ClipWrapUpMixin.handle_failure`

## `src/before_traning/core/video/match.py`

- `11-33` `F` `match_videos`

## `src/before_traning/core/video/matching/builders.py`

- `14-15` `F` `_load_video_package_renamer_config`
- `18-26` `F` `build_video_package_renamer_from_config_or_default`

## `src/before_traning/core/video/matching/matching.py`

- `9-39` `C` `VideoMatchProcessor`
- `12-23` `M` `VideoMatchProcessor.__init__`
- `25-39` `M` `VideoMatchProcessor.run`

## `src/before_traning/core/video/matching/renamer.py`

- `22-187` `C` `VideoPackageRenamer`
- `23-46` `M` `VideoPackageRenamer.__init__`
- `48-52` `M` `VideoPackageRenamer._folder_has_video`
- `54-69` `M` `VideoPackageRenamer._parse_video_time`
- `71-82` `M` `VideoPackageRenamer._list_videos_in_time_order`
- `84-115` `M` `VideoPackageRenamer._pending_folder_names`
- `117-142` `M` `VideoPackageRenamer._build_rename_plan`
- `144-187` `M` `VideoPackageRenamer.run`
- `190-191` `C` `VideoMatchRenamer`

## `src/before_traning/core/video/pipeline.py`

- `10-14` `F` `prepare_videos`

## `src/before_traning/core/video/segment.py`

- `50-57` `F` `_overlap_merge_window_ms`
- `60-63` `F` `_output_directory_name`
- `66-71` `F` `_segment_directory_name`
- `74-150` `F` `_beatmap_payload`
- `153-240` `F` `_segment_row`
- `243-488` `C` `VideoSegmentationProcessor`
- `244-289` `M` `VideoSegmentationProcessor.__init__`
- `291-297` `M` `VideoSegmentationProcessor.progress_message`
- `299-479` `M` `VideoSegmentationProcessor.process_one`
- `481-488` `M` `VideoSegmentationProcessor.handle_failure`
- `491-498` `F` `segment_videos`

## `src/before_traning/main.py`

- `25-26` `F` `_resolve`
- `29-30` `F` `_skip`
- `33-60` `F` `_settings`
- `63-72` `F` `_render`
- `75-83` `F` `_run`
- `87-131` `F` `run_command`
- `135-151` `F` `verify_command`
- `155-175` `F` `match_command`
- `179-196` `F` `clip_command`
- `200-216` `F` `segment_command`
- `220-222` `F` `default_command`
- `225-233` `F` `main`

## `src/before_traning/state/__init__.py`

- `7-12` `F` `__getattr__`

## `src/before_traning/state/manifest_schema.py`

- `9-24` `C` `PackageManifestItem`
- `27-39` `C` `BeatmapDataRecord`

## `src/before_traning/state/process_status.py`

- `25-208` `C` `ProcessStatusManager`
- `26-48` `M` `ProcessStatusManager.__init__`
- `50-56` `M` `ProcessStatusManager._normalize_folder_name`
- `58-59` `M` `ProcessStatusManager._registered_names`
- `61-66` `M` `ProcessStatusManager._assert_registered`
- `68-74` `M` `ProcessStatusManager._require_existing_folder`
- `76-77` `M` `ProcessStatusManager._default_status`
- `79-81` `M` `ProcessStatusManager._validate_step`
- `83-84` `M` `ProcessStatusManager._normalize_status`
- `86-97` `M` `ProcessStatusManager._select_record`
- `99-105` `M` `ProcessStatusManager._has_records`
- `107-121` `M` `ProcessStatusManager._migrate_legacy_status_key`
- `123-128` `M` `ProcessStatusManager._load_legacy_json`
- `130-132` `M` `ProcessStatusManager.get_status_path`
- `134-158` `M` `ProcessStatusManager.load_status`
- `160-178` `M` `ProcessStatusManager.save_status`
- `180-183` `M` `ProcessStatusManager.ensure_status_file`
- `185-188` `M` `ProcessStatusManager.is_step_done`
- `190-196` `M` `ProcessStatusManager.mark_step_done`
- `198-204` `M` `ProcessStatusManager.mark_step_pending`
- `206-208` `M` `ProcessStatusManager.get_steps_summary`

## `src/before_traning/state/segment_schema.py`

- `9-27` `C` `SegmentDatasetItem`

## `src/before_traning/state/status_schema.py`

- `22-34` `C` `ProcessStepStatus`
- `37-56` `F` `normalize_process_steps`
- `59-69` `F` `default_status`
- `72-94` `F` `normalize_status`
- `97-100` `F` `encode_detail`
- `103-109` `F` `decode_detail`
