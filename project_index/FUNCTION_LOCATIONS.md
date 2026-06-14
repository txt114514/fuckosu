# Function Locations

> 自动生成文件，请勿手工修改。运行 `python project_index/build_index.py` 重建。

快速位置表：`343` 个命名函数/方法，附带 `70` 个类定义。
格式为 `起止行  类型  限定名`；路径按模块分组。

快速搜索：`rg -n "符号名" project_index/FUNCTION_LOCATIONS.md`。

## `src/Traning/Lib/audio/matching/matching.py`

- `30-31` `F` `_load_audio_match_experiment_config`
- `34-42` `F` `build_audio_match_experiment_from_config_or_default`
- `45-85` `C` `AudioMatchExperiment`
- `50-85` `M` `AudioMatchExperiment.__init__`
- `88-89` `C` `AudioMatchProcessor`
- `92-94` `F` `main`

## `src/Traning/Lib/audio/matching/preflight.py`

- `9-97` `C` `AudioMatchPreflightMixin`
- `10-15` `M` `AudioMatchPreflightMixin._folder_has_video`
- `17-45` `M` `AudioMatchPreflightMixin._sync_video_matched_status`
- `47-62` `M` `AudioMatchPreflightMixin._pending_folder_names`
- `64-74` `M` `AudioMatchPreflightMixin._candidate_folder_names`
- `76-97` `M` `AudioMatchPreflightMixin._candidate_videos`

## `src/Traning/Lib/audio/matching/steps.py`

- `12-194` `C` `AudioMatchStepsMixin`
- `13-17` `M` `AudioMatchStepsMixin._extract_samples`
- `19-32` `M` `AudioMatchStepsMixin._build_alignment_features`
- `34-69` `M` `AudioMatchStepsMixin._estimate_offset_from_features`
- `71-79` `M` `AudioMatchStepsMixin._result_sort_key`
- `81-173` `M` `AudioMatchStepsMixin._score_pairs`
- `175-194` `M` `AudioMatchStepsMixin._select_greedy_matches`

## `src/Traning/Lib/audio/matching/wrapup.py`

- `9-160` `C` `AudioMatchWrapUpMixin`
- `10-26` `M` `AudioMatchWrapUpMixin._print_greedy_matches`
- `28-133` `M` `AudioMatchWrapUpMixin._apply_matches`
- `135-160` `M` `AudioMatchWrapUpMixin.run`

## `src/Traning/Lib/beatmap/difficulty.py`

- `18-20` `C` `DifficultyEntry`
- `23-134` `C` `DifficultyFileManager`
- `24-42` `M` `DifficultyFileManager.__init__`
- `44-49` `M` `DifficultyFileManager.write_difficulty`
- `51-55` `M` `DifficultyFileManager.read_difficulty`
- `57-103` `M` `DifficultyFileManager.export_one`
- `105-134` `M` `DifficultyFileManager.list_difficulties`
- `137-138` `C` `BeatmapDifficultyProcessor`

## `src/Traning/Lib/beatmap/difficulty_batch.py`

- `6-32` `C` `DifficultyBatchMixin`
- `7-32` `M` `DifficultyBatchMixin.run`

## `src/Traning/Lib/beatmap/folder_store.py`

- `14-170` `C` `BeatmapFolderStore`
- `22-36` `M` `BeatmapFolderStore.__init__`
- `38-46` `M` `BeatmapFolderStore._normalize_folder_name`
- `48-49` `M` `BeatmapFolderStore._registered_names`
- `51-53` `M` `BeatmapFolderStore.is_registered`
- `55-59` `M` `BeatmapFolderStore._assert_registered`
- `61-64` `M` `BeatmapFolderStore.get_folder_path`
- `66-68` `M` `BeatmapFolderStore.folder_exists`
- `70-76` `M` `BeatmapFolderStore._require_existing_folder`
- `78-81` `M` `BeatmapFolderStore.find_files`
- `83-84` `M` `BeatmapFolderStore.find_osu_files`
- `86-91` `M` `BeatmapFolderStore.get_file_path`
- `93-94` `M` `BeatmapFolderStore.file_exists`
- `96-132` `M` `BeatmapFolderStore.write_text`
- `134-151` `M` `BeatmapFolderStore.write_lines`
- `153-164` `M` `BeatmapFolderStore.append_line`
- `166-170` `M` `BeatmapFolderStore.read_text`

## `src/Traning/Lib/beatmap/hit_objects.py`

- `8-11` `C` `HitObject`
- `13-17` `C` `Circle`
- `16-17` `M` `Circle.__post_init__`
- `19-25` `C` `Slider`
- `24-25` `M` `Slider.__post_init__`
- `27-29` `C` `Spinner`
- `28-29` `M` `Spinner.__post_init__`

## `src/Traning/Lib/beatmap/importing/entry.py`

- `8-16` `C` `OsuEntry`

## `src/Traning/Lib/beatmap/importing/importing.py`

- `27-28` `F` `_load_osu_osz_processor_config`
- `31-39` `F` `build_osu_osz_processor_from_config_or_default`
- `42-50` `F` `build_beatmap_import_processor_from_config_or_default`
- `53-82` `C` `OsuOszProcessor`
- `54-82` `M` `OsuOszProcessor.__init__`
- `85-86` `C` `BeatmapImportProcessor`
- `89-91` `F` `main`

## `src/Traning/Lib/beatmap/importing/scanner.py`

- `13-113` `C` `OszScannerMixin`
- `14-19` `M` `OszScannerMixin._is_target_osu`
- `21-34` `M` `OszScannerMixin._read_audio_bytes`
- `36-62` `M` `OszScannerMixin._scan_single_osz`
- `64-113` `M` `OszScannerMixin._scan_all_entries_in_time_order`

## `src/Traning/Lib/beatmap/importing/wrapup.py`

- `4-28` `C` `OszImportWrapUpMixin`
- `5-28` `M` `OszImportWrapUpMixin.run`

## `src/Traning/Lib/beatmap/importing/writer.py`

- `7-67` `C` `OszImportWriterMixin`
- `8-22` `M` `OszImportWriterMixin._rebuild_manifest`
- `24-67` `M` `OszImportWriterMixin._sync_folders_and_copy_files`

## `src/Traning/Lib/beatmap/manifest.py`

- `21-25` `C` `ManifestEntry`
- `28-322` `C` `PackageManifest`
- `31-48` `M` `PackageManifest.__init__`
- `50-60` `M` `PackageManifest._ensure_schema`
- `62-65` `M` `PackageManifest._all_items`
- `67-73` `M` `PackageManifest._normalize_source_name`
- `75-83` `M` `PackageManifest._next_folder_number`
- `85-86` `M` `PackageManifest._folder_name`
- `88-93` `M` `PackageManifest._legacy_osu_filename`
- `95-144` `M` `PackageManifest._rename_legacy_folders`
- `146-159` `M` `PackageManifest._restore_legacy_folders`
- `161-203` `M` `PackageManifest._migrate_legacy_order`
- `205-226` `M` `PackageManifest._migrate_legacy_difficulty_files`
- `228-238` `M` `PackageManifest.export_table`
- `240-275` `M` `PackageManifest.replace`
- `277-284` `M` `PackageManifest.read_folder_names`
- `286-287` `M` `PackageManifest.read_all_folder_names`
- `289-295` `M` `PackageManifest.source_name_for`
- `297-310` `M` `PackageManifest.set_difficulty`
- `312-319` `M` `PackageManifest.difficulty_for`
- `321-322` `M` `PackageManifest.is_active`
- `325-343` `C` `ManifestFolderWalker`
- `326-337` `M` `ManifestFolderWalker.__init__`
- `339-340` `M` `ManifestFolderWalker.read_folder_names`
- `342-343` `M` `ManifestFolderWalker.source_name_for`

## `src/Traning/Lib/beatmap/osu_metadata.py`

- `6-31` `F` `read_section_key`
- `34-35` `F` `read_audio_filename`
- `38-39` `F` `read_overall_difficulty`

## `src/Traning/Lib/beatmap/package.py`

- `14-101` `C` `PackageUpdater`
- `22-42` `M` `PackageUpdater.__init__`
- `44-45` `M` `PackageUpdater.load_manifest_folder_names`
- `47-48` `M` `PackageUpdater.load_registered_names`
- `50-51` `M` `PackageUpdater.replace_manifest`
- `53-54` `M` `PackageUpdater.is_registered`
- `56-71` `M` `PackageUpdater.create_folder_if_registered`
- `73-83` `M` `PackageUpdater.sync_folders_from_manifest`
- `85-101` `M` `PackageUpdater.find_unregistered_existing_folders`

## `src/Traning/Lib/beatmap/timing_points.py`

- `7-15` `C` `OsuOriginalTimingPoint`

## `src/Traning/Lib/beatmap/verification/parser.py`

- `10-212` `C` `VerifyOsuParser`
- `11-35` `M` `VerifyOsuParser.parse_sections`
- `37-44` `M` `VerifyOsuParser.parse_key_value_section`
- `46-68` `M` `VerifyOsuParser.parse_timing_points`
- `70-97` `M` `VerifyOsuParser.get_effective_timing`
- `99-170` `M` `VerifyOsuParser.parse_hitobjects`
- `172-188` `M` `VerifyOsuParser.objects_to_lines`
- `190-212` `M` `VerifyOsuParser.hit_object_to_dict`

## `src/Traning/Lib/beatmap/verification/steps.py`

- `6-73` `C` `VerifyStepsMixin`
- `7-73` `M` `VerifyStepsMixin.process_one`

## `src/Traning/Lib/beatmap/verification/verification.py`

- `27-28` `F` `_load_verify_exporter_config`
- `31-48` `C` `VerifyExporter`
- `32-48` `M` `VerifyExporter.__init__`
- `51-52` `C` `BeatmapVerifyExporter`
- `55-71` `F` `_build_verify_exporter_from_config`
- `74-82` `F` `build_verify_exporter_from_config_or_default`
- `85-88` `F` `build_beatmap_verify_exporter_from_config_or_default`
- `91-93` `F` `main`

## `src/Traning/Lib/beatmap/verification/wrapup.py`

- `6-14` `C` `VerifyWrapUpMixin`
- `7-14` `M` `VerifyWrapUpMixin.handle_failure`

## `src/Traning/Lib/common/batch.py`

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

## `src/Traning/Lib/common/failures.py`

- `11-21` `F` `_error_traceback`
- `24-29` `F` `callable_location`
- `32-42` `F` `exception_location`
- `45-59` `F` `failure_detail`
- `62-70` `F` `exception_detail`
- `73-77` `F` `format_failure`
- `80-81` `F` `format_exception`

## `src/Traning/Lib/common/pathspec.py`

- `9-15` `F` `suffix_pattern`
- `18-19` `F` `suffix_patterns`
- `22-23` `F` `gitwildmatch_spec`
- `26-27` `F` `suffix_spec`
- `30-31` `F` `matches_name`
- `34-35` `F` `filter_files`

## `src/Traning/Lib/tools/ffmpeg.py`

- `38-42` `F` `_command_error_text`
- `45-51` `F` `_run_command`
- `54-57` `F` `run_ffmpeg`
- `60-82` `F` `build_extract_wav_args`
- `85-104` `F` `build_trim_video_args`
- `107-126` `F` `build_segment_video_args`
- `129-148` `F` `build_crop_video_args`
- `151-168` `F` `run_ffprobe_json`
- `171-193` `F` `get_audio_stream_start_time`
- `196-216` `F` `get_media_duration_seconds`
- `219-240` `F` `get_video_size`

## `src/Traning/Lib/video/av_processing/av_processing.py`

- `27-28` `F` `_load_av_correspondence_processor_config`
- `31-39` `F` `build_av_correspondence_processor_from_config_or_default`
- `42-84` `C` `AVCorrespondenceProcessor`
- `48-84` `M` `AVCorrespondenceProcessor.__init__`
- `87-88` `C` `VideoAVProcessor`
- `91-93` `F` `main`

## `src/Traning/Lib/video/av_processing/preflight.py`

- `11-109` `C` `AVPreflightMixin`
- `12-28` `M` `AVPreflightMixin._validate_config`
- `30-39` `M` `AVPreflightMixin._ensure_status_steps_registered`
- `41-59` `M` `AVPreflightMixin._resolve_source_video_path`
- `61-65` `M` `AVPreflightMixin._resolve_song_audio_path`
- `67-68` `M` `AVPreflightMixin._resolve_verify_path`
- `70-97` `M` `AVPreflightMixin._sync_output_status`
- `99-109` `M` `AVPreflightMixin._ensure_required_steps_done`

## `src/Traning/Lib/video/av_processing/steps.py`

- `15-412` `C` `AVCoreStepsMixin`
- `16-24` `M` `AVCoreStepsMixin._extract_audio_to_wav`
- `26-46` `M` `AVCoreStepsMixin._load_wav_samples`
- `48-54` `M` `AVCoreStepsMixin._normalize_series`
- `56-81` `M` `AVCoreStepsMixin._build_feature_series`
- `83-95` `M` `AVCoreStepsMixin._lowpass_samples`
- `97-103` `M` `AVCoreStepsMixin._build_music_refine_series`
- `105-132` `M` `AVCoreStepsMixin._estimate_best_start_frame`
- `134-180` `M` `AVCoreStepsMixin._estimate_offset_seconds`
- `182-195` `M` `AVCoreStepsMixin._parse_verify_hit_times_ms`
- `197-210` `M` `AVCoreStepsMixin._build_verify_click_train`
- `212-261` `M` `AVCoreStepsMixin._estimate_verify_adjustment_seconds`
- `263-285` `M` `AVCoreStepsMixin._validate_trim_window`
- `287-301` `M` `AVCoreStepsMixin._trim_video`
- `303-412` `M` `AVCoreStepsMixin.process_one`

## `src/Traning/Lib/video/av_processing/wrapup.py`

- `6-74` `C` `AVWrapUpMixin`
- `7-11` `M` `AVWrapUpMixin._update_progress`
- `13-14` `M` `AVWrapUpMixin.progress_message`
- `16-58` `M` `AVWrapUpMixin._build_done_detail`
- `60-65` `M` `AVWrapUpMixin._mark_done`
- `67-74` `M` `AVWrapUpMixin.handle_failure`

## `src/Traning/Lib/video/clipping/clipping.py`

- `24-36` `F` `build_fixed_region_video_crop_processor_from_config_or_default`
- `39-112` `C` `FixedRegionVideoCropProcessor`
- `47-52` `M` `FixedRegionVideoCropProcessor.from_settings`
- `54-112` `M` `FixedRegionVideoCropProcessor.__init__`
- `115-116` `C` `VideoClipProcessor`
- `119-121` `F` `main`

## `src/Traning/Lib/video/clipping/geometry.py`

- `8-95` `C` `ClipGeometryMixin`
- `9-10` `M` `ClipGeometryMixin.get_video_size`
- `12-18` `M` `ClipGeometryMixin._scale_crop_coordinate`
- `20-75` `M` `ClipGeometryMixin._resolve_scaled_crop`
- `77-92` `M` `ClipGeometryMixin._validate_crop_bounds`
- `94-95` `M` `ClipGeometryMixin.describe_crop_for_video`

## `src/Traning/Lib/video/clipping/preflight.py`

- `4-32` `C` `ClipPreflightMixin`
- `5-14` `M` `ClipPreflightMixin._ensure_status_steps_registered`
- `16-32` `M` `ClipPreflightMixin._ensure_folder_ready`

## `src/Traning/Lib/video/clipping/steps.py`

- `9-47` `C` `ClipStepsMixin`
- `10-29` `M` `ClipStepsMixin._crop_video_in_place`
- `31-47` `M` `ClipStepsMixin.process_one`

## `src/Traning/Lib/video/clipping/wrapup.py`

- `8-71` `C` `ClipWrapUpMixin`
- `9-10` `M` `ClipWrapUpMixin.progress_message`
- `12-20` `M` `ClipWrapUpMixin._reference_detail`
- `22-37` `M` `ClipWrapUpMixin._mark_cropping`
- `39-58` `M` `ClipWrapUpMixin._mark_done`
- `60-71` `M` `ClipWrapUpMixin.handle_failure`

## `src/Traning/Lib/video/matching/builders.py`

- `14-15` `F` `_load_video_package_renamer_config`
- `18-26` `F` `build_video_package_renamer_from_config_or_default`

## `src/Traning/Lib/video/matching/matching.py`

- `9-39` `C` `VideoMatchProcessor`
- `12-23` `M` `VideoMatchProcessor.__init__`
- `25-39` `M` `VideoMatchProcessor.run`

## `src/Traning/Lib/video/matching/renamer.py`

- `22-187` `C` `VideoPackageRenamer`
- `23-46` `M` `VideoPackageRenamer.__init__`
- `48-52` `M` `VideoPackageRenamer._folder_has_video`
- `54-69` `M` `VideoPackageRenamer._parse_video_time`
- `71-82` `M` `VideoPackageRenamer._list_videos_in_time_order`
- `84-115` `M` `VideoPackageRenamer._pending_folder_names`
- `117-142` `M` `VideoPackageRenamer._build_rename_plan`
- `144-187` `M` `VideoPackageRenamer.run`
- `190-191` `C` `VideoMatchRenamer`

## `src/Traning/Lib/video/segmentation/planner.py`

- `35-47` `C` `ParsedStandardBeatmap`
- `46-47` `M` `ParsedStandardBeatmap.approach_preempt_ms`
- `51-83` `C` `SegmentPlan`
- `66-67` `M` `SegmentPlan.duration_seconds`
- `70-71` `M` `SegmentPlan.pre_context_seconds`
- `74-75` `M` `SegmentPlan.post_context_seconds`
- `78-79` `M` `SegmentPlan.clip_start_ms`
- `82-83` `M` `SegmentPlan.clip_end_ms`
- `86-123` `F` `parse_standard_beatmap`
- `126-127` `F` `parse_standard_hit_objects`
- `130-137` `F` `approach_preempt_ms`
- `140-152` `F` `circle_radius_from_size`
- `155-166` `F` `circle_overlap_ratio`
- `169-188` `F` `_slider_polyline`
- `191-198` `F` `_object_polyline`
- `201-219` `F` `_point_to_segment_distance`
- `222-230` `F` `_orientation`
- `233-255` `F` `_segments_intersect`
- `258-310` `F` `_polyline_distance`
- `313-323` `F` `hit_objects_overlap_ratio`
- `326-395` `F` `group_hit_objects`
- `398-418` `F` `classify_hit_group`
- `421-510` `F` `build_segment_plans`

## `src/Traning/Lib/video/segmentation/segmentation.py`

- `76-579` `C` `VideoSegmentationProcessor`
- `77-118` `M` `VideoSegmentationProcessor.__init__`
- `120-132` `M` `VideoSegmentationProcessor._recover_interrupted_outputs`
- `134-142` `M` `VideoSegmentationProcessor._sync_manifest_table`
- `144-154` `M` `VideoSegmentationProcessor._ensure_status_steps_registered`
- `156-157` `M` `VideoSegmentationProcessor.progress_message`
- `159-160` `M` `VideoSegmentationProcessor._output_directory`
- `162-185` `M` `VideoSegmentationProcessor._output_complete`
- `187-194` `M` `VideoSegmentationProcessor._ensure_required_steps_done`
- `196-200` `M` `VideoSegmentationProcessor._segment_directory_name`
- `202-208` `M` `VideoSegmentationProcessor._overlap_merge_window_ms`
- `210-223` `M` `VideoSegmentationProcessor._write_segment`
- `225-240` `M` `VideoSegmentationProcessor._serialize_hit_object`
- `242-300` `M` `VideoSegmentationProcessor._write_beatmap_data`
- `302-317` `M` `VideoSegmentationProcessor._write_segment_table`
- `319-323` `M` `VideoSegmentationProcessor._remove_output_path`
- `325-465` `M` `VideoSegmentationProcessor._build_output`
- `467-570` `M` `VideoSegmentationProcessor.process_one`
- `572-579` `M` `VideoSegmentationProcessor.handle_failure`

## `src/Traning/conf/field_groups.py`

- `72-73` `F` `group_values`
- `76-78` `F` `assign_group`
- `81-82` `F` `forward_kwargs`

## `src/Traning/conf/legacy_config.py`

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

## `src/Traning/conf/runtime.py`

- `9-12` `F` `ensure_prefect_home`

## `src/Traning/conf/settings.py`

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
- `98-135` `C` `SegmentSettings`
- `109-112` `M` `SegmentSettings._nonnegative_interval`
- `116-119` `M` `SegmentSettings._approach_ratio`
- `123-126` `M` `SegmentSettings._overlap_ratio`
- `130-135` `M` `SegmentSettings._nonnegative_context`
- `138-150` `C` `ProgressSettings`
- `153-186` `C` `Settings`
- `173-174` `M` `Settings.target_root`
- `177-178` `M` `Settings.overwrite`
- `181-182` `M` `Settings.continue_on_error`
- `185-186` `M` `Settings.global_offset_ms`
- `189-199` `F` `_resolve_paths`
- `202-237` `F` `_extract_nested`
- `240-257` `F` `_read_config`
- `260-266` `F` `load_settings`

## `src/Traning/core/beatmap/difficulty.py`

- `13-23` `F` `export_difficulty`

## `src/Traning/core/beatmap/importer.py`

- `11-22` `F` `import_beatmaps`

## `src/Traning/core/beatmap/pipeline.py`

- `9-14` `F` `prepare_beatmaps`

## `src/Traning/core/beatmap/verify.py`

- `12-16` `F` `build_store`
- `19-28` `F` `export_verify`

## `src/Traning/core/flows/pipeline.py`

- `39-45` `C` `PipelineStage`
- `108-116` `F` `_call_stage`
- `119-120` `F` `_enabled`
- `123-144` `F` `_run_stages`
- `148-171` `F` `train_pipeline`
- `174-197` `F` `train_pipeline_direct`
- `200-245` `C` `TemporaryTrainingRunner`
- `201-202` `M` `TemporaryTrainingRunner.__init__`
- `204-245` `M` `TemporaryTrainingRunner.run`

## `src/Traning/core/tasks/__init__.py`

- `4-7` `F` `require_success`

## `src/Traning/core/tasks/av.py`

- `15-16` `F` `av_correspondence_task`

## `src/Traning/core/tasks/clip.py`

- `15-16` `F` `crop_video_task`

## `src/Traning/core/tasks/difficulty.py`

- `15-16` `F` `export_difficulty_task`

## `src/Traning/core/tasks/importer.py`

- `15-16` `F` `import_beatmaps_task`

## `src/Traning/core/tasks/match.py`

- `15-16` `F` `match_videos_task`

## `src/Traning/core/tasks/segment.py`

- `15-16` `F` `segment_videos_task`

## `src/Traning/core/tasks/verify.py`

- `15-16` `F` `export_verify_task`

## `src/Traning/core/video/av.py`

- `13-32` `F` `av_correspondence`

## `src/Traning/core/video/clip.py`

- `11-16` `F` `crop_video`

## `src/Traning/core/video/match.py`

- `11-33` `F` `match_videos`

## `src/Traning/core/video/pipeline.py`

- `10-16` `F` `prepare_videos`

## `src/Traning/core/video/segment.py`

- `11-16` `F` `segment_videos`

## `src/Traning/main.py`

- `25-26` `F` `_resolve`
- `29-30` `F` `_skip`
- `33-60` `F` `_settings`
- `63-72` `F` `_render`
- `75-84` `F` `_run`
- `88-132` `F` `run_command`
- `136-152` `F` `verify_command`
- `156-176` `F` `match_command`
- `180-197` `F` `clip_command`
- `201-217` `F` `segment_command`
- `221-223` `F` `default_command`
- `226-234` `F` `main`

## `src/Traning/state/__init__.py`

- `7-12` `F` `__getattr__`

## `src/Traning/state/manifest_schema.py`

- `9-24` `C` `PackageManifestItem`

## `src/Traning/state/process_status.py`

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

## `src/Traning/state/status_schema.py`

- `22-34` `C` `ProcessStepStatus`
- `37-56` `F` `normalize_process_steps`
- `59-69` `F` `default_status`
- `72-94` `F` `normalize_status`
- `97-100` `F` `encode_detail`
- `103-109` `F` `decode_detail`
