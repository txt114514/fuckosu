from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from time import perf_counter

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Traning.Lib.get_training_data.get_check_data.check_data_main import (
    build_check_data_pipeline_from_config_or_default,
)
from Traning.Lib.get_training_data.setting_loader import (
    DEFAULT_USE_AUDIO_MATCH_EXPERIMENT,
    load_training_settings_or_default,
)
from Traning.Lib.get_training_data.video_clip.video_clip_main import (
    build_video_clip_pipeline_from_config_or_default,
)


def _resolve_optional_bool(default: bool, override: bool | None) -> bool:
    if override is None:
        return default
    return override


def _resolve_skip_flag(default: bool, skip_flag: bool | None) -> bool:
    if skip_flag is None:
        return default
    return False


class TemporaryTrainingRunner:
    """Temporary end-to-end runner for the current training-data pipeline."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path
        self.check_data_pipeline = build_check_data_pipeline_from_config_or_default(
            config_path=config_path,
        )
        self.video_clip_pipeline = build_video_clip_pipeline_from_config_or_default(
            config_path=config_path,
        )

    def _run_stage(self, label: str, stage_func, continue_on_error: bool):
        print()
        print(f"[总流程] 开始 {label}")
        started_at = perf_counter()
        try:
            stage_func()
        except Exception:
            elapsed = perf_counter() - started_at
            print(f"[总流程] 失败 {label} ({elapsed:.2f}s)")
            traceback.print_exc()
            if not continue_on_error:
                raise
            return False

        elapsed = perf_counter() - started_at
        print(f"[总流程] 完成 {label} ({elapsed:.2f}s)")
        return True

    def run(
        self,
        overwrite: bool = False,
        run_check_data: bool = True,
        run_get_files: bool = True,
        run_verify_export: bool = True,
        run_difficulty_export: bool = True,
        run_video_clip: bool = True,
        run_video_init_check: bool = True,
        run_video_match: bool = True,
        run_av_correspondence: bool = True,
        run_clip_stage: bool = True,
        use_audio_match_experiment: bool = DEFAULT_USE_AUDIO_MATCH_EXPERIMENT,
        global_offset_ms: float | None = None,
        continue_on_error: bool = False,
    ) -> dict[str, bool]:
        results: dict[str, bool] = {}

        if run_check_data:
            results["check_data"] = self._run_stage(
                "check_data",
                lambda: self.check_data_pipeline.run(
                    overwrite=overwrite,
                    run_get_files=run_get_files,
                    run_verify_export=run_verify_export,
                    run_difficulty_export=run_difficulty_export,
                ),
                continue_on_error=continue_on_error,
            )

        if run_video_clip:
            results["video_clip"] = self._run_stage(
                "video_clip",
                lambda: self.video_clip_pipeline.run(
                    overwrite=overwrite,
                    run_init_check=run_video_init_check,
                    run_video_match=run_video_match,
                    run_av_correspondence=run_av_correspondence,
                    run_clip_stage=run_clip_stage,
                    use_audio_match_experiment=use_audio_match_experiment,
                    global_offset_ms=global_offset_ms,
                ),
                continue_on_error=continue_on_error,
            )

        return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Temporary runner for the get_training_data pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config.json path used to override file-local defaults.",
    )
    parser.add_argument(
        "--setting",
        type=Path,
        default=None,
        help="Optional setting.json path used to override training_main.py runtime switches.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Overwrite files that are already considered complete.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=None,
        help="Continue to the next top-level stage even if the current stage fails.",
    )
    parser.add_argument(
        "--skip-check-data",
        action="store_true",
        default=None,
        help="Skip the whole check_data stage.",
    )
    parser.add_argument(
        "--skip-get-files",
        action="store_true",
        default=None,
        help="Skip importing .osz/.osu files.",
    )
    parser.add_argument(
        "--skip-verify-export",
        action="store_true",
        default=None,
        help="Skip verify.txt export.",
    )
    parser.add_argument(
        "--skip-difficulty-export",
        action="store_true",
        default=None,
        help="Skip difficulty.txt export.",
    )
    parser.add_argument(
        "--skip-video-clip",
        action="store_true",
        default=None,
        help="Skip the whole video_clip stage.",
    )
    parser.add_argument(
        "--skip-video-init",
        action="store_true",
        default=None,
        help="Skip the video init/status sync stage.",
    )
    parser.add_argument(
        "--skip-video-match",
        action="store_true",
        default=None,
        help="Skip the video matching stage.",
    )
    audio_match_group = parser.add_mutually_exclusive_group()
    audio_match_group.add_argument(
        "--use-audio-match-experiment",
        dest="use_audio_match_experiment",
        action="store_true",
        default=None,
        help="Use the experimental audio-based video matcher instead of the sequence-based matcher.",
    )
    audio_match_group.add_argument(
        "--disable-audio-match-experiment",
        dest="use_audio_match_experiment",
        action="store_false",
        help="Disable audio-based video matching and use the sequence-based matcher.",
    )
    parser.add_argument(
        "--global-offset-ms",
        type=float,
        default=None,
        help="Additional global AV trim offset in milliseconds, applied after audio and verify-based correction.",
    )
    parser.add_argument(
        "--skip-av-correspondence",
        action="store_true",
        default=None,
        help="Skip the AV correspondence stage.",
    )
    parser.add_argument(
        "--skip-clip-stage",
        dest="run_clip_stage",
        action="store_false",
        default=None,
        help="Skip the final clip.py crop stage.",
    )
    parser.add_argument(
        "--run-clip-stage",
        dest="run_clip_stage",
        action="store_true",
        help="Run the final clip.py crop stage. Kept for compatibility.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    settings = load_training_settings_or_default(setting_path=args.setting)

    runner = TemporaryTrainingRunner(config_path=args.config)
    started_at = perf_counter()
    results = runner.run(
        overwrite=_resolve_optional_bool(settings.overwrite, args.overwrite),
        run_check_data=_resolve_skip_flag(settings.run_check_data, args.skip_check_data),
        run_get_files=_resolve_skip_flag(settings.run_get_files, args.skip_get_files),
        run_verify_export=_resolve_skip_flag(
            settings.run_verify_export,
            args.skip_verify_export,
        ),
        run_difficulty_export=_resolve_skip_flag(
            settings.run_difficulty_export,
            args.skip_difficulty_export,
        ),
        run_video_clip=_resolve_skip_flag(settings.run_video_clip, args.skip_video_clip),
        run_video_init_check=_resolve_skip_flag(
            settings.run_video_init_check,
            args.skip_video_init,
        ),
        run_video_match=_resolve_skip_flag(settings.run_video_match, args.skip_video_match),
        run_av_correspondence=_resolve_skip_flag(
            settings.run_av_correspondence,
            args.skip_av_correspondence,
        ),
        run_clip_stage=_resolve_optional_bool(settings.run_clip_stage, args.run_clip_stage),
        use_audio_match_experiment=_resolve_optional_bool(
            settings.use_audio_match_experiment,
            args.use_audio_match_experiment,
        ),
        global_offset_ms=(
            settings.global_offset_ms
            if args.global_offset_ms is None
            else args.global_offset_ms
        ),
        continue_on_error=_resolve_optional_bool(
            settings.continue_on_error,
            args.continue_on_error,
        ),
    )

    elapsed = perf_counter() - started_at
    print()
    print("[总流程] 运行结束")
    for stage_name, success in results.items():
        status = "success" if success else "failed"
        print(f"[总流程] {stage_name}: {status}")
    print(f"[总流程] 总耗时: {elapsed:.2f}s")

    if any(not success for success in results.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
