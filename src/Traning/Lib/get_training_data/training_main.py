from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from time import perf_counter

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Traning.Lib.get_training_data.config_loader import (
    build_from_get_check_data_config_or_default,
    build_from_video_clip_config_or_default,
)
from Traning.Lib.get_training_data.get_check_data.check_data_main import CheckDataPipeline
from Traning.Lib.get_training_data.video_clip.video_clip_main import VideoClipPipeline


class TemporaryTrainingRunner:
    """Temporary end-to-end runner for the current training-data pipeline."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path
        self.check_data_pipeline = build_from_get_check_data_config_or_default(
            CheckDataPipeline,
            config_path=config_path,
            default_builder=CheckDataPipeline,
        )
        self.video_clip_pipeline = build_from_video_clip_config_or_default(
            VideoClipPipeline,
            config_path=config_path,
            default_builder=VideoClipPipeline,
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
        "--overwrite",
        action="store_true",
        help="Overwrite files that are already considered complete.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next top-level stage even if the current stage fails.",
    )
    parser.add_argument(
        "--skip-check-data",
        action="store_true",
        help="Skip the whole check_data stage.",
    )
    parser.add_argument(
        "--skip-get-files",
        action="store_true",
        help="Skip importing .osz/.osu files.",
    )
    parser.add_argument(
        "--skip-verify-export",
        action="store_true",
        help="Skip verify.txt export.",
    )
    parser.add_argument(
        "--skip-difficulty-export",
        action="store_true",
        help="Skip difficulty.txt export.",
    )
    parser.add_argument(
        "--skip-video-clip",
        action="store_true",
        help="Skip the whole video_clip stage.",
    )
    parser.add_argument(
        "--skip-video-init",
        action="store_true",
        help="Skip the video init/status sync stage.",
    )
    parser.add_argument(
        "--skip-video-match",
        action="store_true",
        help="Skip the video matching stage.",
    )
    parser.add_argument(
        "--skip-av-correspondence",
        action="store_true",
        help="Skip the AV correspondence stage.",
    )
    parser.add_argument(
        "--skip-clip-stage",
        dest="run_clip_stage",
        action="store_false",
        help="Skip the final clip.py crop stage.",
    )
    parser.add_argument(
        "--run-clip-stage",
        dest="run_clip_stage",
        action="store_true",
        help="Run the final clip.py crop stage. Kept for compatibility.",
    )
    parser.set_defaults(run_clip_stage=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    runner = TemporaryTrainingRunner(config_path=args.config)
    started_at = perf_counter()
    results = runner.run(
        overwrite=args.overwrite,
        run_check_data=not args.skip_check_data,
        run_get_files=not args.skip_get_files,
        run_verify_export=not args.skip_verify_export,
        run_difficulty_export=not args.skip_difficulty_export,
        run_video_clip=not args.skip_video_clip,
        run_video_init_check=not args.skip_video_init,
        run_video_match=not args.skip_video_match,
        run_av_correspondence=not args.skip_av_correspondence,
        run_clip_stage=args.run_clip_stage,
        continue_on_error=args.continue_on_error,
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
