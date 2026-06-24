from __future__ import annotations

import json
import math
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from before_traning.Lib.beatmap.osz import read_osz_entry
from before_traning.conf import VERIFY_FILENAME, Settings
from before_traning.state.status_schema import STATUS_DB_FILENAME, decode_detail


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MATCHED_MANIFEST = REPO_ROOT / "runs" / "startup" / "matched_samples.json"
STARTUP_MATCHED_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class RawBeatmapCandidate:
    osz_path: Path
    source_name: str
    osu_filename: str
    audio_source_filename: str
    osz_name: str
    osz_mtime_ns: int

    @property
    def identity(self) -> tuple[str, str, int]:
        return (self.source_name, self.osz_name, self.osz_mtime_ns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "osz_path": self.osz_path,
            "source_name": self.source_name,
            "osu_filename": self.osu_filename,
            "audio_source_filename": self.audio_source_filename,
            "osz_name": self.osz_name,
            "osz_mtime_ns": self.osz_mtime_ns,
        }


@dataclass(frozen=True)
class VideoCandidate:
    path: Path
    name: str
    suffix: str
    mtime_ns: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "suffix": self.suffix,
            "mtime_ns": self.mtime_ns,
        }


@dataclass(frozen=True)
class BeforeManifestItem:
    folder_name: str
    source_name: str
    active: bool
    osu_filename: str | None = None
    source_osz_name: str | None = None
    source_mtime_ns: int | None = None

    @property
    def raw_identity(self) -> tuple[str, str | None, int | None]:
        return (self.source_name, self.source_osz_name, self.source_mtime_ns)


@dataclass(frozen=True)
class PendingImportedSample:
    folder_name: str
    source_name: str
    audio_path: Path
    verify_path: Path | None = None
    osz_name: str | None = None
    osz_mtime_ns: int | None = None

    @property
    def sample_key(self) -> str:
        return f"folder:{self.folder_name}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "folder_name": self.folder_name,
            "source_name": self.source_name,
            "audio_path": self.audio_path,
            "verify_path": self.verify_path,
            "osz_name": self.osz_name,
            "osz_mtime_ns": self.osz_mtime_ns,
        }


@dataclass(frozen=True)
class MatchedSample:
    source_name: str
    relation_status: str
    relation_source: str
    detected_at_utc: str = field(default_factory=_utc_now)
    osz_name: str | None = None
    osz_mtime_ns: int | None = None
    folder_name: str | None = None
    video_path: Path | None = None
    video_name: str | None = None
    match_score: float | None = None
    coarse_match_score: float | None = None
    offset_seconds: float | None = None

    @property
    def identity(self) -> tuple[str, str | None, int | None]:
        return (self.source_name, self.osz_name, self.osz_mtime_ns)

    def matches_raw_candidate(self, candidate: RawBeatmapCandidate) -> bool:
        if self.source_name != candidate.source_name:
            return False
        if self.osz_name is not None and self.osz_name != candidate.osz_name:
            return False
        if self.osz_mtime_ns is not None:
            return self.osz_mtime_ns == candidate.osz_mtime_ns
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "relation_status": self.relation_status,
            "relation_source": self.relation_source,
            "detected_at_utc": self.detected_at_utc,
            "osz_name": self.osz_name,
            "osz_mtime_ns": self.osz_mtime_ns,
            "folder_name": self.folder_name,
            "video_path": self.video_path,
            "video_name": self.video_name,
            "match_score": self.match_score,
            "coarse_match_score": self.coarse_match_score,
            "offset_seconds": self.offset_seconds,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "MatchedSample":
        video_path = raw.get("video_path")
        return cls(
            source_name=str(raw["source_name"]),
            relation_status=str(raw.get("relation_status", "matched")),
            relation_source=str(raw.get("relation_source", "startup_manifest")),
            detected_at_utc=str(raw.get("detected_at_utc") or _utc_now()),
            osz_name=_optional_str(raw.get("osz_name")),
            osz_mtime_ns=_optional_int(raw.get("osz_mtime_ns")),
            folder_name=_optional_str(raw.get("folder_name")),
            video_path=Path(video_path) if isinstance(video_path, str) else None,
            video_name=_optional_str(raw.get("video_name")),
            match_score=_optional_float(raw.get("match_score")),
            coarse_match_score=_optional_float(raw.get("coarse_match_score")),
            offset_seconds=_optional_float(raw.get("offset_seconds")),
        )


@dataclass(frozen=True)
class MatchedSampleManifest:
    path: Path
    samples: tuple[MatchedSample, ...] = ()

    @classmethod
    def load(cls, path: Path | None = None) -> "MatchedSampleManifest":
        manifest_path = path or DEFAULT_MATCHED_MANIFEST
        if not manifest_path.is_file():
            return cls(path=manifest_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_samples = payload.get("samples", ())
        samples = tuple(
            MatchedSample.from_mapping(item)
            for item in raw_samples
            if isinstance(item, Mapping) and "source_name" in item
        )
        return cls(path=manifest_path, samples=samples)

    def merged(self, samples: Iterable[MatchedSample]) -> "MatchedSampleManifest":
        by_key: dict[tuple[str, str | None, int | None], MatchedSample] = {
            sample.identity: sample for sample in self.samples
        }
        for sample in samples:
            by_key[sample.identity] = sample
        ordered = tuple(
            sorted(
                by_key.values(),
                key=lambda item: (
                    item.source_name.lower(),
                    item.osz_name or "",
                    item.osz_mtime_ns or -1,
                ),
            )
        )
        return MatchedSampleManifest(path=self.path, samples=ordered)

    def save(self) -> None:
        payload = {
            "schema_version": STARTUP_MATCHED_SCHEMA_VERSION,
            "updated_at_utc": _utc_now(),
            "samples": tuple(sample.as_dict() for sample in self.samples),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f".{self.path.name}.tmp")
        tmp_path.write_text(
            json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    def matches_raw_candidate(self, candidate: RawBeatmapCandidate) -> bool:
        return any(sample.matches_raw_candidate(candidate) for sample in self.samples)


@dataclass(frozen=True)
class MatchProbePair:
    sample_key: str
    source_name: str
    video_path: Path
    video_name: str
    match_score: float
    coarse_match_score: float
    offset_seconds: float
    sample_kind: str
    folder_name: str | None = None
    osz_name: str | None = None
    osz_mtime_ns: int | None = None
    audio_offset_seconds: float | None = None
    verify_adjustment_seconds: float | None = None
    verify_adjustment_ms: float | None = None
    verify_score: float | None = None
    verify_window_ms: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_key": self.sample_key,
            "source_name": self.source_name,
            "video_path": self.video_path,
            "video_name": self.video_name,
            "match_score": self.match_score,
            "coarse_match_score": self.coarse_match_score,
            "offset_seconds": self.offset_seconds,
            "sample_kind": self.sample_kind,
            "folder_name": self.folder_name,
            "osz_name": self.osz_name,
            "osz_mtime_ns": self.osz_mtime_ns,
            "audio_offset_seconds": self.audio_offset_seconds,
            "verify_adjustment_seconds": self.verify_adjustment_seconds,
            "verify_adjustment_ms": self.verify_adjustment_ms,
            "verify_score": self.verify_score,
            "verify_window_ms": self.verify_window_ms,
        }


@dataclass(frozen=True)
class MatchProbeReport:
    min_match_score: float
    pair_count: int
    accepted_matches: tuple[MatchProbePair, ...]
    rejected_matches: tuple[MatchProbePair, ...]
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return bool(self.accepted_matches) and not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_match_score": self.min_match_score,
            "pair_count": self.pair_count,
            "accepted_count": len(self.accepted_matches),
            "rejected_count": len(self.rejected_matches),
            "errors": self.errors,
            "accepted_matches": tuple(item.as_dict() for item in self.accepted_matches),
            "rejected_matches": tuple(item.as_dict() for item in self.rejected_matches[:20]),
            "ok": self.ok,
        }


@dataclass(frozen=True)
class BeforeTrainingSampleInspection:
    matched_manifest_path: Path
    raw_candidates_total: int
    raw_unmatched_candidates: tuple[RawBeatmapCandidate, ...]
    video_candidates: tuple[VideoCandidate, ...]
    pending_imported_samples: tuple[PendingImportedSample, ...]
    recovered_matched_samples: tuple[MatchedSample, ...]
    startup_manifest_samples: tuple[MatchedSample, ...]
    match_probe: MatchProbeReport | None
    issues: tuple[str, ...] = ()

    @property
    def has_unmatched_samples(self) -> bool:
        return bool(self.raw_unmatched_candidates or self.pending_imported_samples)

    @property
    def has_video_candidates(self) -> bool:
        return bool(self.video_candidates)

    @property
    def should_run_before_traning(self) -> bool:
        if not self.has_unmatched_samples or not self.has_video_candidates:
            return False
        if self.match_probe is None:
            return True
        return bool(self.match_probe.accepted_matches)

    @property
    def reason(self) -> str:
        if not self.has_unmatched_samples:
            return "no unmatched beatmap/audio candidates"
        if not self.has_video_candidates:
            return "unmatched beatmap/audio candidates found but no new videos"
        if self.match_probe is None:
            return "unmatched candidates and videos found; match probe skipped"
        if self.match_probe.accepted_matches:
            return "unmatched candidates have accepted audio matches"
        return "match probe found no accepted video/audio relation"

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched_manifest_path": self.matched_manifest_path,
            "raw_candidates_total": self.raw_candidates_total,
            "raw_unmatched_count": len(self.raw_unmatched_candidates),
            "video_candidate_count": len(self.video_candidates),
            "pending_imported_count": len(self.pending_imported_samples),
            "recovered_matched_count": len(self.recovered_matched_samples),
            "startup_manifest_count": len(self.startup_manifest_samples),
            "should_run_before_traning": self.should_run_before_traning,
            "reason": self.reason,
            "issues": self.issues,
            "raw_unmatched_candidates": tuple(
                item.as_dict() for item in self.raw_unmatched_candidates[:20]
            ),
            "video_candidates": tuple(item.as_dict() for item in self.video_candidates[:20]),
            "pending_imported_samples": tuple(
                item.as_dict() for item in self.pending_imported_samples[:20]
            ),
            "match_probe": (
                self.match_probe.as_dict() if self.match_probe is not None else None
            ),
        }


def inspect_before_training_samples(
    settings: Settings,
    *,
    matched_manifest_path: Path | None = None,
    run_match_probe: bool = True,
    min_match_score: float = 0.1,
) -> BeforeTrainingSampleInspection:
    manifest = MatchedSampleManifest.load(matched_manifest_path)
    manifest_items, status_rows, state_issues = _read_before_state(settings)
    recovered = _recover_matched_samples(settings, manifest_items, status_rows)
    known_matched = (*manifest.samples, *recovered)
    raw_candidates, raw_issues = _scan_raw_beatmap_candidates(settings)
    raw_unmatched = _filter_unmatched_raw_candidates(
        raw_candidates,
        manifest_items,
        known_matched,
    )
    pending = _pending_imported_samples(settings, manifest_items, status_rows)
    videos = _scan_video_candidates(settings)
    probe = None
    if run_match_probe and (raw_unmatched or pending) and videos:
        probe = probe_before_training_matches(
            settings,
            raw_unmatched=raw_unmatched,
            pending_imported=pending,
            videos=videos,
            min_match_score=min_match_score,
        )
    return BeforeTrainingSampleInspection(
        matched_manifest_path=manifest.path,
        raw_candidates_total=len(raw_candidates),
        raw_unmatched_candidates=tuple(raw_unmatched),
        video_candidates=tuple(videos),
        pending_imported_samples=tuple(pending),
        recovered_matched_samples=tuple(recovered),
        startup_manifest_samples=manifest.samples,
        match_probe=probe,
        issues=tuple((*state_issues, *raw_issues)),
    )


def recover_matched_sample_manifest(
    settings: Settings,
    *,
    matched_manifest_path: Path | None = None,
) -> MatchedSampleManifest:
    manifest = MatchedSampleManifest.load(matched_manifest_path)
    manifest_items, status_rows, _issues = _read_before_state(settings)
    recovered = _recover_matched_samples(settings, manifest_items, status_rows)
    return manifest.merged(recovered)


def probe_before_training_matches(
    settings: Settings,
    *,
    raw_unmatched: Iterable[RawBeatmapCandidate],
    pending_imported: Iterable[PendingImportedSample],
    videos: Iterable[VideoCandidate],
    min_match_score: float,
) -> MatchProbeReport:
    try:
        return _run_match_probe(
            settings,
            tuple(raw_unmatched),
            tuple(pending_imported),
            tuple(videos),
            min_match_score,
        )
    except Exception as error:
        return MatchProbeReport(
            min_match_score=min_match_score,
            pair_count=0,
            accepted_matches=(),
            rejected_matches=(),
            errors=(f"{type(error).__name__}: {error}",),
        )


def _run_match_probe(
    settings: Settings,
    raw_unmatched: tuple[RawBeatmapCandidate, ...],
    pending_imported: tuple[PendingImportedSample, ...],
    videos: tuple[VideoCandidate, ...],
    min_match_score: float,
) -> MatchProbeReport:
    from before_traning.Lib.video.av_processing.steps import AVCoreStepsMixin

    class _StartupAudioAligner(AVCoreStepsMixin):
        pass

    aligner = _StartupAudioAligner()
    aligner.sample_rate = settings.av.sample_rate
    aligner.envelope_hz = settings.av.envelope_hz
    aligner.refine_hz = min(settings.av.sample_rate, settings.av.refine_hz)
    aligner.refine_search_seconds = settings.av.refine_search_seconds
    aligner.music_lowpass_hz = settings.av.music_lowpass_hz
    aligner.verify_correction_window_ms = settings.av.verify_correction_window_ms

    targets = (
        *(_probe_target_from_raw(settings, item) for item in raw_unmatched),
        *(_probe_target_from_pending(item) for item in pending_imported),
    )
    target_features = {
        target["sample_key"]: _build_probe_features(
            aligner,
            _target_audio_samples(aligner, target),
        )
        for target in targets
    }
    video_features = {
        video.path: _build_probe_features(
            aligner,
            _audio_samples_from_path(aligner, video.path, from_video=True),
        )
        for video in videos
    }

    pairs: list[MatchProbePair] = []
    errors: list[str] = []
    for target in targets:
        features = target_features[str(target["sample_key"])]
        for video in videos:
            try:
                pair = _score_probe_pair(
                    aligner,
                    target,
                    features,
                    video,
                    video_features[video.path],
                )
            except Exception as error:
                errors.append(
                    f"{target['sample_key']} x {video.name}: "
                    f"{type(error).__name__}: {error}"
                )
                continue
            pairs.append(pair)

    selected = _select_greedy_probe_matches(pairs)
    accepted = tuple(pair for pair in selected if pair.match_score >= min_match_score)
    accepted_ids = {(pair.sample_key, pair.video_path) for pair in accepted}
    rejected = tuple(
        pair
        for pair in pairs
        if (pair.sample_key, pair.video_path) not in accepted_ids
    )
    return MatchProbeReport(
        min_match_score=min_match_score,
        pair_count=len(pairs),
        accepted_matches=accepted,
        rejected_matches=rejected,
        errors=tuple(errors),
    )


def _probe_target_from_raw(
    settings: Settings,
    candidate: RawBeatmapCandidate,
) -> dict[str, Any]:
    entry = read_osz_entry(
        candidate.osz_path,
        keyword=settings.file_formats.keyword,
        audio_output_filename=settings.file_management.audio_filename,
    )
    if entry is None:
        raise ValueError(f"{candidate.osz_path} no longer contains target osu entry")
    return {
        "sample_key": f"raw:{candidate.source_name}:{candidate.osz_name}:{candidate.osz_mtime_ns}",
        "sample_kind": "raw_osz",
        "source_name": candidate.source_name,
        "folder_name": None,
        "osz_name": candidate.osz_name,
        "osz_mtime_ns": candidate.osz_mtime_ns,
        "audio_bytes": entry.audio_bytes,
        "audio_suffix": Path(entry.audio_source_filename).suffix or ".mp3",
        "audio_path": None,
        "verify_path": None,
    }


def _probe_target_from_pending(candidate: PendingImportedSample) -> dict[str, Any]:
    return {
        "sample_key": candidate.sample_key,
        "sample_kind": "pending_imported",
        "source_name": candidate.source_name,
        "folder_name": candidate.folder_name,
        "osz_name": candidate.osz_name,
        "osz_mtime_ns": candidate.osz_mtime_ns,
        "audio_bytes": None,
        "audio_suffix": candidate.audio_path.suffix or ".mp3",
        "audio_path": candidate.audio_path,
        "verify_path": candidate.verify_path,
    }


def _score_probe_pair(
    aligner: Any,
    target: Mapping[str, Any],
    target_features: Mapping[str, Any],
    video: VideoCandidate,
    video_features: Mapping[str, Any],
) -> MatchProbePair:
    coarse_start_frame, coarse_score = aligner._estimate_best_start_frame(
        video_features["coarse"],
        target_features["coarse"],
    )
    coarse_offset_seconds = coarse_start_frame / float(aligner.envelope_hz)
    fine_video = video_features["fine"]
    fine_target = target_features["fine"]
    if fine_video.size < fine_target.size:
        raise ValueError(f"{video.path} is shorter than {target['source_name']}")

    search_margin_frames = max(
        1,
        int(round(aligner.refine_search_seconds * aligner.refine_hz)),
    )
    coarse_start_frame_fine = int(round(coarse_offset_seconds * aligner.refine_hz))
    search_start = max(0, coarse_start_frame_fine - search_margin_frames)
    search_end = min(
        fine_video.size,
        coarse_start_frame_fine + fine_target.size + search_margin_frames,
    )
    if search_end - search_start < fine_target.size:
        search_start = max(0, fine_video.size - fine_target.size)
        search_end = fine_video.size

    fine_start_frame, fine_score = aligner._estimate_best_start_frame(
        fine_video[search_start:search_end],
        fine_target,
    )
    audio_offset_seconds = (search_start + fine_start_frame) / float(aligner.refine_hz)
    offset_seconds = audio_offset_seconds
    verify_detail: dict[str, float] = {}
    verify_path = target.get("verify_path")
    if isinstance(verify_path, Path) and verify_path.is_file():
        verify_adjustment = aligner._estimate_verify_adjustment_seconds(
            video_features["transient"],
            verify_path,
            audio_offset_seconds,
        )
        if verify_adjustment is not None:
            verify_adjustment_seconds, verify_detail = verify_adjustment
            offset_seconds += verify_adjustment_seconds

    return MatchProbePair(
        sample_key=str(target["sample_key"]),
        source_name=str(target["source_name"]),
        video_path=video.path,
        video_name=video.name,
        match_score=round(float(fine_score), 6),
        coarse_match_score=round(float(coarse_score), 6),
        offset_seconds=round(float(offset_seconds), 6),
        sample_kind=str(target["sample_kind"]),
        folder_name=_optional_str(target.get("folder_name")),
        osz_name=_optional_str(target.get("osz_name")),
        osz_mtime_ns=_optional_int(target.get("osz_mtime_ns")),
        audio_offset_seconds=round(float(audio_offset_seconds), 6),
        verify_adjustment_seconds=_optional_float(
            verify_detail.get("verify_adjustment_seconds")
        ),
        verify_adjustment_ms=_optional_float(verify_detail.get("verify_adjustment_ms")),
        verify_score=_optional_float(verify_detail.get("verify_score")),
        verify_window_ms=_optional_float(verify_detail.get("verify_window_ms")),
    )


def _select_greedy_probe_matches(
    pairs: Iterable[MatchProbePair],
) -> tuple[MatchProbePair, ...]:
    matches: list[MatchProbePair] = []
    used_samples: set[str] = set()
    used_videos: set[Path] = set()
    for pair in sorted(pairs, key=_probe_sort_key, reverse=True):
        if pair.sample_key in used_samples or pair.video_path in used_videos:
            continue
        used_samples.add(pair.sample_key)
        used_videos.add(pair.video_path)
        matches.append(pair)
    return tuple(matches)


def _probe_sort_key(pair: MatchProbePair) -> tuple[float, float, float, float]:
    verify_score = pair.verify_score if pair.verify_score is not None else float("-inf")
    verify_adjustment = (
        abs(pair.verify_adjustment_ms)
        if pair.verify_adjustment_ms is not None
        else float("inf")
    )
    return (
        pair.match_score,
        verify_score,
        -verify_adjustment,
        pair.coarse_match_score,
    )


def _build_probe_features(aligner: Any, samples: Any) -> dict[str, Any]:
    return {
        "coarse": aligner._build_feature_series(
            samples,
            aligner.envelope_hz,
            mode="energy",
        ),
        "fine": aligner._build_music_refine_series(samples),
        "transient": aligner._build_feature_series(
            samples,
            aligner.refine_hz,
            mode="transient",
        ),
    }


def _target_audio_samples(aligner: Any, target: Mapping[str, Any]) -> Any:
    audio_path = target.get("audio_path")
    if isinstance(audio_path, Path):
        return _audio_samples_from_path(aligner, audio_path, from_video=False)
    audio_bytes = target.get("audio_bytes")
    if not isinstance(audio_bytes, bytes):
        raise ValueError(f"{target['sample_key']} has no readable audio")
    suffix = str(target.get("audio_suffix") or ".mp3")
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / f"source{suffix}"
        source_path.write_bytes(audio_bytes)
        return _audio_samples_from_path(aligner, source_path, from_video=False)


def _audio_samples_from_path(aligner: Any, path: Path, *, from_video: bool) -> Any:
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "audio.wav"
        aligner._extract_audio_to_wav(path, wav_path, from_video=from_video)
        return aligner._load_wav_samples(wav_path)


def _scan_raw_beatmap_candidates(
    settings: Settings,
) -> tuple[list[RawBeatmapCandidate], list[str]]:
    export_dir = settings.file_management.export_dir
    if not export_dir.exists():
        return [], [f"before_traning export_dir does not exist: {export_dir}"]
    candidates: list[RawBeatmapCandidate] = []
    issues: list[str] = []
    for osz_path in sorted(export_dir.glob("*.osz"), key=lambda path: path.name.lower()):
        try:
            entry = read_osz_entry(
                osz_path,
                keyword=settings.file_formats.keyword,
                audio_output_filename=settings.file_management.audio_filename,
            )
        except zipfile.BadZipFile:
            issues.append(f"invalid osz skipped: {osz_path}")
            continue
        except Exception as error:
            issues.append(f"{osz_path}: {type(error).__name__}: {error}")
            continue
        if entry is None:
            continue
        candidates.append(
            RawBeatmapCandidate(
                osz_path=osz_path,
                source_name=entry.osu_base_name,
                osu_filename=entry.osu_filename,
                audio_source_filename=entry.audio_source_filename,
                osz_name=osz_path.name,
                osz_mtime_ns=osz_path.stat().st_mtime_ns,
            )
        )
    return candidates, issues


def _scan_video_candidates(settings: Settings) -> list[VideoCandidate]:
    video_root = settings.file_management.video_root
    if not video_root.exists():
        return []
    suffixes = {suffix.lower() for suffix in settings.file_formats.video_suffixes}
    candidates: list[VideoCandidate] = []
    for path in sorted(video_root.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        candidates.append(
            VideoCandidate(
                path=path,
                name=path.name,
                suffix=path.suffix.lower(),
                mtime_ns=path.stat().st_mtime_ns,
            )
        )
    return candidates


def _filter_unmatched_raw_candidates(
    raw_candidates: Iterable[RawBeatmapCandidate],
    manifest_items: Iterable[BeforeManifestItem],
    known_matched: Iterable[MatchedSample],
) -> tuple[RawBeatmapCandidate, ...]:
    imported_keys = {
        item.raw_identity
        for item in manifest_items
        if item.source_osz_name is not None and item.source_mtime_ns is not None
    }
    matched = tuple(known_matched)
    result: list[RawBeatmapCandidate] = []
    for candidate in raw_candidates:
        key = (candidate.source_name, candidate.osz_name, candidate.osz_mtime_ns)
        if key in imported_keys:
            continue
        if any(sample.matches_raw_candidate(candidate) for sample in matched):
            continue
        result.append(candidate)
    return tuple(result)


def _read_before_state(
    settings: Settings,
) -> tuple[tuple[BeforeManifestItem, ...], dict[tuple[str, str], dict[str, Any]], tuple[str, ...]]:
    items: tuple[BeforeManifestItem, ...] = ()
    statuses: dict[tuple[str, str], dict[str, Any]] = {}
    issues: list[str] = []
    manifest_db = settings.file_management.target_root / settings.file_management.manifest_filename
    status_db = settings.file_management.target_root / STATUS_DB_FILENAME
    if manifest_db.is_file():
        try:
            items = _read_manifest_items(manifest_db)
        except Exception as error:
            issues.append(f"{manifest_db}: {type(error).__name__}: {error}")
    if status_db.is_file():
        try:
            statuses = _read_status_rows(status_db)
        except Exception as error:
            issues.append(f"{status_db}: {type(error).__name__}: {error}")
    return items, statuses, tuple(issues)


def _read_manifest_items(db_path: Path) -> tuple[BeforeManifestItem, ...]:
    with sqlite3.connect(db_path) as connection:
        if not _sqlite_table_exists(connection, "package_manifest_item"):
            return ()
        rows = connection.execute(
            """
            SELECT folder_name, source_name, active, osu_filename,
                   source_osz_name, source_mtime_ns
            FROM package_manifest_item
            ORDER BY sequence, folder_name
            """
        ).fetchall()
    return tuple(
        BeforeManifestItem(
            folder_name=str(row[0]),
            source_name=str(row[1]),
            active=bool(row[2]),
            osu_filename=_optional_str(row[3]),
            source_osz_name=_optional_str(row[4]),
            source_mtime_ns=_optional_int(row[5]),
        )
        for row in rows
    )


def _read_status_rows(db_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    with sqlite3.connect(db_path) as connection:
        if not _sqlite_table_exists(connection, "process_step_status"):
            return {}
        rows = connection.execute(
            """
            SELECT folder_name, step, done, updated_at, detail_json
            FROM process_step_status
            """
        ).fetchall()
    return {
        (str(folder_name), str(step)): {
            "done": bool(done),
            "updated_at": updated_at,
            "detail": decode_detail(detail_json),
        }
        for folder_name, step, done, updated_at, detail_json in rows
    }


def _sqlite_table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _recover_matched_samples(
    settings: Settings,
    manifest_items: Iterable[BeforeManifestItem],
    status_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[MatchedSample, ...]:
    result: list[MatchedSample] = []
    for item in manifest_items:
        folder_has_video = _folder_has_video(settings, item.folder_name)
        video_matched = _status_done(status_rows, item.folder_name, "video_matched")
        video_segmented = _status_done(status_rows, item.folder_name, "video_segmented")
        if not (folder_has_video or video_matched or video_segmented):
            continue
        detail = _status_detail(status_rows, item.folder_name, "video_matched")
        video_path = _video_path_from_detail(detail)
        if video_path is None:
            video_path = _first_folder_video(settings, item.folder_name)
        result.append(
            MatchedSample(
                source_name=item.source_name,
                relation_status="segmented" if video_segmented else "matched",
                relation_source="before_traning_status",
                osz_name=item.source_osz_name,
                osz_mtime_ns=item.source_mtime_ns,
                folder_name=item.folder_name,
                video_path=video_path,
                video_name=video_path.name if video_path is not None else None,
                match_score=_detail_float(detail, "match_score"),
                coarse_match_score=_detail_float(detail, "coarse_match_score"),
                offset_seconds=_detail_float(detail, "offset_seconds"),
            )
        )
    return tuple(result)


def _pending_imported_samples(
    settings: Settings,
    manifest_items: Iterable[BeforeManifestItem],
    status_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[PendingImportedSample, ...]:
    result: list[PendingImportedSample] = []
    target_root = settings.file_management.target_root
    audio_filename = settings.file_management.audio_filename
    for item in manifest_items:
        if not item.active:
            continue
        if _folder_has_video(settings, item.folder_name):
            continue
        if _status_done(status_rows, item.folder_name, "video_matched"):
            continue
        audio_path = target_root / item.folder_name / audio_filename
        if not audio_path.is_file():
            continue
        verify_path = target_root / item.folder_name / VERIFY_FILENAME
        result.append(
            PendingImportedSample(
                folder_name=item.folder_name,
                source_name=item.source_name,
                audio_path=audio_path,
                verify_path=verify_path if verify_path.is_file() else None,
                osz_name=item.source_osz_name,
                osz_mtime_ns=item.source_mtime_ns,
            )
        )
    return tuple(result)


def _folder_has_video(settings: Settings, folder_name: str) -> bool:
    return _first_folder_video(settings, folder_name) is not None


def _first_folder_video(settings: Settings, folder_name: str) -> Path | None:
    folder_path = settings.file_management.target_root / folder_name
    suffixes = {suffix.lower() for suffix in settings.file_formats.video_suffixes}
    if not folder_path.is_dir():
        return None
    for path in sorted(folder_path.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix.lower() in suffixes:
            return path
    return None


def _status_done(
    status_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    folder_name: str,
    step: str,
) -> bool:
    row = status_rows.get((folder_name, step))
    return bool(row and row.get("done"))


def _status_detail(
    status_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    folder_name: str,
    step: str,
) -> Any:
    row = status_rows.get((folder_name, step))
    return row.get("detail") if row is not None else None


def _video_path_from_detail(detail: Any) -> Path | None:
    if not isinstance(detail, Mapping):
        return None
    value = detail.get("video_path") or detail.get("source_video_path")
    return Path(value) if isinstance(value, str) and value else None


def _detail_float(detail: Any, key: str) -> float | None:
    if not isinstance(detail, Mapping):
        return None
    return _optional_float(detail.get(key))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_json_ready(item) for item in value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "BeforeTrainingSampleInspection",
    "DEFAULT_MATCHED_MANIFEST",
    "MatchedSample",
    "MatchedSampleManifest",
    "MatchProbePair",
    "MatchProbeReport",
    "PendingImportedSample",
    "RawBeatmapCandidate",
    "VideoCandidate",
    "inspect_before_training_samples",
    "probe_before_training_matches",
    "recover_matched_sample_manifest",
]
