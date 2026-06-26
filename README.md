# osu video training workspace

This repository prepares osu! gameplay data and trains a video-to-action model.
The current recommended entry is the unified full training flow:

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc '
cd /home/dev/workspace
PYTHONPATH=src:. python -m traning.main full-flow \
  --config configs/model_full_small_vram.yaml \
  --device cuda \
  --resume \
  --resume-policy auto \
  --auto-launch-full \
  --progress-ui rich \
  --progress-language zh-CN
'
```

For a non-training check first:

```bash
PYTHONPATH=src:. python -m traning.main full-flow --mode plan --config configs/model_small_vram.yaml --device cpu
PYTHONPATH=src:. python -m traning.main full-flow --mode dry-run --config configs/model_small_vram.yaml --device cpu --progress-ui off
```

Resume from the latest inheritance package:

```bash
PYTHONPATH=src:. python -m traning.main full-flow --config configs/model_small_vram.yaml --device cuda --resume --resume-policy auto
```

Important outputs are written under `artifacts/training_runs/<run_id>/`: `full_flow_state.json`, `resume_report.json`, `reports/full_flow_report.md`, `ramp/`, `artifacts/`, and `inheritance/`.

More user-facing docs:

- [Quick Start](docs/QUICK_START.md)
- [Training Workflow](docs/TRAINING_WORKFLOW.md)
- [Documentation Index](docs/INDEX.md)

Engineering docs for Codex and maintainers live in [docs/codex/INDEX.md](docs/codex/INDEX.md).
