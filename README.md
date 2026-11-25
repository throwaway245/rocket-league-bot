# Rocket League Multi-Mode PPO Bot

Native C++ PPO training and RLBot integration for a single shared policy (`s`, left-mirrored `MultiModeObs`, 90-action `DefaultAction`) across **1v1, 2v2, and 3v3**, built on the bundled GigaLearnCPP / RLGymCPP / RocketSim stack.

## Current status

Small-model training and foundation distillation are paused. See [DISTILLATION.md](DISTILLATION.md).

- **Canonical snapshot:** [`training-backup/`](training-backup/) (`status: paused`, SHA-256 verified via `training-backup/manifest.json`).
- **Latest resumable learner checkpoint:** `training-backup/learner/7058101983/`, including policy, critic, shared head, optimizer states and running stats; the active reward override is experimental v8.
- **Selected best checkpoint:** `training-backup/best/6898483167/`, with screen `1961.04` and confirmation `1959.82` against the evolving campaign pool. These ratings are not a held-out audit.
- **Distillation teacher:** `training-backup/best/5163287775/` is retained separately to preserve the existing student's teacher contract.
- **Metrics & backups:** Training writes local JSONL metrics; all training-time W&B logging is temporarily disabled. Optional checkpoint backups are separate from telemetry.

## Build and resume

```bash
./setup_linux.sh
# Restricted Linux x86-64 / Python 3.11 CPU fallback:
# bash tools/setup_pypi_cpu.sh

# Inspect paused ratings without starting training:
python3 tools/show_rating.py training-backup

# Restore working copy only (no training started):
python3 tools/resume_training.py --restore-only

# Resume paused training (local metrics; optional checkpoint backup):
python3 tools/resume_training.py

# ...or run the trainer without any helper process:
python3 tools/resume_training.py --without-services
```

See [RANKED_TRAINING.md](RANKED_TRAINING.md) for the evaluation protocol, CLI options, manual post-training export, and recovery details.

## Repository layout

- `src/` — Bot entrypoint (`ExampleMain.cpp`), mode-aware zero-sum rewards, tactical state setter, and RLBot client.
- `GigaLearnCPP/`, `RLBotCPP/`, `rlbot/` — Bundled PPO/RLGymCPP/RocketSim framework and RLBot runner.
- `training-backup/` — Canonical paused campaign (`learner/`, `best/`, immutable `pool/`, `eval/`, `metrics/`, `state.json`, `manifest.json`).
- `tools/` — Campaign driver (`train_ranked.py`), recovery (`resume_training.py`), headless 30-minute GitHub backup worker (`training_service.py`), direct W&B streaming (`ranked_wandb.py`), rating viewer (`show_rating.py`), throughput profiler (`benchmark.py`), CPU wheel bootstrap (`setup_pypi_cpu.sh`), and regression tests (`test_*.py`).

There is no GitHub Actions workflow. Telemetry never touches the repository, so pushing a checkpoint
backup every 30 minutes is the whole of the project's GitHub API usage.

## Deterministic viewing

```bash
build/GigaLearnBot --render --checkpoints training-backup/best --device cpu --modes 1v1,2v2,3v3
```

Streams RocketSimVis UDP frames in a single arena using masked argmax actions without training or saving.
