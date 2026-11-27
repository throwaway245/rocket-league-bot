# Rocket League Multi-Mode PPO Bot

Native C++ PPO training and RLBot integration for a single shared policy (`s`, left-mirrored `MultiModeObs`, 90-action `DefaultAction`) across **1v1, 2v2, and 3v3**, built on the bundled GigaLearnCPP / RLGymCPP / RocketSim stack.

## Current status

Small-model training and foundation distillation are paused. See [DISTILLATION.md](DISTILLATION.md).

- **Latest paused small-model snapshot:** [`training-backup/small-sixteen-hour/`](training-backup/small-sixteen-hour/) contains full learner checkpoint `7745735391`, its opponent pool, state and SHA-256 manifest. The experimental v8 reward override remains active.
- **Selected best:** `training-backup/small-sixteen-hour/best/7545038559/`, with screen `1980.37` and confirmation `1981.57` against the evolving pool; these are not held-out ratings.
- The requested 16-hour run was paused early at the user's request, after approximately 8 hours 15 minutes of wall-clock time (including evaluations); no distillation ran during it.
- The earlier small-model snapshot and distillation teacher remain in `training-backup/`; large distillation phases are preserved under `training-backup/large-distillation/`. Phase 44's monitored policy argmax agreement was approximately 45–49%, below the 90% target.
- Training writes local metrics; training-time W&B is disabled.

## Build and resume

```bash
./setup_linux.sh
# Restricted Linux x86-64 / Python 3.11 CPU fallback:
# bash tools/setup_pypi_cpu.sh

# Inspect paused ratings without starting training:
python3 tools/show_rating.py training-backup/small-sixteen-hour

# Verify and restore the latest snapshot into a working copy (does not start training):
python3 tools/resume_training.py --snapshot training-backup/small-sixteen-hour --restore-only

# Resume explicitly, without a backup service or W&B:
python3 tools/resume_training.py --snapshot training-backup/small-sixteen-hour --without-services
```

The restore command retains an existing working campaign rather than overwriting it; inspect or
move that working copy explicitly if you need to restore a different snapshot. The default backup
service writes to `training-backup/`, so do not enable it if the older snapshot must stay immutable.

```bash
# Original pre-run small-model snapshot (also retained):
python3 tools/resume_training.py --snapshot training-backup --restore-only
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
