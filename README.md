# Rocket League Multi-Mode PPO Bot

Native C++ PPO training and RLBot integration for a single shared policy (`s`, left-mirrored `MultiModeObs`, 90-action `DefaultAction`) across **1v1, 2v2, and 3v3**, built on the bundled GigaLearnCPP / RLGymCPP / RocketSim stack.

## Current status — paused

Training is cleanly paused with no running background processes.

- **Canonical snapshot:** [`training-backup/`](training-backup/) (`status: paused`, SHA-256 verified via `training-backup/manifest.json`).
- **Latest resumable learner checkpoint:** `training-backup/learner/3250007238/` (full policy, critic, shared head, slot encoder, Adam optimizer states, and running stats; mode-aware v3 zero-sum reward recipe).
- **Selected best checkpoint in snapshot:** `training-backup/best/3244539846/` (native sampled actions at temperature `1.0` — Screen `1594.89`, Confirmation `1608.13` against the immutable 5-bot reference pool `aa048b45...`).
- **Target:** `1700.0` combined Elo (`(1v1 + 2v2 + 3v3) / 3`) on fixed-pool screen, confirmation, and held-out audit.
- **Metrics & backups:** Metrics stream **directly** from the training host to Weights & Biases over one HTTPS connection (`tools/ranked_wandb.py`; no GitHub Actions, no telemetry issues, no git commits). The **only** GitHub traffic training creates is a checkpoint backup every 30 minutes (`1800s`) via `tools/training_service.py`, so repository rate limits are never involved in logging. `Overall Steps/Second` is measured over contiguous wall-clock windows so zero training, saving, evaluation, or reporting overhead is excluded.

## Build and resume

```bash
./setup_linux.sh
# Restricted Linux x86-64 / Python 3.11 CPU fallback:
# bash tools/setup_pypi_cpu.sh

# Inspect paused ratings without starting training:
python3 tools/show_rating.py training-backup

# Restore working copy only (no training started):
python3 tools/resume_training.py --restore-only

# Store the W&B API key once for direct streaming (private file outside the repo, mode 600);
# alternatively export WANDB_API_KEY or put it in ~/.netrc. See RANKED_TRAINING.md.
python3 tools/ranked_wandb.py login

# Explicitly resume paused training (30m GitHub backup + direct W&B streaming):
python3 tools/resume_training.py

# Same, but keep telemetry local (no W&B at all):
python3 tools/resume_training.py --no-wandb
# ...or run the trainer without any helper process:
python3 tools/resume_training.py --without-services
```

See [RANKED_TRAINING.md](RANKED_TRAINING.md) for the evaluation protocol, CLI options, W&B streaming, and recovery details.

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
