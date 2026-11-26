# Foundation-v1 and small-to-large distillation

## Frozen model contract

`foundation-v1` is the project's fixed large architecture. Never change its preset in place;
introduce a separately named version and migration plan for incompatible changes. This is a
model/checkpoint ABI, not a claim that the architecture is universally optimal.

- Existing `MultiModeObs`: fixed 1v1/2v2/3v3 representation and teacher's left-mirrored view.
- Shared context-conditioned per-player encoder: **256, 256**; masked mean and max pooling
  separately for teammate and opponent slots. This is permutation-invariant within each team.
- Shared trunk: **1024, 1024, 1024**.
- Actor and critic: separate **512, 512** heads, 90-action lookup policy and scalar value.
- ReLU and layer normalization; no recurrence or attention. Those would require a new version.
- Native model reports **4,167,258** actor parameters including shared trunk/slot encoder;
  critic adds **790,017**, for **4,957,275** total. Optimizers are separate.

The larger trunk provides substantial capacity for future teachers while retaining the tested
multimode input/action contract and inexpensive, order-invariant per-player encoding. It is
roughly ten times the 404,314-parameter small actor. Training this network is materially slower
than the current CPU-sized `s` model; the shape is fixed, not the distillation or PPO recipe.

## Actor-only successor (experimental)

`foundation-policy-v1` keeps the exact `foundation-v1` actor and shared encoder/trunk,
with no critic or critic optimizer. It is for inference and distillation only; PPO is
explicitly rejected. Inference already used only the actor. To migrate an available
`foundation-v1` checkpoint without modifying it, run
`python3 tools/export_actor_only.py OLD_CHECKPOINT NEW_CHECKPOINT`. The destination
must not exist. The export copies and verifies actor weights, carries compatible actor
optimizer state when present, and excludes critic weights and optimizer state. Keep the
original until inference and resumed-distillation equivalence are validated.

Winner trace capture uses five-minute simulated regulation matches, resets kickoffs after
goals without resetting the match score, and discards tied and unfinished matches.
It samples actions from the winning side only, with action masks and matching mirrored
observations. The per-side trajectory collector samples every five simulation decisions
and uses bounded reservoir sampling across completed winning matches. Combined replay
retains at most 16,384 samples per mode. With replay weight 1 and teacher rollout 0, transfer updates are based solely
on recorded winner actions; no fresh teacher probabilities are queried. `--distill-replay-history`
optionally mixes a second immutable compatible shard with the current one.
`--distill-replay-history-dir` can additionally mix 1–8 `.pt` shards from a directory.
Each shard is bounded to 100,000 observations and the combined replay to 250,000;
rows are sampled uniformly from the combined shards. Phase backups can include a gzip-compressed, SHA-256-verified winner replay;
decompress it to an isolated working directory before using it for retention. Replay is
imitation, not PPO. These experimental paths require native regression tests and independent validation;
no candidate should be promoted automatically. Each completed distillation phase must be
verified, saved under `training-backup/large-distillation/`, committed, and pushed before
another phase starts. Run `tools/backup_distillation_phase.py CHECKPOINT --phase N
--match-results RESULTS.json --winner-replay REPLAY.pt` to create a non-overwriting phase
directory containing weights, optimizer state, match evidence, compressed replay, and
SHA-256 checksums. Commit and push it on
the current branch; the helper intentionally does not run Git. `Total Timesteps` counts
student rollout observations, not unique replay actions or PPO steps.

The first new actor was initialized from scratch at seed 20271201; its initial checkpoint
is `phase-0000`. The original 51,132,600-observation student was lost with the ephemeral
workspace and is not represented by this checkpoint.

## First teacher and method

The first teacher is the immutable **best confirmed** small checkpoint
`training-backup/best/5163287775` (confirmed ranking 1883.42 under the old internal protocol).
This is deliberately separate from the latest small checkpoint. Preserve the original backup.
Use a separate output directory and never train or save into `training-backup/`.

The `--distill-teacher` path invokes GigaLearn's `StartTransferLearn`. Compatible multimode
pooled-slot checkpoints (`s`, `m`, `l`, `xl`, or `foundation-v1`) can supply future teachers;
changing a student's teacher requires the explicit `--distill-new-teacher` flag. Keep a separate
immutable snapshot before each such phase. The same observation and action contract means both
networks see the exact same mirrored state and action masks. Training
minimizes teacher-to-student KL divergence over all 90 action probabilities; it is not just hard
argmax imitation. Each batch is generated either by the frozen teacher or the student, to cover
both teacher and student trajectories. The `Total Timesteps` of a distillation checkpoint are
**distillation observations**, not the teacher's prior PPO training steps. The critic is not
transferred and is untrained; do not use this checkpoint as a fully initialized PPO learner
without a separate value-training plan. The checkpoint stores the teacher path and refuses a
resume with another teacher path. Back up/hash both teacher and student when moving them.

Example (no GitHub transport, W&B or Actions):

```bash
GGL_WANDB=0 GGL_METRICS_EVERY=1 GGL_METRICS_DIR=results/foundation-v1/metrics \
  build/GigaLearnBot --model foundation-v1 \
  --distill-teacher training-backup/best/5163287775 \
  --checkpoints results/foundation-v1/checkpoints \
  --device cpu --threads 2 --games 22 --modes 1v1,2v2,3v3 --mode-balance players \
  --distill-batch 1024 --distill-epochs 2 --distill-teacher-rollout 0.75 \
  --lr 0.0001 --max-minutes 60 --ts-per-save 20000 --keep-checkpoints 1 \
  --lean-metrics --fuzzed-kickoffs --tactical-state-prob 0.15
```

For unattended repeated blocks, run `python3 tools/distill_foundation.py`. It trains 15-minute
blocks with a 50/50 teacher/student rollout mixture, checks KL and agreement on fresh seeded
rollouts in all three modes, and continues if the imitation gate fails. If that gate passes, it
runs side-balanced direct matches and matches against two frozen older opponents, comparing the
student to the teacher; it continues if these heuristic match gates fail. On passing both it
pauses for an independent longer audit, without auto-promoting the student. The controller uses
no GitHub or W&B transport. Set `--blocks` to cap the run; Ctrl-C/SIGTERM requests a graceful
checkpoint and stops it. Its logs and JSON results live in `results/foundation-v1/`.

A time limit or SIGTERM saves at the end of the current batch; `--max-timesteps` is an optional
absolute distillation-step limit. Resume with the same model, teacher, output and mirror setting.
The current CLI pins the teacher **path**, not its content hash; verify the backup manifest before
resuming, and don't replace the teacher in place. `--distill-eval` instead measures one fresh
batch without updating or saving the student. Use a seed not used for training, test teacher- and
student-controlled rollouts, and test each mode separately. `Transfer KL` is mean per-state
KL(teacher || student) in nats; `Transfer Learn Accuracy` is argmax agreement. Both are on the
collected batch before updates, not independent proof of competitive strength.

## Promotion gate

Do not describe a student as equally strong until both tests pass:

1. Held-out, independently seeded 1v1, 2v2 and 3v3 rollouts with both rollout controllers:
   KL <= 0.10 nats and argmax agreement >= 90% in each of the six conditions, over multiple
   batches/seeds. These are provisional engineering thresholds, not a guarantee of equal play.
2. Side-balanced direct matches against the teacher, plus evaluation against a fixed opponent
   pool for all three modes, with enough games/goals to bound uncertainty. Report goal shares
   and confidence intervals; require the preselected non-inferiority margin to be met in every
   mode. Direct self-play equality alone can conceal shared weaknesses.

`--eval` supports separate model types by reading each checkpoint's `INFO.json`. Use explicit
seeds and separate mode runs. Never compare an internal Elo directly with other bots' published
ratings. Keep unpromoted students and the teacher clearly distinguished.
