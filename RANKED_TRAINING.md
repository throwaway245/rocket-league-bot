# Ranked Multi-Mode Training, W&B Telemetry, and Recovery

## Resuming and pausing

`training-backup/` holds the single canonical paused snapshot (verified by `training-backup/manifest.json`) and is never trained in directly.

```bash
# Restore into results/ without launching training:
python3 tools/resume_training.py --restore-only

# Verify and resume the finite training schedule (starts the 30m GitHub backup + direct W&B stream):
python3 tools/resume_training.py

# Resume one fixed 15-minute train/evaluate cycle and repeat it indefinitely with a rolling pool of 10 old policies:
python3 tools/resume_training.py --continuous --evolving-opponents

# Keep the checkpoint backup but log nothing to W&B:
python3 tools/resume_training.py --no-wandb

# Resume trainer only (no backup worker, no W&B stream):
python3 tools/resume_training.py --without-services
```

Send `SIGINT` (`Ctrl+C`) or `SIGTERM` once to pause cleanly: the native learner completes the current iteration, writes a full checkpoint, updates `state.json` to `paused`, and exits. In `--continuous` mode the original pilot/final-short-block schedule is ignored: one regular 15-minute training block is followed by evaluation, and the next identical cycle begins from the latest checkpoint. This repeats without a stage-count/time limit; the held-out audit is only part of a finite schedule.

To start a fresh campaign (for example, a 12-hour run = `720` minutes) from the paused checkpoint against the same frozen reference pool:

```bash
python3 tools/train_ranked.py \
  --base training-backup/learner/3250007238 \
  --reference-pool training-backup/pool \
  --out results/weights1700/campaign \
  --minutes 720 --pilot-minutes 15 --block-minutes 30 --rounds 40 \
  --target-ranking 1700 \
  --github-backup --backup-interval 1800 --wandb-online --wandb-interval 10
```

To continue the recipe from the newest canonical checkpoint with a rolling opponent pool that grows from the ten historical policies towards fifteen, in endless train/evaluate cycles:

```bash
python3 tools/train_ranked.py \
  --base training-backup/learner/4876757727 \
  --out results/weights1700-v6/campaign \
  --minutes 4320 --pilot-minutes 15 --block-minutes 30 --rounds 40 \
  --seed 20261121 --audit-seed 4001 \
  --evolving-opponents --opponent-pool-size 15 --promotion-margin 50.0 \
  --continuous --throughput-profile cpu-20k --reward-metrics \
  --lr 9e-5 --critic-lr 0.00022 --epochs 2 --entropy-scale 0.008 \
  --norm-advantages mode --target-kl 0.02 --gamma 0.995 \
  --return-stat-max-count 200000 --random-state-prob 0.0 --tactical-state-prob 0.15 \
  --train-vs-old 0.5 --ts-per-version 20m --max-versions 14 \
  --skill-minutes 25.0 --ts-per-save 10m \
  --mask-entropy --fuzzed-kickoffs --rewards v5 \
  --github-backup --backup-interval 1800 --wandb-online --wandb-interval 10
```

The seed pool is the `policy_versions` folder next to `--base`; the campaign copies it into its own
directory and never trains in the canonical snapshot. In `--continuous` mode the block length, not
`--minutes`, decides how long each training stage runs, and the campaign repeats until it is
interrupted.

## Checkpoint cadence and resuming

- **Saves are derived, not guessed.** `--ts-per-save` is computed per cycle
  (`train_ranked.save_cadence_args`) from the campaign's measured steps/second and capped at
  `--backup-interval × (keep − 1)` and at half a stage, so a kept checkpoint always survives until
  the 30-minute backup worker has published it. A continuous campaign's first stage is only half a
  block long, so `--ts-per-save-after-first` gives it a shorter interval until its own checkpoint
  exists. An explicit `--ts-per-save` still wins, and is recorded in the pinned config.
- **`resume_training.py` replays the pinned recipe token for token.** `train_ranked.py` pins the
  entire config (`train_args`, plan, seeds, pool hash) and refuses to continue when anything
  differs, so the resume builder forwards the stored tokens verbatim (minus the cadence above).
  A regression test asserts the round trip — the previous builder silently dropped `--games`,
  `--mode-balance`, `--fixed-rollout`, `--lean-metrics` and `--keep-checkpoints`, which made resuming
  fail the integrity check.

## Fixed-pool evaluation protocol

`Rating/Ranking = (1v1 Elo + 2v2 Elo + 3v3 Elo) / 3` using native sampled actions at softmax temperature `1.0` (`deterministic=False`, `policy_temperature=1.0`).

- **Frozen opponent pool:** 5 immutable left-mirrored policies in `training-backup/pool/` (SHA-256 `aa048b453da50d8c7c96913692d3e4d70bd141b5247cbd6519fe11589539d81c`).
- **Fixtures:** Balanced across all opponents and both team colors, 36 arenas (12 per mode). The reported rating is the exact endpoint of the final rated round.
- **Tournament Elo:** One Elo update per round per mode, from that round's share of its goals:
  `delta = (C / rounds) * (goals_for / (goals_for + goals_against) - expected)`, `expected = 1 / (1 + 10^((opponent - current) / 400))`, `C = 100` (`--skill-tournament-elo-budget`). The sum over the rounds is at most `C` however many goals, arenas or players a tournament has, so there is no cap on top of the rating — only a fixed budget per tournament.
  Because the update is `C * 0.00144 * (true - current)`, `C` sets how fast the rating follows real improvement: `100` closes about 14 % of the gap per tournament (1/e in ~7 tournaments, 90 % of a 50-point gap in ~16) while a 40-round screen still estimates the change to only about +-2 points.
- **Promotion gate:** Screen on seed `1009` (`40` rounds); if above current best screen, confirm on seed `2003` (`80` rounds). Promotes only when both matching-seed scores improve and their mean gain is at least `+8.0` Elo. The confirmation always runs exactly twice the rounds of the screen it confirms.
- **Held-out audit:** Evaluated once at campaign completion (`80` rounds on the predeclared `--audit-seed`, never used for selection).
- **Round count follows the pool:** The native anchored tournament only accepts a number of rounds that is a multiple of twice the number of loaded opponents, so every member gets the same number of games on both team colors. `--rounds` is therefore a *budget*, not a fixed count: each evaluation derives the smallest multiple of twice the current pool size that is at least the budget. A ten-policy pool rates with `40` rounds, an eleven-policy pool with `44`, up to a fifteen-policy pool with `60`; the confirmation doubles that. The derivation is a pure function of budget and pool size, so a resumed campaign rates exactly the same rounds as the run before it, and every stage records the count it used (`eval_rounds`, `eval_rounds_confirm`).
- **Cache integrity:** Cached `eval/*.json` entries fingerprint `POLICY.lt`, `SHARED_HEAD.lt`, and `SHARED_HEAD_SLOTS.lt` by SHA-256 and validate pool digest and temperature-1 sampled execution.
- **Current Elo only:** A saved result stores the endpoint Elo of each mode (`1v1`, `2v2`, `3v3`, `Ranking`), plus the Elo the tournament started from and the one written back into the policy so a cached result stays verifiable. The per-round rating series and averaged Elo are discarded, so checkpoint JSON stays a compact record of the current rating instead of a rating history.
- **Continuous rating:** The evaluated policy keeps the Elo it already carries — it never starts from the pool's mean or any other shared anchor. The tournament's endpoint is written back into that policy's own `RUNNING_STATS.json` (`skill_ratings`), so the next tournament starts where this one ended, and an accepted pool copy carries exactly the Elo it had when it was accepted. Opponents are only ever read.

### Start-state randomness

Sampled actions at temperature `1.0` are not the only source of variation: every round of a rating tournament also starts from its own randomized kickoff. Skill matches always install `RLGC::FuzzedKickoffState`, seeded per arena from the evaluation seed (`PrepareAnchoredPool`), and each arena re-draws from its own deterministic RNG stream on every reset — so rounds never replay the same start state, and the same seed still reproduces a rating exactly. Training does the same with `--fuzzed-kickoffs`, which `train_ranked.py` always appends and records as `runtime_overrides.fuzzed_kickoffs`. A rating whose `settings.fuzzed_kickoffs` is not `true` is rejected, so a recipe or protocol change cannot silently bring back repeated rounds.

## Reward presets

`--rewards` accepts either explicit `Name=weight` pairs or a named preset; every preset lists *all* weights, so a
preset never inherits a changed default. The resolved weights are logged (`Rewards (preset v4): ...`) and recorded in
the run config as `rewards`, `reward_weights`, and `reward_preset`.

- `v3` — the recipe the 66-cycle campaign was trained with: `Goal=750` plus mode-aware event rewards, every dense
  RLGym-style primitive off.
- `v4` — resumes from a `v3` checkpoint with the dense shaping that RLGym ships by default switched back on, which is
  the intended fix for the ~1795 Elo plateau: the policy was only paid for the outcome of a touch, never for the
  approach. `Air=0.05` and `FaceBall=0.1` pay for aerial play, `TouchAccel=10` for flicks and power shots,
  `TeamSpacing=0.75` for rotation, and `NearestVelocityToBall=0.5` for the nearest player of each team driving at the
  ball. Each is deliberately below the weight the `RewardDefs` table suggests (0.1 / 0.25 / 15 / 1 / 0), because those
  suggestions assume an episode with no event rewards, whereas here every term is layered on top of `Goal=750`.
  `NearestVelocityToBall` stays at half of RLGym's `1.0` since `DirectionalTouch=16` and `ObjectiveBallVel=10` already
  pay for ball advancement. `Goal` remains the anchor and the event rewards move only slightly (`GoalSpeed 120 -> 110`,
  `Save 28 -> 30`, `Shot 18 -> 20`, `ObjectiveBallVel 6.5 -> 10`, `TeamPressure 1.6 -> 2`, `SoloDiscipline 0.55 -> 0.7`,
  `DefensiveCover 0.45 -> 0.6`, `DoubleCommit 0.9 -> 1`, `PickupBoostZS 3.5 -> 4`, `DemoZS 25 -> 15`, `Bump 4 -> 3`).

- `v5` — `v4` without `FaceBall`. `FaceBallReward` is `rotMat.forward . normalize(ball - player)`, paid every step
  no matter what the car is doing, so it is a reward for *looking* at the ball: at weight 0.1 it paid about
  0.032 per step, i.e. ~19-22 per 600-step episode, more than a whole `Shot=20` event, and its cheapest optimum is
  hovering near the ball with the nose pointed at it. Measured on the 4.2B-step policy the raw term averaged 0.32
(on a fresh policy, 0.15), so the per-step figure is the weight times that fraction of steps.

Keep `Goal` first when adding a term: a dense reward that out-earns a goal over an episode makes hovering and ball
chasing optimal, which is exactly the plateau this preset set out to break.

- `v6` — `v5`, rebalanced from a **measurement** of `v5`'s own per-term payouts rather than from reasoning about the
  code. `Learner` reports every active reward under `Rewards/<ClassName>` in the metrics stream (a sample of arenas,
  already multiplied by that term's weight), so weighting the samples by each term's share of the total gives the
  real signal mix. Over 170M timesteps of the live campaign the terms were stable (no drift above 0.006), and the
  weighted mix was:

  | term | raw | weight | contribution | share |
  |---|---|---|---|---|
  | `NearestVelocityToBall` | 0.164 | 0.5 | +0.0820 | **57 %** |
  | `DirectionalTouch` | 0.0040 | 16 | +0.0645 | 45 % |
  | `TeamSpacing` | −0.058 | 0.75 | −0.0434 | **30 %** |
  | `DoubleCommit` | −0.043 | 1 | −0.0425 | **30 %** |
  | `Goal` | ≈0 | 750 | −0.0344 | 24 % |
  | `TouchAccel` | 0.0031 | 10 | +0.0314 | 22 % |
  | `SaveBoost` | 0.407 | 0.05 | +0.0203 | **14 %** |
  | `Air` | 0.338 | 0.05 | +0.0169 | 12 % |

  Three things fell out of that table:

  1. `NearestVelocityToBall` alone was 57 % of the reward. In 1v1 it is *identical* to the plain
     `VelocityPlayerToBall` product, whose cheapest optimum is driving straight at the ball irrespective of angle,
     boost or whether the touch is a good idea — and `v3`/`v4`/`v5` had deliberately switched that family off for
     exactly this reason. Halved to `0.25`; `DirectionalTouch` (16) stays the touch-quality signal.
  2. `TeamSpacing` and `DoubleCommit` were **30 % each**, and both are pure penalties for a robot team standing next
     to itself. A large part of that is the kickoff respawn (cars are *placed* adjacent, so the penalty fires for
     the first fraction of every episode). Both are now distance-gated (`MIN_BALL_DIST = 400`), so they only judge
     shape once the cars have had a moment to separate.
  3. `SaveBoost` was 14 % of the reward at 0.407 raw while being blind to velocity *and* position — its optimum is
     parking on 100 boost anywhere on the pitch, and it out-earned every ball-directed reward. Replaced by
     `GoalSideBoostReward` (`sqrt(boost) * closing speed toward own goal`, only while goal-side of the ball), which
     keeps the "hold boost for the save" incentive and drops the "park on boost" one. At a realistic recovery duty
     cycle it pays ~4 % of the reward instead of 14 %.

  `BehindBall` is now off by default: `clamp(diff/1000, -0.5, 1)` is maximised by a car parked between the ball and
  its own goal at any distance, in every mode, while `DefensiveCover` (2v2/3v3) and `SoloDiscipline` (1v1) already
  pay that positioning *by threat level*, which is the version that actually matters.

  Unchanged on purpose: `Goal=750` stays the anchor, and the whole event/touch-quality family (`Shot 20`, `Save 30`,
  `ScorerAssist 75`, `GoalSpeed 110`, `ObjectiveBallVel 10`, `DirectionalTouch 16`, `KickoffSpeed 6`) is untouched —
  the measurement showed no reason to touch them.

- **Changing rewards mid-campaign is a runtime override, not a recipe change.** `python3 tools/resume_training.py
  --runtime-rewards v6` passes `--runtime-rewards` to `train_ranked.py`, which validates the preset against the bot's
  own tables, records it under `runtime_overrides.rewards` in `state.json`, and appends it *after* the pinned
  `--rewards` so it wins at the trainer's argv. The pinned `train_args` is left alone, so the campaign keeps
  resuming; the next resume picks the override up without the flag.

## Evolving old-policy pool

`--evolving-opponents` migrates/resumes a campaign with a single shared set of historical policies (`--opponent-pool-size`, 15 here) for both `--train-vs-old`/skill matches and anchored evaluation. The initial set is sampled evenly across all available prior policy-version timestamps so it spans the historical range. When history holds fewer policies than requested — a 10 -> 15 increase on a campaign that only ever saved 10 versions, for instance — the pool seeds with what exists and **grows through promotions until it reaches the requested size**, evicting nothing until then; from that point on it is FIFO at a fixed size. Because the pool changes size while the campaign runs, the rating round count is derived from the pool as it is at evaluation time (see *Round count follows the pool* above), so every member still gets the same number of games on both team colors. Native automatic policy saving is effectively disabled; only the campaign's promotion gate changes the set.

After each training block, the candidate is rated against all ten opponents on the screen and confirmation seeds. It is added only if its raw endpoint `Ranking` is at least 50 Elo above the strongest opponent on both evaluations. On promotion it enters the set and, once the set is full, the oldest timestamp is evicted. The incumbent is re-rated whenever the pool changes so comparisons stay on the same pool-specific scale. Pool membership, digest, generation, and strongest-opponent margin are recorded in campaign history. The threshold and pool size are configurable with `--promotion-margin` and `--opponent-pool-size`.

### Immutable old policies

Every policy version accepted into the pool is an immutable snapshot: its weights and a small `STATS.json` holding only the Elo it had at acceptance (`skill_ratings`) plus the mirror flag are written exactly once, when the copy is created. This does not apply to the active learner — its weights keep training, and its own Elo is continuous: native sessions move it, and every anchored evaluation tournament writes its endpoint back into the learner's `RUNNING_STATS.json`. Only the accepted copy is frozen. Nothing rewrites an accepted copy afterwards — no Elo updates from later skill matches, no metadata refreshes — so old policy folders stay byte-identical and backup commits stop churning.

`train_ranked.py` pins a SHA-256 manifest of every version (`frozen_versions` in `state.json`), re-checks it at each cycle boundary, and compares the trainer's `learner/policy_versions/` copy against the pool after every training block. Any re-rated or re-annotated old version aborts the cycle instead of being silently accepted, and the GitHub backup worker refuses to publish a snapshot that no longer matches the pinned manifest. Identical copies are never recopied, and FIFO eviction only removes whole versions.

In this mode, GitHub `training-backup/` stores only the latest full model checkpoint, the exact old-policy set under `learner/policy_versions/`, and the small state/manifest metadata needed to verify and resume; it does not duplicate the pool, best model, evaluation cache, or training metric logs. Local campaign history/metrics remain in `results/...`.

The reported Elo values use the exact final rating endpoint from a match/session, not a cross-round smoothed or interpolated series. W&B logs each rating event as a discrete sample (`Elo/...` for the native tracker and `Rating/...` for the anchored candidate evaluation); other sparse scalar diagnostics may still be window-averaged.

Inspect saved ratings anytime with:

```bash
python3 tools/show_rating.py training-backup
```

Verify that the ten old policies are still byte-identical to their creation record (exit code `1` on any rewrite):

```bash
python3 tools/verify_frozen_versions.py results/weights1700-v3/campaign
```

## Throughput accounting (`Overall Steps/Second`)

`Overall Steps/Second` uses contiguous monotonic wall-clock intervals with zero excluded CPU or wall time:

- Inside `GigaLearnCPP` (`Learner.cpp` and `python_scripts/metric_receiver.py`), each emitted window spans from process start (first report) or the exact end of the previous emitted report through opponent switching, rollout collection, inference, trajectory packing, GAE, per-mode advantage normalization, PPO updates, skill-tracker rating matches, policy-version writes, checkpoint saves, and metric serialization/flushing (`Iteration Time` and `Overhead Time` are logged alongside `Collection Time` and `Consumption Time`).
- Across ranked campaign stages (`train_ranked.py` and `ranked_wandb.py`), stage `steps_per_second` and stage-boundary `Overall Steps/Second` also fold in subprocess startup/teardown, fixed-pool screen/confirmation evaluations, and checkpoint promotion copies.

## Suspension ray casts

The vehicles' suspension rays are cast by Bullet's `btDefaultVehicleRaycaster`, which ray tests the whole world through RocketSim's voxel broadphase. Two specialized versions were implemented and measured, and both were reverted, because neither of them made the training faster:

- A specialized ray caster that checks the bounding box of a candidate before its exact ray test. The check only skips objects whose exact ray test is cheap anyway (the triangle meshes and the convex shapes), and the exact test of a candidate that survives it is still Bullet's, so the check mostly adds work. Measured on the same rays, with both casters timed individually and the profiler's own timers turned off for both of them: the specialized caster needed 445.4 ns per ray, Bullet's 437.9 ns per ray, i.e. it was 1.7% slower. Measured over the whole trainer, with the two variants alternating over repeated runs, so that the drift of the host cannot favour one of them: the median throughput ratio was 0.982x and the mean 0.978x, i.e. parity at best.
- The analytic suspension collision grid of RocketSim (`SuspensionCollisionGrid`), which resolves the ray against the arena's planes instead of ray testing the world. It fell back to the full Bullet ray test for every ray, because the cell a ray starts in always contains the car that owns the ray, and the ray cost 1.28x what it costs without the grid.

An analytic intersection of the ray with the arena's planes was implemented as well, in the same arithmetic as Bullet's generic path for a concave shape. It was not faster, because it does the same work: the shapes' ray test builds two triangles around the ray and intersects them.

### The direct ray test of a static plane

The ray test of a `btStaticPlaneShape` no longer intersects the ray with the two triangles that the shape builds around it. `btCollisionWorld::rayTestSingleInternal` does the same arithmetic without the triangles:

- The ray is put into the local frame of the shape and the square of the two triangles is built exactly as `btStaticPlaneShape::processAllTriangles` builds it: its size is the half diagonal of the bounding box of the ray, its center is the projection of the middle of the ray onto the plane, and its sides follow from `btPlaneSpace1`.
- The plane of the first triangle, and the distances of the two ends of the ray to it, are computed exactly as `btTriangleRaycastCallback::processTriangle` computes them. The crossing of the ray, its fraction and the side that the ray comes from therefore follow from the same numbers as in the generic path, in the same arithmetic.
- The two triangles only decide which of the two of them is the hit, which the distance of the crossing point to the diagonal of the square gives. The edges of the triangles accept a hit with the same tolerance as the generic path (0.02% of the radius of the square). Close to the diagonal both triangles accept the hit, and the generic path then keeps the one with the smaller distance; that rare case is decided with the exact edge tests of the generic path.
- The normal of the hit is the normal of the triangle, flipped when the ray comes from the side that the normal of the shape points to, and the `LocalShapeInfo` of the hit has the same part (0) and the same triangle (0 or 1).

The ray test is therefore the same ray test as the generic one, with less work: it does not build the second triangle, it does not intersect the ray with the triangles, it does not normalize a normal per triangle, and it does not transform the ray into the local frame twice.

`GGL_DIRECT_PLANE=0` turns the direct ray test off and lets the generic path do the work, so that the two can be compared in the same binary.

The result is the same hit, down to the last bit. `GGL_VERIFY_PLANE=1` computes every plane ray twice, once directly and once with the generic path of Bullet, and compares them: whether they hit at all, the fraction, the point, the normal and the metadata. `tools/verify_plane.py` runs the trainer with it and fails when the program did not test any ray, did not hit any plane or found a difference. Over 133 million plane rays and 14.6 million hits there is no difference at all: the fraction, the point and the normal are the same numbers, not only the same values.

The metadata is the one exception, and only where the generic path has no stable answer: it decides the triangle with an edge test that subtracts the coordinates of the projected center of the square, which are as large as the arena is, so its answer changes when the center moves by a unit in the last place. The verification measures that: it moves the center of the square by a few units in the last place and checks whether the generic path still gives the same triangle. All 33 differences of a run of 133 million rays are such a case, and the direct ray test is the more precise one of the two.

The direct ray test is faster. Measured with the two settings alternating over three repetitions, so that the drift of the host cannot favour one of them, and with the medians of the time per candidate (`tools/bench_rays.py`):

| Shape of the candidate | Generic path | Direct path | |
| --- | --- | --- | --- |
| The arena's planes | 106.1 ns | 90.8 ns | 0.86x |
| The arena's collision mesh | 267.0 ns | 268.5 ns | 1.00x |
| The cars' compound shapes | 321.4 ns | 326.4 ns | 0.98x |
| The cars' hit boxes | 201.4 ns | 201.4 ns | 1.00x |
| The ball | 130.6 ns | 131.4 ns | 0.99x |

The planes are 48% of the ray test candidates, so the ray tests of the whole world get about 7% cheaper. The physics step of the environment is what the ray tests are in, and it is about a fifth of an iteration, so the training gets about 1.5% faster. The throughput of the whole trainer cannot be compared reliably on this host: two runs of the same binary differ by 15% to 50% because of the drift of the host, which is larger than the effect. `tools/bench_paired.py` runs the two settings in the order A, B, B, A and reports the CPU time of the child, which is the steadier of the two measurements. Over eight such repetitions the direct ray test was the faster one in five, both in the throughput of the whole trainer and in its CPU time; the median of the ratios was 0.997x for the throughput and 1.025x for the CPU time. The ray test measurement above is the one that resolves the difference, because it does not have the drift of the host in it.

Where the time of the ray tests goes (`GGL_RAY_STATS=1` prints it when the program ends):

| Shape of the candidate | Share of the candidates | Time of its ray test |
| --- | --- | --- |
| The arena's planes | 48% | 91 ns each |
| The arena's collision mesh | 22% | 269 ns each |
| The cars' compound shapes | 10% | 326 ns each |
| The cars' hit boxes | 11% | 201 ns each |
| The ball | 11% | 131 ns each |

The ray casts are about a third of the physics, and the physics is about a sixth of an iteration, so even a ray test that costs nothing would only make the training about 15% faster. The reason is that the rest of the iteration is matrix products that run at 72 GFLOP/s (the inference) and 101 GFLOP/s (the PPO update), which is 76% and 100% of what a single core of this host reaches. A real speedup therefore has to come from doing less work (fewer or smaller matrix products, fewer physics ticks), not from making the ray casts cheaper.

## Profiling

`GGL_PROFILE=1` turns on the timers of `GigaLearnCPP/RLGymCPP/src/RLGymCPP/Util/Profile.h`. They measure the phases of the collection, of the PPO update and of the environment step, and they are reported as `Prof/…` next to the other metrics. The counters do nothing while the profiler is off, so a production run has no atomic operations and no clock reads in its hot paths.

`GGL_RAY_STATS=1` collects the number of the ray test candidates per shape type and the time that their ray tests took, and prints them when the program ends.

`GGL_DIRECT_PLANE=0` lets Bullet's generic path test the static planes instead of the direct ray test that is described above, and `GGL_VERIFY_PLANE=1` compares the two of them on every plane ray and prints the result when the program ends.

Three tools measure and check all of this, and they all run the repository's own training recipe:

- `tools/verify_plane.py` runs the trainer with the comparison turned on and fails when the two paths differ. It is the check that the direct ray test is the generic ray test.
- `tools/bench_rays.py` runs the two settings next to each other, with the order reversed in every repetition, and reports the time that the ray tests of every shape type needed per candidate.
- `tools/bench_paired.py` runs the two settings in the order A, B, B, A over the whole trainer and reports the throughput and the CPU time of the child process, which does not include the time that the process waits for the host.

## Automatic 30-minute GitHub backups

This is the only GitHub traffic a campaign creates.

`--github-backup` runs `tools/training_service.py` as a headless background worker (no HTTP server or web UI). Every `--backup-interval` seconds (default `1800` = 30 minutes, respected across restarts via `manifest.json`), it stages the latest checkpoint and minimal campaign state. In evolving-pool mode it includes exactly the current `learner/policy_versions/` set (10 by default), excludes duplicate pool/best/eval/metrics copies, hashes the snapshot, and pushes a verified commit to the current branch. The legacy fixed-pool backup format is retained for campaigns that have not enabled evolving opponents.

The cadence is enforced in one place, so one push per 30 minutes is a hard ceiling:

- **`seconds_until_backup_due` gates every attempt** on the last success or on
  `training-backup/manifest.json`, so a restart, a stage boundary or a re-launched worker cannot
  create a burst of commits. A failed push is retried at most once a minute, and only for the
  commit that is already made.
- **Nothing else in training talks to GitHub.** Metrics, ratings, evaluations and the W&B stream
  never touch the repository, so no logging path can add API calls between backups. The worker
  also writes `backup-status.json` (last backed-up checkpoint, next due time) when `--status-file`
  is passed; it is informational and never triggers an extra request.
- **Retention vs. cadence:** the trainer keeps its newest checkpoints and the worker pushes one
  complete checkpoint every 30 minutes, so a checkpoint that lived for less than one interval is
  not published. That is the intended trade-off: `--ts-per-save` (e.g. `10m`) and the stage length
  (e.g. `15-30m`) keep checkpoints alive for at least one backup window, and a hard stop is
  recovered from the newest local checkpoint plus the last published snapshot.

## Weights & Biases reporting

Metrics leave the training host over exactly one connection: HTTPS to W&B. Telemetry never touches GitHub.

- **Direct streaming (`--wandb-online`, the default for `tools/resume_training.py`):**
  `tools/ranked_wandb.py stream` tails the campaign's own durable files (`state.json`, `metrics/*.jsonl`,
  `eval/*.json`) and logs every training row, stage rating and held-out audit sample straight into a single
  resumable online W&B run (`step_metric='Total Timesteps'`). It holds no trainer lock, never rewrites campaign
  state, and keeps its delivery cursor locally in `<out>/.relay_cursor.json`, so a restart never logs a step twice.
- **Offline export (`--wandb-offline`, or `tools/ranked_wandb.py export`):** writes one closed offline run that can
  be synced later. No credentials and no network are needed.
- **Nothing goes through GitHub.** There is no `wandb-live.yml` workflow, no telemetry issue or comment feed, no
  `gh api` call, no sealed-key handshake and no telemetry commit. The earlier GitHub relay existed only to bridge a
  sandbox without W&B egress; it turned a metrics stream into repository API traffic and was removed, because
  pushing metrics through GitHub is what produced the rate limits and the account restriction.
- Reward-component diagnostics are sampled during training and appear under `Rewards/…`; use `python3 tools/resume_training.py --reward-metrics` to enable them on a resumed campaign without changing its frozen PPO/evaluation recipe. `--player-metrics` retains sampled per-player telemetry under `Player/…`, but it measured about 7 % throughput loss on the two-vCPU runner (5-iteration repeats: 19.4-21.3k → 18.0-19.3k steps/s) and an evolving campaign's backup stores no `metrics/`, so its output is not kept — leave it off unless you are actively reading `Player/…` locally. Sparse JSONL/W&B output windows average ordinary numeric diagnostics across skipped iterations (while keeping cumulative counters and the timestep axis at their latest values); rating events are immediate and are never window-averaged. On this two-vCPU runner, `python3 tools/resume_training.py --throughput-profile cpu-20k` selects the fastest measured all-mode profile (64 games, one PPO epoch, one collection-inference thread); clean benchmarks peak near 19.5k overall SPS, though shared-host variability can lower that. The stream maps the native skill tracker's `Rating/…` rows to `Elo/…`, uses `Rating/…`/`Confirmation/…` for the anchored evaluations, and publishes neither `Live/…` nor `Stream/…` transport fields.
### Where the W&B API key lives

Store it **once**, in a private file in your own config directory, outside the repository:

```bash
python3 tools/ranked_wandb.py login          # prompts, input hidden; writes ~/.config/wandb/api_key
```

That path is the default and it is deliberately chosen:

| Property | Why |
| --- | --- |
| `~/.config/wandb/api_key`, mode `600` | Survives restarts, container recreation (with a mounted home/volume) and `git clone`s. Only your user can read it. |
| Outside the working tree | The 30-minute backup worker only ever stages `training-backup/` (`git add -- <dest>`), and the streamer refuses to *write* a key inside the repository at all — so the key cannot leak into a commit or a backup. |
| Not `/dev/shm` | tmpfs is wiped on restart; a key there has to be re-entered every session. |
| Not `.secrets/` in the repo | Convenient but tied to a checkout: it dies with the working copy and lives one careless `git add -A` away from being published. |

Resolution order, first hit wins (`tools/ranked_wandb.py`):

1. `WANDB_API_KEY` in the environment — the right choice for a container, systemd unit or CI that injects
   secrets at launch, and the only one that never touches disk.
2. `--api-key-file`, default `~/.config/wandb/api_key`; point it elsewhere with `RANKED_WANDB_KEY_FILE`
   (e.g. a Docker secret mount such as `/run/secrets/wandb_api_key`). A file that is group/world readable
   or a symlink is **refused** rather than trusted.
3. `~/.netrc` (`machine api.wandb.ai …`) — the standard home for machine credentials.
4. whatever `wandb login` last wrote (W&B stores it in `~/.netrc`).

No credential means no stream: the streamer stops with an instruction instead of logging anonymously.
`pip install wandb` is required; training itself never needs it. Use `--no-wandb` (`resume_training.py`) to
keep the checkpoint backup but log nothing.

For a long-lived host, `systemd`/Docker is the cleanest home: inject `WANDB_API_KEY` as a secret at launch,
or mount a `0600` key file and set `RANKED_WANDB_KEY_FILE` to it. Rotate by writing the new key over the old
file (`login` overwrites atomically) and restarting the streamer.

### If the training host has no direct W&B egress

Fix the host's egress, or export offline and sync from a machine that has it:

```bash
python3 tools/ranked_wandb.py export results/weights1700/campaign --out results/weights1700/wandb-offline
wandb sync results/weights1700/wandb-offline
```

Do not route metrics through the repository. A metrics feed on GitHub is API traffic against your own account,
and a 15-second poll is enough to hit rate limits and get the account flagged.

## Regression tests

```bash
python3 -m unittest discover -s tools -p 'test_*.py' -v

# After building build/GigaLearnBot:
build/GigaLearnTests
build/GigaLearnMultiModeTests
python3 tools/test_native_rollout.py
python3 tools/test_native_render.py
```
