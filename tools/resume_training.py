#!/usr/bin/env python3
"""Verify the canonical paused snapshot, restore a working copy, and explicitly resume.
Use --restore-only to prepare without starting training. Never trains in training-backup/.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'training-backup'
WORK = ROOT / 'results/weights1700/campaign'


def verify(snapshot):
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        path = (snapshot / name).resolve()
        if not path.is_relative_to(snapshot.resolve()):
            raise ValueError(f'Unsafe manifest path: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Backup hash mismatch: {name}')
    return manifest


def restore(snapshot=SNAPSHOT, work=WORK):
    manifest = verify(snapshot)
    if work.exists():
        if not (work / 'state.json').exists():
            raise ValueError('Existing work directory is not a campaign; refusing to overwrite')
        print('Existing working campaign retained; resume it rather than overwrite newer progress.')
    else:
        work.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot, work, ignore=shutil.ignore_patterns('manifest.json'))
        print(f'Restored checkpoint {manifest["checkpoint"]} into {work}')
    state = json.loads((work / 'state.json').read_text())
    runtime = state.get('runtime_overrides') or {}
    if runtime.get('evolving_opponents'):
        # Compact GitHub snapshots keep just the latest model and its ten opponents. Rebuild
        # local-only evaluation/telemetry directories without duplicating model snapshots.
        versions = work / 'learner/policy_versions'
        if not versions.is_dir():
            raise ValueError('Backup is missing its evolving old-policy pool')
        pool = work / 'pool'
        if not pool.exists():
            shutil.copytree(versions, pool)
        digest = hashlib.sha256()
        for path in sorted(pool.glob('*/*')):
            if path.is_file():
                digest.update(str(path.relative_to(pool)).encode() + b'\0' + path.read_bytes())
        state['pool_sha256'] = digest.hexdigest()
        for folder in ('eval', 'metrics', 'best'):
            (work / folder).mkdir(parents=True, exist_ok=True)
        (work / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
    return state


# Native recipe tokens that train_ranked.py forwards itself. Everything else the preset pins
# (--games, --mode-balance, --ts-per-itr, --fixed-rollout, --lean-metrics, --keep-checkpoints, ...)
# travels as passthrough, exactly like running the recipe by hand.
TRAINER_FLAGS_WITH_VALUE = {
    '--threads', '--minibatch-size', '--batch-size', '--lr', '--critic-lr', '--epochs',
    '--entropy-scale', '--norm-advantages', '--target-kl', '--gamma', '--gae-lambda',
    '--return-stat-max-count', '--random-state-prob', '--tactical-state-prob', '--train-vs-old',
    '--ts-per-version', '--max-versions', '--skill-minutes', '--ts-per-save', '--rewards',
}
TRAINER_FLAGS = {'--mask-entropy', '--fuzzed-kickoffs', '--pfsp'}
# Owned by the runtime save cadence below, never replayed from the preset.
CADENCE_FLAGS = {'--ts-per-save'}


def split_recipe(train_args):
    """Split a pinned native recipe into (trainer flags, native passthrough)."""
    trainer, native, i = [], [], 0
    args = list(train_args)
    while i < len(args):
        flag = args[i]
        if flag in TRAINER_FLAGS_WITH_VALUE:
            value = args[i + 1] if i + 1 < len(args) else ''
            if flag not in CADENCE_FLAGS:
                trainer += [flag, value]
            i += 2
            continue
        if flag in TRAINER_FLAGS:
            trainer.append(flag)
            i += 1
            continue
        value = args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith('--') else None
        native.append(flag)
        if value is not None:
            native.append(value)
            i += 2
        else:
            i += 1
    return trainer, native


def native_passthrough(train_args):
    return split_recipe(train_args)[1]


def build_resume_command(state, work, without_services=False, backup_interval=1800,
                         reward_metrics=False, player_metrics=False, throughput_profile=None,
                         continuous=False, evolving_opponents=False, opponent_pool_size=None,
                         promotion_margin=None, wandb_online=True, wandb_interval=10.0,
                         runtime_rewards=None):
    config = state['config']
    cmd = [sys.executable, str(ROOT / 'tools/train_ranked.py'), '--base', config['base'],
           '--out', str(work.relative_to(ROOT)), '--minutes', str(sum(config['minutes'])),
           '--pilot-minutes', str(config['minutes'][0]),
           '--block-minutes', str(max(config['minutes'][1:] or config['minutes'])),
           '--rounds', str(config['rounds']), '--seed', str(config['seed']),
           '--audit-seed', str(config['audit_seed'])]
    if state.get('target_ranking'):
        cmd += ['--target-ranking', str(state['target_ranking'])]
    if config.get('reference_pool'):
        cmd += ['--reference-pool', config['reference_pool']]
    if config.get('extra_versions'):
        cmd += ['--extra-versions', *config['extra_versions']]
    # Replay the stored recipe exactly: the tokens train_ranked.py exposes as its own flags, plus
    # everything else (--games, --mode-balance, --fixed-rollout, --lean-metrics, --keep-checkpoints,
    # ...) as native passthrough. train_ranked.py pins the whole config and refuses to resume when
    # it differs, so nothing may be dropped or rewritten here. --ts-per-save is the one exception:
    # the derived save cadence owns it.
    trainer_flags, native_flags = split_recipe(config['train_args'])
    for flag in trainer_flags:
        cmd.append('--minibatch' if flag == '--minibatch-size' else flag)
    rewards = runtime_rewards or (state.get('runtime_overrides') or {}).get('rewards')
    if rewards:
        # Rewards are a runtime override, not part of the pinned recipe, so changing them keeps the
        # campaign resumable (train_ranked appends this after the pinned --rewards, so it wins).
        cmd += ['--runtime-rewards', rewards]
    if reward_metrics or (state.get('runtime_overrides') or {}).get('reward_metrics'):
        cmd.append('--reward-metrics')
    if player_metrics or (state.get('runtime_overrides') or {}).get('player_metrics'):
        cmd.append('--player-metrics')
    profile = throughput_profile or (state.get('runtime_overrides') or {}).get('throughput_profile')
    if profile:
        cmd += ['--throughput-profile', profile]
    if continuous or (state.get('runtime_overrides') or {}).get('continuous'):
        cmd.append('--continuous')
    runtime = state.get('runtime_overrides') or {}
    if evolving_opponents or runtime.get('evolving_opponents'):
        cmd.append('--evolving-opponents')
        cmd += ['--opponent-pool-size', str(opponent_pool_size if opponent_pool_size is not None
                                            else runtime.get('opponent_pool_size', 10))]
        cmd += ['--promotion-margin', str(promotion_margin if promotion_margin is not None
                                           else runtime.get('promotion_margin', 50.0))]
    if not without_services:
        # 30-minute GitHub checkpoint backups; metrics go straight to W&B, never through GitHub.
        cmd += ['--github-backup', '--backup-interval', str(backup_interval)]
        if wandb_online:
            cmd += ['--wandb-online', '--wandb-interval', str(wandb_interval)]
    if native_flags:
        # Native passthrough last: train_ranked forwards everything after `--` to the trainer, so
        # the preset travels unchanged instead of being rejected as an unknown driver flag.
        cmd.extend(['--', *native_flags])
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore-only', action='store_true')
    parser.add_argument('--without-services', action='store_true',
                        help='Run trainer only, without the 30-minute GitHub backup or W&B streaming')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Keep the GitHub checkpoint backup but do not stream metrics to W&B')
    parser.add_argument('--reward-metrics', action='store_true',
                        help='Enable and persist sampled reward-component telemetry on the resumed campaign')
    parser.add_argument('--runtime-rewards',
                        help='Change the running campaign\'s reward preset/weights (e.g. v6). Leaves the pinned '
                             'recipe intact, so the campaign still resumes.')
    parser.add_argument('--player-metrics', action='store_true',
                        help='Enable and persist sampled per-player telemetry on the resumed campaign')
    parser.add_argument('--throughput-profile', choices=('cpu-20k',),
                        help='Use the benchmarked two-vCPU profile targeting about 20k overall steps/second')
    parser.add_argument('--continuous', action='store_true',
                        help='Continue with repeated training/evaluation blocks indefinitely, until interrupted')
    parser.add_argument('--evolving-opponents', action='store_true',
                        help='Enable the rolling historical opponent pool gated by Elo')
    parser.add_argument('--opponent-pool-size', type=int,
                        help='Number of old policies in the evolving pool (default: 10)')
    parser.add_argument('--promotion-margin', type=float,
                        help='Required Elo lead over the pool strongest policy (default: 50)')
    parser.add_argument('--backup-interval', type=int, default=1800,
                        help='Minimum seconds between GitHub auto-backups (default: 1800 = 30 minutes)')
    parser.add_argument('--wandb-interval', type=float, default=10.0,
                        help='Seconds between direct W&B stream polls (default: 10)')
    args = parser.parse_args()
    saved = json.loads((SNAPSHOT / 'state.json').read_text())
    work = ROOT / saved.get('working_directory', str(WORK.relative_to(ROOT)))
    if not work.resolve().is_relative_to((ROOT / 'results').resolve()):
        raise ValueError('Saved working campaign must be under results/, never the canonical backup')
    state = restore(work=work)
    if args.restore_only:
        return
    if state['status'] == 'complete':
        print('Campaign complete; create a new experiment to train further.')
        return
    cmd = build_resume_command(state, work, without_services=args.without_services,
                               backup_interval=args.backup_interval, reward_metrics=args.reward_metrics,
                               player_metrics=args.player_metrics, throughput_profile=args.throughput_profile,
                               continuous=args.continuous, evolving_opponents=args.evolving_opponents,
                               opponent_pool_size=args.opponent_pool_size,
                               promotion_margin=args.promotion_margin,
                               wandb_online=not args.no_wandb, wandb_interval=args.wandb_interval,
                               runtime_rewards=args.runtime_rewards)
    print('Resuming only because this script was explicitly invoked.', flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
