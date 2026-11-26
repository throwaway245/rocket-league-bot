#!/usr/bin/env python3
"""Create an immutable, verified, Git-trackable actor-only phase backup.

This only prepares the backup. The operator must commit and push the phase before
starting the next one; it deliberately does not supervise training or invoke Git.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / 'training-backup' / 'large-distillation'
REQUIRED = ('INFO.json', 'RUNNING_STATS.json', 'POLICY.lt', 'POLICY_OPTIM.lt',
            'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt', 'SHARED_HEAD_OPTIM.lt')


def sha(path):
    with path.open('rb') as f:
        digest = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def save_phase(checkpoint, phase, match_results, winner_replay=None):
    checkpoint = checkpoint.resolve(strict=True)
    if phase < 0:
        raise ValueError('phase must be nonnegative')
    info = json.loads((checkpoint / 'INFO.json').read_text())
    stats = json.loads((checkpoint / 'RUNNING_STATS.json').read_text())
    if info.get('model') != 'foundation-policy-v1' or info.get('obs') != 'MultiModeObs':
        raise ValueError('not an actor-only multimode foundation checkpoint')
    if any((checkpoint / f).exists() for f in ('CRITIC.lt', 'CRITIC_OPTIM.lt')):
        raise ValueError('actor-only checkpoint unexpectedly contains critic weights')
    if not isinstance(stats.get('total_timesteps'), int):
        raise ValueError('missing total_timesteps')
    for name in REQUIRED:
        if not (checkpoint / name).is_file():
            raise ValueError(f'missing {name}')
    if phase and not match_results:
        raise ValueError('completed phase requires match results')
    BACKUP.mkdir(parents=True, exist_ok=True)
    target = BACKUP / f'phase-{phase:04d}'
    if target.exists():
        raise ValueError(f'backup already exists: {target}')
    with tempfile.TemporaryDirectory(prefix='.phase-', dir=BACKUP) as work:
        stage = Path(work) / 'stage'
        stage.mkdir()
        actor = stage / 'checkpoint'
        actor.mkdir()
        hashes = {}
        for name in REQUIRED:
            src, dst = checkpoint / name, actor / name
            shutil.copy2(src, dst)
            hashes[name] = sha(dst)
            if hashes[name] != sha(src):
                raise IOError(f'verification failed: {name}')
        replay_metadata = None
        if winner_replay:
            winner_replay = winner_replay.resolve(strict=True)
            replay_path = stage / 'winner-replay.pt.gz'
            with winner_replay.open('rb') as src, replay_path.open('wb') as raw:
                with gzip.GzipFile(filename='', mode='wb', compresslevel=1, fileobj=raw, mtime=0) as dst:
                    shutil.copyfileobj(src, dst)
            decoded_hash = hashlib.sha256()
            with gzip.open(replay_path, 'rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    decoded_hash.update(block)
            if decoded_hash.hexdigest() != sha(winner_replay):
                raise IOError('winner replay verification failed')
            replay_metadata = {'compressed_sha256': sha(replay_path), 'uncompressed_sha256': decoded_hash.hexdigest()}
        manifest = {'phase': phase, 'Total Timesteps': stats['total_timesteps'], 'total_timesteps': stats['total_timesteps'],
                    'model': info['model'], 'checkpoint_sha256': hashes, 'winner_replay': replay_metadata,
                    'match_results': json.loads(match_results.read_text()) if match_results else None}
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        os.rename(stage, target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--phase', type=int, required=True)
    parser.add_argument('--match-results', type=Path)
    parser.add_argument('--winner-replay', type=Path, help='Compress and verify immutable winning-action replay for retention')
    args = parser.parse_args()
    print(save_phase(args.checkpoint, args.phase, args.match_results, args.winner_replay))


if __name__ == '__main__':
    main()
