#!/usr/bin/env python3
"""Copy a foundation-v1 checkpoint to its actor-only ABI without altering the source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def export(source: Path, destination: Path):
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists():
        raise ValueError('destination already exists')
    # Refuse output inside the source so the source is always immutable.
    if destination.resolve().is_relative_to(source):
        raise ValueError('destination must be outside the source checkpoint')
    info = json.loads((source / 'INFO.json').read_text())
    if info.get('model') != 'foundation-v1' or info.get('obs') != 'MultiModeObs':
        raise ValueError('expected a full multimode foundation-v1 checkpoint')
    names = ['POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt', 'RUNNING_STATS.json']
    # Optimizer state is optional for inference, but preserve it if it exists for resume.
    names += [n for n in ('POLICY_OPTIM.lt', 'SHARED_HEAD_OPTIM.lt') if (source / n).is_file()]
    for name in names:
        if not (source / name).is_file():
            raise ValueError(f'missing actor file: {name}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.actor-export-', dir=destination.parent) as tmp:
        tmp = Path(tmp)
        for name in names:
            shutil.copy2(source / name, tmp / name)
            if hashlib.sha256((source / name).read_bytes()).digest() != hashlib.sha256((tmp / name).read_bytes()).digest():
                raise IOError(f'copy verification failed for {name}')
        info['model'] = 'foundation-policy-v1'
        info['actor_only_export'] = 'foundation-v1'
        (tmp / 'INFO.json').write_text(json.dumps(info, indent=4) + '\n')
        # mkdir is atomic and refuses an existing destination, then transfer files.
        # Incomplete exports are removed on failure; source is never modified.
        destination.mkdir()
        try:
            for file in tmp.iterdir():
                os.replace(file, destination / file.name)
        except BaseException:
            shutil.rmtree(destination)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
    print(f'Actor-only checkpoint: {args.destination}; source retained at {args.source}')


if __name__ == '__main__':
    main()
