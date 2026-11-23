#!/usr/bin/env python3
"""Compares the time that the ray tests of two settings of the same binary need.

The host that a benchmark runs on is shared and its speed drifts by more than the
difference that is measured, so the two settings are run next to each other and the
order is reversed in every repetition. Every repetition measures the time that the ray
tests of every shape type took, which is what the comparison is about: the number of
candidates is large enough for the time per candidate of a single run to be precise,
and the repetitions cancel the drift of the host.

    tools/bench_rays.py A=NAME=VALUE B=NAME=VALUE [--repetitions 4] [--minutes 1]

The example below compares the direct ray test of the static planes with the generic
path of Bullet:

    tools/bench_rays.py GGL_DIRECT_PLANE=1 GGL_DIRECT_PLANE=0
"""
import argparse
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import default_checkpoints, parse_env, prepare_checkpoints, run_recipe  # noqa: E402

SHAPE_NAMES = {0: 'box', 8: 'sphere', 21: 'triangle mesh', 28: 'static plane', 31: 'compound'}
LINE = re.compile(r'shape type\s+(\d+):\s+(\d+) candidates,\s+([0-9.]+) s')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('a', help='The environment of the first setting, e.g. GGL_DIRECT_PLANE=1')
    parser.add_argument('b', help='The environment of the second setting, e.g. GGL_DIRECT_PLANE=0')
    parser.add_argument('--repetitions', type=int, default=4, help='Repetitions of the pair (default: 4)')
    parser.add_argument('--minutes', type=float, default=1.0, help='Minutes of every run (default: 1)')
    parser.add_argument('--checkpoints', type=Path, default=None,
                        help='Folder with the checkpoint to train from (default: the newest of the repository)')
    parser.add_argument('--fresh', action='store_true', help='Train from an untrained network')
    args = parser.parse_args()

    env_a, env_b = parse_env(args.a), parse_env(args.b)
    source = None if args.fresh else (args.checkpoints or default_checkpoints())
    if not args.fresh and source is None:
        raise SystemExit('No checkpoint found; pass --checkpoints or --fresh')

    extra = ['--max-minutes', str(max(0.1, args.minutes)), '--ts-per-save', '1b',
             '--keep-checkpoints', '1']
    if source is None:
        extra.append('--new-run')

    per_shape = {'A': {}, 'B': {}}
    for rep in range(args.repetitions):
        order = ['A', 'B'] if rep % 2 == 0 else ['B', 'A']
        for name in order:
            env = env_a if name == 'A' else env_b
            work = None
            run_extra = list(extra)
            if source is not None:
                work, target = prepare_checkpoints(source)
                run_extra += ['--checkpoints', target]
            try:
                stdout, _rows, _wall, _cpu = run_recipe(run_extra, {**env, 'GGL_RAY_STATS': '1'})
            finally:
                if work:
                    import shutil
                    shutil.rmtree(work, ignore_errors=True)
            for line in stdout.splitlines():
                match = LINE.search(line)
                if not match:
                    continue
                shape_type, candidates, seconds = int(match.group(1)), int(match.group(2)), float(match.group(3))
                if candidates <= 0:
                    continue
                ns = seconds * 1e9 / candidates
                per_shape[name].setdefault(shape_type, []).append(ns)
                print(f'  rep {rep + 1} {name}: shape {shape_type:2d} '
                      f'{SHAPE_NAMES.get(shape_type, "?"):14s} {candidates:10d} candidates '
                      f'{ns:8.1f} ns each', flush=True)

    print()
    print(f'{"shape":16s} {"A ns":>10s} {"B ns":>10s} {"B/A":>8s}')
    for shape_type in sorted(per_shape['A']):
        if shape_type not in per_shape['B']:
            continue
        a = statistics.median(per_shape['A'][shape_type])
        b = statistics.median(per_shape['B'][shape_type])
        print(f'{SHAPE_NAMES.get(shape_type, str(shape_type)):16s} {a:10.1f} {b:10.1f} {b / a:8.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
