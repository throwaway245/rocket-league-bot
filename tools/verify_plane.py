#!/usr/bin/env python3
"""Checks that the direct ray test of a static plane gives the same hit as Bullet's.

With GGL_VERIFY_PLANE=1 every ray against a static plane is computed twice: once with
the direct ray test and once with the generic path of Bullet, which builds the two
triangles of the shape around the ray and intersects them. The two have to agree on
whether they hit at all, on the fraction of the hit, on its point, on its normal and on
its metadata (the part and the triangle of the shape).

The run only passes when the program really tested planes: a run without a single ray or
without a single hit is a failure, not a success.

    tools/verify_plane.py [--minutes 2] [--checkpoints DIR] [--fresh]
"""
import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import ROOT, default_checkpoints, prepare_checkpoints  # noqa: E402

PATTERN = re.compile(
    r'plane verification: (\d+) rays, (\d+) hits, (\d+) mismatches '
    r'\(hit (\d+), fraction (\d+), point (\d+), normal (\d+), metadata (\d+), '
    r'unstable (\d+)\), '
    r'max fraction difference ([0-9.e+-]+), max point difference ([0-9.e+-]+), '
    r'max normal difference ([0-9.e+-]+), max metadata mismatch offset ([0-9.e+-]+)')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--minutes', type=float, default=2.0,
                        help='How long the trainer runs (default: 2 minutes)')
    parser.add_argument('--checkpoints', type=Path, default=None,
                        help='Folder with the checkpoint to train from (default: the newest of the repository)')
    parser.add_argument('--fresh', action='store_true',
                        help='Train from an untrained network instead of from a checkpoint')
    args = parser.parse_args()

    from bench_common import run_recipe

    work = None
    extra = ['--max-minutes', str(max(0.1, args.minutes)), '--ts-per-save', '1b',
             '--keep-checkpoints', '1']
    if args.fresh:
        extra.append('--new-run')
    else:
        source = args.checkpoints or default_checkpoints()
        if source is None:
            raise SystemExit('No checkpoint found; pass --checkpoints or --fresh')
        work, target = prepare_checkpoints(source)
        # The recipe writes into the folder that it is given, so it is run on the copy.
        extra = ['--max-minutes', str(max(0.1, args.minutes)), '--ts-per-save', '1b',
                 '--keep-checkpoints', '1', '--checkpoints', target]

    env = {'GGL_VERIFY_PLANE': '1', 'GGL_DIRECT_PLANE': '1'}
    try:
        stdout, _rows, _wall, _cpu = run_recipe(extra, env)
    finally:
        if work:
            shutil.rmtree(work, ignore_errors=True)

    lines = [line for line in stdout.splitlines() if 'plane verification' in line]
    for line in lines:
        print(line)
    # The details of the rays on which the two paths disagree, if there are any.
    for line in stdout.splitlines():
        if '[btCollisionWorld]' in line and 'plane verification' not in line:
            print(line)
    if not lines:
        print('NO VERIFICATION OUTPUT FOUND')
        print(stdout[-3000:])
        return 1

    match = PATTERN.search(lines[-1])
    if not match:
        print('COULD NOT READ THE VERIFICATION OUTPUT')
        return 1

    (rays, hits, mismatches, hit_mismatches, fraction_mismatches, point_mismatches,
     normal_mismatches, metadata_mismatches, unstable_metadata) = (int(match.group(i)) for i in range(1, 10))
    max_fraction, max_point, max_normal, max_offset = (float(match.group(i)) for i in (10, 11, 12, 13))

    print()
    print(f'rays checked            : {rays}')
    print(f'plane hits              : {hits}')
    print(f'mismatches              : {mismatches}')
    print(f'  hit existence         : {hit_mismatches}')
    print(f'  fraction              : {fraction_mismatches}')
    print(f'  point                 : {point_mismatches}')
    print(f'  normal                : {normal_mismatches}')
    print(f'  metadata              : {metadata_mismatches}')
    print(f'  unstable metadata     : {unstable_metadata} '
          '(the generic path gives a different triangle for the same ray)')
    print(f'max fraction difference : {max_fraction:.3e}')
    print(f'max point difference    : {max_point:.3e}')
    print(f'max normal difference   : {max_normal:.3e}')
    print(f'max metadata offset     : {max_offset:.3e} (of the radius of the square)')

    problems = []
    if rays <= 0:
        problems.append('no plane ray was tested at all')
    if hits <= 0:
        problems.append('no plane ray hit a plane')
    if mismatches != 0:
        problems.append(f'{mismatches} mismatches')
    if problems:
        print('VERIFICATION FAILED: ' + '; '.join(problems))
        return 1

    print('VERIFICATION PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
