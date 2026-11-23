#!/usr/bin/env python3
"""Paired A/B of two environment settings of the same binary.

The host that a benchmark runs on is shared and its speed drifts by more than the
difference that is measured, so the two variants are run next to each other and the
order of the runs is reversed in every repetition. Every repetition runs the two
variants in the order A, B, B, A (or B, A, A, B), so that a drift of the host that is
the same over the repetition cancels out.

What is reported is the ratio of every repetition, because the two runs of a repetition
are close to each other in time. The CPU time of the child process is reported as well:
it is the same measurement as the wall clock, but it does not include the time that the
process waits for the host.

    tools/bench_paired.py A=NAME=VALUE B=NAME=VALUE [--repetitions 4] [--minutes 2]

The example below compares the direct ray test of the static planes with the generic
path of Bullet:

    tools/bench_paired.py GGL_DIRECT_PLANE=1 GGL_DIRECT_PLANE=0
"""
import argparse
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import default_checkpoints, parse_env, prepare_checkpoints, run_recipe, throughput  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('a', help='The environment of the first setting, e.g. GGL_DIRECT_PLANE=1')
    parser.add_argument('b', help='The environment of the second setting, e.g. GGL_DIRECT_PLANE=0')
    parser.add_argument('--repetitions', type=int, default=4, help='Repetitions of the pair (default: 4)')
    parser.add_argument('--minutes', type=float, default=2.0, help='Minutes of every run (default: 2)')
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

    pairs = []
    for rep in range(args.repetitions):
        order = ['A', 'B', 'B', 'A'] if rep % 2 == 0 else ['B', 'A', 'A', 'B']
        runs = {'A': [], 'B': []}
        for name in order:
            env = env_a if name == 'A' else env_b
            work = None
            run_extra = list(extra)
            if source is not None:
                work, target = prepare_checkpoints(source)
                run_extra += ['--checkpoints', target]
            try:
                _stdout, rows, wall, cpu = run_recipe(run_extra, env)
            finally:
                if work:
                    shutil.rmtree(work, ignore_errors=True)
            result = throughput(rows, wall, cpu)
            runs[name].append(result)
            print(f'  rep {rep + 1} {name}: {result["overall_sps"]:8.0f} steps/s overall, '
                  f'{result["cpu_ms_per_1k_steps"]:.2f} ms cpu/1k steps', flush=True)

        # The runs of a repetition do the same work, so the ratio of their times is the
        # inverse of the ratio of their throughput. The CPU time is the steadier of the two.
        a_sps = statistics.mean(r['overall_sps'] for r in runs['A'])
        b_sps = statistics.mean(r['overall_sps'] for r in runs['B'])
        a_cpu = statistics.mean(r['cpu_ms_per_1k_steps'] for r in runs['A'])
        b_cpu = statistics.mean(r['cpu_ms_per_1k_steps'] for r in runs['B'])
        pairs.append((b_sps / a_sps, a_cpu / b_cpu if b_cpu else 0.0))
        print(f'  rep {rep + 1}: overall {pairs[-1][0]:.4f}x  cpu work {pairs[-1][1]:.4f}x', flush=True)

    overall = [p[0] for p in pairs]
    cpu = [p[1] for p in pairs]
    print()
    print('pairs (B/A):', ' '.join(f'{r:.4f}' for r in overall))
    print('cpu pairs   :', ' '.join(f'{r:.4f}' for r in cpu))
    print(f'overall  median {statistics.median(overall):.4f}x  mean {statistics.mean(overall):.4f}x')
    print(f'cpu work median {statistics.median(cpu):.4f}x  mean {statistics.mean(cpu):.4f}x  '
          '(B needs this much less CPU per step)')
    print(f'overall  faster in {sum(1 for r in overall if r > 1)} of {len(overall)} repetitions')
    print(f'cpu work faster in {sum(1 for r in cpu if r > 1)} of {len(cpu)} repetitions')
    return 0


if __name__ == '__main__':
    sys.exit(main())
