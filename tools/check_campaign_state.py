#!/usr/bin/env python3
"""Checks that a campaign's state file agrees with its metrics log and checkpoints.

A campaign's state.json holds the stage bookkeeping, not the live step count: the
`pending` block records the step count at the *start* of the stage that is still
running, and the live step count lives in the learner directory's newest numbered
checkpoint. This script puts the three sources side by side and reports whether they
agree, so that a paused campaign can be resumed without guessing.

    tools/check_campaign_state.py results/weights1700-v6/campaign
"""
import argparse, json, pathlib, sys


def newest_checkpoint(learner):
    if not learner.is_dir():
        return None
    steps = [int(d.name) for d in learner.iterdir() if d.is_dir() and d.name.isdigit()]
    return max(steps) if steps else None


def metrics_rows(metrics_dir):
    rows = []
    if not metrics_dir.is_dir():
        return rows
    for path in sorted(metrics_dir.glob('*.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def check(campaign):
    state_path = campaign / 'state.json'
    if not state_path.is_file():
        raise SystemExit(f'No state file: {state_path}')
    state = json.loads(state_path.read_text())
    rows = metrics_rows(campaign / 'metrics')
    ckpt = newest_checkpoint(campaign / 'learner')

    pending = state.get('pending') or {}
    stage = pending.get('stage')
    start = pending.get('start_steps')
    metric_steps = int(rows[-1]['Total Timesteps']) if rows else None
    iterations = int(rows[-1]['Total Iterations']) if rows else None

    print(f'campaign        : {campaign}')
    print(f'status          : {state.get("status")}')
    print(f'stages finished : {len(state.get("history") or [])}')
    print(f'pending stage   : {stage} (trained={pending.get("trained")}, '
          f'{pending.get("seconds", 0):.1f}s of stage time)')
    print(f'stage start     : {start}')
    print(f'newest ckpt     : {ckpt}')
    print(f'metrics steps   : {metric_steps} (iteration {iterations})')
    print()

    problems = []
    if start is not None and ckpt is not None and ckpt < start:
        problems.append('The newest checkpoint is older than the stage start.')
    if metric_steps is not None and ckpt is not None and metric_steps != ckpt:
        problems.append(f'The metrics log ({metric_steps}) and the newest checkpoint ({ckpt}) differ.')
    if start is not None and metric_steps is not None and metric_steps < start:
        problems.append('The metrics log is older than the stage start.')
    if pending.get('trained') and stage is not None and len(state.get('history') or []) <= stage:
        problems.append('The stage is marked trained but has no history entry.')

    if problems:
        for p in problems:
            print('PROBLEM:', p)
    else:
        print('The state file, the metrics log and the checkpoints agree.')
        if pending.get('trained') is False and metric_steps is not None and start is not None:
            done = metric_steps - start
            seconds = pending.get('seconds', 0) or 0
            print(f'The stage produced {done} steps in {seconds:.1f}s '
                  f'({done / seconds:.0f} steps/s) and was interrupted before it was marked trained,')
            print('so the campaign resumes the training from the newest checkpoint.')
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=pathlib.Path)
    args = parser.parse_args()
    sys.exit(check(args.campaign))


if __name__ == '__main__':
    main()
