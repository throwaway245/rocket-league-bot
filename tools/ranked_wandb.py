#!/usr/bin/env python3
"""Stream a ranked training campaign straight into Weights & Biases.

Every metric leaves the machine over a single direct HTTPS connection to W&B
(`api.wandb.ai`). Nothing about telemetry touches GitHub: there is no Actions
workflow, no issue/comment feed, no REST or CLI GitHub call, no key handshake and
no Git commit. The only GitHub traffic this repository produces is the 30-minute
checkpoint backup worker (`tools/training_service.py`).

Two ways to use it:

  stream  Tail a running campaign's durable files (`state.json`, `metrics/*.jsonl`,
          `eval/*.json`) and log every training row, rating stage and held-out audit
          sample into a single resumable online run. Read-only: the trainer is never
          locked and its state is never altered.
  export  Write one closed, syncable offline run from a finished (or stopped)
          campaign. No network access and no credentials are needed; sync it later.

The delivery cursor (`<out>/.relay_cursor.json`) makes both paths idempotent, so a
restarted stream never logs a step twice and never re-logs a completed stage.

Examples
    python3 tools/ranked_wandb.py stream results/weights1700/campaign --out results/weights1700/wandb
    python3 tools/ranked_wandb.py export results/weights1700/campaign --out results/weights1700/wandb-offline
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

PROJECT = 'rocket-league-ggl'
GROUP = 'weights-1700'
CURSOR_NAME = '.relay_cursor.json'
RATING_PREFIX = 'Rating/'


def start(directory, config, name, online=False, run_id=None, project=PROJECT, group=GROUP):
    """Open a W&B run. Offline runs need neither credentials nor network."""
    try:
        import wandb  # optional dependency; JSON-only training never needs it
    except ImportError:
        raise SystemExit('The `wandb` package is required for W&B logging: pip install wandb '
                         '(training itself never needs it; metrics stay in metrics/*.jsonl)')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    run = wandb.init(mode='online' if online else 'offline', id=run_id,
                     resume='allow' if online and run_id else None,
                     project=project, group=group, name=name, dir=str(directory.resolve()),
                     config=config,
                     settings=wandb.Settings(console='off', disable_code=True, disable_git=True,
                                             x_disable_stats=True, x_disable_meta=True, init_timeout=30))
    run.define_metric('Total Timesteps')
    run.define_metric('*', step_metric='Total Timesteps')
    return run


def metric_name(key):
    """`Rating/...` rows come from the native skill tracker, so file them under `Elo/`."""
    return 'Elo/' + key.removeprefix(RATING_PREFIX) if key.startswith(RATING_PREFIX) else key


def read_rows(campaign, after_step=-1):
    """Yield completed metric rows newer than `after_step`, oldest first.

    A row still being appended by the trainer has no trailing newline yet and is skipped
    until the next poll.
    """
    rows = {}
    metrics_dir = Path(campaign) / 'metrics'
    if metrics_dir.is_dir():
        for path in sorted(metrics_dir.glob('*.jsonl')):
            for line in path.read_text(errors='replace').splitlines(keepends=True):
                if not line.endswith('\n') or not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = row.get('Total Timesteps')
                if step is not None and int(step) > after_step:
                    rows[int(step)] = row
    return [rows[step] for step in sorted(rows)]


def stage_records(state):
    """One dict per completed evaluation stage, keyed by its step."""
    records = {}
    for entry in state.get('history') or []:
        step = int(entry['steps'])
        record = {f'Rating/{k}': v for k, v in (entry.get('screen') or {}).items()}
        record.update({f'Confirmation/{k}': v for k, v in (entry.get('confirm') or {}).items()})
        record['Campaign/Stage'] = entry.get('stage')
        record['Campaign/Promoted'] = int(bool(entry.get('promoted')))
        if entry.get('opponent_pool_steps') is not None:
            record['Campaign/Opponent Pool Size'] = len(entry['opponent_pool_steps'])
            record['Campaign/Opponent Pool Generation'] = entry.get('opponent_pool_generation', 0)
            record['Campaign/Opponent Pool Promoted'] = int(bool(entry.get('opponent_pool_promoted')))
            if entry.get('strongest_opponent_ranking') is not None:
                record['Campaign/Strongest Opponent Elo'] = entry['strongest_opponent_ranking']
                record['Campaign/Pool Promotion Margin'] = entry.get('opponent_pool_margin')
        if entry.get('steps_per_second') is not None:
            record['Campaign/Steps Per Second'] = entry['steps_per_second']
        records[step] = record
    return records


def log_audit(run, state):
    audit = state.get('audit') or {}
    if not audit:
        return
    history = state.get('history') or []
    for name, step in [('Parent', (state.get('config') or {}).get('parent_steps')),
                       ('Best', (state.get('best') or {}).get('steps')),
                       ('Latest', history[-1]['steps'] if history else None)]:
        for metric, value in (audit.get(str(step)) or {}).items():
            run.summary[f'Audit/{name} {metric}'] = value


def summary_fields(state, campaign):
    best = state.get('best') or {}
    fields = {
        'Campaign/Status': state.get('status', 'unknown'),
        'Campaign/Target Ranking': state.get('target_ranking'),
        'Campaign/Stages Completed': max(0, len(state.get('history') or []) - 1),
    }
    if best:
        fields['Campaign/Best Steps'] = best.get('steps')
        for metric, value in (best.get('screen') or {}).items():
            fields[f'Best/Screen {metric}'] = value
        for metric, value in (best.get('confirm') or {}).items():
            fields[f'Best/Confirm {metric}'] = value
    return {k: v for k, v in fields.items() if v is not None}


def campaign_run_id(campaign, config):
    """Stable id so a restarted stream continues the same W&B run instead of forking it."""
    ident = {'campaign': str(Path(campaign).resolve())}
    for key in ('base', 'parent_steps', 'seed', 'audit_seed', 'protocol'):
        if config.get(key) is not None:
            ident[key] = config[key]
    digest = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    return f'rlbot-{config.get("parent_steps", "campaign")}-{digest}'


def read_cursor(directory):
    path = Path(directory) / CURSOR_NAME
    if not path.is_file():
        return -1, set()
    try:
        cursor = json.loads(path.read_text())
    except (OSError, ValueError):
        return -1, set()
    return int(cursor.get('last_step', -1)), {int(s) for s in cursor.get('stages', [])}


def write_cursor(directory, last_step, stages):
    path = Path(directory) / CURSOR_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'last_step': int(last_step), 'stages': sorted(int(s) for s in stages)}) + '\n')
    tmp.replace(path)


def pending_events(campaign, state, last_step, logged_stages):
    """Ordered, idempotent `(step, metrics)` pairs plus the stages they complete."""
    events = {}
    for row in read_rows(campaign, after_step=last_step):
        step = int(row['Total Timesteps'])
        events[step] = {metric_name(k): v for k, v in row.items()
                        if k != '_time' and v is not None and not isinstance(v, bool)}
    new_stages = set()
    for step, record in stage_records(state).items():
        stage = int(record['Campaign/Stage'])
        if stage in logged_stages:
            continue
        events.setdefault(step, {}).update(record)
        new_stages.add(stage)
    for step, record in events.items():
        record['Total Timesteps'] = step  # the W&B step metric has to be on every record
    ordered = [(step, events[step]) for step in sorted(events)]
    return ordered, new_stages


def stream(campaign, directory, interval=10.0, project=PROJECT, group=GROUP, run_name='three-mode-ranking-live',
           idle_timeout_minutes=60.0, max_minutes=None, once=False, stop_statuses=('complete', 'paused', 'failed')):
    """Tail durable campaign JSON and log it directly to online W&B until it stops moving."""
    campaign = Path(campaign)
    directory = Path(directory)
    state = json.loads((campaign / 'state.json').read_text())  # the writer renames atomically
    config = state.get('config') or {}
    last_step, stages = read_cursor(directory)
    run = start(directory, config, run_name, online=True,
                run_id=campaign_run_id(campaign, config), project=project, group=group)
    print(f'Online W&B run: {run.url}', flush=True)
    deadline = None if max_minutes is None else time.monotonic() + max_minutes * 60.0
    idle_since = None
    try:
        while True:
            try:
                state = json.loads((campaign / 'state.json').read_text())
            except (OSError, ValueError):
                state = None
            if state is not None:
                ordered, new_stages = pending_events(campaign, state, last_step, stages)
                for step, record in ordered:
                    run.log(record)
                    last_step = max(last_step, step)
                stages |= new_stages
                if ordered or new_stages:
                    idle_since = None
                    print(f'Logged {len(ordered)} step(s), +{len(new_stages)} stage(s) -> '
                          f'last_step={last_step:,} status={state.get("status")}', flush=True)
                elif idle_since is None:
                    idle_since = time.monotonic()
                write_cursor(directory, last_step, stages)
                for key, value in summary_fields(state, campaign).items():
                    run.summary[key] = value
                log_audit(run, state)
                status = state.get('status')
                if once or status in stop_statuses:
                    print(f'Campaign status is `{status}`; closing the W&B stream.', flush=True)
                    break
                if idle_since is not None and time.monotonic() - idle_since >= idle_timeout_minutes * 60.0:
                    print(f'No new metrics for {idle_timeout_minutes:g}m; closing the W&B stream.', flush=True)
                    break
            if deadline is not None and time.monotonic() >= deadline:
                print('Reached --max-minutes; the run stays resumable by id.', flush=True)
                break
            time.sleep(interval)
    finally:
        write_cursor(directory, last_step, stages)
        run.finish()
    return run


def export(campaign, directory, project=PROJECT, group=GROUP):
    """Write one closed offline run from a campaign's completed stages. No network needed."""
    campaign = Path(campaign)
    state = json.loads((campaign / 'state.json').read_text())
    if not state.get('history'):
        raise SystemExit('Wait for at least one completed evaluation')
    last = int(state['history'][-1]['steps'])
    run = start(directory, state.get('config') or {}, f'ranking-{last}', project=project, group=group)
    failed = True
    try:
        # Merge metric rows and evaluation stages into one step-ordered series, exactly like the
        # live stream, so an exported run and a streamed run plot the same way.
        ordered, _ = pending_events(campaign, state, last_step=-1, logged_stages=set())
        for step, record in ordered:
            run.log(record)
        log_audit(run, state)
        for key, value in summary_fields(state, campaign).items():
            run.summary[key] = value
        failed = False
    finally:
        run.finish(exit_code=int(failed))
    print(f'Finished offline W&B run: {Path(run.dir).parent}')
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)

    st = sub.add_parser('stream', help='Tail a running campaign straight into online W&B')
    st.add_argument('campaign', type=Path, help='Campaign directory containing state.json and metrics/')
    st.add_argument('--out', type=Path, required=True, help='Local W&B cache and cursor directory')
    st.add_argument('--interval', type=float, default=10.0, help='Seconds between polls (default: 10)')
    st.add_argument('--idle-timeout-minutes', type=float, default=60.0,
                    help='Close the run after this long without new metrics (default: 60)')
    st.add_argument('--max-minutes', type=float, default=None, help='Optional runtime budget')
    st.add_argument('--once', action='store_true', help='Log what is pending and exit')
    st.add_argument('--run-name', default='three-mode-ranking-live')
    st.add_argument('--project', default=PROJECT)
    st.add_argument('--group', default=GROUP)

    ex = sub.add_parser('export', help='Write a syncable offline run from a finished campaign')
    ex.add_argument('campaign', type=Path)
    ex.add_argument('--out', type=Path, required=True, help='Destination for the offline run')
    ex.add_argument('--project', default=PROJECT)
    ex.add_argument('--group', default=GROUP)

    args = parser.parse_args()
    if getattr(args, 'interval', 1) <= 0:
        parser.error('--interval must be positive')
    if args.mode == 'stream':
        stream(args.campaign, args.out, interval=args.interval, project=args.project, group=args.group,
               run_name=args.run_name, idle_timeout_minutes=args.idle_timeout_minutes,
               max_minutes=args.max_minutes, once=args.once)
    else:
        export(args.campaign, args.out, project=args.project, group=args.group)


if __name__ == '__main__':
    main()
