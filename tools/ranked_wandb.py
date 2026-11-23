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

The API key comes from the first of these that exists, so the same command works
whether the operator configured it once or exported it for one run:

  1. `WANDB_API_KEY` in the environment (best for a container/systemd unit that
     injects secrets at launch).
  2. `--api-key-file`, default `~/.config/wandb/api_key` — a private file in the
     user's config directory, outside the repository. Create it with
     `python3 tools/ranked_wandb.py login` (set `RANKED_WANDB_KEY_FILE` to put it
     somewhere else).
  3. `~/.netrc`, the standard home for machine credentials (`api.wandb.ai`).
  4. `wandb login`, i.e. W&B's own store in `~/.netrc`.

No key is ever read from, or written to, the repository: this file contains no
GitHub transport and the 30-minute backup worker only stages `training-backup/`.

Examples
    python3 tools/ranked_wandb.py login                      # store the key once, private file
    python3 tools/ranked_wandb.py stream results/weights1700/campaign --out results/weights1700/wandb
    python3 tools/ranked_wandb.py export results/weights1700/campaign --out results/weights1700/wandb-offline
"""
import argparse
import getpass
import hashlib
import json
import netrc
import os
from pathlib import Path
import stat
import sys
import time

PROJECT = 'rocket-league-ggl'
GROUP = 'weights-1700'
CURSOR_NAME = '.relay_cursor.json'
RATING_PREFIX = 'Rating/'
NETRC_HOST = 'api.wandb.ai'

# Where a stored key lives by default. Deliberately outside the repository (`~` is the
# user's own config directory), private to the user, and independent of the W&B SDK's
# own settings so a `wandb` upgrade can never silently relocate it.
DEFAULT_KEY_FILE = Path('~/.config/wandb/api_key').expanduser()


def key_file():
    """The key file in use: `RANKED_WANDB_KEY_FILE`, else `~/.config/wandb/api_key`."""
    override = os.environ.get('RANKED_WANDB_KEY_FILE', '').strip()
    return Path(override).expanduser() if override else DEFAULT_KEY_FILE


def default_netrc():
    return Path('~/.netrc').expanduser()


def _read_key_file(path):
    """Read a key file, refusing one that is group/world readable or a symlink."""
    path = Path(path)
    try:
        info = path.lstat()
    except OSError:
        return ''
    if stat.S_ISLNK(info.st_mode):
        return ''
    if info.st_mode & (stat.S_IRGRP | stat.S_IROTH):
        print(f'Ignoring {path}: it is readable by other users (chmod 600 it).', file=sys.stderr)
        return ''
    try:
        return path.read_text(encoding='utf-8').strip()
    except OSError:
        return ''


def _read_netrc(path):
    path = Path(path)
    if not path.is_file():
        return ''
    try:
        auth = netrc.netrc(str(path)).authenticators(NETRC_HOST)
    except (OSError, netrc.NetrcParseError):
        return ''
    return (auth[2] or '').strip() if auth else ''


def resolve_api_key(explicit=''):
    """Return `(key, source)` from env, key file, netrc — never from the repository."""
    if explicit:
        return explicit.strip(), 'explicit'
    env_val = os.environ.get('WANDB_API_KEY', '').strip()
    if env_val:
        return env_val, 'environment'
    path = key_file()
    stored = _read_key_file(path)
    if stored:
        return stored, str(path)
    stored = _read_netrc(default_netrc())
    if stored:
        return stored, str(default_netrc())
    return '', 'none'


def store_api_key(key, path=None):
    """Write `key` to a private (0600) file outside the repository and return its path."""
    path = Path(path).expanduser() if path is not None else key_file()
    repo = Path(__file__).resolve().parents[1].resolve()
    if path.resolve().is_relative_to(repo):
        raise SystemExit(f'Refusing to store the W&B key inside the repository: {path}\n'
                         'Use ~/.config/wandb/api_key (default) or an environment variable.')
    key = key.strip()
    if not key:
        raise SystemExit('Empty key; nothing written.')
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    # Create with 0600 from the start, so the key is never briefly world readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, (key + '\n').encode('utf-8'))
    finally:
        os.close(fd)
    os.chmod(path, 0o600)
    return path


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


def authenticate(explicit_key=''):
    """Make sure W&B has a credential, using the first source that has one.

    Returns the source of the key (`environment`, a file path, `explicit` or `netrc`). The key
    itself is handed to the SDK through the environment and is never printed.
    """
    import wandb
    key, source = resolve_api_key(explicit_key)
    if not key:
        raise SystemExit(
            'No W&B credential found. Store one once with\n'
            f'    python3 tools/ranked_wandb.py login\n'
            f'(it is kept private in {key_file()}, or set RANKED_WANDB_KEY_FILE), '
            'or export WANDB_API_KEY for this run.')
    os.environ['WANDB_API_KEY'] = key
    # relogin=True rewrites ~/.netrc so the SDK and this process agree on the credential.
    wandb.login(key=key, relogin=True)
    return source


def stream(campaign, directory, interval=10.0, project=PROJECT, group=GROUP, run_name='three-mode-ranking-live',
           idle_timeout_minutes=60.0, max_minutes=None, once=False, stop_statuses=('complete', 'paused', 'failed'),
           api_key='', offline_fallback=False):
    """Tail durable campaign JSON and log it directly to online W&B until it stops moving.

    With `offline_fallback`, a host that cannot reach W&B (or has no credential) keeps logging
    into a local, syncable run under `<directory>/offline` instead of losing the metrics. Nothing
    is retried in a loop: the reason is printed once, and `wandb sync` uploads the run later.
    """
    campaign = Path(campaign)
    directory = Path(directory)
    state = json.loads((campaign / 'state.json').read_text())  # the writer renames atomically
    config = state.get('config') or {}
    last_step, stages = read_cursor(directory)
    run, source = None, 'none'
    try:
        source = authenticate(api_key)
        run = start(directory, config, run_name, online=True,
                    run_id=campaign_run_id(campaign, config), project=project, group=group)
        print(f'Online W&B run: {run.url} (credential: {source})', flush=True)
    except BaseException as exc:
        if not offline_fallback or isinstance(exc, KeyboardInterrupt):
            raise
        # Common on sandboxes that block W&B egress. Keep the data, do not spin.
        print(f'W&B is not usable from this host ({type(exc).__name__}: {str(exc)[:200]}).\n'
              f'Logging into an OFFLINE run under {directory / "offline"} instead; '
              f'sync it later with: wandb sync {directory / "offline"}', flush=True)
        run = start(directory / 'offline', config, run_name, online=False, project=project, group=group)
        source = 'offline'
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

    lg = sub.add_parser('login', help='Store the W&B API key once, in a private file outside the repo')
    lg.add_argument('--api-key-file', type=Path, default=None,
                    help=f'Destination (default: {DEFAULT_KEY_FILE}; env RANKED_WANDB_KEY_FILE)')
    lg.add_argument('--key', default=None,
                    help='Key value; omit to be prompted (kept out of shell history)')

    ex = sub.add_parser('export', help='Write a syncable offline run from a finished campaign')
    ex.add_argument('campaign', type=Path)
    ex.add_argument('--out', type=Path, required=True, help='Destination for the offline run')
    ex.add_argument('--project', default=PROJECT)
    ex.add_argument('--group', default=GROUP)

    st.add_argument('--api-key-file', type=Path, default=None,
                    help='Key file override (default: env RANKED_WANDB_KEY_FILE or ~/.config/wandb/api_key)')
    st.add_argument('--offline-fallback', action='store_true',
                    help='If W&B is unreachable or unauthenticated, log into <out>/offline instead of '
                         'exiting (sync later with `wandb sync`); never loses metrics')

    args = parser.parse_args()
    if getattr(args, 'interval', 1) <= 0:
        parser.error('--interval must be positive')
    if args.mode == 'login':
        path = args.api_key_file or key_file()
        if args.key:
            key = args.key
        elif not sys.stdin.isatty():
            key = sys.stdin.readline()
        else:
            key = getpass.getpass('W&B API key (input hidden): ')
        stored = store_api_key(key, path)
        print(f'Stored W&B API key in {stored} (mode 600). It is outside the repository and '
              'never committed by the backup worker.')
        return
    if args.mode == 'stream':
        if args.api_key_file is not None:
            os.environ['RANKED_WANDB_KEY_FILE'] = str(args.api_key_file)
        stream(args.campaign, args.out, interval=args.interval, project=args.project, group=args.group,
               run_name=args.run_name, idle_timeout_minutes=args.idle_timeout_minutes,
               max_minutes=args.max_minutes, once=args.once,
               offline_fallback=args.offline_fallback)
    else:
        export(args.campaign, args.out, project=args.project, group=args.group)


if __name__ == '__main__':
    main()
