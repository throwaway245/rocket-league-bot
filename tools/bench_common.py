#!/usr/bin/env python3
"""Shared helpers of the benchmark and verification tools.

The tools run the repository's own training recipe (`train_1v1_2v2_3v3.sh`), so that
they measure exactly what the training measures, and they read the metrics that the
trainer writes to `GGL_METRICS_DIR`.
"""
import json
import os
import resource
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = 'train_1v1_2v2_3v3.sh'


def parse_env(spec):
    """Turns `A=1,B=2` into a dictionary."""
    env = {}
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise SystemExit(f'Expected NAME=VALUE, got "{part}"')
        name, value = part.split('=', 1)
        env[name.strip()] = value.strip()
    return env


def default_checkpoints():
    """The newest checkpoint of the repository, or None when there is none."""
    for candidate in ('training-backup/learner', 'results/weights1700-v6/campaign/learner'):
        folder = ROOT / candidate
        if not folder.is_dir():
            continue
        if any(name.isdecimal() for name in os.listdir(folder)):
            return folder
    return None


def newest_checkpoint(folder):
    names = [name for name in os.listdir(folder) if name.isdecimal() and (Path(folder) / name).is_dir()]
    if not names:
        raise SystemExit(f'No checkpoint in {folder}')
    return max(names, key=int)


def prepare_checkpoints(source):
    """Copies the newest checkpoint and its old-policy pool into a fresh folder."""
    work = tempfile.mkdtemp(prefix='ggl_tool_')
    target = os.path.join(work, 'checkpoints')
    os.makedirs(target)
    name = newest_checkpoint(source)
    shutil.copytree(os.path.join(source, name), os.path.join(target, name))
    pool = os.path.join(source, 'policy_versions')
    if os.path.isdir(pool):
        shutil.copytree(pool, os.path.join(target, 'policy_versions'))
    return work, target


def run_recipe(args, env, timeout=3600):
    """Runs the training recipe and returns its output, its metrics, its wall and CPU time."""
    work = tempfile.mkdtemp(prefix='ggl_run_')
    try:
        checkpoints = os.path.join(work, 'checkpoints')
        os.makedirs(checkpoints)
        metrics = os.path.join(work, 'metrics')

        run_env = dict(os.environ)
        run_env.update(env)
        run_env['GGL_METRICS_DIR'] = metrics
        # Every iteration, so that the measurement has enough rows to work with.
        run_env['GGL_METRICS_EVERY'] = '1'
        run_env.setdefault('OMP_WAIT_POLICY', 'passive')
        run_env.pop('GGL_WANDB', None)

        # The recipe ends in `exec`, so the shell is replaced by the trainer and the CPU
        # time of the child is the CPU time of the training.
        cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.time()
        proc = subprocess.run(['bash', str(ROOT / RECIPE), '--checkpoints', checkpoints] + list(args),
                              cwd=ROOT, env=run_env, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              timeout=timeout)
        wall = time.time() - start
        cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (cpu_after.ru_utime - cpu_before.ru_utime) + (cpu_after.ru_stime - cpu_before.ru_stime)
        if proc.returncode != 0:
            print(proc.stdout[-3000:])
            raise SystemExit(f'The trainer failed with exit code {proc.returncode}')

        rows = []
        if os.path.isdir(metrics):
            for name in sorted(os.listdir(metrics)):
                if not name.endswith('.jsonl'):
                    continue
                with open(os.path.join(metrics, name)) as f:
                    rows += [json.loads(line) for line in f if line.strip()]
        return proc.stdout, rows, wall, cpu
    finally:
        shutil.rmtree(work, ignore_errors=True)


def throughput(rows, wall, cpu):
    """The throughput of a run: the steps per second of the wall clock and of the CPU."""
    if len(rows) < 3:
        raise SystemExit('Too few iterations were completed to measure a throughput')
    steps = sum(r.get('Collected Timesteps', 0) for r in rows[1:])
    span = rows[-1]['_time'] - rows[1]['_time']
    if span <= 0 or steps <= 0:
        raise SystemExit('The run did not collect any steps')
    return {
        'overall_sps': steps / span,
        'cpu_ms_per_1k_steps': 1e6 * cpu / steps,
        'wall_seconds': wall,
        'cpu_seconds': cpu,
        'steps': steps,
    }
