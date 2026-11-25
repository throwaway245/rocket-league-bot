#!/usr/bin/env python3
"""Run bounded distillation blocks with read-only imitation and match checks.

Ctrl-C/SIGTERM requests a native checkpoint and stops; no GitHub or W&B services.
Results and logs are local under results/foundation-v1. Never auto-promotes a model.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from resume_training import verify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/foundation-v1'
CHECKPOINTS = OUT / 'checkpoints'
TEACHER = ROOT / 'training-backup/best/5163287775'
BOT = ROOT / 'build/GigaLearnBot'
MODES = ('1v1', '2v2', '3v3')
active = None
stopping = False


def stop(signum, frame):
    global stopping
    stopping = True
    if active is not None and active.poll() is None:
        active.send_signal(signal.SIGTERM)


def latest_checkpoint():
    paths = [p for p in CHECKPOINTS.iterdir() if p.is_dir() and p.name.isdecimal()]
    if not paths:
        raise RuntimeError('No distillation checkpoint was saved')
    return max(paths, key=lambda p: int(p.name))


def run(command, log, env):
    global active
    with Path(log).open('a') as out:
        # Isolate the native process group so the controller forwards a stop signal exactly once.
        active = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        code = active.wait()
        active = None
    if code and not stopping:
        raise RuntimeError(f'Command failed with exit code {code}; see {log}')
    return code


def train_args(minutes, seed):
    return [str(BOT), '--model', 'foundation-v1', '--distill-teacher', str(TEACHER),
            '--checkpoints', str(CHECKPOINTS), '--device', 'cpu', '--threads', '2', '--games', '22',
            '--modes', ','.join(MODES), '--mode-balance', 'players', '--distill-batch', '1024',
            '--distill-epochs', '2', '--distill-teacher-rollout', '0.5', '--lr', '0.0001',
            '--max-minutes', str(minutes), '--ts-per-save', '20000', '--keep-checkpoints', '1',
            '--lean-metrics', '--fuzzed-kickoffs', '--tactical-state-prob', '0.15', '--rewards', 'v6',
            '--seed', str(seed)]


def evaluate(phase, env):
    checkpoint = latest_checkpoint()
    folder = OUT / 'validation' / checkpoint.name
    folder.mkdir(parents=True, exist_ok=True)
    measurements = []
    # Different from all training seeds. Use both teacher and student trajectories, separately by mode.
    for mode in MODES:
        for teacher in (0, 1):
            for seed in (20269921, 20269922):
                if stopping:
                    return None
                log = folder / f'{mode}-rollout{teacher}-seed{seed}.log'
                command = [str(BOT), '--model', 'foundation-v1', '--distill-teacher', str(TEACHER),
                           '--checkpoints', str(CHECKPOINTS), '--device', 'cpu', '--threads', '2',
                           '--games', '8', '--modes', mode, '--distill-batch', '4096',
                           '--distill-epochs', '1', '--distill-teacher-rollout', str(teacher),
                           '--distill-eval', '--lean-metrics', '--no-metrics', '--fuzzed-kickoffs',
                           '--tactical-state-prob', '0.15', '--rewards', 'v6', '--seed', str(seed)]
                # One invocation per file so a restart cannot accidentally parse a previous result.
                log.unlink(missing_ok=True)
                run(command, log, env)
                if stopping:
                    return None
                text = log.read_text(errors='replace')
                kl = re.findall(r'Transfer KL:\s*([0-9.eE+-]+)', text)
                accuracy = re.findall(r'Transfer Learn Accuracy:\s*([0-9.eE+-]+)', text)
                if not kl or not accuracy:
                    raise RuntimeError(f'Missing validation metrics in {log}')
                measurements.append({'mode': mode, 'teacher_rollout': teacher, 'seed': seed,
                                     'kl': float(kl[-1]), 'argmax_agreement': float(accuracy[-1])})
    passed = all(m['kl'] <= 0.10 and m['argmax_agreement'] >= 0.90 for m in measurements)
    result = {'phase': phase, 'checkpoint': int(checkpoint.name), 'measurements': measurements,
              'imitation_gate_passed': passed, 'match_gate_passed': False,
              'promoted': False, 'note': 'No automatic promotion; fixed-pool and independent match confirmation required.'}
    (folder / 'imitation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"Checkpoint {checkpoint.name}: imitation gate {'PASS' if passed else 'not yet passed'}", flush=True)
    return result


def match_checks(checkpoint, folder, env):
    """Side-balanced direct and fixed-opponent games; no confidence claims or promotion."""
    pool = ROOT / 'training-backup/learner/policy_versions'
    versions = sorted((p for p in pool.iterdir() if p.is_dir() and p.name.isdecimal()),
                      key=lambda p: int(p.name))
    if len(versions) < 2:
        raise RuntimeError('Need at least two frozen small-policy opponents for the match gate')
    # Version snapshots have STATS but not INFO; supply the known s observation/model metadata.
    prepared = ROOT / 'build/foundation-fixed-opponents'
    prepared.mkdir(parents=True, exist_ok=True)
    opponents = []
    for version in (versions[0], versions[-1]):
        dest = prepared / version.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(version, dest)
        shutil.copy2(TEACHER / 'INFO.json', dest / 'INFO.json')
        opponents.append(dest)
    def match(a, b, label, seed):
        path = folder / f'{label}-{seed}.json'
        log = folder / f'{label}-{seed}.log'
        command = [str(BOT), '--eval', str(a), '--vs', str(b), '--eval-modes', ','.join(MODES),
                   '--eval-minutes', '30', '--eval-arenas', '12', '--eval-max-seconds', '60',
                   '--eval-finish-games', '--seed', str(seed), '--threads', '2', '--eval-out', str(path)]
        path.unlink(missing_ok=True)
        log.unlink(missing_ok=True)
        run(command, log, env)
        if stopping:
            return None
        return json.loads(path.read_text())['modes']
    seeds = (20269931, 20269932)  # Not used for training or imitation checks.
    direct = []
    pool_results = []
    for seed in seeds:
        result = match(checkpoint, TEACHER, 'direct', seed)
        if result is None:
            return None
        direct.append(result)
        for opponent in opponents:
            student = match(checkpoint, opponent, f'student-vs-{opponent.name}', seed)
            if student is None:
                return None
            teacher = match(TEACHER, opponent, f'teacher-vs-{opponent.name}', seed)
            if teacher is None:
                return None
            pool_results.append({'opponent': opponent.name, 'seed': seed, 'student': student, 'teacher': teacher})
    def goals(result, side, mode):
        return int(result[mode][side]['goals'])
    def score(result, mode):
        a, b = goals(result, 'a', mode), goals(result, 'b', mode)
        return a / (a + b) if a + b else None
    summary = {}
    passed = True
    for mode in MODES:
        a = sum(goals(x, 'a', mode) for x in direct)
        b = sum(goals(x, 'b', mode) for x in direct)
        direct_share = a / (a + b) if a + b else 0
        fixed = []
        for item in pool_results:
            ss = score(item['student'], mode)
            ts = score(item['teacher'], mode)
            fixed.append(None if ss is None or ts is None else ss - ts)
        # Heuristic gates, not a formal statistical confidence interval.
        mode_pass = a + b >= 80 and direct_share >= 0.45 and all(x is not None and x >= -0.08 for x in fixed)
        passed &= mode_pass
        summary[mode] = {'direct_goals_student': a, 'direct_goals_teacher': b,
                         'direct_goal_share': direct_share, 'fixed_pool_share_differences': fixed,
                         'passed': mode_pass}
    result = {'checkpoint': int(checkpoint.name), 'seeds': seeds, 'mode_results': summary,
              'match_gate_passed': passed, 'promoted': False,
              'note': 'Heuristic gates only; independent longer match audit is still needed.'}
    (folder / 'matches.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block-minutes', type=float, default=15)
    parser.add_argument('--blocks', type=int, default=0, help='Maximum blocks; 0 means continue until stopped')
    args = parser.parse_args()
    if args.block_minutes <= 0 or args.blocks < 0:
        parser.error('block minutes must be positive and blocks nonnegative')
    if not BOT.is_file():
        parser.error('Build GigaLearnBot first')
    verify(ROOT / 'training-backup')
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GGL_WANDB='0', GGL_METRICS_EVERY='1',
               GGL_METRICS_DIR=str(OUT / 'metrics'))
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with (OUT / '.controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A foundation controller is already running')
        # The native trainer has no lock of its own. Refuse to race another native writer.
        for proc in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                argv = proc.read_bytes().split(b'\0')
                if any(b'GigaLearnBot' in a for a in argv) and b'--checkpoints' in argv:
                    idx = argv.index(b'--checkpoints')
                    if idx + 1 < len(argv) and Path(os.fsdecode(argv[idx + 1])).resolve() == CHECKPOINTS.resolve():
                        raise SystemExit('Another native trainer or validator uses the foundation checkpoint directory')
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
        phase = int(time.time())
        completed = 0
        while not stopping and (args.blocks == 0 or completed < args.blocks):
            verify(ROOT / 'training-backup')
            print(f'Starting distillation block {phase}', flush=True)
            run(train_args(args.block_minutes, phase), OUT / 'controller-train.log', env)
            if stopping:
                break
            result = evaluate(phase, env)
            if result is None:
                break
            if result['imitation_gate_passed']:
                print('Imitation gate passed; checking direct and fixed-pool matches.', flush=True)
                checkpoint = latest_checkpoint()
                match = match_checks(checkpoint, OUT / 'validation' / checkpoint.name, env)
                if match is None:
                    break
                if match['match_gate_passed']:
                    print('Heuristic match gates passed; pausing for independent audit (not auto-promoted).', flush=True)
                    break
                print('Match gates not yet passed; continuing distillation.', flush=True)
            completed += 1
            phase += 1
    print('Foundation distillation controller stopped.', flush=True)


if __name__ == '__main__':
    main()
