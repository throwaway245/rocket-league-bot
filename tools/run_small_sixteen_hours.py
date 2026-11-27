#!/usr/bin/env python3
"""Run the restored small PPO campaign for 16 wall-clock hours, with 25-minute checks.

Backups are separate from the immutable starting snapshot and large distillation phases.
W&B and distillation are disabled. Failure or early exit is reported, never called success.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import resume_training
import training_service

DURATION = 16 * 60 * 60
INTERVAL = 1500
CAMPAIGN = ROOT / 'results/weights1700-v6/campaign'
DEST = ROOT / 'training-backup/small-sixteen-hour'
BRANCH = 'arena/9887d5e6-rocket-league-bot'


def main():
    os.chdir(ROOT)
    if subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() != BRANCH:
        raise RuntimeError('Wrong branch')
    state = json.loads((CAMPAIGN / 'state.json').read_text())
    command = resume_training.build_resume_command(state, CAMPAIGN, without_services=True, wandb_online=False)
    env = os.environ.copy()
    env.update(GGL_WANDB='0', OMP_WAIT_POLICY='passive')
    print('Launching small PPO campaign, no distillation or W&B:', command, flush=True)
    child = subprocess.Popen(command, cwd=ROOT, env=env)
    start = time.monotonic()
    print('Training started UTC:', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)
    finished = threading.Event()
    backup_state = {}

    def report_and_backup():
        checkpoints = sorted((p for p in (CAMPAIGN / 'learner').iterdir() if p.name.isdecimal()), key=lambda p: int(p.name))
        free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
        print(f'Monitor elapsed={time.monotonic()-start:.0f}s child={child.poll()} latest={checkpoints[-1].name if checkpoints else "none"} free_bytes={free}', flush=True)
        try:
            training_service.backup(CAMPAIGN, DEST, BRANCH, backup_state)
            print('Separate checkpoint backup:', json.dumps(backup_state), flush=True)
        except Exception as exc:
            print('BACKUP FAILED (will retry):', repr(exc), flush=True)

    def monitor():
        while not finished.is_set():
            # The requested monitoring cadence is a literal sleep 1500.
            sleeper = subprocess.Popen(['sleep', str(INTERVAL)])
            while sleeper.poll() is None and not finished.wait(5):
                pass
            if sleeper.poll() is None:
                sleeper.terminate()
                sleeper.wait()
            if finished.is_set():
                break
            report_and_backup()
        report_and_backup()

    thread = threading.Thread(target=monitor, daemon=False)
    thread.start()
    try:
        remaining = max(0, DURATION - (time.monotonic() - start))
        try:
            rc = child.wait(timeout=remaining)
            print(f'ERROR: campaign exited before 16 hours, exit={rc}, elapsed={time.monotonic()-start:.0f}s', flush=True)
            result = 1
        except subprocess.TimeoutExpired:
            print('16-hour wall-clock interval elapsed; requesting graceful save.', flush=True)
            child.send_signal(signal.SIGTERM)
            try:
                rc = child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                print('ERROR: campaign did not finish graceful shutdown within 600 seconds.', flush=True)
                rc = 1
            result = 0 if rc == 0 else 1
            print(f'Campaign shutdown exit={rc}, elapsed={time.monotonic()-start:.0f}s', flush=True)
    finally:
        finished.set()
        thread.join()
    return result


if __name__ == '__main__':
    sys.exit(main())
