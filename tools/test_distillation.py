"""Regression checks for the fixed large model and native distillation path."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BOT = ROOT / 'build/GigaLearnBot'
TEACHER = ROOT / 'training-backup/best/5163287775'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DistillationTest(unittest.TestCase):
    def test_foundation_shape_is_versioned(self):
        source = (ROOT / 'src/ExampleMain.cpp').read_text()
        self.assertRegex(source, r'"foundation-v1"[^\n]+\{ 256, 256 \}[^\n]+\{ 1024, 1024, 1024 \}[^\n]+\{ 512, 512 \}[^\n]+\{ 512, 512 \}')

    @unittest.skipUnless(BOT.is_file() and TEACHER.is_dir() and (ROOT / 'collision_meshes').is_dir(),
                         'native executable, collision meshes and teacher required')
    def test_transfer_stop_and_read_only_validation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as tmp:
            out = Path(tmp) / 'student'
            before = digest(TEACHER / 'POLICY.lt')
            cmd = [str(BOT), '--model', 'foundation-v1', '--distill-teacher', str(TEACHER),
                   '--checkpoints', str(out), '--device', 'cpu', '--threads', '2', '--games', '11',
                   '--modes', '1v1,2v2,3v3', '--mode-balance', 'players', '--distill-batch', '72',
                   '--distill-epochs', '1', '--distill-teacher-rollout', '1', '--max-timesteps', '72',
                   '--no-metrics', '--lean-metrics', '--seed', '1984']
            env = dict(os.environ, GGL_WANDB='0')
            run = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=90)
            self.assertEqual(run.returncode, 0, run.stdout[-4000:] + run.stderr[-2000:])
            ckpt = max((p for p in out.iterdir() if p.name.isdecimal()), key=lambda p: int(p.name))
            info = json.loads((ckpt / 'INFO.json').read_text())
            self.assertEqual(info['model'], 'foundation-v1')
            self.assertEqual(info['distill_teacher'], str(TEACHER.resolve()))
            stats = json.loads((ckpt / 'RUNNING_STATS.json').read_text())
            self.assertEqual(stats['mirror_x'], -1)
            student = digest(ckpt / 'POLICY.lt')
            val = subprocess.run(cmd + ['--max-timesteps', '0', '--distill-eval', '--seed', '1985'],
                                 cwd=ROOT, env=env, text=True, capture_output=True, timeout=90)
            self.assertEqual(val.returncode, 0, val.stdout[-4000:] + val.stderr[-2000:])
            match = re.search(r'Transfer KL:\s*([0-9.eE+-]+)', val.stdout)
            self.assertIsNotNone(match, val.stdout[-2000:])
            self.assertTrue(math.isfinite(float(match.group(1))))
            self.assertEqual(student, digest(ckpt / 'POLICY.lt'))
            self.assertEqual(before, digest(TEACHER / 'POLICY.lt'))
            self.assertEqual(len(list(out.iterdir())), 1)


if __name__ == '__main__':
    unittest.main()
