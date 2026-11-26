"""End-to-end native smoke test for match winner replay and actor-only learning."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BOT = ROOT / 'build/GigaLearnBot'
SMALL = ROOT / 'training-backup/learner/7178045919'
LARGE = ROOT / 'training-backup/large-distillation/phase-0000/checkpoint'
CANDIDATES = ROOT / 'results/distillation-candidates'
MESH = ROOT / 'collision_meshes/soccar/mesh_15.cmf'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(BOT.is_file() and SMALL.is_dir() and LARGE.is_dir() and MESH.is_file(),
                     'native build, simulation assets and both model checkpoints required')
class WinnerDistillationTest(unittest.TestCase):
    def test_match_replay_actor_only_and_isolated_small(self):
        CANDIDATES.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=CANDIDATES, prefix='smoke-') as tmp:
            out = Path(tmp)
            traces, result = out / 'winner-traces', out / 'match.json'
            env = dict(os.environ, GGL_WANDB='0')
            def run(cmd, success=True):
                proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=180)
                if success:
                    self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-1500:])
                else:
                    self.assertNotEqual(proc.returncode, 0)
                return proc
            small_before = digest(SMALL / 'POLICY.lt')
            large_before = digest(LARGE / 'POLICY.lt')
            run([str(BOT), '--eval', str(SMALL), '--vs', str(LARGE), '--eval-modes', '1v1',
                 '--eval-minutes', '10', '--eval-arenas', '2', '--eval-finish-games', '--seed', '2048999',
                 '--threads', '2', '--eval-winner-traces', str(traces), '--eval-out', str(result)])
            scores = json.loads(result.read_text())
            self.assertEqual(scores['match_seconds'], 300)
            self.assertGreater(scores['match_wins_a'], 0)
            replay = traces / 'winner-a.pt'
            replay_hash = digest(replay)
            candidate = out / 'large'
            shutil.copytree(LARGE, candidate / '0')
            train = [str(BOT), '--model', 'foundation-policy-v1', '--distill-teacher', str(SMALL),
                     '--checkpoints', str(candidate), '--device', 'cpu', '--threads', '2', '--games', '11',
                     '--modes', '1v1,2v2,3v3', '--mode-balance', 'players', '--distill-batch', '72',
                     '--distill-epochs', '1', '--distill-teacher-rollout', '0', '--distill-new-teacher',
                     '--distill-replay-in', str(replay), '--distill-replay-weight', '1',
                     '--max-timesteps', '72', '--no-metrics', '--seed', '99']
            trained = run(train)
            self.assertIn('Transfer Replay KL', trained.stdout)
            self.assertNotIn('Old Policy Entropy:', trained.stdout)
            self.assertEqual(replay_hash, digest(replay))
            self.assertEqual(small_before, digest(SMALL / 'POLICY.lt'))
            newest = max((p for p in candidate.iterdir() if p.name.isdecimal()), key=lambda p: int(p.name))
            self.assertGreater(int(newest.name), 0)
            self.assertNotEqual(digest(newest / 'POLICY.lt'), large_before)
            self.assertFalse((newest / 'CRITIC.lt').exists())
            self.assertFalse((newest / 'CRITIC_OPTIM.lt').exists())
            # This policy is not permitted to enter PPO.
            run([str(BOT), '--model', 'foundation-policy-v1', '--checkpoints', str(candidate),
                 '--device', 'cpu', '--no-metrics', '--max-timesteps', '1000'], success=False)
            self.assertEqual(large_before, digest(LARGE / 'POLICY.lt'))


if __name__ == '__main__':
    unittest.main()
