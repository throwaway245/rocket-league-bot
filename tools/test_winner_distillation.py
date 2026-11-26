"""End-to-end native smoke test for match winner replay and actor-only learning."""
import hashlib
import json
import os
import re
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
            self.assertEqual(scores['tactical_restart_probability'], 0)
            self.assertGreater(scores['match_wins_a'], 0)
            tactical_result = out / 'tactical.json'
            run([str(BOT), '--eval', str(SMALL), '--vs', str(LARGE), '--eval-modes', '1v1',
                 '--eval-minutes', '10', '--eval-arenas', '2', '--eval-finish-games', '--seed', '2049000',
                 '--threads', '2', '--eval-winner-traces', str(out / 'tactical-traces'),
                 '--eval-tactical-restart-prob', '1', '--eval-out', str(tactical_result)])
            tactical = json.loads(tactical_result.read_text())
            self.assertEqual(tactical['tactical_restart_probability'], 1)
            self.assertEqual(tactical['match_wins_a'] + tactical['match_wins_b'] + tactical['match_draws'], 2)
            single_result = out / 'single.json'
            run([str(BOT), '--eval', str(SMALL), '--vs', str(LARGE), '--eval-modes', '1v1',
                 '--eval-single-match', '--seed', '2049001', '--threads', '2', '--eval-winner-policy-targets',
                 '--eval-winner-traces', str(out / 'single-traces'), '--eval-out', str(single_result)])
            single = json.loads(single_result.read_text())
            self.assertEqual(single['match_wins_a'] + single['match_wins_b'] + single['match_draws'], 1)
            self.assertEqual(single['num_arenas'], 1)
            self.assertEqual(single['match_wins_a'], 1)
            self.assertTrue((out / 'single-traces' / 'winner-a-policy.pt').is_file())
            replay = traces / 'winner-a.pt'
            replay_hash = digest(replay)
            history_dir = out / 'history'
            history_dir.mkdir()
            shutil.copy2(replay, history_dir / 'older.pt')
            candidate = out / 'large'
            shutil.copytree(LARGE, candidate / '0')
            train = [str(BOT), '--model', 'foundation-policy-v1', '--distill-teacher', str(SMALL),
                     '--checkpoints', str(candidate), '--device', 'cpu', '--threads', '2', '--games', '11',
                     '--modes', '1v1,2v2,3v3', '--mode-balance', 'players', '--distill-batch', '72',
                     '--distill-epochs', '1', '--distill-teacher-rollout', '0', '--distill-new-teacher',
                     '--distill-replay-in', str(replay), '--distill-replay-history', str(replay),
                     '--distill-replay-history-dir', str(history_dir), '--distill-replay-weight', '1',
                     '--max-timesteps', '72', '--no-metrics', '--seed', '99']
            trained = run(train)
            self.assertIn('Transfer Replay KL', trained.stdout)
            self.assertIn('Transfer Replay Action Agreement', trained.stdout)
            self.assertIn('winner replay observations', trained.stdout)
            self.assertNotIn('Old Policy Entropy:', trained.stdout)
            self.assertEqual(replay_hash, digest(replay))
            self.assertEqual(small_before, digest(SMALL / 'POLICY.lt'))
            newest = max((p for p in candidate.iterdir() if p.name.isdecimal()), key=lambda p: int(p.name))
            self.assertGreater(int(newest.name), 0)
            self.assertNotEqual(digest(newest / 'POLICY.lt'), large_before)
            self.assertFalse((newest / 'CRITIC.lt').exists())
            self.assertFalse((newest / 'CRITIC_OPTIM.lt').exists())
            before_eval = digest(newest / 'POLICY.lt')
            evaluated = run([str(BOT), '--model', 'foundation-policy-v1', '--distill-teacher', str(SMALL),
                             '--checkpoints', str(candidate), '--device', 'cpu', '--threads', '2', '--games', '11',
                             '--modes', '1v1', '--distill-batch', '72', '--distill-epochs', '1',
                             '--distill-teacher-rollout', '0', '--distill-replay-in', str(replay),
                             '--distill-replay-weight', '1', '--distill-eval', '--no-metrics', '--seed', '99'])
            self.assertIn('Transfer Replay Action Agreement:', evaluated.stdout)
            self.assertIn('Transfer Replay Policy Argmax Agreement:', evaluated.stdout)
            self.assertIn('Transfer Replay Teacher Action Agreement:', evaluated.stdout)
            self.assertIn('Transfer Replay Teacher KL:', evaluated.stdout)
            self.assertIn('Transfer Replay Rows:', evaluated.stdout)
            self.assertNotIn('Old Policy Entropy:', evaluated.stdout)
            self.assertEqual(before_eval, digest(newest / 'POLICY.lt'))
            self.assertEqual(replay_hash, digest(replay))
            # Identical policies must agree on argmax on every replay row and have zero KL.
            small_candidate = out / 'small'
            shutil.copytree(SMALL, small_candidate / '0')
            identical = run([str(BOT), '--model', 's', '--distill-teacher', str(SMALL),
                             '--checkpoints', str(small_candidate), '--device', 'cpu', '--threads', '2',
                             '--games', '11', '--modes', '1v1', '--distill-batch', '72', '--distill-epochs', '1',
                             '--distill-teacher-rollout', '0', '--distill-replay-in', str(replay),
                             '--distill-replay-weight', '1', '--distill-eval', '--distill-new-teacher',
                             '--distill-small-candidate', '--no-metrics', '--seed', '99'])
            self.assertIn('Transfer Replay Policy Argmax Agreement: 1', identical.stdout)
            self.assertIn('Transfer Replay Teacher KL: 0', identical.stdout)
            policy_replay = out / 'single-traces' / 'winner-a-policy.pt'
            policy_eval = run([str(BOT), '--model', 's', '--distill-teacher', str(SMALL),
                               '--checkpoints', str(small_candidate), '--device', 'cpu', '--threads', '2',
                               '--games', '11', '--modes', '1v1', '--distill-batch', '72', '--distill-epochs', '1',
                               '--distill-teacher-rollout', '0', '--distill-replay-in', str(policy_replay),
                               '--distill-replay-weight', '1', '--distill-eval', '--distill-new-teacher',
                               '--distill-small-candidate', '--no-metrics', '--seed', '99'])
            self.assertIn('Transfer Replay Policy Argmax Agreement: 1', policy_eval.stdout)
            replay_kl = re.search(r'^Transfer Replay KL: ([\deE+.-]+)$', policy_eval.stdout, re.MULTILINE)
            self.assertIsNotNone(replay_kl)
            self.assertLess(abs(float(replay_kl.group(1))), 1e-6)
            self.assertEqual(small_before, digest(small_candidate / '0' / 'POLICY.lt'))
            # This policy is not permitted to enter PPO.
            run([str(BOT), '--model', 'foundation-policy-v1', '--checkpoints', str(candidate),
                 '--device', 'cpu', '--no-metrics', '--max-timesteps', '1000'], success=False)
            self.assertEqual(large_before, digest(LARGE / 'POLICY.lt'))


if __name__ == '__main__':
    unittest.main()
