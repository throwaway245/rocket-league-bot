import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import resume_training
import train_ranked
from resume_training import restore, verify


def strip_cadence(tokens):
    """A recipe without --ts-per-save, which the derived save cadence owns."""
    out, i = [], 0
    while i < len(tokens):
        if tokens[i] == '--ts-per-save':
            i += 2
            continue
        out.append(tokens[i])
        i += 1
    return out


class RestoreTests(unittest.TestCase):
    def make_snapshot(self, path):
        path.mkdir()
        state = b'{"status":"paused","pending":{"seconds":1366.75}}'
        (path / 'state.json').write_bytes(state)
        (path / 'manifest.json').write_text(json.dumps({'checkpoint': 123,
            'sha256': {'state.json': hashlib.sha256(state).hexdigest()}}))

    def test_restore_verifies_and_never_overwrites_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, work = root / 'snapshot', root / 'work'
            self.make_snapshot(snapshot)
            result = restore(snapshot, work)
            self.assertEqual(result['pending']['seconds'], 1366.75)
            self.assertFalse((work / 'manifest.json').exists())
            (work / 'state.json').write_text('{"status":"newer"}')
            self.assertEqual(restore(snapshot, work)['status'], 'newer')
            self.assertEqual(json.loads((snapshot / 'state.json').read_text())['status'], 'paused')

    def test_compact_evolving_backup_rebuilds_local_only_campaign_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, work = root / 'snapshot', root / 'work'
            (snapshot / 'learner/9000').mkdir(parents=True)
            versions = snapshot / 'learner/policy_versions'
            versions.mkdir()
            for step in range(10):
                version = versions / str(step)
                version.mkdir()
                for name in ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt'):
                    (version / name).write_bytes(f'{step}:{name}'.encode())
                (version / 'STATS.json').write_text(json.dumps({'skill_ratings': {
                    '1v1': 1500 + step, '2v2': 1500 + step, '3v3': 1500 + step}}))
            state = {'status': 'paused', 'runtime_overrides': {
                'evolving_opponents': True, 'opponent_pool_size': 10}}
            (snapshot / 'state.json').write_text(json.dumps(state))
            hashes = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in snapshot.rglob('*') if p.is_file()}
            (snapshot / 'manifest.json').write_text(json.dumps({'checkpoint': 9000, 'sha256': hashes}))
            restored = restore(snapshot, work)
            self.assertEqual(len(list((work / 'pool').iterdir())), 10)
            self.assertTrue(all((work / folder).is_dir() for folder in ('eval', 'metrics', 'best')))
            self.assertEqual(restored['pool_sha256'], train_ranked.pool_digest(work / 'pool'))

    def test_tampered_backup_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, work = root / 'snapshot', root / 'work'
            self.make_snapshot(snapshot)
            (snapshot / 'state.json').write_text('corrupt')
            with self.assertRaises(ValueError):
                restore(snapshot, work)
            self.assertFalse(work.exists())

    def test_restore_only_uses_saved_campaign_path_without_starting_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / 'snapshot'
            self.make_snapshot(snapshot)
            state = json.loads((snapshot / 'state.json').read_text())
            state['working_directory'] = 'results/weights1700/campaign'
            payload = json.dumps(state).encode()
            (snapshot / 'state.json').write_bytes(payload)
            (snapshot / 'manifest.json').write_text(json.dumps({'checkpoint': 123,
                'sha256': {'state.json': hashlib.sha256(payload).hexdigest()}}))
            with patch.object(resume_training, 'ROOT', root), patch.object(resume_training, 'SNAPSHOT', snapshot), \
                 patch.object(resume_training, 'WORK', root / 'results/old'), \
                 patch.object(resume_training, 'restore', wraps=lambda **kw: restore(snapshot=snapshot, **kw)), \
                 patch.object(resume_training.subprocess, 'run') as launch, \
                 patch('sys.argv', ['resume_training.py', '--restore-only']):
                resume_training.main()
            self.assertTrue((root / 'results/weights1700/campaign/state.json').exists())
            self.assertFalse((root / 'results/old').exists())
            launch.assert_not_called()

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_snapshot(root / 'snapshot')
            (root / 'snapshot/manifest.json').write_text(json.dumps({'sha256': {'../outside': 'bad'}}))
            with self.assertRaises(ValueError):
                verify(root / 'snapshot')

    def state(self):
        """A campaign record whose pinned recipe was actually produced by the driver.

        Building `train_args` through `train_ranked.recipe_train_args` is what makes the round-trip
        test meaningful: a hand-written recipe is not what the trainer would have pinned.
        """
        args = train_ranked.build_arg_parser().parse_known_args([
            '--base', 'checkpoints/3244539846', '--out', 'results/x',
            '--threads', '2', '--minibatch', '1008',
            '--lr', '0.00011', '--critic-lr', '0.0002', '--epochs', '2',
            '--entropy-scale', '0.0025', '--norm-advantages', 'mode', '--target-kl', '0.02',
            '--tactical-state-prob', '0.22', '--train-vs-old', '0.3', '--ts-per-save', '10m',
            '--mask-entropy', '--fuzzed-kickoffs', '--pfsp', '--rewards', 'Goal=600,Save=65',
            '--skill-minutes', '15',
        ])[0]
        return {
            'target_ranking': 1700.0,
            'runtime_overrides': {'reward_metrics': True, 'player_metrics': True,
                                  'throughput_profile': 'cpu-20k', 'continuous': True,
                                  'evolving_opponents': True, 'opponent_pool_size': 10,
                                  'promotion_margin': 50.0},
            'config': {
                'base': 'checkpoints/3244539846',
                'minutes': [12, 20, 20],
                'rounds': 40,
                'seed': 20261118,
                'audit_seed': 9007,
                'reference_pool': 'training-backup/pool',
                'extra_versions': ['checkpoints/3137998278'],
                'train_args': train_ranked.recipe_train_args(args),
            },
        }

    def replay(self, cmd):
        """Parse a resume command the way train_ranked.main() does.

        Everything up to `--` is driver flags; everything after it is the native recipe the driver
        forwards to the trainer (--games, --mode-balance, --ts-per-itr, --fixed-rollout,
        --lean-metrics, --keep-checkpoints).
        """
        argv = [str(a) for a in cmd[2:]]
        cut = argv.index('--')
        return train_ranked.build_arg_parser().parse_args(argv[:cut]), argv[cut + 1:]

    def build(self, **kwargs):
        return resume_training.build_resume_command(
            self.state(), resume_training.ROOT / 'results/weights1700-v3/campaign', **kwargs)

    def test_build_resume_command_reproduces_the_pinned_recipe(self):
        state = self.state()
        cmd = self.build()
        self.assertNotIn('--monitor', cmd)
        known, passthrough = self.replay(cmd)
        # --ts-per-save is owned by the derived save cadence; every other token has to match, or
        # train_ranked.py refuses to resume ("Campaign settings or frozen pool changed").
        self.assertEqual(strip_cadence(train_ranked.recipe_train_args(known)),
                         strip_cadence(state['config']['train_args']))
        # The preset the driver does not pin itself travels as native passthrough. The old builder
        # dropped these, which made every resume fail the config check.
        self.assertEqual(passthrough, ['--games', '88', '--mode-balance', 'players',
                                       '--ts-per-itr', '48384', '--fixed-rollout',
                                       '--lean-metrics', '--keep-checkpoints', '2'])
        for token in ('--github-backup', '--backup-interval', '1800', '--wandb-online',
                      '--mask-entropy', '--fuzzed-kickoffs', '--pfsp',
                      '--norm-advantages', 'mode', '--epochs', '2',
                      '--critic-lr', '0.0002', '--tactical-state-prob', '0.22',
                      '--extra-versions', 'checkpoints/3137998278', '--reward-metrics',
                      '--player-metrics', '--throughput-profile', 'cpu-20k', '--continuous',
                      '--evolving-opponents', '--opponent-pool-size', '10',
                      '--promotion-margin', '50.0'):
            self.assertIn(token, cmd)

    def test_restored_recipe_still_matches_after_a_full_round_trip(self):
        """The stored recipe is exactly what the pinned recipe builder produces for it."""
        state = self.state()
        known, _ = self.replay(self.build())
        produced = strip_cadence(train_ranked.recipe_train_args(known))
        self.assertEqual(produced, strip_cadence(state['config']['train_args']))
        # Sanity: the builder is deterministic, so a second resume is byte-identical.
        self.assertEqual(self.build(), self.build())

    def test_without_services_drops_backup_and_stream_but_keeps_the_recipe(self):
        state = self.state()
        cmd = self.build(without_services=True)
        self.assertNotIn('--github-backup', cmd)
        self.assertNotIn('--wandb-online', cmd)
        known, _ = self.replay(cmd)
        self.assertEqual(strip_cadence(train_ranked.recipe_train_args(known)),
                         strip_cadence(state['config']['train_args']))

    def test_runtime_rewards_are_a_runtime_override_not_a_recipe_change(self):
        """A reward change must leave the pinned recipe alone, or the campaign stops resuming."""
        state = self.state()
        cmd = self.build(runtime_rewards='v6')
        self.assertEqual(cmd[cmd.index('--runtime-rewards') + 1], 'v6')
        # The pinned recipe still carries the original preset, and the override is appended after it
        # so it wins at the trainer's argv level.
        self.assertIn('Goal=600,Save=65', cmd)   # the pinned recipe's own preset, unchanged
        # It must come after the pinned --rewards, which is what makes the later flag win.
        self.assertGreater(cmd.index('--runtime-rewards'),
                           cmd.index('--rewards'))
        known, passthrough = self.replay(cmd)
        self.assertEqual(strip_cadence(train_ranked.recipe_train_args(known)),
                         strip_cadence(state['config']['train_args']))
        # ... and train_ranked persists it as a runtime override, so the next resume picks it up from
        # state.json without the flag (asserted through the builder, not by re-running the campaign).
        remembered = self.state()
        remembered['runtime_overrides']['rewards'] = 'v6'
        self.assertIn('--runtime-rewards',
                      resume_training.build_resume_command(
                          remembered, resume_training.ROOT / 'results/weights1700-v3/campaign'))

    def test_reward_specs_are_validated_against_the_bots_own_tables(self):
        train_ranked.validate_rewards('v6')
        train_ranked.validate_rewards('Goal=750,Save=30')
        for bad in ('v7', 'Goal=750,NotAReward=2', 'Goal=750,garbage'):
            with self.assertRaises(ValueError):
                train_ranked.validate_rewards(bad)

    def test_no_wandb_keeps_the_backup_worker(self):
        cmd = self.build(wandb_online=False)
        self.assertIn('--github-backup', cmd)
        self.assertNotIn('--wandb-online', cmd)



if __name__ == '__main__':
    unittest.main()
