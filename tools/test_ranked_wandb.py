import json
from pathlib import Path
import sys
import tempfile
import unittest
import unittest.mock
from unittest.mock import patch
import ranked_wandb


class FakeRun:
    url = 'mock'
    dir = '/tmp/fake-wandb-run'

    def __init__(self):
        self.summary = {}
        self.rows = []
        self.closed = False

    def log(self, row):
        self.rows.append(row)

    def finish(self, exit_code=None):
        self.closed = True

    def define_metric(self, *args, **kwargs):
        pass


def write_campaign(root: Path, status='complete', rating_key='Rating/Ranking'):
    (root / 'metrics').mkdir(parents=True)
    (root / 'metrics/a.jsonl').write_text(
        json.dumps({'Total Timesteps': 2, rating_key: 1500}) + '\n' + '{"partial":')
    (root / 'state.json').write_text(json.dumps({
        'config': {'parent_steps': 100}, 'status': status, 'best': {'steps': 1},
        'target_ranking': 1700.0,
        'history': [{'stage': 0, 'steps': 1, 'screen': {'Ranking': 1400}, 'confirm': None,
                     'promoted': True}],
    }))


class StreamTests(unittest.TestCase):
    def test_backfill_partial_line_and_metric_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_campaign(root)
            run = FakeRun()
            with patch.object(ranked_wandb, 'authenticate', return_value='test'), \
                    patch.object(ranked_wandb, 'start', return_value=run) as start:
                ranked_wandb.stream(root, root / 'out', interval=0.01, idle_timeout_minutes=0.001)
            self.assertTrue(start.call_args.kwargs['online'])
            self.assertEqual([r['Total Timesteps'] for r in run.rows], [1, 2])
            self.assertEqual(run.rows[0]['Rating/Ranking'], 1400)
            self.assertEqual(run.rows[1]['Elo/Ranking'], 1500)
            self.assertFalse(any(key.startswith('Live/') for row in run.rows for key in row))
            self.assertFalse(any(key.startswith('Stream/') for key in run.summary))
            self.assertTrue((root / 'out/.relay_cursor.json').is_file())
            self.assertEqual(run.summary['Campaign/Status'], 'complete')
            self.assertTrue(run.closed)
            # Resuming the stream doesn't replay rows or completed evaluations.
            run.rows.clear()
            with patch.object(ranked_wandb, 'authenticate', return_value='test'), \
                    patch.object(ranked_wandb, 'start', return_value=run):
                ranked_wandb.stream(root, root / 'out', interval=0.01, idle_timeout_minutes=0.001)
            self.assertEqual(run.rows, [])

    def test_stream_logs_new_stages_as_they_appear_and_keeps_running_while_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_campaign(root, status='training')
            run = FakeRun()
            seen = {}

            def fake_log(row):
                run.rows.append(row)
                # Simulate the trainer finishing a stage while the stream is polling.
                if len(run.rows) == 2 and 'Campaign/Stage' not in seen:
                    seen['Campaign/Stage'] = True
                    state = json.loads((root / 'state.json').read_text())
                    state['history'].append({'stage': 1, 'steps': 2, 'screen': {'Ranking': 1500},
                                             'confirm': {'Ranking': 1510}, 'promoted': True})
                    (root / 'state.json').write_text(json.dumps(state))

            run.log = fake_log
            with patch.object(ranked_wandb, 'authenticate', return_value='test'), \
                    patch.object(ranked_wandb, 'start', return_value=run):
                ranked_wandb.stream(root, root / 'out', interval=0.01, max_minutes=0.02)
            stages = [r['Campaign/Stage'] for r in run.rows if 'Campaign/Stage' in r]
            self.assertEqual(stages, [0, 1])
            confirm = [r for r in run.rows if 'Confirmation/Ranking' in r]
            self.assertEqual(confirm[0]['Confirmation/Ranking'], 1510)
            self.assertFalse(run.summary['Campaign/Status'] != 'training')


class OfflineFallbackTests(unittest.TestCase):
    """A host that cannot reach W&B must keep syncable metrics, not lose them."""

    def test_unreachable_wandb_falls_back_to_a_local_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_campaign(root)
            run = FakeRun()
            calls = []

            def fake_start(directory, *a, **kw):
                calls.append((Path(directory), kw.get('online')))
                if kw.get('online'):
                    raise ConnectionError('TLS/SSL connection has been closed (EOF)')
                return run

            with patch.object(ranked_wandb, 'authenticate', return_value='test'), \
                    patch.object(ranked_wandb, 'start', side_effect=fake_start):
                ranked_wandb.stream(root, root / 'out', interval=0.01, idle_timeout_minutes=0.001,
                                    offline_fallback=True)
            self.assertEqual(calls[0][1], True)
            self.assertEqual(calls[1][0], root / 'out/offline')
            self.assertFalse(calls[1][1])           # offline run
            self.assertEqual([r['Total Timesteps'] for r in run.rows], [1, 2])
            self.assertTrue(run.closed)

    def test_without_the_flag_a_network_failure_still_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_campaign(root)
            with patch.object(ranked_wandb, 'authenticate', return_value='test'), \
                    patch.object(ranked_wandb, 'start', side_effect=ConnectionError('blocked')):
                with self.assertRaises(ConnectionError):
                    ranked_wandb.stream(root, root / 'out', interval=0.01, idle_timeout_minutes=0.001)


class ExportTests(unittest.TestCase):
    def test_export_writes_completed_rows_and_stages_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_campaign(root)
            run = FakeRun()
            with patch.object(ranked_wandb, 'start', return_value=run) as start:
                ranked_wandb.export(root, root / 'out')
            # Export writes an offline run: no credentials, no network.
            self.assertNotIn('online', start.call_args.kwargs)
            self.assertEqual([r['Total Timesteps'] for r in run.rows], [1, 2])
            self.assertEqual(run.summary['Campaign/Status'], 'complete')
            self.assertTrue(run.closed)

    def test_export_refuses_a_campaign_without_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_campaign(root)
            state = json.loads((root / 'state.json').read_text())
            state['history'] = []
            (root / 'state.json').write_text(json.dumps(state))
            with self.assertRaises(SystemExit):
                ranked_wandb.export(root, root / 'out')


class NoGithubTransportTests(unittest.TestCase):
    """Metrics must reach W&B over one direct HTTPS connection, never through GitHub."""

    def setUp(self):
        self.tools = Path(ranked_wandb.__file__).resolve().parent
        self.root = self.tools.parent

    def test_the_github_relay_is_gone(self):
        self.assertFalse((self.tools / 'wandb_live_bridge.py').exists())
        self.assertFalse((self.tools / 'test_wandb_live_bridge.py').exists())
        self.assertEqual(list((self.root / '.github').rglob('*wandb*')), [])

    def test_streamer_uses_no_github_api_or_subprocess(self):
        source = (self.tools / 'ranked_wandb.py').read_text()
        for needle in ('api.github.com', 'gh api', 'subprocess', 'GITHUB_TOKEN'):
            self.assertNotIn(needle, source)
        trainer = (self.tools / 'train_ranked.py').read_text()
        self.assertNotIn('--wandb-bridge', trainer)
        self.assertNotIn('wandb_live_bridge', trainer)

    def test_wandb_is_imported_lazily_so_json_only_training_needs_no_sdk(self):
        source = (self.tools / 'ranked_wandb.py').read_text()
        top_level = [line for line in source.splitlines()[:40] if line.startswith('import ') or line.startswith('from ')]
        self.assertFalse(any('wandb' in line for line in top_level))


if __name__ == '__main__':
    unittest.main()


class ApiKeyStorageTests(unittest.TestCase):
    """The key must persist in one private place outside the repository — never in git."""

    def setUp(self):
        import os
        self.os = os
        self._env = dict(os.environ)
        self._patch = patch.object(ranked_wandb, 'DEFAULT_KEY_FILE',
                                   Path(os.environ['HOME']) / 'test-wandb-key')
        self._patch.start()
        os.environ.pop('WANDB_API_KEY', None)
        os.environ.pop('RANKED_WANDB_KEY_FILE', None)

    def tearDown(self):
        self._patch.stop()
        self.os.environ.clear()
        self.os.environ.update(self._env)

    def test_login_writes_a_private_file_outside_the_repository(self):
        import stat
        with tempfile.TemporaryDirectory() as home:
            key_file = Path(home) / '.config/wandb/api_key'
            path = ranked_wandb.store_api_key('wandb_v1_test_key\n', key_file)
            self.assertEqual(path, key_file)
            self.assertEqual(key_file.read_text().strip(), 'wandb_v1_test_key')
            mode = stat.S_IMODE(key_file.stat().st_mode)
            self.assertEqual(mode, 0o600)
            self.assertEqual(stat.S_IMODE(key_file.parent.stat().st_mode), 0o700)
            self.assertFalse(Path(ranked_wandb.__file__).resolve().parents[1]
                             .resolve().is_relative_to(key_file.parent))

    def test_stored_key_is_found_without_any_environment_variable(self):
        with tempfile.TemporaryDirectory() as home:
            key_file = Path(home) / '.config/wandb/api_key'
            ranked_wandb.store_api_key('stored-key', key_file)
            self.os.environ['RANKED_WANDB_KEY_FILE'] = str(key_file)
            self.assertEqual(ranked_wandb.resolve_api_key(), ('stored-key', str(key_file)))

    def test_environment_wins_over_the_stored_file(self):
        with tempfile.TemporaryDirectory() as home:
            key_file = Path(home) / 'key'
            ranked_wandb.store_api_key('stored-key', key_file)
            self.os.environ['RANKED_WANDB_KEY_FILE'] = str(key_file)
            self.os.environ['WANDB_API_KEY'] = 'env-key'
            self.assertEqual(ranked_wandb.resolve_api_key(), ('env-key', 'environment'))

    def test_world_readable_key_file_is_refused(self):
        with tempfile.TemporaryDirectory() as home:
            key_file = Path(home) / 'key'
            key_file.write_text('leaky-key')
            key_file.chmod(0o644)
            self.os.environ['RANKED_WANDB_KEY_FILE'] = str(key_file)
            with patch('sys.stderr'):
                self.assertEqual(ranked_wandb.resolve_api_key()[0], '')

    def test_symlinked_key_file_is_refused(self):
        with tempfile.TemporaryDirectory() as home:
            real = Path(home) / 'real'; real.write_text('k'); real.chmod(0o600)
            link = Path(home) / 'link'; link.symlink_to(real)
            self.os.environ['RANKED_WANDB_KEY_FILE'] = str(link)
            self.assertEqual(ranked_wandb.resolve_api_key()[0], '')

    def test_key_netrc_entry_is_used_last(self):
        with tempfile.TemporaryDirectory() as home:
            netrc_path = Path(home) / '.netrc'
            netrc_path.write_text('machine api.wandb.ai login user password netrc-key\n')
            netrc_path.chmod(0o600)
            self.os.environ['RANKED_WANDB_KEY_FILE'] = str(Path(home) / 'missing')
            with patch.object(ranked_wandb, 'default_netrc', return_value=netrc_path):
                self.assertEqual(ranked_wandb.resolve_api_key(), ('netrc-key', str(netrc_path)))

    def test_refuses_to_write_a_key_inside_the_repository(self):
        repo = Path(ranked_wandb.__file__).resolve().parents[1]
        with self.assertRaises(SystemExit):
            ranked_wandb.store_api_key('k', repo / '.secrets/wandb_api_key')

    def test_authenticate_fails_loudly_instead_of_streaming_anonymously(self):
        with tempfile.TemporaryDirectory() as home:
            self.os.environ['RANKED_WANDB_KEY_FILE'] = str(Path(home) / 'missing')
            with patch.dict('sys.modules', {'wandb': type(sys)('wandb')}), \
                    patch.object(ranked_wandb, 'default_netrc', return_value=Path(home) / 'none'):
                with self.assertRaises(SystemExit) as caught:
                    ranked_wandb.authenticate()
            self.assertIn('login', str(caught.exception))

    def test_resolved_key_is_handed_to_wandb_without_being_echoed(self):
        fake = type(sys)('wandb')
        fake.login = unittest.mock.MagicMock()
        with tempfile.TemporaryDirectory() as home:
            key_file = Path(home) / 'key'
            ranked_wandb.store_api_key('secret-key', key_file)
            self.os.environ['RANKED_WANDB_KEY_FILE'] = str(key_file)
            with patch.dict('sys.modules', {'wandb': fake}):
                self.assertEqual(ranked_wandb.authenticate(), str(key_file))
            fake.login.assert_called_once_with(key='secret-key', relogin=True)
            self.assertEqual(self.os.environ['WANDB_API_KEY'], 'secret-key')
