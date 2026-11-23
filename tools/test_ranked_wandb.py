import json
from pathlib import Path
import tempfile
import unittest
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
            with patch.object(ranked_wandb, 'start', return_value=run) as start:
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
            with patch.object(ranked_wandb, 'start', return_value=run):
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
            with patch.object(ranked_wandb, 'start', return_value=run):
                ranked_wandb.stream(root, root / 'out', interval=0.01, max_minutes=0.02)
            stages = [r['Campaign/Stage'] for r in run.rows if 'Campaign/Stage' in r]
            self.assertEqual(stages, [0, 1])
            confirm = [r for r in run.rows if 'Confirmation/Ranking' in r]
            self.assertEqual(confirm[0]['Confirmation/Ranking'], 1510)
            self.assertFalse(run.summary['Campaign/Status'] != 'training')


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
