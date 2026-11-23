"""Campaign state reconciliation tests (no Torch, W&B or simulator needed)."""
import json
from pathlib import Path
import tempfile
import unittest

import check_campaign_state as check


def write_campaign(root, steps, metrics_steps, ckpt_steps, trained=False, stage=1):
    (root / 'metrics').mkdir(parents=True)
    (root / 'learner').mkdir(parents=True)
    state = {
        'status': 'paused',
        'history': [{'stage': 0}],
        'pending': {'stage': stage, 'start_steps': steps, 'seconds': 825.6, 'trained': trained},
    }
    (root / 'state.json').write_text(json.dumps(state))
    rows = [{'Total Timesteps': float(metrics_steps), 'Total Iterations': 150858}]
    (root / 'metrics' / 'run.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    for s in ckpt_steps:
        (root / 'learner' / str(s)).mkdir()
    return root


class CampaignStateTests(unittest.TestCase):
    def test_consistent_campaign_reports_no_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_campaign(Path(tmp) / 'c', 4876757727, 4889162217, [4880003907, 4889162217])
            self.assertEqual(check.check(root), 0)

    def test_metrics_and_checkpoint_disagreement_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_campaign(Path(tmp) / 'c', 4876757727, 4889162217, [4880003907])
            self.assertEqual(check.check(root), 1)

    def test_checkpoint_older_than_the_stage_start_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_campaign(Path(tmp) / 'c', 4876757727, 4876806237, [4876757727])
            self.assertEqual(check.check(root), 1)

    def test_trained_stage_without_a_history_entry_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_campaign(Path(tmp) / 'c', 4876757727, 4889162217,
                                  [4880003907, 4889162217], trained=True, stage=1)
            self.assertEqual(check.check(root), 1)


if __name__ == '__main__':
    unittest.main()
