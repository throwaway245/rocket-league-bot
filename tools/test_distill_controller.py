import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import distill_foundation as controller


class ControllerTest(unittest.TestCase):
    def test_match_gate_requires_direct_and_fixed_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher = root / 'teacher'
            teacher.mkdir()
            (teacher / 'INFO.json').write_text('{"model":"s","obs":"MultiModeObs"}')
            pool = root / 'training-backup/learner/policy_versions'
            for n in ('100', '200'):
                p = pool / n
                p.mkdir(parents=True)
                (p / 'STATS.json').write_text('{}')
            folder = root / 'results'
            folder.mkdir()
            checkpoint = root / '123'
            checkpoint.mkdir()
            def fake_run(command, log, env):
                a = Path(command[command.index('--eval') + 1])
                b = Path(command[command.index('--vs') + 1])
                # Student can defeat a weak pool but cannot pass if it loses badly to teacher.
                ga, gb = ((10, 100) if a == checkpoint and b == teacher else (60, 40))
                modes = {m: {'a': {'goals': ga}, 'b': {'goals': gb}} for m in controller.MODES}
                out = Path(command[command.index('--eval-out') + 1])
                out.write_text(json.dumps({'modes': modes}))
                return 0
            with patch.object(controller, 'ROOT', root), patch.object(controller, 'TEACHER', teacher), \
                    patch.object(controller, 'run', side_effect=fake_run), patch.object(controller, 'stopping', False):
                result = controller.match_checks(checkpoint, folder, {})
            self.assertFalse(result['match_gate_passed'])
            self.assertFalse(result['promoted'])
            self.assertFalse(result['mode_results']['1v1']['passed'])

    def test_training_has_no_remote_services(self):
        argv = controller.train_args(10, 123)
        self.assertNotIn('--github-backup', argv)
        self.assertNotIn('--wandb', argv)
        self.assertEqual(argv[argv.index('--model') + 1], 'foundation-v1')
        self.assertEqual(argv[argv.index('--distill-teacher-rollout') + 1], '0.5')


if __name__ == '__main__':
    unittest.main()
