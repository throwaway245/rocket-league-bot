import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import backup_distillation_phase as backup


class DistillationBackupTest(unittest.TestCase):
    def test_verified_and_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / 'src'
            src.mkdir()
            for name in backup.REQUIRED:
                (src / name).write_bytes(name.encode())
            (src / 'INFO.json').write_text(json.dumps({'model': 'foundation-policy-v1', 'obs': 'MultiModeObs'}))
            (src / 'RUNNING_STATS.json').write_text(json.dumps({'total_timesteps': 123}))
            results = root / 'matches.json'
            results.write_text('{"seed": 42}')
            with patch.object(backup, 'BACKUP', root / 'backups'):
                dest = backup.save_phase(src, 1, results)
                data = json.loads((dest / 'manifest.json').read_text())
                self.assertEqual(data['total_timesteps'], 123)
                self.assertEqual(data['match_results']['seed'], 42)
                self.assertEqual(data['checkpoint_sha256']['POLICY.lt'], backup.sha(src / 'POLICY.lt'))
                with self.assertRaises(ValueError):
                    backup.save_phase(src, 1, results)
                (src / 'CRITIC.lt').touch()
                with self.assertRaises(ValueError):
                    backup.save_phase(src, 2, results)


if __name__ == '__main__':
    unittest.main()
