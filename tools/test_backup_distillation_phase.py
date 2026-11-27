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
            replay = root / 'replay.pt'
            replay.write_bytes(b'original replay')
            with patch.object(backup, 'BACKUP', root / 'backups'):
                dest = backup.save_phase(src, 1, results, replay)
                data = json.loads((dest / 'manifest.json').read_text())
                self.assertEqual(data['total_timesteps'], 123)
                self.assertEqual(data['match_results']['seed'], 42)
                self.assertEqual(data['checkpoint_sha256']['POLICY.lt'], backup.sha(src / 'POLICY.lt'))
                self.assertEqual(data['winner_replay']['uncompressed_sha256'], backup.sha(replay))
                self.assertEqual(data['winner_replay']['compressed_sha256'], backup.sha(dest / 'winner-replay.pt.gz'))
                with self.assertRaises(ValueError):
                    backup.save_phase(src, 1, results)
                (src / 'CRITIC.lt').touch()
                with self.assertRaises(ValueError):
                    backup.save_phase(src, 2, results)


if __name__ == '__main__':
    unittest.main()
