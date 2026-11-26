import json
from pathlib import Path
import tempfile
import unittest

from export_actor_only import export


class ExportActorOnlyTest(unittest.TestCase):
    def test_preserves_actor_and_source_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, dest = root / 'source', root / 'output'
            source.mkdir()
            info = {'model': 'foundation-v1', 'obs': 'MultiModeObs'}
            (source / 'INFO.json').write_text(json.dumps(info))
            for name in ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt', 'RUNNING_STATS.json', 'CRITIC.lt', 'CRITIC_OPTIM.lt', 'POLICY_OPTIM.lt'):
                (source / name).write_bytes(name.encode())
            export(source, dest)
            self.assertEqual(json.loads((dest / 'INFO.json').read_text())['model'], 'foundation-policy-v1')
            self.assertEqual(json.loads((source / 'INFO.json').read_text()), info)
            self.assertTrue((source / 'CRITIC.lt').is_file())
            self.assertFalse((dest / 'CRITIC.lt').exists())
            self.assertFalse((dest / 'CRITIC_OPTIM.lt').exists())
            for name in ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt', 'POLICY_OPTIM.lt'):
                self.assertEqual((source / name).read_bytes(), (dest / name).read_bytes())
            with self.assertRaises(ValueError):
                export(source, dest)

    def test_rejects_missing_files_and_nested_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'
            source.mkdir()
            (source / 'INFO.json').write_text(json.dumps({'model': 'foundation-v1', 'obs': 'MultiModeObs'}))
            with self.assertRaises(ValueError):
                export(source, source / 'output')
            with self.assertRaises(ValueError):
                export(source, Path(tmp) / 'output')
            self.assertFalse((Path(tmp) / 'output').exists())


if __name__ == '__main__':
    unittest.main()
