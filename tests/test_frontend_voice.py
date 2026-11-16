"""学习模式纯语音对话前端逻辑测试：node 真跑 web.html 语音函数（无 node 时跳过）。"""
import shutil
import subprocess
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / 'frontend_voice.mjs'


@unittest.skipUnless(shutil.which('node'), '需要 node 运行前端行为测试')
class FrontendVoiceTests(unittest.TestCase):
    def test_voice_mode_behavior(self):
        proc = subprocess.run(['node', str(SCRIPT)], capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn('FRONTEND_VOICE=PASS', proc.stdout)


if __name__ == '__main__':
    unittest.main()
