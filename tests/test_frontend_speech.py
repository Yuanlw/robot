"""前端语音行为测试：用 node 真跑 web.html 的 scheduleSpeech（不靠字符串断言）。

回归 2026-11-14 静默无声事故。node 不在 PATH 时跳过（CI/无 node 机器），
但本机验收必须装 node 跑全量：超时必出声 / 取消必静默 / 正常路径播本地音频。
"""
import shutil
import subprocess
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / 'frontend_speech.mjs'


@unittest.skipUnless(shutil.which('node'), '需要 node 运行前端行为测试')
class FrontendSpeechTests(unittest.TestCase):
    def test_schedule_speech_behavior(self):
        proc = subprocess.run(['node', str(SCRIPT)], capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn('FRONTEND_SPEECH=PASS', proc.stdout)


if __name__ == '__main__':
    unittest.main()
