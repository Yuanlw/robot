"""人脸识别升级（A-E 五项）单元测试：纯逻辑 + FakeFR 驱动，不依赖真实模型。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.db import init_db, connection
from app.face import (FaceService, _parse_samples, _capped, _vote, _yaw_asym,
                      CORE_SAMPLES, LEARN_NOVELTY)
from app.config import settings


class PureLogicTests(unittest.TestCase):
    def test_parse_samples_new_legacy_lbph_garbage(self):
        self.assertEqual(_parse_samples(json.dumps({"backend": "dlib", "samples": [[1.0], [2.0]]}).encode()), [[1.0], [2.0]])
        self.assertEqual(_parse_samples(json.dumps({"backend": "dlib", "values": [1.0, 2.0]}).encode()), [[1.0, 2.0]])  # 旧格式兼容
        self.assertEqual(_parse_samples(json.dumps({"backend": "lbph", "model": "x"}).encode()), [])
        self.assertEqual(_parse_samples(b'not-json'), [])
        self.assertEqual(_parse_samples(None), [])

    def test_capped_keeps_core_and_newest(self):
        samples = [[float(i)] for i in range(10)]
        out = _capped(samples, cap=4)
        self.assertEqual(len(out), 4)
        self.assertEqual(out[:CORE_SAMPLES], samples[:CORE_SAMPLES])  # 录入核心永久保留
        self.assertEqual(out[-1], samples[-1])                       # 最新学习样本保留
        self.assertEqual(_capped([[1.0]], cap=8), [[1.0]])           # 未超限原样

    def test_vote_rules(self):
        self.assertEqual(_vote([(1, .3), (1, .35), (None, .9)]), (1, 2, 3))   # 3帧2中
        self.assertEqual(_vote([(1, .3), (2, .3), (3, .3)]), (None, 1, 3))    # 三帧三人 → 未知
        self.assertEqual(_vote([(1, .3)]), (1, 1, 1))                          # 单帧
        self.assertEqual(_vote([(1, .3), (2, .3)]), (None, 1, 2))             # 2帧不一致 → 未知
        self.assertEqual(_vote([(1, .3), (1, .4)]), (1, 2, 2))                # 2帧一致
        self.assertEqual(_vote([]), (None, 0, 0))
        self.assertEqual(_vote([(None, None)]), (None, 0, 1))

    def test_yaw_asym(self):
        sym = {'left_eye': [(10, 10), (12, 10)], 'right_eye': [(20, 10), (22, 10)],
               'nose_tip': [(15, 16), (16, 18), (16, 18)]}
        self.assertLess(_yaw_asym(sym), 0.05)
        asym = {'left_eye': [(10, 10), (12, 10)], 'right_eye': [(20, 10), (22, 10)],
                'nose_tip': [(10, 16), (10, 18), (10, 18)]}  # 鼻子贴左眼 → 大侧脸
        self.assertGreater(_yaw_asym(asym), 0.5)
        self.assertEqual(_yaw_asym({}), 1.0)  # 缺关键点 → 拒绝


class FakeFR:
    """替身 face_recognition：enc_map[path] 决定该"照片"的特征向量。"""

    def __init__(self, enc_map=None):
        self.enc_map = enc_map or {}

    def load_image_file(self, path):
        return str(path)

    def face_locations(self, img):
        return [(0, 100, 100, 0)]

    def face_encodings(self, img, locations=None):
        return self.enc_map.get(str(img), [])

    def face_landmarks(self, img):
        return []

    def face_distance(self, known, enc):
        return np.linalg.norm(np.asarray(known, dtype=float) - np.asarray(enc, dtype=float), axis=1)


def zeros():
    return [0.0] * 128


def offset(v):
    e = zeros()
    e[0] = v
    return e


class FaceServiceCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='robot-face-')
        self.addCleanup(self.tmp.cleanup)
        p = patch('app.db.DB_PATH', Path(self.tmp.name) / 'robot.db')
        p.start()
        self.addCleanup(p.stop)
        init_db()
        self.svc = FaceService()
        self.svc.init_face_db()
        gate = patch.object(FaceService, '_quality_gate', return_value=(True, ''))
        gate.start()
        self.addCleanup(gate.stop)

    def use_fake(self, enc_map):
        fake = FakeFR(enc_map)
        p = patch.object(FaceService, '_dlib', return_value=fake)
        p.start()
        self.addCleanup(p.stop)
        return fake

    def enroll_person(self, name, path, enc):
        fake = self.use_fake({path: [np.array(enc)]})
        return self.svc.enroll(name, path)


class EnrollTests(FaceServiceCase):
    def test_enroll_then_append_samples(self):
        r1 = self.enroll_person('袁玥', '/p1.jpg', offset(0.0))
        self.assertEqual(r1['samples'], 1)
        fake = FakeFR({'/p2.jpg': [np.array(offset(0.3))]})
        with patch.object(FaceService, '_dlib', return_value=fake):
            r2 = self.svc.enroll('袁玥', '/p2.jpg', append=True)
        self.assertEqual(r2['samples'], 2)
        self.assertEqual(self.svc.sample_counts()[r1['id']], 2)

    def test_append_duplicate_rejected(self):
        self.enroll_person('袁玥', '/p1.jpg', offset(0.0))
        fake = FakeFR({'/p2.jpg': [np.array(offset(0.01))]})  # 距离 0.01 < LEARN_NOVELTY
        with patch.object(FaceService, '_dlib', return_value=fake):
            with self.assertRaises(ValueError):
                self.svc.enroll('袁玥', '/p2.jpg', append=True)

    def test_append_to_unknown_name_becomes_first_enroll(self):
        fake = FakeFR({'/p9.jpg': [np.array(offset(0.0))]})
        with patch.object(FaceService, '_dlib', return_value=fake):
            r = self.svc.enroll('新成员', '/p9.jpg', append=True)
        self.assertEqual(r['samples'], 1)

    def test_cap_rolls_and_keeps_core(self):
        self.enroll_person('袁玥', '/p0.jpg', offset(0.0))
        for i in range(1, 12):  # 追加 11 个互不重复的样本（间隔 0.1 > LEARN_NOVELTY）
            fake = FakeFR({f'/p{i}.jpg': [np.array(offset(0.1 + i * 0.1))]})
            with patch.object(FaceService, '_dlib', return_value=fake):
                self.svc.enroll('袁玥', f'/p{i}.jpg', append=True)
        counts = list(self.svc.sample_counts().values())
        self.assertEqual(counts[0], settings.face_max_samples)
        with connection() as c:
            payload = c.execute("SELECT face_encoding FROM people").fetchone()[0]
        samples = _parse_samples(payload)
        self.assertEqual(samples[0], offset(0.0))  # 首个录入核心仍在


class RecognizeTests(FaceServiceCase):
    def test_min_distance_across_samples_and_unknown(self):
        # 录入两个差异较大的样本
        self.enroll_person('袁玥', '/p1.jpg', offset(0.0))
        fake = FakeFR({'/p2.jpg': [np.array(offset(0.6))]})
        with patch.object(FaceService, '_dlib', return_value=fake):
            self.svc.enroll('袁玥', '/p2.jpg', append=True)
        # 查询靠近第二个样本（距离 0.12）→ 应命中（单样本旧逻辑会算出 0.6 判未知）
        fake = FakeFR({'/q.jpg': [np.array(offset(0.48))]})
        with patch.object(FaceService, '_dlib', return_value=fake):
            r = self.svc.recognize('/q.jpg')
        self.assertEqual(r['faces'][0]['name'], '袁玥')
        self.assertLess(r['faces'][0]['distance'], settings.face_threshold)
        # 远离所有样本 → 未知（1.5 距 0.0/0.6/已学样本都 > 阈值）
        fake2 = FakeFR({'/q2.jpg': [np.array(offset(1.5))]})
        with patch.object(FaceService, '_dlib', return_value=fake2):
            r2 = self.svc.recognize('/q2.jpg')
        self.assertIsNone(r2['faces'][0]['person_id'])

    def test_incremental_learning(self):
        self.enroll_person('袁玥', '/p1.jpg', offset(0.0))
        before = self.svc.sample_counts()
        pid = list(before)[0]
        # 高置信（0.38 ≤ 0.40）且够新（>0.08）→ 自动学习入池
        fake = FakeFR({'/q.jpg': [np.array(offset(0.38))]})
        with patch.object(FaceService, '_dlib', return_value=fake):
            self.svc.recognize('/q.jpg')
        self.assertEqual(self.svc.sample_counts()[pid], before[pid] + 1)
        # 低置信（0.45 > 0.40，但仍命中识别）→ 不学习
        fake = FakeFR({'/q2.jpg': [np.array(offset(0.45))]})
        with patch.object(FaceService, '_dlib', return_value=fake):
            self.svc.recognize('/q2.jpg')
        self.assertEqual(self.svc.sample_counts()[pid], before[pid] + 1)

    def test_burst_vote_and_log(self):
        self.enroll_person('袁玥', '/p1.jpg', offset(0.0))
        fake = FakeFR({'/f0.jpg': [np.array(offset(0.2))],   # 命中
                       '/f1.jpg': [np.array(offset(0.25))],  # 命中
                       '/f2.jpg': [np.array(offset(0.9))]})  # 未知
        with patch.object(FaceService, '_dlib', return_value=fake):
            r = self.svc.recognize_burst(['/f0.jpg', '/f1.jpg', '/f2.jpg'])
        self.assertEqual(r['faces'][0]['name'], '袁玥')
        self.assertEqual((r['frames'], r['matched_frames']), (3, 2))
        self.assertAlmostEqual(r['faces'][0]['distance'], 0.2, places=3)  # 胜者帧中的最小距离
        # E：识别距离已落库
        with connection() as c:
            rows = c.execute("SELECT person_id,distance,frames,matched_frames FROM face_recog_log WHERE purpose='recognize'").fetchall()
        self.assertTrue(any(row[2] == 3 and row[3] == 2 for row in rows))

    def test_burst_disagree_returns_unknown(self):
        self.enroll_person('袁玥', '/p1.jpg', offset(0.0))
        self.enroll_person('Chris', '/c1.jpg', offset(0.6))
        fake = FakeFR({'/f0.jpg': [np.array(offset(0.1))],    # → 袁玥
                       '/f1.jpg': [np.array(offset(0.55))],   # → Chris (|0.55-0.6|=0.05)
                       '/f2.jpg': [np.array(offset(1.5))]})  # → 未知（距两人都远）
        with patch.object(FaceService, '_dlib', return_value=fake):
            r = self.svc.recognize_burst(['/f0.jpg', '/f1.jpg', '/f2.jpg'])
        self.assertEqual(r['faces'], [])
        self.assertEqual(r['matched_frames'], 1)

    def test_stats_shape(self):
        self.enroll_person('袁玥', '/p1.jpg', offset(0.0))
        fake = FakeFR({'/q.jpg': [np.array(offset(0.2))], '/q2.jpg': [np.array(offset(0.9))]})
        with patch.object(FaceService, '_dlib', return_value=fake):
            self.svc.recognize('/q.jpg')
            self.svc.recognize('/q2.jpg')
        st = self.svc.stats(7)
        self.assertEqual(st['total'], 2)
        self.assertEqual(st['matched'], 1)
        self.assertEqual(st['match_rate'], 0.5)
        self.assertIsNotNone(st['matched_distance']['p50'])


if __name__ == '__main__':
    unittest.main()


class MultiFaceTests(FaceServiceCase):
    def test_burst_rejects_multi_face_frames(self):
        self.enroll_person('袁玥', '/p1.jpg', offset(0.0))
        fake = FakeFR({'/f0.jpg': [np.array(offset(0.15))],                    # 单人命中
                       '/f1.jpg': [np.array(offset(0.2)), np.array(offset(0.25))],  # 两张脸 → 废票
                       '/f2.jpg': [np.array(offset(0.18))]})                   # 单人命中
        with patch.object(FaceService, '_dlib', return_value=fake):
            r = self.svc.recognize_burst(['/f0.jpg', '/f1.jpg', '/f2.jpg'])
        self.assertEqual(r['multi_face_frames'], 1)
        self.assertEqual(r['faces'][0]['name'], '袁玥')  # 2/3 有效票一致仍认定
        self.assertEqual(r['matched_frames'], 2)

    def test_burst_majority_multi_returns_unknown(self):
        self.enroll_person('袁玥', '/p1.jpg', offset(0.0))
        fake = FakeFR({'/f0.jpg': [np.array(offset(0.1)), np.array(offset(0.15))],
                       '/f1.jpg': [np.array(offset(0.1)), np.array(offset(0.15))],
                       '/f2.jpg': [np.array(offset(0.2))]})
        with patch.object(FaceService, '_dlib', return_value=fake):
            r = self.svc.recognize_burst(['/f0.jpg', '/f1.jpg', '/f2.jpg'])
        self.assertEqual(r['multi_face_frames'], 2)
        self.assertEqual(r['faces'], [])  # 有效票不足多数 → 宁可未知
