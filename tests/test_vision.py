"""坐姿守护单元测试（纯逻辑 + 数值入库；不依赖真实摄像头）。"""
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from app.db import init_db, connection
from app import study
from app import vision as v
from app import study_behavior


def landmarks(tilt_deg=0.0, shoulder_dy=0.0, width=0.3, nose_y=0.3):
    """构造 33 点姿态：耳朵线倾斜 tilt，肩线差 shoulder_dy，肩宽 width。"""
    lm = [(0.0, 0.0, 0.0)] * 33
    lm[v.NOSE] = (0.5, nose_y, 0.0)
    import math
    dx, dy = 0.05 * math.cos(math.radians(tilt_deg)), 0.05 * math.sin(math.radians(tilt_deg))
    lm[v.LEFT_EAR] = (0.5 - dx, 0.35 - dy, 0.0)
    lm[v.RIGHT_EAR] = (0.5 + dx, 0.35 + dy, 0.0)
    lm[v.LEFT_SHOULDER] = (0.5 - width / 2, 0.6, 0.0)
    lm[v.RIGHT_SHOULDER] = (0.5 + width / 2, 0.6 + shoulder_dy, 0.0)
    for i in range(33):
        if lm[i] == (0.0, 0.0, 0.0):
            lm[i] = (0.5, 0.5, 0.0)
    return lm


class MetricsTests(unittest.TestCase):
    def test_head_roll_detected(self):
        m = v.compute_metrics(landmarks(tilt_deg=15))
        self.assertIsNotNone(m)
        self.assertGreater(m['head_roll'], 10)
        self.assertIn('头歪', v.check_issues(m, None))

    def test_upright_no_issue(self):
        m = v.compute_metrics(landmarks())
        self.assertEqual(v.check_issues(m, None), [])

    def test_near_and_low_vs_baseline(self):
        base = {'shoulder_width': 0.3, 'head_drop': 0.3}
        near = v.compute_metrics(landmarks(width=0.45))
        self.assertIn('离屏过近', v.check_issues(near, base))
        low = v.compute_metrics(landmarks(nose_y=0.52))  # head_drop=0.6-0.52→0.08 < 0.62*0.3
        self.assertIn('头过低', v.check_issues(low, base))

    def test_invalid_landmarks(self):
        self.assertIsNone(v.compute_metrics([]))


class TrackerTests(unittest.TestCase):
    def test_streak_then_warn_then_cooldown(self):
        t = v.WarningTracker()
        now = datetime(2026, 11, 12, 10, 0)
        for i in range(2):
            r = t.record(1, ['头歪'], True, now + timedelta(seconds=5 * (i + 1)))
            self.assertEqual(r['warnings'], [])
        r = t.record(1, ['头歪'], True, now + timedelta(seconds=15))
        self.assertEqual(r['warnings'], ['头歪'])
        # 冷却 10 分钟内不再提醒（重新攒满 3 帧也不行）
        for i in range(3):
            r = t.record(1, ['头歪'], True, now + timedelta(minutes=5))
            self.assertEqual(r['warnings'], [])
        # 清一次重置连击，冷却过后再次满连击才提醒
        t.record(1, [], True, now + timedelta(minutes=6))
        for i in range(3):
            r = t.record(1, ['头歪'], True, now + timedelta(minutes=11))
        self.assertEqual(r['warnings'], ['头歪'])

    def test_intermittent_issue_does_not_accumulate_across_clear(self):
        t = v.WarningTracker()
        base = datetime(2026, 11, 12, 10, 0)
        t.record(1, ['头歪'], True, base)
        t.record(1, ['头歪'], True, base + timedelta(seconds=5))
        t.record(1, [], True, base + timedelta(seconds=10))   # 坐正一次
        r = t.record(1, ['头歪'], True, base + timedelta(seconds=15))
        self.assertEqual(r['warnings'], [])

    def test_absent_pauses_and_return_resumes(self):
        t = v.WarningTracker()
        base = datetime(2026, 11, 12, 10, 0)
        self.assertFalse(t.record(1, [], False, base)['left'])
        self.assertFalse(t.record(1, [], False, base + timedelta(seconds=5))['left'])
        r = t.record(1, [], False, base + timedelta(seconds=10))
        self.assertTrue(r['left'])
        self.assertFalse(t.record(1, [], True, base + timedelta(seconds=15))['returned'])
        r = t.record(1, [], True, base + timedelta(seconds=20))
        self.assertTrue(r['returned'])


class StudyCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='robot-vision-')
        self.addCleanup(self.tmp.cleanup)
        p = patch('app.db.DB_PATH', Path(self.tmp.name) / 'robot.db')
        p.start()
        self.addCleanup(p.stop)
        init_db()
        study.init_study_db()
        v.init_vision_db()


class PersistenceTests(StudyCase):
    def test_frame_stores_only_numbers(self):
        sess = study.start_session(1)
        m = v.compute_metrics(landmarks(tilt_deg=12))
        vid = v.log_frame(sess['id'], 1, True, m, ['头歪'], ['头歪'], '头歪提醒')
        with connection() as c:
            row = c.execute("SELECT head_roll,issues,warning_sent FROM study_vision_events WHERE id=?", (vid,)).fetchone()
        self.assertGreater(row[0], 10)
        self.assertEqual(row[1], '头歪')
        with connection() as c2:
            cols = [r[1] for r in c2.execute("PRAGMA table_info(study_vision_events)").fetchall()]
        self.assertNotIn('image', cols)
        self.assertNotIn('photo', cols)

    def test_baseline_roundtrip(self):
        m = v.compute_metrics(landmarks())
        self.assertIsNone(v.get_baseline(1))
        v.save_baseline(1, m)
        self.assertEqual(v.get_baseline(1)['shoulder_width'], m['shoulder_width'])

    def test_mark_responded_within_window(self):
        sess = study.start_session(1)
        m = v.compute_metrics(landmarks(tilt_deg=12))
        v.log_frame(sess['id'], 1, True, m, ['头歪'], ['头歪'], '提醒')
        with connection() as c:
            c.execute("UPDATE study_vision_events SET created_at=? ",
                      ((datetime.now().astimezone() - timedelta(minutes=1)).isoformat(),))
        v.mark_responded(1, ['头歪'])
        with connection() as c:
            responded = c.execute("SELECT user_responded FROM study_vision_events").fetchone()[0]
        self.assertEqual(responded, 1)

    def test_old_warning_not_marked_responded(self):
        sess = study.start_session(1)
        m = v.compute_metrics(landmarks(tilt_deg=12))
        v.log_frame(sess['id'], 1, True, m, ['头歪'], ['头歪'], '提醒')
        with connection() as c:
            c.execute("UPDATE study_vision_events SET created_at=?",
                      ((datetime.now().astimezone() - timedelta(minutes=30)).isoformat(),))
        v.mark_responded(1, ['头歪'])
        with connection() as c:
            responded = c.execute("SELECT user_responded FROM study_vision_events").fetchone()[0]
        self.assertEqual(responded, 0)

    def test_behavior_pattern(self):
        sess = study.start_session(1)
        m = v.compute_metrics(landmarks(tilt_deg=12))
        for _ in range(4):
            v.log_frame(sess['id'], 1, True, m, ['头歪'], [], None)
        v.log_frame(sess['id'], 1, False, None, [], [], None)
        p = study_behavior.analyze_behavior_pattern(1)
        self.assertTrue(p['has_data'])
        self.assertEqual(p['poor_frames'], 4)
        self.assertIn('头歪', p['common_issues'])


if __name__ == '__main__':
    unittest.main()


class RelativeThresholdTests(unittest.TestCase):
    def test_angle_issues_use_personal_baseline(self):
        # 基线本身带 7°/6°（摄像头仰拍的系统性偏差）
        base = {'shoulder_width': 0.3, 'head_drop': 0.3, 'head_roll': 7.0, 'shoulder_roll': 6.0}
        m12 = v.compute_metrics(landmarks(tilt_deg=12))
        # 绝对阈值(10°)会误报；相对阈值 = max(10, 7+8) = 15 → 不报
        self.assertNotIn('头歪', v.check_issues(m12, base))
        m18 = v.compute_metrics(landmarks(tilt_deg=18))
        self.assertIn('头歪', v.check_issues(m18, base))  # 18 > 15 真问题

    def test_no_baseline_falls_back_absolute(self):
        m = v.compute_metrics(landmarks(tilt_deg=12))
        self.assertIn('头歪', v.check_issues(m, None))


class CalibrationGateTests(unittest.TestCase):
    def test_reasonable_baseline_passes(self):
        # Chris 昨晚真实基线（shoulder_roll=16.2）应通过门槛，由相对判定吸收
        self.assertIsNone(v.validate_calibration({'head_roll': 8.8, 'shoulder_roll': 16.2}))

    def test_extreme_frame_rejected(self):
        self.assertIsNotNone(v.validate_calibration({'head_roll': 20.0, 'shoulder_roll': 5.0}))
        self.assertIsNotNone(v.validate_calibration({'head_roll': 3.0, 'shoulder_roll': 30.0}))
