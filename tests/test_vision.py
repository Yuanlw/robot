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


class JudgeTests(unittest.TestCase):
    """决策层（六级漏斗）测试：severe 连击/冷却/预算、mild 静默、质量门、系统怀疑。"""

    def _rec(self, j, sev, present=True, quality=True, now=None, pid=1, sid=9):
        return j.record(pid, sid, sev, present, quality, now)

    def test_severe_streak_cooldown_budget(self):
        j = v.PostureJudge()
        base = datetime(2026, 11, 13, 10, 0)
        sev = {'头歪': 'severe'}
        for i in range(2):
            self.assertEqual(self._rec(j, sev, now=base + timedelta(seconds=5 * (i + 1)))['warnings'], [])
        r = self._rec(j, sev, now=base + timedelta(seconds=15))
        self.assertEqual(r['warnings'], ['头歪'])  # 连续3帧重度 → 开口（第1次）
        for _ in range(3):  # 冷却期内即使连击满也不开口
            r = self._rec(j, sev, now=base + timedelta(minutes=2))
        self.assertEqual(r['warnings'], [])
        self._rec(j, {}, now=base + timedelta(minutes=3))  # 坐正一次清连击
        for _ in range(3):
            r = self._rec(j, sev, now=base + timedelta(minutes=6))
        self.assertEqual(r['warnings'], ['头歪'])  # 冷却已过 → 第2次
        self._rec(j, {}, now=base + timedelta(minutes=7))
        for _ in range(3):
            r = self._rec(j, sev, now=base + timedelta(minutes=12))
        self.assertEqual(r['warnings'], [])        # 预算(2次)用完
        self.assertEqual(r['suppressed'], 'nag_limit')

    def test_mild_never_speaks(self):
        j = v.PostureJudge()
        now = datetime(2026, 11, 13, 10, 0)
        for i in range(10):
            r = self._rec(j, {'头歪': 'mild', '肩不平': 'mild'}, now=now + timedelta(seconds=5 * i))
        self.assertEqual(r['warnings'], [])

    def test_low_quality_frames_ignored(self):
        j = v.PostureJudge()
        now = datetime(2026, 11, 13, 10, 0)
        for i in range(6):
            r = self._rec(j, {'头歪': 'severe'}, quality=False, now=now + timedelta(seconds=5 * i))
        self.assertEqual(r['warnings'], [])
        self.assertEqual(r['stats']['quality_frames'], 0)

    def test_intermittent_severe_resets_streak(self):
        j = v.PostureJudge()
        base = datetime(2026, 11, 13, 10, 0)
        self._rec(j, {'头歪': 'severe'}, now=base)
        self._rec(j, {'头歪': 'severe'}, now=base + timedelta(seconds=5))
        self._rec(j, {}, now=base + timedelta(seconds=10))  # 坐正一次
        r = self._rec(j, {'头歪': 'severe'}, now=base + timedelta(seconds=15))
        self.assertEqual(r['warnings'], [])

    def test_suspect_calibration_suppresses_and_reset_clears(self):
        j = v.PostureJudge()
        now = datetime(2026, 11, 13, 10, 0)
        suspect_new_seen = False
        for i in range(20):  # 20帧高质量、70% poor（mild 也计入 poor）
            sev = {'头歪': 'mild'} if i % 10 < 7 else {}
            r = self._rec(j, sev, now=now + timedelta(seconds=5 * i))
            suspect_new_seen |= r['suspect_new']
        self.assertTrue(r['stats']['suspect'])
        self.assertTrue(suspect_new_seen)
        self.assertEqual(r['suppressed'], 'suspect_calibration')
        for _ in range(4):  # 抑制后即使重度连击也闭嘴
            r = self._rec(j, {'头歪': 'severe'}, now=now + timedelta(minutes=5))
        self.assertEqual(r['warnings'], [])
        j.reset(1)  # 重新校准 → 清零重来
        r = self._rec(j, {}, now=now + timedelta(minutes=6))
        self.assertFalse(r['stats']['suspect'])

    def test_session_switch_resets_budget(self):
        j = v.PostureJudge()
        now = datetime(2026, 11, 13, 10, 0)
        for k in range(3):
            j.record(1, 9, {'头歪': 'severe'}, True, True, now + timedelta(seconds=5 * k))
        for k in range(3):
            j.record(1, 9, {'头歪': 'severe'}, True, True, now + timedelta(minutes=6))
        for k in range(3):  # 换新会话 → 预算重置
            r = j.record(1, 10, {'头歪': 'severe'}, True, True, now + timedelta(minutes=12))
        self.assertEqual(r['warnings'], ['头歪'])

    def test_absent_pauses_and_return_resumes(self):
        j = v.PostureJudge()
        base = datetime(2026, 11, 13, 10, 0)
        self.assertFalse(j.record(1, 9, {}, False, False, base)['left'])
        self.assertFalse(j.record(1, 9, {}, False, False, base + timedelta(seconds=5))['left'])
        self.assertTrue(j.record(1, 9, {}, False, False, base + timedelta(seconds=10))['left'])
        self.assertFalse(j.record(1, 9, {}, True, True, base + timedelta(seconds=15))['returned'])
        self.assertTrue(j.record(1, 9, {}, True, True, base + timedelta(seconds=20))['returned'])


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


class SeverityTests(unittest.TestCase):
    def test_severity_levels_angles(self):
        base = {'shoulder_width': 0.3, 'head_drop': 0.3, 'head_roll': 0.0, 'shoulder_roll': 0.0}
        mild = v.compute_metrics(landmarks(tilt_deg=12))   # 10<12≤15 → mild
        self.assertEqual(v.issue_severities(mild, base).get('头歪'), 'mild')
        severe = v.compute_metrics(landmarks(tilt_deg=18))  # >15 → severe
        self.assertEqual(v.issue_severities(severe, base).get('头歪'), 'severe')

    def test_severity_levels_distance(self):
        base = {'shoulder_width': 0.3, 'head_drop': 0.3, 'head_roll': 0.0, 'shoulder_roll': 0.0}
        mild = v.compute_metrics(landmarks(width=0.36))    # 1.12<1.2≤1.30 → mild
        self.assertEqual(v.issue_severities(mild, base).get('离屏过近'), 'mild')
        severe = v.compute_metrics(landmarks(width=0.42))  # 1.4 > 1.30 → severe
        self.assertEqual(v.issue_severities(severe, base).get('离屏过近'), 'severe')


class MultiPersonTests(unittest.TestCase):
    """多人帧冻结判定（2026-11-13 多人场景 review）。"""

    def test_multi_frames_freeze_and_notice(self):
        j = v.PostureJudge()
        base = datetime(2026, 11, 13, 20, 0)
        sev = {'头歪': 'severe'}
        # 第1帧多人：冻结，不提醒，无说明
        r1 = j.record(1, 9, sev, True, True, base, multi=True)
        self.assertEqual(r1['warnings'], [])
        self.assertEqual(r1['suppressed'], 'multi_person')
        self.assertFalse(r1['multi_notice'])
        # 第2帧连续多人：触发一次说明
        r2 = j.record(1, 9, sev, True, True, base + timedelta(seconds=5), multi=True)
        self.assertTrue(r2['multi_notice'])
        # 冷却期内不再说明
        r3 = j.record(1, 9, sev, True, True, base + timedelta(seconds=10), multi=True)
        self.assertFalse(r3['multi_notice'])
        # 多人帧不积累姿态统计
        self.assertEqual(r3['stats']['quality_frames'], 0)

    def test_multi_frames_do_not_trigger_absent(self):
        j = v.PostureJudge()
        base = datetime(2026, 11, 13, 20, 0)
        for i in range(5):  # 连续多人帧不该被当成"没人"
            r = j.record(1, 9, {}, False, False, base + timedelta(seconds=5 * i), multi=True)
        self.assertFalse(r['left'])

    def test_normal_frames_resume_after_multi(self):
        j = v.PostureJudge()
        base = datetime(2026, 11, 13, 20, 0)
        j.record(1, 9, {}, True, True, base, multi=True)
        j.record(1, 9, {}, True, True, base + timedelta(seconds=5), multi=True)
        # 恢复单人后，severe 连击照常工作
        for i in range(3):
            r = j.record(1, 9, {'头歪': 'severe'}, True, True, base + timedelta(seconds=10 + 5 * i))
        self.assertEqual(r['warnings'], ['头歪'])
        self.assertEqual(r['stats']['quality_frames'], 3)


class MultiStatsTests(StudyCase):
    def test_behavior_excludes_multi_frames(self):
        from app import study_behavior
        sess = study.start_session(1)
        m = {'head_roll': 2.0, 'shoulder_roll': 1.0, 'shoulder_width': 0.3, 'head_drop': 0.25}
        for _ in range(3):
            v.log_frame(sess['id'], 1, True, m, [], [], None)                    # good
        for _ in range(2):
            v.log_frame(sess['id'], 1, True, m, ['头歪'], [], None, quality='multi')  # 多人帧
        p = study_behavior.analyze_behavior_pattern(1)
        self.assertEqual(p['frames'], 5)
        self.assertEqual(p['multi_frames'], 2)
        self.assertEqual(p['posture_good_ratio'], 1.0)  # 多人帧不进比率
