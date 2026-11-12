"""学习陪伴单元测试（第 0 期状态机 + 第一期画像/统计/报告）。"""
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from app.db import init_db, connection
from app import study
from app import study_profiler, study_report
from app import vision as vision_module


class StudyCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='robot-study-')
        self.addCleanup(self.tmp.cleanup)
        p = patch('app.db.DB_PATH', Path(self.tmp.name) / 'robot.db')
        p.start()
        self.addCleanup(p.stop)
        init_db()
        study.init_study_db()
        vision_module.init_vision_db()

    def make_session(self, person_id, started, status='active', study_seconds=0, break_seconds=0):
        with connection() as c:
            cur = c.execute(
                "INSERT INTO study_sessions(person_id,started_at,ended_at,study_seconds,break_seconds,status) VALUES(?,?,?,?,?,?)",
                (person_id, started.isoformat(), started.isoformat() if status == 'completed' else None,
                 study_seconds, break_seconds, status))
            sid = cur.lastrowid
            c.execute("INSERT INTO study_events(session_id,event_type,created_at) VALUES(?,'start',?)",
                      (sid, started.isoformat()))
        return sid


class LifecycleTests(StudyCase):
    def test_start_pause_resume_end_persists_seconds(self):
        session = study.start_session(1)
        with self.assertRaises(ValueError):
            study.start_session(1)
        study.pause_session(session['id'])
        study.resume_session(session['id'])
        ended = study.end_session(session['id'])
        self.assertEqual(ended['status'], 'completed')
        self.assertIsNotNone(ended['ended_at'])
        self.assertGreaterEqual(ended['study_seconds'], 0)
        self.assertIsNone(study.get_active_session(1))

    def test_rate_session_records_quality(self):
        session = study.start_session(1)
        study.rate_session(session['id'], 8, ['晚上', '有点吵'])
        with connection() as c:
            row = c.execute("SELECT quality_score,context_tags FROM study_events "
                            "WHERE session_id=? AND event_type='rate'", (session['id'],)).fetchone()
        self.assertEqual(row[0], 8)
        self.assertIn('晚上', row[1])
        with self.assertRaises(ValueError):
            study.rate_session(session['id'], 11)


class ReminderDecisionTests(StudyCase):
    def test_due_break_reminder_with_spoken_text(self):
        started = datetime.now().astimezone() - timedelta(minutes=45)
        sid = self.make_session(1, started)
        needs = study.sessions_needing_reminder(40, 5)
        self.assertEqual([(n['session']['id'], n['type']) for n in needs], [(sid, 'break_time')])
        self.assertIn('起来活动', needs[0]['spoken'])

    def test_extension_pushes_due_time(self):
        started = datetime.now().astimezone() - timedelta(minutes=41)
        sid = self.make_session(1, started)
        study.record_reminder(sid, 'extend', '10')
        self.assertEqual(study.sessions_needing_reminder(40, 5), [])
        # 11 分钟后到期（用固定 now 验证推后的到点生效）
        later = datetime.now().astimezone() + timedelta(minutes=11)
        needs = study.sessions_needing_reminder(40, 5, now=later)
        self.assertEqual([n['type'] for n in needs], ['break_time'])

    def test_gentle_reminder_once_for_extend_prone(self):
        started = datetime.now().astimezone() - timedelta(minutes=37)
        sid = self.make_session(1, started)
        resolver = lambda pid: {'focus_minutes': 40, 'break_minutes': 5, 'extend_prone': True}
        needs = study.sessions_needing_reminder(40, 5, threshold_resolver=resolver)
        self.assertEqual([n['type'] for n in needs], ['gentle'])
        study.record_reminder(sid, 'gentle', '3')
        self.assertEqual(study.sessions_needing_reminder(40, 5, threshold_resolver=resolver), [])

    def test_break_phase_resumes(self):
        started = datetime.now().astimezone() - timedelta(minutes=50)
        sid = self.make_session(1, started)
        study.pause_session(sid)  # break 事件在现在
        self.assertEqual(study.sessions_needing_reminder(40, 5), [])
        later = datetime.now().astimezone() + timedelta(minutes=6)
        needs = study.sessions_needing_reminder(40, 5, now=later)
        self.assertEqual([n['type'] for n in needs], ['resume_time'])


class StatsTests(StudyCase):
    def test_period_stats_and_snapshot_upsert(self):
        now = datetime.now().astimezone()
        self.make_session(1, now - timedelta(days=1, hours=2), 'completed',
                          study_seconds=3000, break_seconds=600)
        self.make_session(1, now - timedelta(days=1, hours=9), 'completed',
                          study_seconds=4200, break_seconds=0)
        self.make_session(1, now - timedelta(hours=1), 'active', study_seconds=0)
        start = now - timedelta(days=2)
        stats = study.compute_period_stats(1, start, now + timedelta(minutes=1), 'weekly')
        self.assertEqual(stats['total_sessions'], 3)
        self.assertEqual(stats['completed_sessions'], 2)
        self.assertEqual(stats['total_study_minutes'], (3000 + 4200) // 60)
        self.assertEqual(stats['max_focus_minutes'], 70)
        self.assertRegex(stats['best_time_slot'], r'^\d{2}:00-\d{2}:00$')
        # 快照写入且重复生成不报错（INSERT OR REPLACE）
        with connection() as c:
            rows = c.execute("SELECT COUNT(*) FROM study_analytics WHERE person_id=1").fetchone()[0]
        self.assertEqual(rows, 1)
        study.compute_period_stats(1, start, now + timedelta(minutes=1), 'weekly')
        with connection() as c:
            rows = c.execute("SELECT COUNT(*) FROM study_analytics WHERE person_id=1").fetchone()[0]
        self.assertEqual(rows, 1)

    def test_analytics_table_columns(self):
        # 旧库迁移路径：新列存在
        with connection() as c:
            cols = {r[1] for r in c.execute("PRAGMA table_info(study_events)").fetchall()}
        self.assertIn('quality_score', cols)
        self.assertIn('context_tags', cols)


class ProfileTests(StudyCase):
    def test_cold_start_uses_defaults(self):
        profile = study_profiler.build_profile(99)
        self.assertEqual(profile['optimal_focus_minutes'], 40)
        self.assertFalse(profile['extend_prone'])
        self.assertEqual(profile['trend'], 'new')
        self.assertEqual(profile['sessions_analyzed'], 0)

    def test_optimal_from_history_rounds_to_five(self):
        now = datetime.now().astimezone()
        for minutes in (50, 45, 40):
            self.make_session(1, now - timedelta(days=1), 'completed', study_seconds=minutes * 60)
        profile = study_profiler.build_profile(1)
        self.assertEqual(profile['optimal_focus_minutes'], 45)
        self.assertEqual(profile['sessions_analyzed'], 3)

    def test_threshold_resolver_shape(self):
        resolver = study_profiler.thresholds_for_reminder(1)
        self.assertEqual(set(resolver), {'focus_minutes', 'break_minutes', 'extend_prone'})

    def test_suggest_returns_actions(self):
        result = study_profiler.suggest(1)
        self.assertTrue(result['suggestions'])


class ReportTests(StudyCase):
    def test_week_report_markdown(self):
        now = datetime.now().astimezone()
        self.make_session(1, now - timedelta(days=1, hours=3), 'completed',
                          study_seconds=2700, break_seconds=300)
        report = study_report.build_report(1, '袁玥', 'week')
        self.assertIn('学习报告', report['markdown'])
        self.assertIn('45 分钟', report['markdown'])
        self.assertIn('袁玥', report['markdown'])
        self.assertEqual(report['stats']['completed_sessions'], 1)

    def test_today_report_empty(self):
        report = study_report.build_report(1, '袁玥', 'today')
        self.assertIn('这个周期还没有学习记录', report['markdown'])


if __name__ == '__main__':
    unittest.main()
