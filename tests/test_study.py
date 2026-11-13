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
        from app import agent_store
        agent_store.init_agent_db()

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


class ReconcileTests(StudyCase):
    def _backdate(self, table, column_value, sql_where, ts):
        with connection() as c:
            c.execute(f"UPDATE {table} SET created_at=? WHERE {sql_where}", (ts.isoformat(), column_value))

    def test_overdue_session_auto_ends(self):
        now = datetime.now().astimezone()
        sid = self.make_session(1, now - timedelta(hours=7))
        actions = study.reconcile_sessions(now)
        self.assertEqual([(a[0], a[2]) for a in actions], [(sid, 'end')])
        self.assertEqual(study.get_session(sid)['status'], 'completed')

    def test_idle_session_auto_ends(self):
        now = datetime.now().astimezone()
        sid = self.make_session(1, now - timedelta(hours=3))
        actions = study.reconcile_sessions(now)
        self.assertEqual([(a[0], a[2], a[3]) for a in actions], [(sid, 'end', '长时间无活动')])

    def test_fresh_session_untouched(self):
        now = datetime.now().astimezone()
        sid = self.make_session(1, now - timedelta(minutes=10))
        self.assertEqual(study.reconcile_sessions(now), [])
        self.assertEqual(study.get_session(sid)['status'], 'active')

    def test_vision_stale_pauses_then_fresh_frame_resumes(self):
        from app import vision as vision_module
        now = datetime.now().astimezone()
        sid = self.make_session(1, now - timedelta(minutes=30))
        m = {'head_roll': 2.0, 'shoulder_roll': 1.0, 'shoulder_width': 0.3, 'head_drop': 0.25}
        # 守护曾出帧，但最后一帧在 15 分钟前（页面被关）
        vision_module.log_frame(sid, 1, True, m, [], [], None)
        with connection() as c:
            c.execute("UPDATE study_vision_events SET created_at=?", ((now - timedelta(minutes=15)).isoformat(),))
        actions = study.reconcile_sessions(now)
        self.assertEqual([(a[0], a[2]) for a in actions], [(sid, 'auto_pause')])
        self.assertEqual(study.get_session(sid)['status'], 'paused')
        # 帧恢复流动（刚出一帧）→ 自动续跑
        vision_module.log_frame(sid, 1, True, m, [], [], None)
        actions = study.reconcile_sessions(datetime.now().astimezone())
        self.assertEqual([(a[0], a[2]) for a in actions], [(sid, 'auto_resume')])
        self.assertEqual(study.get_session(sid)['status'], 'active')

    def test_manual_pause_not_resumed_by_frames(self):
        from app import vision as vision_module
        now = datetime.now().astimezone()
        sid = self.make_session(1, now - timedelta(minutes=10))
        study.pause_session(sid)  # 手动休息（无 auto_pause 标记）
        m = {'head_roll': 2.0, 'shoulder_roll': 1.0, 'shoulder_width': 0.3, 'head_drop': 0.25}
        vision_module.log_frame(sid, 1, True, m, [], [], None)
        actions = study.reconcile_sessions(now)
        self.assertEqual(actions, [])
        self.assertEqual(study.get_session(sid)['status'], 'paused')


class ReviewFixTests(StudyCase):
    """2026-11-13 产品逻辑 review 修复的回归测试。"""

    def _add_person(self, pid, name):
        with connection() as c:
            c.execute("INSERT OR IGNORE INTO people(id,name,created_at) VALUES(?,?,?)",
                      (pid, name, '2026-01-01T00:00:00+08:00'))

    def _metrics(self):
        return {'head_roll': 2.0, 'shoulder_roll': 1.0, 'shoulder_width': 0.3, 'head_drop': 0.25}

    def test_auto_paused_session_gets_no_resume_reminder(self):
        # P0 回归：视觉断连暂停的会话，绝不能被 resume_time 提醒唤醒（否则每5分钟死循环播报）
        from app import vision as vision_module
        now = datetime.now().astimezone()
        sid = self.make_session(1, now - timedelta(minutes=50))
        vision_module.log_frame(sid, 1, True, self._metrics(), [], [], None)
        with connection() as c:
            c.execute("UPDATE study_vision_events SET created_at=?", ((now - timedelta(minutes=20)).isoformat(),))
        actions = study.reconcile_sessions(now)
        self.assertEqual([a[2] for a in actions], ['auto_pause'])
        # 休息早已"超时"，但 auto_pause 会话不参与定时提醒
        self.assertEqual(study.sessions_needing_reminder(40, 5, now=now + timedelta(minutes=10)), [])
        # 帧恢复后仍能由 reconcile 续跑
        vision_module.log_frame(sid, 1, True, self._metrics(), [], [], None)
        actions = study.reconcile_sessions(datetime.now().astimezone())
        self.assertEqual([a[2] for a in actions], ['auto_resume'])

    def test_manual_guardian_off_not_auto_paused(self):
        from app import vision as vision_module
        now = datetime.now().astimezone()
        sid = self.make_session(1, now - timedelta(minutes=30))
        vision_module.log_frame(sid, 1, True, self._metrics(), [], [], None)
        with connection() as c:
            c.execute("UPDATE study_vision_events SET created_at=?", ((now - timedelta(minutes=15)).isoformat(),))
        study.mark_vision_toggle(1, False)  # 孩子手动关掉守护 ≠ 离座
        self.assertEqual(study.reconcile_sessions(now), [])
        self.assertEqual(study.get_session(sid)['status'], 'active')

    def test_rateable_window(self):
        sess = study.start_session(1)
        study.end_session(sess['id'])
        self.assertIsNotNone(study.get_rateable_session(1))  # 刚结束可评分
        with connection() as c:
            c.execute("UPDATE study_sessions SET ended_at=?",
                      ((datetime.now().astimezone() - timedelta(hours=2)).isoformat(),))
        self.assertIsNone(study.get_rateable_session(1))    # 2小时前结束的不行

    def test_spoken_includes_person_name(self):
        self._add_person(1, '袁玥')
        now = datetime.now().astimezone()
        self.make_session(1, now - timedelta(minutes=45))
        needs = study.sessions_needing_reminder(40, 5, now=now)
        self.assertIn('袁玥', needs[0]['spoken'])

    def test_delete_person_ends_study_session(self):
        from app import identity as identity_mod
        self._add_person(7, '测试娃')
        study.start_session(7)
        identity_mod.delete_person(7)
        self.assertIsNone(study.get_current_session(7))
        with connection() as c:
            status = c.execute("SELECT status FROM study_sessions WHERE person_id=7").fetchone()[0]
            sid = c.execute("SELECT id FROM study_sessions WHERE person_id=7").fetchone()[0]
            reason = c.execute("SELECT event_data FROM study_events WHERE session_id=? AND event_type='end' ORDER BY id DESC LIMIT 1", (sid,)).fetchone()[0]
        self.assertEqual(status, 'completed')
        self.assertIn('成员删除', reason)
