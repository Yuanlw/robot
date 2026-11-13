"""坐姿守护（第二期 2A）：MediaPipe Pose 本地姿态分析。

隐私设计（红线）：
- 画面帧只在本进程内存中解码分析，不落盘、不上传；
- 数据库只保存数值特征（角度/比例）、问题标签和提醒时间，不含任何图像信息；
- 默认关闭，孩子可在页面随时关掉。

结构：compute_metrics / check_issues / WarningTracker 是纯函数/纯状态，
不依赖 mediapipe，可单测；VisionService 负责模型加载与解码，mediapipe
缺失时 available()=False，功能整体优雅降级。
"""
import json
import math
import threading
from datetime import datetime, timedelta
from .db import connection, now_iso

# MediaPipe Pose 33 点索引
NOSE, LEFT_EAR, RIGHT_EAR = 0, 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

HEAD_ROLL_LIMIT = 10.0       # 头部侧倾角度上限（无基线时的绝对值）
SHOULDER_ROLL_LIMIT = 8.0    # 肩膀水平差上限（无基线时的绝对值）
POSTURE_MARGIN = 8.0         # 相对基线的容差：角度超过"个人校准值+此裕度"才算问题
SEVERE_MARGIN = 5.0          # 分级：超过相对限再 +5° 才算重度（只有重度有资格开口）
NEAR_RATIO = 1.12            # 肩宽相对基线超过此比例 = 离屏过近（轻度）
NEAR_SEVERE_RATIO = 1.30     # 离屏过近的重度线
HEAD_DROP_RATIO = 0.62       # 鼻-肩垂直距离低于基线此比例 = 头过低（轻度）
HEAD_DROP_SEVERE_RATIO = 0.48  # 头过低的重度线
MIN_STREAK = 3               # 连续 N 帧同一重度问题才提醒（防瞬时动作/抖动误报）
WARNING_COOLDOWN_MIN = 5     # 同一问题提醒冷却（验证期从 10 分钟放宽到 5，便于采样）
MAX_SPEAK_PER_ISSUE = 2      # 同一问题单场会话最多开口次数，超过只记录不唠叨
SUSPECT_POOR_RATIO = 0.6     # 高质量帧 poor 占比超过此值 → 怀疑基线/机位而非孩子
SUSPECT_MIN_FRAMES = 20      # 占比判定所需最少高质量帧样本
VISIBILITY_MIN = 0.6         # 关键点平均可见度低于此值的帧不参与任何判定
ABSENT_PAUSE_FRAMES = 3      # 连续 N 帧无人 → 暂停计时
PRESENT_RESUME_FRAMES = 2    # 暂停后连续 N 帧有人 → 恢复计时
# 校准质量门槛：基线帧本身超过这些角度说明机位太歪或没坐正，拒绝入库
CALIBRATE_HEAD_ROLL_MAX = 15.0
CALIBRATE_SHOULDER_ROLL_MAX = 22.0


def compute_metrics(landmarks) -> dict | None:
    """landmarks: [(x,y,z),...] 归一化坐标。返回姿态数值特征（无图像信息）。"""
    try:
        nose = landmarks[NOSE]
        l_ear, r_ear = landmarks[LEFT_EAR], landmarks[RIGHT_EAR]
        l_sh, r_sh = landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
    except (IndexError, TypeError):
        return None
    def deg(a, b):
        return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    head_roll = abs(deg(l_ear, r_ear))
    if head_roll > 90:
        head_roll = 180 - head_roll
    shoulder_roll = abs(deg(l_sh, r_sh))
    if shoulder_roll > 90:
        shoulder_roll = 180 - shoulder_roll
    width = abs(r_sh[0] - l_sh[0])
    mid_shoulder_y = (l_sh[1] + r_sh[1]) / 2
    head_drop = mid_shoulder_y - nose[1]  # y 向下为正：头越低（趴桌）值越小
    if width <= 0.01 or head_drop <= 0:
        return None
    return {"head_roll": round(head_roll, 1), "shoulder_roll": round(shoulder_roll, 1),
            "shoulder_width": round(width, 4), "head_drop": round(head_drop, 4)}


def issue_severities(metrics: dict, baseline: dict) -> dict:
    """判定每个问题的严重度：{问题: 'mild'|'severe'}。

    决策层第③级：轻度只记录进数据（供报告分析），重度才有资格进入
    持续性判定和开口流程。2026-11-13 review：把"测量"和"判断"分开。
    """
    out = {}
    head_limit = HEAD_ROLL_LIMIT
    shoulder_limit = SHOULDER_ROLL_LIMIT
    if baseline:
        head_limit = max(HEAD_ROLL_LIMIT, baseline.get('head_roll', 0) + POSTURE_MARGIN)
        shoulder_limit = max(SHOULDER_ROLL_LIMIT, baseline.get('shoulder_roll', 0) + POSTURE_MARGIN)
    if metrics['head_roll'] > head_limit:
        out['头歪'] = 'severe' if metrics['head_roll'] > head_limit + SEVERE_MARGIN else 'mild'
    if metrics['shoulder_roll'] > shoulder_limit:
        out['肩不平'] = 'severe' if metrics['shoulder_roll'] > shoulder_limit + SEVERE_MARGIN else 'mild'
    if baseline:
        if metrics['shoulder_width'] > baseline['shoulder_width'] * NEAR_RATIO:
            out['离屏过近'] = 'severe' if metrics['shoulder_width'] > baseline['shoulder_width'] * NEAR_SEVERE_RATIO else 'mild'
        if metrics['head_drop'] < baseline['head_drop'] * HEAD_DROP_RATIO:
            out['头过低'] = 'severe' if metrics['head_drop'] < baseline['head_drop'] * HEAD_DROP_SEVERE_RATIO else 'mild'
    return out


def check_issues(metrics: dict, baseline: dict) -> list[str]:
    """问题标签列表（含轻度和重度）。兼容旧调用方；开口决策用 issue_severities。"""
    return list(issue_severities(metrics, baseline).keys())


def validate_calibration(metrics: dict) -> str | None:
    """校准质量门槛：基线帧本身角度过大说明机位歪/没坐正，返回错误文案；合格返回 None。"""
    if metrics['head_roll'] > CALIBRATE_HEAD_ROLL_MAX:
        return f"校准帧头侧倾 {metrics['head_roll']:.0f}°，超出 {CALIBRATE_HEAD_ROLL_MAX:.0f}°——请坐直、摆正摄像头后重拍"
    if metrics['shoulder_roll'] > CALIBRATE_SHOULDER_ROLL_MAX:
        return f"校准帧肩线差 {metrics['shoulder_roll']:.0f}°，超出 {CALIBRATE_SHOULDER_ROLL_MAX:.0f}°——请把摄像头摆平（不要仰拍/侧拍）后重拍"
    return None


WARNING_TEXT = {
    '头歪': '头有点歪了，把脖子立直吧。',
    '肩不平': '肩膀一高一低，坐端正一下。',
    '离屏过近': '离屏幕太近了，往后退一点，保护眼睛。',
    '头过低': '头低得有点多，坐直一点会更好看也更舒服。',
}


class PostureJudge:
    """姿态判断决策层（六级漏斗）：检测层只产数字，开口之前必须过这里。

    ① 质量门：低可见度帧不参与判定、不清连击（防遮挡/侧身闪烁）
    ② 相对限：issue_severities 已完成（个人基线+裕度）
    ③ 分级：只有 severe 进入连击与开口；mild 只入库供报告
    ④ 持续性：severe 连续 MIN_STREAK 帧才成立（滤瞬时动作与抖动）
    ⑤ 预算：同一问题单场最多开口 MAX_SPEAK_PER_ISSUE 次，之后只记录（唠叨无益）
    ⑥ 系统怀疑：高质量帧≥SUSPECT_MIN_FRAMES 且 poor 占比>SUSPECT_POOR_RATIO →
       判定"基线/机位错了，不是孩子错了"→ 停全部姿态播报、提示重校准；
       重新校准（reset）后清零重来。2026-11-13 昨晚 64% poor 率就是这个场景。

    到/离座迁移与旧版一致（连续 3 帧无人暂停、回来 2 帧恢复）。
    纯内存状态、确定性规则，可单测。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict[int, dict] = {}

    def _session_state(self, person_id: int, session_id) -> dict:
        st = self._state.setdefault(person_id, {})
        if st.get('session_id') != session_id:
            absent = st.get('_absent')  # 到/离座状态跨会话保留
            st.clear()
            st.update({'session_id': session_id, 'warned': {},
                       'quality_frames': 0, 'poor_frames': 0, 'suspect': False})
            if absent:
                st['_absent'] = absent
        return st

    def record(self, person_id: int, session_id, severities: dict, present: bool,
               quality_ok: bool, now: datetime = None) -> dict:
        """返回 {'warnings':[标签], 'left', 'returned', 'suppressed', 'suspect_new', 'stats'}。"""
        now = now or datetime.now().astimezone()
        with self._lock:
            st = self._session_state(person_id, session_id)
            warnings, suppressed, suspect_new, nag_limited = [], None, False, False

            # 到/离座迁移
            absent = st.setdefault('_absent', {'streak': 0, 'paused': False, 'present_streak': 0})
            left = returned = False
            if not present:
                absent['present_streak'] = 0
                absent['streak'] += 1
                if absent['streak'] >= ABSENT_PAUSE_FRAMES and not absent['paused']:
                    absent['paused'] = True
                    left = True
            else:
                absent['streak'] = 0
                if absent['paused']:
                    absent['present_streak'] += 1
                    if absent['present_streak'] >= PRESENT_RESUME_FRAMES:
                        absent['paused'] = False
                        absent['present_streak'] = 0
                        returned = True
                else:
                    absent['present_streak'] = 0

            # ①质量门：低质量帧跳过全部姿态判定（不清连击，避免遮挡闪烁重置）
            if present and quality_ok:
                st['quality_frames'] += 1
                if severities:
                    st['poor_frames'] += 1

                # ⑥系统怀疑
                if not st['suspect'] and st['quality_frames'] >= SUSPECT_MIN_FRAMES \
                        and st['poor_frames'] / st['quality_frames'] > SUSPECT_POOR_RATIO:
                    st['suspect'] = True
                    suspect_new = True
                if st['suspect']:
                    suppressed = 'suspect_calibration'
                else:
                    # ③④⑤：重度问题连击 + 冷却 + 单场预算
                    for issue in WARNING_TEXT:
                        entry = st['warned'].setdefault(issue, {'streak': 0, 'last_warn': None, 'spoken': 0})
                        if severities.get(issue) == 'severe':
                            entry['streak'] += 1
                            cooled = (entry['last_warn'] is None or
                                      (now - entry['last_warn']).total_seconds() >= WARNING_COOLDOWN_MIN * 60)
                            if entry['streak'] >= MIN_STREAK and cooled:
                                entry['streak'] = 0
                                if entry['spoken'] < MAX_SPEAK_PER_ISSUE:
                                    warnings.append(issue)
                                    entry['last_warn'] = now
                                    entry['spoken'] += 1
                                else:
                                    nag_limited = True
                        else:
                            entry['streak'] = 0
                    if nag_limited and not warnings:
                        suppressed = 'nag_limit'

            return {'warnings': warnings, 'left': left, 'returned': returned,
                    'suppressed': suppressed, 'suspect_new': suspect_new,
                    'stats': {'quality_frames': st['quality_frames'],
                              'poor_frames': st['poor_frames'], 'suspect': st['suspect']}}

    def reset(self, person_id: int):
        """重新校准/会话收口时清零（含 suspect 抑制）。"""
        with self._lock:
            self._state.pop(person_id, None)


judge = PostureJudge()


def init_vision_db():
    with connection() as c:
        c.execute('''CREATE TABLE IF NOT EXISTS study_vision_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            person_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            present INTEGER NOT NULL,
            head_roll REAL, shoulder_roll REAL,
            shoulder_width REAL, head_drop REAL,
            posture_quality TEXT,
            issues TEXT,
            warning_sent TEXT,
            user_responded INTEGER DEFAULT 0
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS study_vision_baselines (
            person_id INTEGER PRIMARY KEY,
            features TEXT NOT NULL,
            created_at TEXT NOT NULL
        )''')


def get_baseline(person_id: int) -> dict | None:
    with connection() as c:
        row = c.execute("SELECT features FROM study_vision_baselines WHERE person_id=?", (person_id,)).fetchone()
    return json.loads(row[0]) if row else None


def save_baseline(person_id: int, metrics: dict):
    with connection() as c:
        c.execute("INSERT OR REPLACE INTO study_vision_baselines(person_id,features,created_at) VALUES(?,?,?)",
                  (person_id, json.dumps(metrics), now_iso()))


def log_frame(session_id, person_id, present, metrics, issues, warning_labels,
              warning_text) -> int:
    """仅数值与标签入库；图像立即丢弃。返回事件 id（未入库返回 0）。"""
    with connection() as c:
        cur = c.execute(
            "INSERT INTO study_vision_events(session_id,person_id,created_at,present,head_roll,shoulder_roll,"
            "shoulder_width,head_drop,posture_quality,issues,warning_sent) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (session_id, person_id, now_iso(), 1 if present else 0,
             metrics.get('head_roll') if metrics else None,
             metrics.get('shoulder_roll') if metrics else None,
             metrics.get('shoulder_width') if metrics else None,
             metrics.get('head_drop') if metrics else None,
             'absent' if not present else ('poor' if issues else 'good'),
             ','.join(issues) if issues else None,
             warning_text))
        return cur.lastrowid


def mark_responded(person_id: int, cleared_issues: list[str], now: datetime = None):
    """问题在提醒后 3 分钟内消失 → 记为已响应（用于纠正率统计）。"""
    now = now or datetime.now().astimezone()
    window = (now - timedelta(minutes=3)).isoformat()
    with connection() as c:
        for issue in cleared_issues:
            c.execute(
                "UPDATE study_vision_events SET user_responded=1 WHERE person_id=? AND user_responded=0 "
                "AND warning_sent IS NOT NULL AND issues LIKE ? AND created_at>=?",
                (person_id, f"%{issue}%", window))


class VisionService:
    def __init__(self):
        self._pose = None
        self._error = None

    def available(self) -> bool:
        try:
            import mediapipe  # noqa: F401
            return True
        except ImportError:
            return False

    def details(self) -> dict:
        return {"available": self.available(), "error": self._error}

    def _model(self):
        if self._pose is None:
            import mediapipe as mp
            self._pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=0,
                                                min_detection_confidence=0.5)
        return self._pose

    def analyze_bytes(self, data: bytes) -> tuple[dict | None, bool, bool]:
        """bytes(JPEG/PNG) → (metrics, present, quality_ok)。帧不落盘；异常抛出由调用方处理。

        quality_ok：5 个关键点（鼻/双耳/双肩）平均可见度 ≥ VISIBILITY_MIN。
        遮挡、侧身、画面残缺时 MediaPipe 仍会外推出地标，角度不可信——决策层第①级。
        """
        import cv2
        import numpy as np
        arr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError('无法解码图片，请重新拍摄')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._model().process(rgb)
        if not result.pose_landmarks:
            return None, False, False
        lms = result.pose_landmarks.landmark
        metrics = compute_metrics([(p.x, p.y, p.z) for p in lms])
        key_vis = [lms[i].visibility for i in (NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER)]
        quality_ok = metrics is not None and (sum(key_vis) / len(key_vis)) >= VISIBILITY_MIN
        return metrics, True, quality_ok
