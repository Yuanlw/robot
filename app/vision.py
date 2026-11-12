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

HEAD_ROLL_LIMIT = 10.0       # 头部侧倾角度上限
SHOULDER_ROLL_LIMIT = 8.0    # 肩膀水平差上限
NEAR_RATIO = 1.12            # 肩宽相对基线超过此比例 = 离屏过近
HEAD_DROP_RATIO = 0.62       # 鼻-肩垂直距离低于基线此比例 = 头过低/趴桌
MIN_STREAK = 3               # 连续 N 帧同一问题才提醒（防误报）
WARNING_COOLDOWN_MIN = 10    # 同一问题提醒冷却
ABSENT_PAUSE_FRAMES = 3      # 连续 N 帧无人 → 暂停计时
PRESENT_RESUME_FRAMES = 2    # 暂停后连续 N 帧有人 → 恢复计时


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


def check_issues(metrics: dict, baseline: dict) -> list[str]:
    """对照基线输出问题标签列表。无基线时只做绝对角度判断。"""
    issues = []
    if metrics['head_roll'] > HEAD_ROLL_LIMIT:
        issues.append('头歪')
    if metrics['shoulder_roll'] > SHOULDER_ROLL_LIMIT:
        issues.append('肩不平')
    if baseline:
        if metrics['shoulder_width'] > baseline['shoulder_width'] * NEAR_RATIO:
            issues.append('离屏过近')
        if metrics['head_drop'] < baseline['head_drop'] * HEAD_DROP_RATIO:
            issues.append('头过低')
    return issues


WARNING_TEXT = {
    '头歪': '头有点歪了，把脖子立直吧。',
    '肩不平': '肩膀一高一低，坐端正一下。',
    '离屏过近': '离屏幕太近了，往后退一点，保护眼睛。',
    '头过低': '头低得有点多，坐直一点会更好看也更舒服。',
}


class WarningTracker:
    """连续帧判定 + 冷却，纯内存状态，可单测。同时处理到/离座迁移。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict[int, dict] = {}

    def record(self, person_id: int, issues: list[str], present: bool,
               now: datetime = None) -> dict:
        """返回 {'warnings': [标签], 'left': bool, 'returned': bool}。"""
        now = now or datetime.now().astimezone()
        with self._lock:
            st = self._state.setdefault(person_id, {})
            warnings = []
            for issue, text in WARNING_TEXT.items():
                bad = issue in issues
                entry = st.setdefault(issue, {'streak': 0, 'last_warn': None})
                if bad:
                    entry['streak'] += 1
                    cooled = entry['last_warn'] is None or (now - entry['last_warn']).total_seconds() >= WARNING_COOLDOWN_MIN * 60
                    if entry['streak'] >= MIN_STREAK and cooled:
                        warnings.append(issue)
                        entry['last_warn'] = now
                        entry['streak'] = 0
                else:
                    entry['streak'] = 0
            # 到/离座
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
            return {'warnings': warnings, 'left': left, 'returned': returned}

    def reset(self, person_id: int):
        with self._lock:
            self._state.pop(person_id, None)


tracker = WarningTracker()


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

    def analyze_bytes(self, data: bytes) -> tuple[dict | None, bool]:
        """bytes(JPEG/PNG) → (metrics, present)。帧不落盘；异常抛出由调用方处理。"""
        import cv2
        import numpy as np
        arr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError('无法解码图片，请重新拍摄')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._model().process(rgb)
        if not result.pose_landmarks:
            return None, False
        lm = [(p.x, p.y, p.z) for p in result.pose_landmarks.landmark]
        return compute_metrics(lm), True
