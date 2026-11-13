"""Local face enrollment and recognition with a dlib or OpenCV backend.

2026-11-13 五项升级（对标 Face ID 的软件可追部分）：
A 多样本录入：dlib 特征从单样本升级为样本池（前 CORE_SAMPLES 个为录入核心，
  上限 face_max_samples），识别取最小距离——角度鲁棒性主要来源；
B 多帧投票：recognize_burst 对 1~5 帧投票，3 帧至少 2 帧一致才认定，
  消掉单帧模糊/眨眼/半侧脸的偶发失败；
C 增量学习：高置信识别（距离 ≤ face_learn_threshold）且特征足够新时滚动追加
  （FIFO 保留录入核心），适应孩子成长/剪发/眼镜——特征仍只存本机；
D 录入质量门槛：清晰度(Laplacian)/脸部占比/正脸度(68点偏航不对称)，
  首次录入要求近正脸，追加角度样本放宽到轻侧脸；
E 距离落库：每次识别写 face_recog_log（含未识别帧的最佳距离），
  阈值调优从拍脑袋变成看分布。
LBPH 保留为存量成员兼容（单模板，不参与 A/C；重录后自动升级 dlib）。
"""

import base64
import json
import math
import tempfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from .config import settings
from .db import connection, now_iso

CORE_SAMPLES = 2          # 样本池中永久保留的录入核心样本数
LEARN_NOVELTY = 0.08      # 新特征与既有样本最小距离低于此值视为重复，不入库
BLUR_MIN = 70.0           # 人脸区域 Laplacian 方差下限
FACE_MIN_ABS = 96         # 人脸框最小像素宽
FACE_MIN_RATIO = 0.15     # 人脸框相对画面宽度最小占比
FRONTAL_ASYM_MAX = 0.30   # 首次录入的偏航不对称上限（近正脸）
APPEND_ASYM_MAX = 0.55    # 追加角度样本的上限（允许轻侧脸）


def _parse_samples(payload) -> list:
    """解析存储的特征：新格式 samples 列表 / 旧格式单 values / lbph → []。"""
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict) or data.get("backend") != "dlib":
        return []
    if isinstance(data.get("samples"), list):
        return [s for s in data["samples"] if isinstance(s, list)]
    if isinstance(data.get("values"), list):
        return [data["values"]]  # 旧单样本格式
    return []


def _capped(samples: list, cap: int = None) -> list:
    """样本池滚动上限：保留前 CORE_SAMPLES 个录入核心，超出时淘汰最旧的学习样本。"""
    cap = cap or settings.face_max_samples
    if len(samples) <= cap:
        return samples
    learned = samples[CORE_SAMPLES:]
    return samples[:CORE_SAMPLES] + learned[-(cap - CORE_SAMPLES):]


def _vote(frame_results: list, ) -> tuple:
    """帧级投票：frame_results=[(person_id|None, distance)]。

    返回 (winner|None, matched, n)。n>=2 时要求命中帧 >= ceil((n+1)/2)
    （3 帧至少 2 帧、2 帧必须一致），否则宁可判未知。
    """
    n = len(frame_results)
    matched = [(pid, d) for pid, d in frame_results if pid is not None]
    if n == 0 or not matched:
        return None, 0, n
    counts = Counter(pid for pid, _ in matched)
    winner, cnt = counts.most_common(1)[0]
    need = math.ceil((n + 1) / 2)
    if cnt < need:
        return None, cnt, n
    return winner, cnt, n


def _yaw_asym(landmarks: dict) -> float:
    """68 点地标的偏航不对称度：|d(鼻,左眼)-d(鼻,右眼)|/眼距。0=正脸，越大越侧。"""
    def center(pts):
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    try:
        left, right = center(landmarks['left_eye']), center(landmarks['right_eye'])
        nose_pts = landmarks['nose_tip']
        nose = nose_pts[len(nose_pts) // 2] if isinstance(nose_pts, list) else nose_pts
    except (KeyError, IndexError, TypeError):
        return 1.0
    eye_d = math.dist(left, right)
    if eye_d <= 0:
        return 1.0
    return abs(math.dist(nose, left) - math.dist(nose, right)) / eye_d


class FaceService:
    def _dlib(self):
        try:
            import face_recognition
            return face_recognition
        except ImportError:
            return None

    def _opencv(self):
        try:
            import cv2
            import numpy as np
            return (cv2, np) if hasattr(cv2, "face") else None
        except ImportError:
            return None

    # ---------- 存储 ----------

    def init_face_db(self):
        with connection() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS face_recog_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                person_id INTEGER,
                distance REAL,
                backend TEXT,
                frames INTEGER DEFAULT 1,
                matched_frames INTEGER DEFAULT 0,
                enrolled_samples INTEGER DEFAULT 0,
                purpose TEXT DEFAULT 'recognize'
            )''')

    def _save(self, name: str, payload: bytes) -> dict:
        with connection() as conn:
            conn.execute(
                "INSERT INTO people(name,face_encoding,created_at) VALUES(?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET face_encoding=excluded.face_encoding, created_at=excluded.created_at",
                (name, payload, now_iso()),
            )
            person_id = conn.execute("SELECT id FROM people WHERE name=?", (name,)).fetchone()[0]
        return {"id": person_id, "name": name}

    def _people(self):
        with connection() as conn:
            return conn.execute("SELECT id,name,face_encoding FROM people WHERE face_encoding IS NOT NULL").fetchall()

    def sample_counts(self) -> dict:
        out = {}
        for row in self._people():
            samples = _parse_samples(row["face_encoding"])
            out[row["id"]] = len(samples) if samples else 1  # lbph 单模板记 1
        return out

    def _log(self, person_id, distance, backend, frames=1, matched=0, samples=0, purpose='recognize'):
        try:
            with connection() as conn:
                conn.execute(
                    "INSERT INTO face_recog_log(created_at,person_id,distance,backend,frames,matched_frames,enrolled_samples,purpose) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (now_iso(), person_id, distance, backend, frames, matched, samples, purpose))
        except Exception:
            pass  # 记录失败绝不阻断识别

    # ---------- 录入（A 多样本 + D 质量门槛） ----------

    def _quality_gate(self, image_path: str, append: bool, fr) -> tuple:
        """返回 (ok, reason)。cv2 不可用时放行（门槛不应把录入堵死）。"""
        backend = self._opencv()
        if not backend:
            return True, ''
        cv2, _ = backend
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            return False, '无法读取照片'
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        locations = fr.face_locations(rgb)
        if len(locations) != 1:
            return False, '照片必须恰好包含一个人脸'
        top, right, bottom, left = locations[0]
        face_w = right - left
        if face_w < max(FACE_MIN_ABS, w * FACE_MIN_RATIO):
            return False, f'脸太小（约{face_w}px），请靠近摄像头一些'
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray[top:bottom, left:right], cv2.CV_64F).var())
        if blur < BLUR_MIN:
            return False, f'照片模糊（清晰度{blur:.0f}），请拿稳再拍'
        landmarks = fr.face_landmarks(rgb)
        asym = _yaw_asym(landmarks[0]) if landmarks else 1.0
        limit = APPEND_ASYM_MAX if append else FRONTAL_ASYM_MAX
        if asym > limit:
            return False, ('请正对摄像头，不要侧脸' if not append
                           else '只需轻微转头（左右15°左右），不要大侧脸')
        return True, ''

    def enroll(self, name: str, image_path: str, append: bool = False) -> dict:
        dlib = self._dlib()
        if dlib:
            ok, reason = self._quality_gate(image_path, append, dlib)
            if not ok:
                raise ValueError(reason)
            image = dlib.load_image_file(image_path)
            encodings = dlib.face_encodings(image)
            if len(encodings) != 1:
                raise ValueError("注册照片必须恰好包含一个人脸")
            new = encodings[0]
            existing = []
            if append:
                with connection() as conn:
                    row = conn.execute("SELECT face_encoding FROM people WHERE name=?", (name,)).fetchone()
                existing = _parse_samples(row["face_encoding"]) if row else []
                if not existing:
                    append = False  # 对象不存在或旧 LBPH 成员：按首次录入（升级 dlib）
            if append:
                dists = dlib.face_distance(existing, new)
                if len(dists) and float(min(dists)) < LEARN_NOVELTY:
                    raise ValueError("这个角度和已有样本太像了，换个方向再拍")
                samples = _capped(existing + [new.tolist()])
            else:
                samples = [new.tolist()]
            result = self._save(name, json.dumps({"backend": "dlib", "samples": samples}).encode())
            result['samples'] = len(samples)
            self._log(result['id'], None, 'dlib', purpose='enroll' + ('_append' if append else ''),
                      samples=len(samples))
            return result

        if append:
            raise ValueError("当前后端不支持追加样本，请先重新录入升级")
        cv2, np, crops = self._opencv_faces(image_path)
        if len(crops) != 1:
            raise ValueError("注册照片必须恰好包含一个清晰正脸")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train([crops[0]], np.array([0], dtype=np.int32))
        with tempfile.NamedTemporaryFile(suffix=".yml") as model_file:
            recognizer.write(model_file.name)
            model = base64.b64encode(Path(model_file.name).read_bytes()).decode()
        return self._save(name, json.dumps({"backend": "lbph", "model": model}).encode())

    def _opencv_faces(self, image_path: str):
        backend = self._opencv()
        if not backend:
            raise RuntimeError("人脸识别依赖不可用，请安装 opencv-contrib-python")
        cv2, np = backend
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("无法读取照片")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
        boxes = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        crops = [cv2.resize(gray[y:y + h, x:x + w], (200, 200)) for x, y, w, h in boxes]
        return cv2, np, crops

    # ---------- 识别（A 最小距离 + C 增量学习 + E 落库） ----------

    def _best_match(self, fr, people, enc):
        """返回 (person_id, name, distance, samples)；无候选时 (None,None,None,[])。"""
        best = (None, None, None, [])
        for row in people:
            samples = _parse_samples(row["face_encoding"])
            if not samples:
                continue
            dists = fr.face_distance(samples, enc)
            d = float(min(dists))
            if best[2] is None or d < best[2]:
                best = (row["id"], row["name"], d, samples)
        return best

    def _maybe_learn(self, fr, person_id, distance, samples, enc) -> bool:
        """C 增量学习：高置信 + 特征够新 → 滚动入池。"""
        if not settings.face_learn or person_id is None or distance is None:
            return False
        if distance > settings.face_learn_threshold:
            return False
        dists = fr.face_distance(samples, enc)
        if len(dists) and float(min(dists)) < LEARN_NOVELTY:
            return False  # 已有几乎相同的样本，不重复入库
        new_samples = _capped(samples + [enc.tolist()])
        with connection() as conn:
            conn.execute("UPDATE people SET face_encoding=? WHERE id=?",
                         (json.dumps({"backend": "dlib", "samples": new_samples}).encode(), person_id))
        return True

    def recognize(self, image_path: str) -> dict:
        dlib = self._dlib()
        if dlib:
            image = dlib.load_image_file(image_path)
            locations = dlib.face_locations(image)
            encodings = dlib.face_encodings(image, locations)
            people = self._people()
            results = []
            for enc in encodings:
                n_samples = 0
                pid, name, dist, samples = self._best_match(dlib, people, enc)
                if dist is None or dist > settings.face_threshold:
                    pid, name = None, "未知"
                elif self._maybe_learn(dlib, pid, dist, samples, enc):
                    n_samples = len(samples) + 1
                else:
                    n_samples = len(samples)
                results.append({"person_id": pid, "name": name,
                                "distance": round(dist, 4) if dist is not None else None})
                self._log(pid, dist, 'dlib', samples=n_samples)
            return {"faces": results, "count": len(results), "backend": "dlib"}

        cv2, _, crops = self._opencv_faces(image_path)
        people = self._people()
        models = []
        for row in people:
            data = json.loads(row["face_encoding"])
            if data.get("backend") != "lbph":
                continue
            with tempfile.NamedTemporaryFile(suffix=".yml") as model_file:
                Path(model_file.name).write_bytes(base64.b64decode(data["model"]))
                model = cv2.face.LBPHFaceRecognizer_create()
                model.read(model_file.name)
                models.append((row, model))
        results = []
        for crop in crops:
            best_name, best_confidence, best_id = "未知", 999.0, None
            for row, model in models:
                _, confidence = model.predict(crop)
                if confidence < best_confidence:
                    best_name, best_confidence, best_id = row["name"], float(confidence), row["id"]
            if best_confidence > settings.lbph_threshold:
                best_name, best_id = "未知", None
            results.append({"person_id": best_id, "name": best_name, "confidence": round(best_confidence, 2)})
            self._log(best_id, best_confidence, 'lbph')
        return {"faces": results, "count": len(results), "backend": "opencv-lbph"}

    def recognize_burst(self, image_paths: list) -> dict:
        """B 多帧投票识别：同一人连拍多帧，多数一致才认定。"""
        dlib = self._dlib()
        frame_results = []  # [(pid|None, dist)] 投票用
        frame_info = []     # [(pid, name, dist, samples, enc)] 命中帧详情
        if dlib:
            people = self._people()
            for path in image_paths:
                try:
                    image = dlib.load_image_file(str(path))
                    encodings = dlib.face_encodings(image)
                except Exception:
                    encodings = []
                if not encodings:
                    frame_results.append((None, None))
                    continue
                enc = encodings[0]
                pid, name, dist, samples = self._best_match(dlib, people, enc)
                hit = dist is not None and dist <= settings.face_threshold
                frame_results.append((pid if hit else None, dist))
                if hit:
                    frame_info.append((pid, name, dist, samples, enc))
            winner, matched, n = _vote(frame_results)
            if winner is not None:
                # 胜者的距离/姓名必须取自胜者帧中最好的那帧，而不是全局最小帧
                _, name, dist, samples, enc = min(frame_info, key=lambda f: f[2] if f[0] == winner else float('inf'))
                self._maybe_learn(dlib, winner, dist, samples, enc)
                self._log(winner, dist, 'dlib', frames=n, matched=matched, samples=len(samples))
                return {"faces": [{"person_id": winner, "name": name, "distance": round(dist, 4)}],
                        "count": 1, "backend": "dlib", "frames": n, "matched_frames": matched}
            best_dist = min((r[1] for r in frame_results if r[1] is not None), default=None)
            self._log(None, best_dist, 'dlib', frames=n, matched=matched)
            return {"faces": [], "count": 0, "backend": "dlib",
                    "frames": n, "matched_frames": matched}

        # LBPH 兜底：逐帧单张识别后投票（无增量学习）
        for path in image_paths:
            r = self.recognize(str(path))
            face = r["faces"][0] if r["faces"] else {}
            frame_results.append((face.get("person_id"), face.get("confidence")))
        winner, matched, n = _vote(frame_results)
        if winner is None:
            return {"faces": [], "count": 0, "backend": "opencv-lbph", "frames": n, "matched_frames": matched}
        name = next((r["name"] for r in self._people() if r["id"] == winner), "未知")
        return {"faces": [{"person_id": winner, "name": name}], "count": 1,
                "backend": "opencv-lbph", "frames": n, "matched_frames": matched}

    # ---------- E 统计 ----------

    def stats(self, days: int = 7) -> dict:
        since = (datetime.now().astimezone() - timedelta(days=days)).isoformat()
        try:
            with connection() as conn:
                rows = conn.execute(
                    "SELECT person_id,distance,backend,frames,matched_frames FROM face_recog_log "
                    "WHERE created_at>=? AND purpose='recognize'", (since,)).fetchall()
        except Exception:
            return {"error": "face_recog_log 不存在", "total": 0}
        matched = [r["distance"] for r in rows if r["person_id"] is not None and r["distance"] is not None]
        unknown = [r["distance"] for r in rows if r["person_id"] is None and r["distance"] is not None]

        def pct(vals):
            if not vals:
                return None
            vals = sorted(vals)
            pick = lambda q: round(vals[min(len(vals) - 1, int(q * len(vals)))], 4)
            return {"min": round(vals[0], 4), "p50": pick(0.5), "p90": pick(0.9), "max": round(vals[-1], 4)}

        return {
            "since_days": days, "total": len(rows),
            "matched": len(matched), "unknown": len(unknown),
            "match_rate": round(len(matched) / len(rows), 2) if rows else None,
            "matched_distance": pct(matched),
            "unknown_best_distance": pct(unknown),
            "face_threshold": settings.face_threshold,
            "by_backend": dict(Counter(r["backend"] for r in rows)),
        }
