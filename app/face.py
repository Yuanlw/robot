"""Local face enrollment and recognition with a dlib or OpenCV backend."""

import base64
import json
import tempfile
from pathlib import Path

from .config import settings
from .db import connection, now_iso


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

    def _save(self, name: str, payload: bytes) -> dict:
        with connection() as conn:
            cur = conn.execute(
                "INSERT INTO people(name,face_encoding,created_at) VALUES(?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET face_encoding=excluded.face_encoding, created_at=excluded.created_at",
                (name, payload, now_iso()),
            )
            person_id = cur.lastrowid or conn.execute("SELECT id FROM people WHERE name=?", (name,)).fetchone()[0]
        return {"id": person_id, "name": name}

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

    def enroll(self, name: str, image_path: str) -> dict:
        dlib = self._dlib()
        if dlib:
            image = dlib.load_image_file(image_path)
            encodings = dlib.face_encodings(image)
            if len(encodings) != 1:
                raise ValueError("注册照片必须恰好包含一个人脸")
            return self._save(name, json.dumps({"backend": "dlib", "values": encodings[0].tolist()}).encode())

        cv2, np, crops = self._opencv_faces(image_path)
        if len(crops) != 1:
            raise ValueError("注册照片必须恰好包含一个清晰正脸")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train([crops[0]], np.array([0], dtype=np.int32))
        with tempfile.NamedTemporaryFile(suffix=".yml") as model_file:
            recognizer.write(model_file.name)
            model = base64.b64encode(Path(model_file.name).read_bytes()).decode()
        return self._save(name, json.dumps({"backend": "lbph", "model": model}).encode())

    def recognize(self, image_path: str) -> dict:
        dlib = self._dlib()
        if dlib:
            image = dlib.load_image_file(image_path)
            locations = dlib.face_locations(image)
            encodings = dlib.face_encodings(image, locations)
            with connection() as conn:
                people = conn.execute("SELECT id,name,face_encoding FROM people WHERE face_encoding IS NOT NULL").fetchall()
            known = []
            for row in people:
                data = json.loads(row["face_encoding"])
                if data.get("backend") == "dlib":
                    known.append((row, data["values"]))
            results = []
            for encoding in encodings:
                best_name, best_distance = "未知", 1.0
                for row, values in known:
                    distance = dlib.face_distance([values], encoding)[0]
                    if distance < best_distance:
                        best_name, best_distance = row["name"], float(distance)
                if best_distance > settings.face_threshold:
                    best_name = "未知"
                results.append({"name": best_name, "distance": round(best_distance, 4)})
            return {"faces": results, "count": len(results), "backend": "dlib"}

        cv2, _, crops = self._opencv_faces(image_path)
        with connection() as conn:
            people = conn.execute("SELECT id,name,face_encoding FROM people WHERE face_encoding IS NOT NULL").fetchall()
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
            best_name, best_confidence = "未知", 999.0
            for row, model in models:
                _, confidence = model.predict(crop)
                if confidence < best_confidence:
                    best_name, best_confidence = row["name"], float(confidence)
            if best_confidence > settings.lbph_threshold:
                best_name = "未知"
            results.append({"name": best_name, "confidence": round(best_confidence, 2)})
        return {"faces": results, "count": len(results), "backend": "opencv-lbph"}
