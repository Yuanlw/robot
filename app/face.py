"""Face enrollment and recognition with an optional face-recognition backend."""

import base64
import json
from pathlib import Path

from .config import settings
from .db import connection, now_iso


class FaceService:
    def _backend(self):
        try:
            import face_recognition
            return face_recognition
        except ImportError as exc:
            raise RuntimeError("人脸识别未安装，请安装 face-recognition 和 OpenCV") from exc

    def enroll(self, name: str, image_path: str) -> dict:
        face_recognition = self._backend()
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) != 1:
            raise ValueError("注册照片必须恰好包含一个人脸")
        payload = json.dumps(encodings[0].tolist()).encode()
        with connection() as conn:
            cur = conn.execute(
                "INSERT INTO people(name,face_encoding,created_at) VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET face_encoding=excluded.face_encoding",
                (name, payload, now_iso()),
            )
            person_id = cur.lastrowid or conn.execute("SELECT id FROM people WHERE name=?", (name,)).fetchone()[0]
        return {"id": person_id, "name": name}

    def recognize(self, image_path: str) -> dict:
        face_recognition = self._backend()
        image = face_recognition.load_image_file(image_path)
        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, locations)
        with connection() as conn:
            people = conn.execute("SELECT id,name,face_encoding FROM people WHERE face_encoding IS NOT NULL").fetchall()
        known = [(row, face_recognition.face_encodings if False else json.loads(row["face_encoding"])) for row in people]
        results = []
        for encoding in encodings:
            best_name, best_distance = "未知", 1.0
            for row, values in known:
                distance = face_recognition.face_distance([values], encoding)[0]
                if distance < best_distance:
                    best_name, best_distance = row["name"], float(distance)
            if best_distance > settings.face_threshold:
                best_name = "未知"
            results.append({"name": best_name, "distance": round(best_distance, 4)})
        return {"faces": results, "count": len(results)}
