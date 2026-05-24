import json
import os
import shutil
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from deepface import DeepFace

from config import EMBEDDINGS_FILE, KNOWN_FACES_DIR, DEEPFACE_MODEL, DEEPFACE_DETECTOR_BACKEND


class FaceDatabase:
    def __init__(self) -> None:
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
        self.people: Dict[str, List[List[float]]] = {}
        self.load()

    @staticmethod
    def clean_name(name: str) -> str:
        return "".join(ch for ch in name.strip() if ch.isalnum() or ch in (" ", "_", "-"))[:60]

    def load(self) -> None:
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
                self.people = json.load(f)
        else:
            self.people = {}

    def save(self) -> None:
        with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.people, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize_crop(face_bgr: np.ndarray) -> np.ndarray:
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Empty face crop")
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_AREA)

    def embed(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            img = self._normalize_crop(face_bgr)
            reps = DeepFace.represent(
                img_path=img,
                model_name=DEEPFACE_MODEL,
                detector_backend=DEEPFACE_DETECTOR_BACKEND,
                enforce_detection=False,
            )
            if not reps:
                return None
            emb = np.array(reps[0]["embedding"], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm == 0:
                return None
            return emb / norm
        except Exception as exc:
            print(f"[FaceDatabase] embed error: {exc}")
            return None

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(a, b))

    def recognize_embedding(self, emb: np.ndarray, threshold: float) -> Tuple[str, float]:
        best_name = "Unknown Face"
        best_distance = 999.0
        for name, embeddings in self.people.items():
            for stored in embeddings:
                dist = self.cosine_distance(emb, np.array(stored, dtype=np.float32))
                if dist < best_distance:
                    best_distance = dist
                    best_name = name
        if best_distance <= threshold:
            return best_name, best_distance
        return "Unknown Face", best_distance

    def recognize(self, face_bgr: np.ndarray, threshold: float) -> Tuple[str, float]:
        emb = self.embed(face_bgr)
        if emb is None:
            return "Unknown Face", 999.0
        return self.recognize_embedding(emb, threshold)

    def add_person_image(self, name: str, face_bgr: np.ndarray) -> bool:
        clean_name = self.clean_name(name)
        if not clean_name:
            return False
        emb = self.embed(face_bgr)
        if emb is None:
            return False
        person_dir = os.path.join(KNOWN_FACES_DIR, clean_name)
        os.makedirs(person_dir, exist_ok=True)
        filename = os.path.join(person_dir, f"{int(time.time() * 1000)}.jpg")
        cv2.imwrite(filename, face_bgr)
        self.people.setdefault(clean_name, []).append(emb.tolist())
        self.save()
        return True

    def rename_person(self, old_name: str, new_name: str) -> Tuple[bool, str]:
        old_name = self.clean_name(old_name)
        new_name = self.clean_name(new_name)
        if not old_name or old_name not in self.people:
            return False, "Person not found."
        if not new_name:
            return False, "New name is required."
        if new_name != old_name and new_name in self.people:
            return False, "The new name already exists."
        self.people[new_name] = self.people.pop(old_name)
        old_dir = os.path.join(KNOWN_FACES_DIR, old_name)
        new_dir = os.path.join(KNOWN_FACES_DIR, new_name)
        if os.path.exists(old_dir) and old_dir != new_dir:
            os.makedirs(os.path.dirname(new_dir), exist_ok=True)
            if os.path.exists(new_dir):
                shutil.rmtree(new_dir)
            os.rename(old_dir, new_dir)
        self.save()
        return True, "Person renamed successfully."

    def delete_person(self, name: str) -> Tuple[bool, str]:
        name = self.clean_name(name)
        if not name or name not in self.people:
            return False, "Person not found."
        self.people.pop(name, None)
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
        self.save()
        return True, "Person deleted successfully."

    def clear_samples(self, name: str) -> Tuple[bool, str]:
        name = self.clean_name(name)
        if not name or name not in self.people:
            return False, "Person not found."
        self.people[name] = []
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
        os.makedirs(person_dir, exist_ok=True)
        self.save()
        return True, "Samples cleared successfully."

    def list_people(self) -> List[Dict[str, int]]:
        return [{"name": name, "samples": len(embs)} for name, embs in sorted(self.people.items())]
