import os
import queue
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
from flask import Flask, Response, jsonify, render_template, request

from config import (
    CAMERA_INDEX,
    FACE_CONFIDENCE,
    FACE_DETECTION_EVERY_N_FRAMES,
    FACE_MATCH_THRESHOLD,
    FACE_RECOGNITION_EVERY_N_FRAMES,
    FACE_RECOGNITION_MIN_SECONDS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    HAND_CONFIDENCE,
    JPEG_QUALITY,
    MAX_FACE_RECOGNITION_QUEUE,
    TARGET_FPS,
    TRACK_TTL_SECONDS,
    UNKNOWN_FACES_DIR,
)
from src.face_db import FaceDatabase
from src.hand_utils import add_custom_gesture, delete_custom_gesture, hand_payload, load_custom_gestures

app = Flask(__name__)

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_db = FaceDatabase()
state_lock = threading.Lock()
unknown_lock = threading.Lock()
unknown_faces: Dict[str, Dict] = {}
latest_state: Dict = {
    "faces": [],
    "face": "No Face",
    "face_distance": None,
    "unknown_available": False,
    "unknown_faces": [],
    "hands": [],
    "people": face_db.list_people(),
    "custom_gestures": load_custom_gestures(),
    "frame": {"width": FRAME_WIDTH, "height": FRAME_HEIGHT},
    "performance": {"mode": "async", "recognition_queue": 0},
}


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


class CameraProcessor:
    """Realtime camera pipeline.

    The video stream must never wait for DeepFace. MediaPipe detection runs in the
    capture thread, while face identity recognition runs in a separate worker.
    Each tracked face owns its own label/distance, so multiple faces are not
    copied from the same previous recognition result.
    """

    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera. Change CAMERA_INDEX or check camera permissions.")

        self.face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=FACE_CONFIDENCE)
        self.hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=HAND_CONFIDENCE,
            min_tracking_confidence=HAND_CONFIDENCE,
        )

        self.frame_no = 0
        self.frame_lock = threading.Lock()
        self.latest_jpeg: Optional[bytes] = None
        self.running = True

        self.tracks: Dict[int, Dict] = {}
        self.next_track_id = 1
        self.last_detected_boxes: List[Tuple[int, int, int, int]] = []

        self.recognition_queue: "queue.Queue[Dict]" = queue.Queue(maxsize=MAX_FACE_RECOGNITION_QUEUE)
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self.capture_thread.start()
        self.recognition_thread.start()

    @staticmethod
    def _safe_box(relative_box, width: int, height: int) -> Tuple[int, int, int, int]:
        x = int(relative_box.xmin * width)
        y = int(relative_box.ymin * height)
        w = int(relative_box.width * width)
        h = int(relative_box.height * height)
        pad_x = int(w * 0.18)
        pad_y = int(h * 0.25)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)
        return x1, y1, x2, y2

    @staticmethod
    def _box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @staticmethod
    def _remember_unknown(face_crop, box, distance: float) -> str:
        os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)
        unknown_id = str(uuid.uuid4())[:8]
        file_path = os.path.join(UNKNOWN_FACES_DIR, f"{unknown_id}.jpg")
        cv2.imwrite(file_path, face_crop)
        with unknown_lock:
            unknown_faces[unknown_id] = {
                "id": unknown_id,
                "crop": face_crop,
                "box": box,
                "distance": distance,
                "created_at": time.time(),
            }
            if len(unknown_faces) > 20:
                oldest = sorted(unknown_faces.values(), key=lambda item: item["created_at"])[0]["id"]
                unknown_faces.pop(oldest, None)
        return unknown_id

    def _assign_tracks(self, boxes: List[Tuple[int, int, int, int]]) -> List[Dict]:
        now = time.time()
        # Remove stale tracks.
        for tid in list(self.tracks):
            if now - self.tracks[tid].get("last_seen", 0) > TRACK_TTL_SECONDS:
                self.tracks.pop(tid, None)

        assigned_tracks = set()
        results = []
        for box in boxes:
            best_tid = None
            best_score = 0.0
            for tid, track in self.tracks.items():
                if tid in assigned_tracks:
                    continue
                score = box_iou(box, track["box"])
                if score > best_score:
                    best_score = score
                    best_tid = tid
            if best_tid is None or best_score < 0.18:
                best_tid = self.next_track_id
                self.next_track_id += 1
                self.tracks[best_tid] = {
                    "track_id": best_tid,
                    "box": box,
                    "name": "Recognizing...",
                    "distance": None,
                    "unknown": False,
                    "unknown_id": None,
                    "pending": False,
                    "last_recognized": 0.0,
                    "last_seen": now,
                }
            else:
                self.tracks[best_tid]["box"] = box
                self.tracks[best_tid]["last_seen"] = now
            assigned_tracks.add(best_tid)
            results.append(self.tracks[best_tid])
        return results

    def _enqueue_recognition(self, track: Dict, face_crop) -> None:
        now = time.time()
        if track.get("pending"):
            return
        if now - track.get("last_recognized", 0.0) < FACE_RECOGNITION_MIN_SECONDS:
            return
        if face_crop is None or face_crop.size == 0:
            return
        try:
            # Keep queue fresh. If full, drop the oldest pending job.
            if self.recognition_queue.full():
                try:
                    old = self.recognition_queue.get_nowait()
                    old_tid = old.get("track_id")
                    if old_tid in self.tracks:
                        self.tracks[old_tid]["pending"] = False
                except queue.Empty:
                    pass
            track["pending"] = True
            self.recognition_queue.put_nowait({"track_id": track["track_id"], "crop": face_crop.copy(), "box": track["box"], "queued_at": now})
        except queue.Full:
            track["pending"] = False

    def _recognition_loop(self) -> None:
        while self.running:
            try:
                item = self.recognition_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            track_id = item["track_id"]
            crop = item["crop"]
            label, distance = face_db.recognize(crop, FACE_MATCH_THRESHOLD)
            unknown_id = None
            if label == "Unknown Face" and crop.size > 0:
                unknown_id = self._remember_unknown(crop, item["box"], distance)
            track = self.tracks.get(track_id)
            if track:
                track.update({
                    "name": label,
                    "distance": distance,
                    "unknown": label == "Unknown Face",
                    "unknown_id": unknown_id,
                    "pending": False,
                    "last_recognized": time.time(),
                })
            self.recognition_queue.task_done()

    def _detect_face_boxes(self, rgb, width: int, height: int) -> List[Tuple[int, int, int, int]]:
        if self.frame_no % FACE_DETECTION_EVERY_N_FRAMES != 0 and self.last_detected_boxes:
            return self.last_detected_boxes
        results = self.face_detector.process(rgb)
        boxes: List[Tuple[int, int, int, int]] = []
        if results.detections:
            detections = sorted(
                results.detections,
                key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height,
                reverse=True,
            )
            for det in detections:
                box = self._safe_box(det.location_data.relative_bounding_box, width, height)
                x1, y1, x2, y2 = box
                if (x2 - x1) > 35 and (y2 - y1) > 35:
                    boxes.append(box)
        self.last_detected_boxes = boxes
        return boxes

    def _capture_loop(self) -> None:
        min_delay = 1.0 / max(1, TARGET_FPS)
        while self.running:
            started = time.time()
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_no += 1

            boxes = self._detect_face_boxes(rgb, width, height)
            tracks = self._assign_tracks(boxes)
            faces_payload = []

            for idx, track in enumerate(tracks):
                x1, y1, x2, y2 = track["box"]
                face_crop = frame[y1:y2, x1:x2]
                if self.frame_no % FACE_RECOGNITION_EVERY_N_FRAMES == 0 or track.get("name") == "Recognizing...":
                    self._enqueue_recognition(track, face_crop)
                color = (0, 200, 0)
                if track.get("unknown"):
                    color = (0, 0, 255)
                elif track.get("pending") or track.get("name") == "Recognizing...":
                    color = (255, 170, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                name = track.get("name", "Recognizing...")
                distance = track.get("distance")
                label = name if distance is None else f"{name} ({distance:.2f})"
                cv2.putText(frame, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)
                faces_payload.append({
                    "index": idx + 1,
                    "track_id": track["track_id"],
                    "id": track.get("unknown_id"),
                    "name": name,
                    "distance": distance,
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "center": self._box_center(track["box"]),
                    "unknown": bool(track.get("unknown")),
                    "pending": bool(track.get("pending")),
                })

            hands_payload = []
            hand_results = self.hands_detector.process(rgb)
            if hand_results.multi_hand_landmarks:
                handedness_list = hand_results.multi_handedness or []
                for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    handedness = "Right"
                    if idx < len(handedness_list):
                        handedness = handedness_list[idx].classification[0].label
                    payload = hand_payload(hand_landmarks, handedness)
                    hands_payload.append(payload)
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    wrist = hand_landmarks.landmark[0]
                    tx, ty = int(wrist.x * width), int(wrist.y * height)
                    pinch = "PINCH" if payload["pinch"]["active"] else "OPEN"
                    cv2.putText(frame, f"{handedness}: {payload['fingers']} - {payload['gesture']} - {pinch}", (tx, max(30, ty - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 170, 0), 2)

            first_face = faces_payload[0] if faces_payload else None
            with state_lock:
                latest_state.update({
                    "faces": faces_payload,
                    "face": first_face["name"] if first_face else "No Face",
                    "face_distance": first_face.get("distance") if first_face else None,
                    "unknown_available": any(f["unknown"] for f in faces_payload),
                    "unknown_faces": [
                        {"id": f["id"], "distance": f["distance"], "box": f["box"], "track_id": f["track_id"]}
                        for f in faces_payload if f.get("id")
                    ],
                    "hands": hands_payload,
                    "people": face_db.list_people(),
                    "custom_gestures": load_custom_gestures(),
                    "frame": {"width": width, "height": height},
                    "performance": {
                        "mode": "async",
                        "recognition_queue": self.recognition_queue.qsize(),
                        "face_threshold": FACE_MATCH_THRESHOLD,
                        "target_fps": TARGET_FPS,
                    },
                    "time": time.time(),
                })

            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                with self.frame_lock:
                    self.latest_jpeg = buffer.tobytes()

            elapsed = time.time() - started
            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self.frame_lock:
            return self.latest_jpeg

    def release(self) -> None:
        self.running = False
        time.sleep(0.15)
        self.cap.release()
        self.face_detector.close()
        self.hands_detector.close()


camera: Optional[CameraProcessor] = None


def get_camera() -> CameraProcessor:
    global camera
    if camera is None:
        camera = CameraProcessor()
    return camera


def generate_frames():
    cam = get_camera()
    last = None
    while True:
        jpeg = cam.get_latest_jpeg()
        if jpeg is None:
            time.sleep(0.03)
            continue
        # Send frames continuously, even when the camera thread is processing.
        if jpeg is last:
            time.sleep(0.01)
        last = jpeg
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    with state_lock:
        return jsonify(latest_state)


@app.route("/api/register_face", methods=["POST"])
def api_register_face():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    unknown_id = (payload.get("unknown_id") or "").strip()
    if not name:
        return jsonify({"ok": False, "message": "Person name is required."}), 400
    with unknown_lock:
        if unknown_id:
            item = unknown_faces.get(unknown_id)
        else:
            item = sorted(unknown_faces.values(), key=lambda v: v["created_at"], reverse=True)[0] if unknown_faces else None
    if not item:
        return jsonify({"ok": False, "message": "No unknown face snapshot is available."}), 400
    ok = face_db.add_person_image(name, item["crop"])
    if not ok:
        return jsonify({"ok": False, "message": "Failed to register face. Move closer and try again."}), 500
    # Force fresh recognition after registration.
    for track in get_camera().tracks.values():
        track["last_recognized"] = 0
        track["pending"] = False
    return jsonify({"ok": True, "message": f"{name} registered successfully.", "people": face_db.list_people()})


@app.route("/api/add_face_sample", methods=["POST"])
def api_add_face_sample():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    unknown_id = (payload.get("unknown_id") or "").strip()
    if not name:
        return jsonify({"ok": False, "message": "Select a person first."}), 400
    with unknown_lock:
        item = unknown_faces.get(unknown_id) if unknown_id else None
        if item is None and unknown_faces:
            item = sorted(unknown_faces.values(), key=lambda v: v["created_at"], reverse=True)[0]
    if not item:
        return jsonify({"ok": False, "message": "No face snapshot is available."}), 400
    ok = face_db.add_person_image(name, item["crop"])
    if not ok:
        return jsonify({"ok": False, "message": "Failed to add sample."}), 500
    return jsonify({"ok": True, "message": "Sample added successfully.", "people": face_db.list_people()})


@app.route("/api/rename_person", methods=["POST"])
def api_rename_person():
    payload = request.get_json(silent=True) or {}
    ok, message = face_db.rename_person(payload.get("old_name", ""), payload.get("new_name", ""))
    return jsonify({"ok": ok, "message": message, "people": face_db.list_people()}), (200 if ok else 400)


@app.route("/api/delete_person", methods=["POST"])
def api_delete_person():
    payload = request.get_json(silent=True) or {}
    ok, message = face_db.delete_person(payload.get("name", ""))
    return jsonify({"ok": ok, "message": message, "people": face_db.list_people()}), (200 if ok else 400)


@app.route("/api/clear_samples", methods=["POST"])
def api_clear_samples():
    payload = request.get_json(silent=True) or {}
    ok, message = face_db.clear_samples(payload.get("name", ""))
    return jsonify({"ok": ok, "message": message, "people": face_db.list_people()}), (200 if ok else 400)


@app.route("/api/custom_gestures", methods=["GET", "POST", "DELETE"])
def api_custom_gestures():
    if request.method == "GET":
        return jsonify({"gestures": load_custom_gestures()})
    payload = request.get_json(silent=True) or {}
    if request.method == "POST":
        ok, message, gestures = add_custom_gesture(payload.get("name", ""), payload.get("states", {}))
    else:
        ok, message, gestures = delete_custom_gesture(payload.get("name", ""))
    return jsonify({"ok": ok, "message": message, "gestures": gestures}), (200 if ok else 400)


@app.route("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    try:
        get_camera()
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
    finally:
        if camera is not None:
            camera.release()
