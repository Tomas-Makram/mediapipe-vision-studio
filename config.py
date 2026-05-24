import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "face_embeddings.json")
CUSTOM_GESTURES_FILE = os.path.join(DATA_DIR, "custom_gestures.json")
UNKNOWN_FACES_DIR = os.path.join(DATA_DIR, "unknown_faces")

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "960"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "540"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "72"))
TARGET_FPS = int(os.getenv("TARGET_FPS", "24"))

FACE_CONFIDENCE = float(os.getenv("FACE_CONFIDENCE", "0.60"))
HAND_CONFIDENCE = float(os.getenv("HAND_CONFIDENCE", "0.60"))

# 0.68 was too permissive for multiple faces and can mark different people as the same person.
# For Facenet512 with normalized cosine distance, 0.38-0.45 is a safer practical range.
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.42"))

# Face detection remains light. DeepFace recognition is heavy and runs in a background worker.
FACE_DETECTION_EVERY_N_FRAMES = int(os.getenv("FACE_DETECTION_EVERY_N_FRAMES", "2"))
FACE_RECOGNITION_EVERY_N_FRAMES = int(os.getenv("FACE_RECOGNITION_EVERY_N_FRAMES", "28"))
FACE_RECOGNITION_MIN_SECONDS = float(os.getenv("FACE_RECOGNITION_MIN_SECONDS", "1.2"))
MAX_FACE_RECOGNITION_QUEUE = int(os.getenv("MAX_FACE_RECOGNITION_QUEUE", "8"))
TRACK_TTL_SECONDS = float(os.getenv("TRACK_TTL_SECONDS", "2.0"))

DEEPFACE_MODEL = os.getenv("DEEPFACE_MODEL", "Facenet512")
DEEPFACE_DETECTOR_BACKEND = os.getenv("DEEPFACE_DETECTOR_BACKEND", "skip")
