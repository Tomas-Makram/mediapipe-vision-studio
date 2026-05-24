import json
import os
from typing import Dict, List, Tuple

from config import CUSTOM_GESTURES_FILE

FINGER_KEYS = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_LABELS = {
    "thumb": "Thumb",
    "index": "Index",
    "middle": "Middle",
    "ring": "Ring",
    "pinky": "Pinky",
}


def _is_thumb_open(lm, handedness: str) -> bool:
    if handedness.lower().startswith("right"):
        return lm[4].x < lm[3].x
    return lm[4].x > lm[3].x


def finger_states(hand_landmarks, handedness: str) -> Dict[str, bool]:
    lm = hand_landmarks.landmark
    return {
        "thumb": _is_thumb_open(lm, handedness),
        "index": lm[8].y < lm[6].y,
        "middle": lm[12].y < lm[10].y,
        "ring": lm[16].y < lm[14].y,
        "pinky": lm[20].y < lm[18].y,
    }


def count_fingers(hand_landmarks, handedness: str) -> int:
    return sum(1 for v in finger_states(hand_landmarks, handedness).values() if v)


def distance(lm, a: int, b: int) -> float:
    dx = lm[a].x - lm[b].x
    dy = lm[a].y - lm[b].y
    dz = lm[a].z - lm[b].z
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def pinch_info(hand_landmarks) -> Dict[str, float | bool]:
    lm = hand_landmarks.landmark
    palm = max(distance(lm, 0, 9), 0.0001)
    d = distance(lm, 4, 8)
    ratio = d / palm
    return {"active": ratio < 0.34, "distance": float(d), "ratio": float(ratio)}


def pointer_info(hand_landmarks) -> Dict[str, float]:
    tip = hand_landmarks.landmark[8]
    return {"x": float(tip.x), "y": float(tip.y), "z": float(tip.z)}


def load_custom_gestures() -> List[Dict]:
    os.makedirs(os.path.dirname(CUSTOM_GESTURES_FILE), exist_ok=True)
    if not os.path.exists(CUSTOM_GESTURES_FILE):
        return []
    with open(CUSTOM_GESTURES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_custom_gestures(gestures: List[Dict]) -> None:
    os.makedirs(os.path.dirname(CUSTOM_GESTURES_FILE), exist_ok=True)
    with open(CUSTOM_GESTURES_FILE, "w", encoding="utf-8") as f:
        json.dump(gestures, f, ensure_ascii=False, indent=2)


def add_custom_gesture(name: str, states: Dict[str, bool]) -> Tuple[bool, str, List[Dict]]:
    clean_name = name.strip()[:60]
    if not clean_name:
        return False, "Gesture name is required.", load_custom_gestures()
    if any(k not in states for k in FINGER_KEYS):
        return False, "All five finger states are required.", load_custom_gestures()
    gestures = [g for g in load_custom_gestures() if g.get("name") != clean_name]
    gestures.append({"name": clean_name, "states": {k: bool(states[k]) for k in FINGER_KEYS}})
    save_custom_gestures(gestures)
    return True, "Gesture saved successfully.", gestures


def delete_custom_gesture(name: str) -> Tuple[bool, str, List[Dict]]:
    gestures = load_custom_gestures()
    filtered = [g for g in gestures if g.get("name") != name]
    if len(filtered) == len(gestures):
        return False, "Gesture not found.", gestures
    save_custom_gestures(filtered)
    return True, "Gesture deleted successfully.", filtered


def classify_gesture(hand_landmarks, handedness: str) -> str:
    lm = hand_landmarks.landmark
    states = finger_states(hand_landmarks, handedness)
    for custom in load_custom_gestures():
        if custom.get("states") == states:
            return custom.get("name", "Custom Gesture")
    total = sum(states.values())
    if total == 0:
        return "Fist"
    if total == 5:
        return "Open Palm"
    if states["index"] and states["middle"] and not states["ring"] and not states["pinky"]:
        return "Peace"
    if states["thumb"] and not states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
        return "Thumbs Up"
    if distance(lm, 4, 8) < distance(lm, 0, 9) * 0.35 and (states["middle"] or states["ring"] or states["pinky"]):
        return "OK / Pinch"
    return "Unclassified"


def hand_payload(hand_landmarks, handedness: str) -> Dict:
    states = finger_states(hand_landmarks, handedness)
    return {
        "hand": handedness,
        "fingers": sum(states.values()),
        "gesture": classify_gesture(hand_landmarks, handedness),
        "finger_states": states,
        "finger_details": [{"key": k, "label": FINGER_LABELS[k], "open": states[k]} for k in FINGER_KEYS],
        "pinch": pinch_info(hand_landmarks),
        "pointer": pointer_info(hand_landmarks),
    }
