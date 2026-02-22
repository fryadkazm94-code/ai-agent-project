import time
from deepface import DeepFace


class EmotionAgent:
    def __init__(self, cooldown_s: float = 1.0):
        self.last_ts = 0.0
        self.cooldown_s = cooldown_s

    def run(self, face_crop_bgr):
        now = time.time()
        if now - self.last_ts < self.cooldown_s:
            return None

        self.last_ts = now

        try:
            result = DeepFace.analyze(
                face_crop_bgr,
                actions=["emotion"],
                enforce_detection=False
            )
            if isinstance(result, list):
                result = result[0]

            emotion = result["dominant_emotion"]
            conf = float(result["emotion"][emotion])

            return {"emotion": emotion, "confidence": conf}
        except Exception:
            return None