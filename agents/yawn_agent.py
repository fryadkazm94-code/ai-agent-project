import time
import cv2
import mediapipe as mp
import math


class YawnAgent:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Landmark indices (MediaPipe FaceMesh)
        self.UP = 13   # upper inner lip
        self.LO = 14   # lower inner lip
        self.LC = 61   # left mouth corner
        self.RC = 291  # right mouth corner

        self.yawn_start = None

        # Tune these if needed
        self.MAR_THRESHOLD = 0.08
        self.YAWN_MIN_SECONDS = 1.6

    def run(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(frame_rgb)

        if not res.multi_face_landmarks:
            self.yawn_start = None
            return {"yawn": False, "duration": 0.0, "mar": 0.0}

        lm = res.multi_face_landmarks[0].landmark
        h, w = frame_bgr.shape[:2]

        def pt(i):
            return (lm[i].x * w, lm[i].y * h)

        def dist(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        up = pt(self.UP)
        lo = pt(self.LO)
        lc = pt(self.LC)
        rc = pt(self.RC)

        vertical = dist(up, lo)
        horizontal = dist(lc, rc)
        mar = vertical / max(horizontal, 1e-6)

        now = time.time()
        if mar > self.MAR_THRESHOLD:
            if self.yawn_start is None:
                self.yawn_start = now
            duration = now - self.yawn_start
            is_yawn = duration >= self.YAWN_MIN_SECONDS
            return {"yawn": is_yawn, "duration": duration, "mar": mar}
        else:
            self.yawn_start = None
            return {"yawn": False, "duration": 0.0, "mar": mar}
        