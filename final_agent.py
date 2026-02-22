from pathlib import Path
import sys
import time
import json
import subprocess
import threading

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from agents.sensor_agent import FaceDetectionAgent
from agents.yawn_agent import YawnAgent
from agents.decision_agent import MoodDecisionAgent
from agents.action_agent import ActionAgent


WINDOW_SECONDS = 30             
EMOTION_SAMPLE_INTERVAL = 8.0   
SHOW_CAMERA = True              


def get_env_python(env_name: str) -> Path:
    p = ROOT / env_name / "Scripts" / "python.exe"  
    if p.exists():
        return p
    p = ROOT / env_name / "bin" / "python"          
    if p.exists():
        return p
    return Path()


def call_emotion_worker(face_crop_bgr, emotion_python: Path, worker_script: Path, temp_img_path: Path):
  
    try:
        ok = cv2.imwrite(str(temp_img_path), face_crop_bgr)
        if not ok:
            print("[EMOTION] Failed to write temp image")
            return None

        result = subprocess.run(
            [str(emotion_python), str(worker_script), str(temp_img_path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print("[EMOTION] Worker failed:", result.stderr.strip() or result.stdout.strip())
            return None

        lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
        if not lines:
            return None

        data = json.loads(lines[-1])
        if isinstance(data, dict) and "emotion" in data:
            return data

        return None

    except subprocess.TimeoutExpired:
        print("[EMOTION] Worker timed out")
        return None
    except Exception as e:
        print(f"[EMOTION] Error calling worker: {e}")
        return None


def start_emotion_job(state, face_crop, emotion_python, worker_script, temp_img_path, window_seq: int):
    """
    Non-blocking emotion call so camera loop doesn't freeze.
    """
    if state.get("emotion_busy", False):
        return

    state["emotion_busy"] = True
    crop_copy = face_crop.copy()

    def _job():
        try:
            emo = call_emotion_worker(crop_copy, emotion_python, worker_script, temp_img_path)

            if window_seq != state.get("window_seq"):
                return

            if emo:
                state["emotion_samples"].append(emo)
                state["last_emotion_text"] = f"{emo['emotion']} ({emo['confidence']:.1f})"
                print(f"[EMOTION] {state['last_emotion_text']}")
            else:
                state["last_emotion_text"] = "none"
        finally:
            state["emotion_busy"] = False

    threading.Thread(target=_job, daemon=True).start()


def summarize_emotions(samples):
    if not samples:
        return None

    counts = {}
    best_conf = {}

    for s in samples:
        emo = s.get("emotion")
        conf = float(s.get("confidence", 0.0))
        if not emo:
            continue

        counts[emo] = counts.get(emo, 0) + 1
        best_conf[emo] = max(best_conf.get(emo, 0.0), conf)

    if not counts:
        return None

    winner = sorted(
        counts.keys(),
        key=lambda e: (counts[e], best_conf.get(e, 0.0)),
        reverse=True
    )[0]

    return {"emotion": winner, "confidence": best_conf.get(winner, 0.0)}


def new_window_state():
    return {
        "window_start_ts": None,
        "window_seq": 0,                  
        "last_emotion_sample_ts": 0.0,
        "emotion_samples": [],
        "emotion_busy": False,            
        "yawn_any": False,
        "max_yawn_duration": 0.0,
        "max_mar": 0.0,
        "last_emotion_text": "none",
        "last_decision_text": "none"
    }


def reset_window(state, yawn_agent=None):
    state["window_start_ts"] = None
    state["window_seq"] += 1
    state["last_emotion_sample_ts"] = 0.0
    state["emotion_samples"] = []
    state["yawn_any"] = False
    state["max_yawn_duration"] = 0.0
    state["max_mar"] = 0.0
    state["last_emotion_text"] = "none"

    # reset yawn memory when face disappears
    if yawn_agent is not None:
        if hasattr(yawn_agent, "yawn_start"):
            yawn_agent.yawn_start = None
        if hasattr(yawn_agent, "mouth_open"):
            yawn_agent.mouth_open = False
        if hasattr(yawn_agent, "mar_ema"):
            yawn_agent.mar_ema = None


def clamp_crop(frame, bbox):
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop


def draw_overlay(frame, bbox, face_present, remaining, state, yawn_text):
    if bbox:
        x, y, bw, bh = bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    line1 = f"Face: {'YES' if face_present else 'NO'} | Timer: {max(0, int(remaining))}s"
    line2 = f"Emotion: {state['last_emotion_text']} | {yawn_text}"
    line3 = f"Last Decision: {state['last_decision_text']}"
    line4 = "ESC = Exit"

    cv2.putText(frame, line1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, line2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, line3, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    cv2.putText(frame, line4, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def main():
    emotion_python = get_env_python("emotion_env")
    if not emotion_python:
        print("[ERROR] emotion_env python not found")
        print("Expected: emotion_env\\Scripts\\python.exe")
        return

    worker_script = ROOT / "tests" / "emotion_worker.py"
    if not worker_script.exists():
        print(f"[ERROR] Missing worker script: {worker_script}")
        return

    temp_dir = ROOT / "temp"
    temp_dir.mkdir(exist_ok=True)
    temp_img_path = temp_dir / "emotion_face.jpg"

    face_agent = FaceDetectionAgent()
    yawn_agent = YawnAgent()
    decision_agent = MoodDecisionAgent()
    action_agent = ActionAgent(log_path=str(ROOT / "logs" / "events.log"))

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Camera not found")
        return

    state = new_window_state()

    print("[INFO] Final multi-agent system started.")
    print("[INFO] Face present = start 30s window | Face lost = reset timer")
    print("[INFO] Press ESC to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Camera frame not received")
                break

            now = time.time()
            bbox = face_agent.run(frame)
            face_present = bbox is not None

            yawn_text = "Yawn: no_data"
            remaining = 0

            if not face_present:
                reset_window(state, yawn_agent=yawn_agent)
                yawn_text = "Yawn: reset"

            else:
                # Start window if needed
                if state["window_start_ts"] is None:
                    state["window_start_ts"] = now
                    state["window_seq"] += 1
                    state["last_emotion_sample_ts"] = 0.0
                    state["emotion_samples"] = []
                    state["yawn_any"] = False
                    state["max_yawn_duration"] = 0.0
                    state["max_mar"] = 0.0
                    print("[INFO] Face detected -> 30s window started")

                elapsed = now - state["window_start_ts"]
                remaining = WINDOW_SECONDS - elapsed

                # --- Yawn (continuous) ---
                yawn_out = yawn_agent.run(frame)
                if yawn_out:
                    state["yawn_any"] = state["yawn_any"] or bool(yawn_out.get("yawn", False))
                    state["max_yawn_duration"] = max(state["max_yawn_duration"], float(yawn_out.get("duration", 0.0)))
                    state["max_mar"] = max(state["max_mar"], float(yawn_out.get("mar", 0.0)))
                    yawn_text = (
                        f"Yawn: {state['yawn_any']} "
                        f"(dur={state['max_yawn_duration']:.1f}s mar={state['max_mar']:.3f})"
                    )

                # --- Emotion (non-blocking background job) ---
                if (now - state["last_emotion_sample_ts"]) >= EMOTION_SAMPLE_INTERVAL and not state["emotion_busy"]:
                    face_crop = clamp_crop(frame, bbox)
                    if face_crop is not None:
                        start_emotion_job(
                            state=state,
                            face_crop=face_crop,
                            emotion_python=emotion_python,
                            worker_script=worker_script,
                            temp_img_path=temp_img_path,
                            window_seq=state["window_seq"]
                        )
                    else:
                        state["last_emotion_text"] = "crop_failed"

                    state["last_emotion_sample_ts"] = now

                # --- End of 30s window -> decision + action ---
                if elapsed >= WINDOW_SECONDS:
                    emotion_info = summarize_emotions(state["emotion_samples"])
                    yawn_info = {
                        "yawn": state["yawn_any"],
                        "duration": state["max_yawn_duration"],
                        "mar": state["max_mar"]
                    }

                    decision = decision_agent.run(emotion_info, yawn_info)
                    action_agent.run(decision)

                    state["last_decision_text"] = f"{decision['state']} ({decision['reason']})"
                    print("[DECISION]", state["last_decision_text"])

                    # Start a fresh 30s window immediately (face is still present)
                    state["window_start_ts"] = now
                    state["window_seq"] += 1
                    state["last_emotion_sample_ts"] = 0.0
                    state["emotion_samples"] = []
                    state["yawn_any"] = False
                    state["max_yawn_duration"] = 0.0
                    state["max_mar"] = 0.0

            if SHOW_CAMERA:
                draw_overlay(frame, bbox, face_present, remaining, state, yawn_text)
                cv2.imshow("Final Multi-Agent System", frame)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()