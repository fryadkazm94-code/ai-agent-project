import os
import sys
import time
import importlib
import cv2

# ----------------------------
# PATH FIX (makes all imports work)
# ----------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(ROOT, "agents")
TESTS_DIR = os.path.join(ROOT, "tests")

for p in (ROOT, AGENTS_DIR, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


# ----------------------------
# Helpers
# ----------------------------
def _press_esc_to_exit_loop(win_name: str):
    return (cv2.waitKey(1) & 0xFF) == 27  # ESC


def _safe_import(module_name: str):
    """
    Import a module by name. If it was imported before, reload it so
    re-running a choice works (especially for test scripts).
    """
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


# ----------------------------
# Menu actions
# ----------------------------
def run_face_detection():
    from sensor_agent import FaceDetectionAgent

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found / not accessible.")
        return

    agent = FaceDetectionAgent()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = agent.run(frame)
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Detection", frame)
        if _press_esc_to_exit_loop("Face Detection"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_emotion_once_test():
    """
    Runs your existing DeepFace test (single frame) safely.
    """
    # This file is at project root in your uploads list: test_emotion.py
    # It reads one frame, prints emotions, then exits.
    _safe_import("test_emotion")


def run_emotion_agent_once():
    """
    Runs your wrapper EmotionAgent test (single frame) safely.
    """
    # This file is at project root in your uploads list: run_emotion_agent.py
    _safe_import("run_emotion_agent")


def run_yawn_live():
    """
    Runs your existing yawn live window safely.
    """
    # This file is at project root in your uploads list: run_yawn_agent.py
    _safe_import("run_yawn_agent")


def run_decision_demo():
    """
    Runs your decision test samples safely.
    """
    # This file is at project root in your uploads list: test_decision_agent.py
    _safe_import("test_decision_agent")


def run_action_demo():
    """
    Runs your action test (prints + logs + optional beep) safely.
    """
    # This file is at project root in your uploads list: test_action_agent.py
    _safe_import("test_action_agent")


def run_full_system():
    """
    Full integrated pipeline:
    Camera -> FaceDetection -> EmotionAgent (on face crop) + YawnAgent (on frame)
           -> MoodDecisionAgent -> ActionAgent
    """
    from sensor_agent import FaceDetectionAgent
    from analysis_agent import EmotionAgent
    from yawn_agent import YawnAgent
    from decision_agent import MoodDecisionAgent
    from action_agent import ActionAgent

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found / not accessible.")
        return

    face_agent = FaceDetectionAgent()
    emotion_agent = EmotionAgent(cooldown_s=1.0)
    yawn_agent = YawnAgent()
    decision_agent = MoodDecisionAgent()
    action_agent = ActionAgent()

    last_emotion_info = None
    last_action_ts = 0.0
    ACTION_COOLDOWN_S = 1.0  # avoid spamming logs every frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Face bbox + crop (for emotion)
        bbox = face_agent.run(frame)
        face_crop = None
        if bbox:
            x, y, w, h = bbox
            face_crop = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 2) Run perception agents
        yawn_info = yawn_agent.run(frame)

        emotion_info = None
        if face_crop is not None and face_crop.size > 0:
            emotion_info = emotion_agent.run(face_crop)

        if emotion_info is not None:
            last_emotion_info = emotion_info

        # 3) Decision
        decision = decision_agent.run(last_emotion_info, yawn_info)

        # 4) Action (cooldown)
        now = time.time()
        if now - last_action_ts >= ACTION_COOLDOWN_S:
            action_agent.run(decision)
            last_action_ts = now

        # 5) Overlay info
        emo_txt = "Emotion: none"
        if last_emotion_info:
            emo_txt = f"Emotion: {last_emotion_info.get('emotion')} ({last_emotion_info.get('confidence', 0):.1f})"

        yawn_txt = f"MAR: {yawn_info.get('mar', 0):.3f} | Yawn: {yawn_info.get('yawn')} | Dur: {yawn_info.get('duration', 0):.1f}s"
        state_txt = f"STATE: {decision.get('state', 'unknown').upper()}"

        cv2.putText(frame, emo_txt, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, yawn_txt, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, state_txt, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("FULL SYSTEM (ESC to quit)", frame)
        if _press_esc_to_exit_loop("FULL SYSTEM (ESC to quit)"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ----------------------------
# CLI Menu
# ----------------------------
def main():
    while True:
        print("\n==== AI Multi-Agent Launcher ====")
        print("1) Face Detection (live)")
        print("2) Emotion (DeepFace test: single frame)")
        print("3) EmotionAgent wrapper (single frame)")
        print("4) Yawn Detection (live)")
        print("5) Decision Agent (demo samples)")
        print("6) Action Agent (demo print/log/beep)")
        print("7) FULL SYSTEM (all agents together)")
        print("0) Exit")

        choice = input("Select option: ").strip()

        try:
            if choice == "1":
                run_face_detection()
            elif choice == "2":
                run_emotion_once_test()
            elif choice == "3":
                run_emotion_agent_once()
            elif choice == "4":
                run_yawn_live()
            elif choice == "5":
                run_decision_demo()
            elif choice == "6":
                run_action_demo()
            elif choice == "7":
                run_full_system()
            elif choice == "0":
                print("Bye 👋")
                break
            else:
                print("Invalid choice.")
        except Exception as e:
            print("\n[ERROR] Something failed while running that option:")
            print(e)


if __name__ == "__main__":
    main()