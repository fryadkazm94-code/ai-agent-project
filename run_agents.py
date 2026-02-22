from pathlib import Path
import subprocess
import sys


# Project root = folder where this file exists
ROOT = Path(__file__).resolve().parent


def get_env_python(env_name: str) -> Path:
    """
    Returns the python executable inside a virtual environment.
    Supports Windows and Unix-style venv layouts.
    """
    # Windows
    win_path = ROOT / env_name / "Scripts" / "python.exe"
    if win_path.exists():
        return win_path

    # Linux/macOS (just in case)
    unix_path = ROOT / env_name / "bin" / "python"
    if unix_path.exists():
        return unix_path

    return Path()  # empty path if not found


def run_script(script_rel_path: str, python_exe: Path | None = None, title: str = "") -> None:
    """
    Runs a script using the given python executable.
    Uses absolute paths + cwd=ROOT so it works even if terminal is opened elsewhere.
    """
    script_path = ROOT / script_rel_path

    if title:
        print(f"\n=== {title} ===")

    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return

    if python_exe is None:
        python_exe = Path(sys.executable)

    if not python_exe.exists():
        print(f"[ERROR] Python executable not found: {python_exe}")
        return

    print(f"[INFO] Python: {python_exe}")
    print(f"[INFO] Script : {script_path}")

    try:
        result = subprocess.run(
            [str(python_exe), str(script_path)],
            cwd=str(ROOT)
        )

        if result.returncode != 0:
            print(f"[WARN] Script exited with code {result.returncode}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C).")
    except Exception as e:
        print(f"[ERROR] Failed to run script: {e}")


def run_face() -> None:
    face_python = get_env_python("face_env")
    if not face_python:
        print("[ERROR] face_env python not found. Expected:")
        print(f"  {ROOT / 'face_env' / 'Scripts' / 'python.exe'}")
        return

    run_script("agents/sensor_agent.py", face_python, "Face Detection Agent")


def run_emotion() -> None:
    emotion_python = get_env_python("emotion_env")
    if not emotion_python:
        print("[ERROR] emotion_env python not found. Expected:")
        print(f"  {ROOT / 'emotion_env' / 'Scripts' / 'python.exe'}")
        return

    run_script("tests/run_emotion_agent.py", emotion_python, "Emotion Recognition Agent")


def run_yawn() -> None:
    face_python = get_env_python("face_env")
    if not face_python:
        print("[ERROR] face_env python not found. Expected:")
        print(f"  {ROOT / 'face_env' / 'Scripts' / 'python.exe'}")
        return

    run_script("tests/run_yawn_agent.py", face_python, "Yawn Detection Agent")


def run_decision_demo() -> None:
    # No heavy deps; use current Python
    run_script("tests/test_decision_agent.py", Path(sys.executable), "Decision Agent Demo")


def run_action_demo() -> None:
    # No heavy deps; use current Python
    run_script("tests/test_action_agent.py", Path(sys.executable), "Action Agent Demo")


def main() -> None:
    while True:
        print("\n==== AI Multi-Agent Launcher ====")
        print("1 - Face Detection Agent")
        print("2 - Emotion Recognition Agent")
        print("3 - Yawn Detection Agent")
        print("4 - Decision Agent (Demo)")
        print("5 - Action Agent (Demo)")
        print("6 - Exit")

        choice = input("Select option: ").strip().lower()

        if choice == "1":
            run_face()
        elif choice == "2":
            run_emotion()
        elif choice == "3":
            run_yawn()
        elif choice == "4":
            run_decision_demo()
        elif choice == "5":
            run_action_demo()
        elif choice in {"6", "q", "quit", "exit"}:
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1-6.")


if __name__ == "__main__":
    main()