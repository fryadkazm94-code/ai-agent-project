import sys
import os
import json
import cv2

# Optional: reduce TensorFlow logs a bit
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from agents.analysis_agent import EmotionAgent


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "missing_image_path"}))
        sys.exit(1)

    image_path = sys.argv[1]

    frame = cv2.imread(image_path)
    if frame is None:
        print(json.dumps({"error": "cannot_read_image"}))
        sys.exit(1)

    agent = EmotionAgent(cooldown_s=0.0)  # worker should not self-throttle
    result = agent.run(frame)

    # Print only JSON to stdout so final_agent can parse it
    print(json.dumps(result))


if __name__ == "__main__":
    main()