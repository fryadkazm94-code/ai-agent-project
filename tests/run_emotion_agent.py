import sys
import os

# make project root visible to python
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
from agents.analysis_agent import EmotionAgent

# open webcam
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
cap.release()

if not ret:
    print("Camera not working")
    exit()

agent = EmotionAgent()

result = agent.run(frame)

print("Emotion result:", result)
