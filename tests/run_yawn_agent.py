
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
from agents.yawn_agent import YawnAgent

cap = cv2.VideoCapture(0)
agent = YawnAgent()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out = agent.run(frame)

    text = f"MAR: {out['mar']:.3f} | Yawn: {out['yawn']} | Dur: {out['duration']:.1f}s"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Yawn Agent", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

