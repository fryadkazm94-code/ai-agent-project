import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents.decision_agent import MoodDecisionAgent

agent = MoodDecisionAgent()

samples = [
    ({"emotion": "happy", "confidence": 85}, {"yawn": False, "duration": 0.0, "mar": 0.02}),
    ({"emotion": "sad", "confidence": 75}, {"yawn": False, "duration": 0.0, "mar": 0.03}),
    ({"emotion": "neutral", "confidence": 90}, {"yawn": True, "duration": 2.0, "mar": 0.12}),
    (None, {"yawn": False, "duration": 0.0, "mar": 0.0}),
]

for i, (emo, yawn) in enumerate(samples, 1):
    print(i, agent.run(emo, yawn))