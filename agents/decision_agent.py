class MoodDecisionAgent:
    """
    Combines EmotionAgent + YawnAgent outputs into a simple mood/state decision.
    This is rule-based (no ML), which is totally valid for "agent" behavior.
    """

    def run(self, emotion_info, yawn_info):
        emotion = (emotion_info or {}).get("emotion", "unknown")
        econf = float((emotion_info or {}).get("confidence", 0.0))

        yawn = bool((yawn_info or {}).get("yawn", False))
        ydur = float((yawn_info or {}).get("duration", 0.0))
        mar = float((yawn_info or {}).get("mar", 0.0))

        # ---- RULES ----
        # Rule 1: If yawning (sustained mouth open), mark as drowsy
        if yawn or ydur >= 1.6:
            return {
                "state": "drowsy",
                "reason": f"yawn_detected duration={ydur:.1f}s mar={mar:.3f}"
            }

        # Rule 2: If emotion is confident and negative, mark as stressed
        negative = {"angry", "fear", "sad", "disgust"}
        if emotion in negative and econf >= 60.0:
            return {
                "state": "stressed",
                "reason": f"emotion={emotion} conf={econf:.1f}"
            }

        # Rule 3: If emotion is confident and positive, mark as engaged
        if emotion == "happy" and econf >= 60.0:
            return {
                "state": "engaged",
                "reason": f"emotion=happy conf={econf:.1f}"
            }

        # Rule 4: If we have weak/no signals, mark as normal/unknown
        if emotion == "unknown":
            return {"state": "unknown", "reason": "no_emotion_detected"}

        return {"state": "normal", "reason": f"emotion={emotion} conf={econf:.1f}"}
    
    