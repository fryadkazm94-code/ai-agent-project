from datetime import datetime
import os
import time
import threading

# Desktop notifications (toast)
try:
    from plyer import notification
    PLYER_AVAILABLE = True
except Exception:
    PLYER_AVAILABLE = False


class ActionAgent:
    """
    Receives the final decision from MoodDecisionAgent
    and performs smart actions:
      - logs to file
      - desktop notifications (toast)
      - starts a break timer (drowsy)
      - tracks focus sessions (engaged)
    """

    def __init__(
        self,
        log_path="logs/events.log",
        break_seconds=120,     # 2 minutes (set to 5 for testing)
        stress_cooldown=30     # avoid spamming notifications
    ):
        self.log_path = log_path
        self.break_seconds = break_seconds
        self.stress_cooldown = stress_cooldown

        # Internal state memory
        self.last_state = None
        self.last_stress_action_ts = 0.0
        self.drowsy_count = 0
        self.focus_start_ts = None
        self.break_timer_active = False

        # Make sure logs folder exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _now_str(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write_log(self, message: str):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def _log_and_print(self, text: str):
        msg = f"[{self._now_str()}] {text}"
        print(msg)
        self._write_log(msg)
        return msg

    def _notify(self, title: str, message: str, timeout: int = 5):
        """
        Sends a desktop notification (toast). If plyer is not installed,
        it just logs the notification content.
        """
        self._log_and_print(f"NOTIFY | title={title} | msg={message}")

        if not PLYER_AVAILABLE:
            self._log_and_print("NOTIFY_SKIPPED | reason=plyer_not_installed")
            return

        try:
            notification.notify(
                title=title,
                message=message,
                timeout=timeout
            )
        except Exception as e:
            self._log_and_print(f"NOTIFY_FAILED | error={e}")

    def _start_break_timer(self):
        """
        Starts a non-blocking break timer thread.
        Prevents multiple timers from running at the same time.
        """
        if self.break_timer_active:
            self._log_and_print("ACTION=break_timer_skipped | reason=already_running")
            return

        self.break_timer_active = True

        def _timer_job():
            try:
                self._log_and_print(f"ACTION=break_timer_started | duration={self.break_seconds}s")
                self._notify(
                    "Break Time",
                    f"You look tired. Take a short break ({self.break_seconds}s).",
                    timeout=5
                )

                time.sleep(self.break_seconds)

                self._log_and_print("ACTION=break_timer_finished | message=break_over")
                self._notify(
                    "Break Over",
                    "Time to continue. Welcome back 👌",
                    timeout=5
                )
            finally:
                self.break_timer_active = False

        t = threading.Thread(target=_timer_job, daemon=True)
        t.start()

    def _stress_notification(self):
        """
        Sends a stress-support notification with cooldown
        to avoid repeated spam.
        """
        now = time.time()
        if now - self.last_stress_action_ts < self.stress_cooldown:
            remaining = int(self.stress_cooldown - (now - self.last_stress_action_ts))
            self._log_and_print(f"ACTION=stress_notify_skipped | cooldown_remaining={remaining}s")
            return

        self.last_stress_action_ts = now
        self._log_and_print("ACTION=stress_notify_sent")
        self._notify(
            "Stress Check",
            "You seem stressed. Pause and take 5 deep breaths.",
            timeout=6
        )

    def _start_focus_session(self):
        if self.focus_start_ts is None:
            self.focus_start_ts = time.time()
            self._log_and_print("ACTION=focus_session_started")
            self._notify(
                "Focus Mode",
                "You look engaged. Keep going 🔥",
                timeout=4
            )
        else:
            self._log_and_print("ACTION=focus_session_already_active")

    def _end_focus_session_if_active(self, reason="state_changed"):
        if self.focus_start_ts is None:
            return

        duration = int(time.time() - self.focus_start_ts)
        self.focus_start_ts = None
        self._log_and_print(f"ACTION=focus_session_ended | duration={duration}s | reason={reason}")

        self._notify(
            "Focus Session Ended",
            f"Session duration: {duration} seconds.",
            timeout=4
        )

    def _smart_action(self, state: str):
        """
        Smart trigger logic based on state.
        """

        # If leaving "engaged", close the focus session
        if self.last_state == "engaged" and state != "engaged":
            self._end_focus_session_if_active(reason=f"left_engaged_to_{state}")

        if state == "drowsy":
            self.drowsy_count += 1
            self._start_break_timer()

            # Escalation if repeated drowsy detections
            if self.drowsy_count >= 3:
                self._log_and_print("ACTION=escalation_warning | message=repeated_drowsy_detected")
                self._notify(
                    "Repeated Drowsiness",
                    "You looked drowsy multiple times. Consider a longer break.",
                    timeout=6
                )

        elif state == "stressed":
            self._stress_notification()

        elif state == "engaged":
            self._start_focus_session()

        elif state in {"normal", "unknown"}:
            self.drowsy_count = 0
            self._log_and_print(f"ACTION=none | state={state}")

        else:
            self._log_and_print(f"ACTION=no_rule | state={state}")

        self.last_state = state

    def run(self, decision: dict):
        """
        decision example:
        {"state": "drowsy", "reason": "yawn_detected duration=2.0s mar=0.120"}
        """
        state = (decision or {}).get("state", "unknown")
        reason = (decision or {}).get("reason", "no_reason")

        # Main state log
        state_msg = f"STATE={state.upper()} | {reason}"
        self._log_and_print(state_msg)

        # Smart trigger action
        self._smart_action(state)

        return state_msg