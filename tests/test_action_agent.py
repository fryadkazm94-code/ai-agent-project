import sys
import os

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from agents.action_agent import ActionAgent


def run_case(agent, title, decision):
    print(f"\n--- {title} ---")
    print(f"Input decision: {decision}")
    msg = agent.run(decision)
    print(f"Returned msg : {msg}")
    return msg


def main():
    # Use a separate test log so you don't mix with real logs
    test_log_path = os.path.join(PROJECT_ROOT, "logs", "test_action_agent.log")

    # Optional: clear old test log
    if os.path.exists(test_log_path):
        os.remove(test_log_path)

    agent = ActionAgent(log_path=test_log_path)

    # Different test scenarios (normal + edge cases)
    test_cases = [
        ("Normal state", {"state": "normal", "reason": "emotion=neutral conf=90"}),
        ("Stressed state", {"state": "stressed", "reason": "emotion=sad conf=82"}),
        ("Drowsy state", {"state": "drowsy", "reason": "yawn_detected duration=2.3s mar=0.12"}),
        ("Engaged state", {"state": "engaged", "reason": "emotion=happy conf=95"}),
        ("Unknown custom state", {"state": "focused", "reason": "custom_state_from_test"}),
        ("Missing reason", {"state": "normal"}),
        ("Missing state", {"reason": "state_missing_in_input"}),
        ("Empty dict", {}),
        ("None input", None),
    ]

    results = []
    for title, decision in test_cases:
        msg = run_case(agent, title, decision)
        results.append(msg)

    # Verify log file was created and count lines
    print("\n=== Log File Check ===")
    if os.path.exists(test_log_path):
        with open(test_log_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"Log file path : {test_log_path}")
        print(f"Lines written : {len(lines)}")
        print(f"Expected      : {len(test_cases)}")

        # Show last few log lines
        print("\nLast log entries:")
        for line in lines[-5:]:
            print(line)

        if len(lines) == len(test_cases):
            print("\n✅ PASS: ActionAgent logged all test cases.")
        else:
            print("\n⚠️ WARNING: Number of log lines does not match number of test cases.")
    else:
        print("❌ FAIL: Test log file was not created.")


if __name__ == "__main__":
    main()