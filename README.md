# ai-agent-project

# AI Multi-Agent Mood Assistant (Laptop Camera Based)

## Project Idea (Simple Explanation)

This project is a **multi-agent system** made of smaller agents (sub-agents) that work together.

The system uses the **laptop camera** to check the user every **30 seconds** and then decides what action to take.

It is designed to feel like a simple smart assistant:

- If the user looks **drowsy** (yawning a lot) → suggest a break
- If the user looks **stressed** → send a calming notification
- If the user looks **engaged/focused** → log a focus session

---

## Why This Is a Multi-Agent System

Instead of one big program doing everything, we split the work into **5 agents**, each with one responsibility:

1. **Face Detection Agent**  
   Detects if a face is visible in the camera.

2. **Emotion Agent**  
   Analyzes facial expression (happy, neutral, angry, etc.).

3. **Yawn Detection Agent**  
   Detects yawning / mouth-open duration.

4. **Decision Agent**  
   Combines emotion + yawn results and decides the final state:
   - `drowsy`
   - `stressed`
   - `engaged`
   - `normal`

5. **Action Agent**  
   Takes action based on the decision:
   - Notifications
   - Break timer
   - Logging focus sessions

---

## How the Final System Works (Step by Step)

### Main Logic (`final_agent.py`)

1. Open the webcam
2. Check if a face is visible
3. If **no face**:
   - reset the 30-second timer to 0
   - reset current measurements
4. If **face is detected**:
   - start a **30-second measurement window**
   - continuously track yawning
   - sample emotion every few seconds
5. After 30 seconds:
   - run the **Decision Agent**
   - run the **Action Agent**
6. Start a new 30-second cycle (if the face is still present)

---

## Project Folder Structure

```text
ai-agent-project/
│
├── agents/
│   ├── sensor_agent.py        # Face detection
│   ├── analysis_agent.py      # Emotion analysis
│   ├── yawn_agent.py          # Yawn detection
│   ├── decision_agent.py      # Decision logic
│   └── action_agent.py        # Notifications / actions
│
├── tests/
│   ├── run_emotion_agent.py
│   ├── run_yawn_agent.py
│   ├── test_decision_agent.py
│   ├── test_action_agent.py
│   └── emotion_worker.py      # Runs emotion analysis in emotion_env
│
├── logs/
│   └── events.log             # System event logs
│
├── face_env/                  # Virtual environment for camera + MediaPipe
├── emotion_env/               # Virtual environment for DeepFace/TensorFlow
│
├── run_agents.py              # Menu launcher for testing sub-agents
├── final_agent.py             # Final integrated multi-agent system
└── README.md

<!--  -->


to run the project, first of all install the project files, and put them in partition c
then run the next codes in the visual studio terminal

cd C:\ai-agent-project
face_env\Scripts\python final_agent.py

if you want to test the agents agent by agent, we already prepared a file for that task
which is called tests, you run the next command in the terminal and it will
provide you with the tests options

cd C:\ai-agent-project
face_env\Scripts\python run_agents.py
```
