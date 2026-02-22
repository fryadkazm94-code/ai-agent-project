# AI Multi-Agent Mood Assistant (Laptop Camera Based)

---

## Installing and Running

before all you need to install the next environment files and libraries

run in terminal:
cd C:\ai-agent-project
face_env\Scripts\python -m pip install opencv-python mediapipe plyer playsound

emotion_env\Scripts\python -m pip install tensorflow deepface opencv-python

face_env\Scripts\python final_agent.py

#running the program

to run the program, enter the next command in the terminal

cd C:\ai-agent-project
face_env\Scripts\python final_agent.py

---

## Program Explanation

After running the command, the system opens the laptop camera and starts working automatically.

The program checks the user every **30 seconds**.

1. The system first detects if a face exists in front of the camera.
   - If no face is detected → the timer resets and waits.
   - If a face appears → a 30-second measurement window starts.

2. During the 30 seconds:
   - The program monitors mouth movement to detect yawning.
   - The program analyzes facial expression to estimate emotion.

3. After the 30 seconds:
   - The information is sent to the **Decision Agent**.
   - The Decision Agent determines the user state:

   - **Drowsy** → frequent yawning detected
   - **Stressed** → negative emotions detected
   - **Engaged** → positive/focused emotion detected
   - **Normal** → none of the above

4. The **Action Agent** then performs an action:
   - shows a desktop notification
   - suggests a short break
   - or logs a focus session in the log file

The process repeats continuously while the program is running.

You can stop the program at any time by pressing **ESC**.

---

## Testing Individual Agents

If you want to test each agent separately instead of the full system, run:
cd C:\ai-agent-project
face_env\Scripts\python run_agents.py

A menu will appear allowing you to test:

- face detection
- emotion detection
- yawn detection
- decision logic
- action behavior

This shows that each agent works individually and also as part of the complete system.

---

## Project Purpose

This project demonstrates a **multi-agent software system**.  
Instead of one large program, the system is divided into smaller agents, each with a single responsibility, and they cooperate to make a final decision and action automatically.

The goal is to demonstrate system design and integration of AI tools, not to train AI models from scratch.
