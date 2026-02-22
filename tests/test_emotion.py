import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read from camera")
    exit()

# DeepFace expects a face in the image; enforce_detection=False avoids hard crash
result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

if isinstance(result, list):
    result = result[0]

print("Dominant emotion:", result.get("dominant_emotion"))
print("Emotion scores:", result.get("emotion"))