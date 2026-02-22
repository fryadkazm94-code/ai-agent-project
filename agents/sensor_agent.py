import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection


class FaceDetectionAgent:
    def __init__(self):
        self.fd = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

    def run(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results = self.fd.process(frame_rgb)

        if not results.detections:
            return None

        detection = max(results.detections, key=lambda d: d.score[0])
        bbox = detection.location_data.relative_bounding_box

        h, w, _ = frame_bgr.shape

        x = max(int(bbox.xmin * w), 0)
        y = max(int(bbox.ymin * h), 0)
        bw = max(int(bbox.width * w), 1)
        bh = max(int(bbox.height * h), 1)

        return (x, y, bw, bh)


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not found")
        exit()

    agent = FaceDetectionAgent()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = agent.run(frame)

        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Face Detection Agent", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()