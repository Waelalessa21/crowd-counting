from ultralytics import YOLO
import cv2
import numpy as np

class CrowdCountingModel:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def match_person(self, centroid, tracked, threshold=50):
        for pid, data in tracked.items():
            prev = data["centroid"]
            if np.linalg.norm(np.array(centroid) - np.array(prev)) < threshold:
                return pid
        return None

    def process(self, video_path, state):
        cap = cv2.VideoCapture(video_path)
        frame_id = 0

        if "tracked_people" not in state:
            state.tracked_people = {}
        if "next_id" not in state:
            state.next_id = 0
        if "total_unique_people" not in state:
            state.total_unique_people = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            results = self.model(frame, verbose=False)[0]
            detections = []

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                    detections.append((x1, y1, x2, y2, centroid))

            for x1, y1, x2, y2, centroid in detections:
                matched_id = self.match_person(centroid, state["tracked_people"])

                if matched_id is not None:
                    person = state["tracked_people"][matched_id]
                    person["centroid"] = centroid
                    person["last_seen"] = frame_id
                else:
                    state["tracked_people"][state["next_id"]] = {
                        "centroid": centroid,
                        "last_seen": frame_id
                    }
                    state["next_id"] += 1
                    state["total_unique_people"] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            state["tracked_people"] = {
                pid: data for pid, data in state["tracked_people"].items()
                if frame_id - data["last_seen"] < 60
            }

            cv2.putText(frame, f"People Seen: {state['total_unique_people']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)

            yield frame

        cap.release()
