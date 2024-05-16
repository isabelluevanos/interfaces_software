import threading
from queue import Queue

import cv2
import numpy as np
import torch
from sort import Sort

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
frame_queue = Queue()
result_queue = Queue()


def object_detection():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        results = model(frame)
        results = results.xyxy[0].numpy()
        people_det = results[results[:, 5] == 0]

        dets = []
        for *xyxy, conf, cls in people_det:
            x1, y1, x2, y2 = map(int, xyxy)
            dets.append([x1, y1, x2, y2, conf])
        dets = np.array(dets)

        result_queue.put(dets)
    result_queue.put(None)


def tracking_and_display():
    mov_tracker = Sort()
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        dets = result_queue.get()
        if dets is None:
            break

        trackers = mov_tracker.update(dets)

        for d in trackers:
            x1, y1, x2, y2, track_id = map(int, d[:5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                str(track_id),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                thickness=2,
            )

        cv2.imshow("view", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()


def main():
    cap = cv2.VideoCapture(r"test.mp4")

    detection_thread = threading.Thread(target=object_detection)
    display_thread = threading.Thread(target=tracking_and_display)

    detection_thread.start()
    display_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)
            break
        frame_queue.put(frame)

    cap.release()
    detection_thread.join()
    display_thread.join()


if __name__ == "__main__":
    main()
