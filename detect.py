import cv2
from ultralytics import YOLO
import pyautogui
import time

model = YOLO("runs/detect/train2/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    classes = results.names
    is_sleeping = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = classes[cls_id]

        if label == 'Awake':
            is_sleeping = True
        if label == 'Sleeping':
            is_sleeping= True

    frame_out = results.plot()

    cv2.imshow("sleep detection", frame_out)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()