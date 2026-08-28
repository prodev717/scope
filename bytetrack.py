import cv2
from ultralytics import YOLO

# Local YOLO model
model = YOLO("yolo11n.pt")

# Open video
cap = cv2.VideoCapture("face.mpg")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO + ByteTrack
    results = model.track(
        frame,
        tracker="bytetrack.yaml",
        persist=True,
        conf=0.50,
        verbose=False
    )

    # Draw detections + tracking IDs
    annotated_frame = results[0].plot()

    annotated_frame = cv2.resize(annotated_frame, (640, 480))
    # Display using OpenCV
    cv2.imshow("Border Surveillance", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()