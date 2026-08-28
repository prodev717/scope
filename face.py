import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Load ArcFace + face detector locally
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture("face.mpg")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect faces and generate ArcFace embeddings
    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)

        # ArcFace embedding
        embedding = face.embedding

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Display
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    frame = cv2.resize(frame, (640, 480))

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()