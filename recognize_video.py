import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Configuration
DATABASE_PATH = "face_database.npz"
SIMILARITY_THRESHOLD = 0.40  # Cosine similarity threshold (0.35-0.50 recommended for ArcFace)

# Load saved database
data = np.load(DATABASE_PATH)
known_embeddings = data["embeddings"]  # Shape: (N, 512)
known_names = data["names"]            # Shape: (N,)

# Initialize InsightFace
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture("face.mpg")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)

        # Normalize target embedding
        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)

        label = "Unknown"
        
        if len(known_embeddings) > 0:
            # Cosine similarity against all registered faces (matrix multiplication)
            similarities = np.dot(known_embeddings, embedding)
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]

            if best_score >= SIMILARITY_THRESHOLD:
                label = f"{known_names[best_idx]} ({best_score:.2f})"

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()