import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Configuration
FACES_DIR = "known_faces"       # Path to your folder containing face images
SAVE_PATH = "face_database.npz" # Output file for saved embeddings

# Initialize InsightFace
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

known_embeddings = []
known_names = []

# Process each image in the directory
for filename in os.listdir(FACES_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(FACES_DIR, filename)
    name = os.path.splitext(filename)[0]  # Use image filename as person name
    
    img = cv2.imread(img_path)
    if img is None:
        continue

    faces = app.get(img)

    if len(faces) == 0:
        print(f"[WARNING] No face found in {filename}")
        continue

    # Pick the largest face if multiple faces exist in the photo
    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

    # Extract and L2 normalize embedding
    embedding = face.embedding
    embedding = embedding / np.linalg.norm(embedding)

    known_embeddings.append(embedding)
    known_names.append(name)
    print(f"[REGISTERED] {name}")

# Save to a compressed NumPy format
np.savez_compressed(
    SAVE_PATH, 
    embeddings=np.array(known_embeddings), 
    names=np.array(known_names)
)

print(f"\nSuccessfully saved {len(known_names)} faces to '{SAVE_PATH}'")