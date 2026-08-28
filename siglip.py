import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel


MODEL_NAME = "google/siglip-base-patch16-224"

# Load model locally
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()


# Text classes
texts = [
    "a normal person walking",
    "a person crossing a restricted area",
    "a person loitering",
    "a person carrying a suspicious bag",
    "an abandoned bag",
    "a vehicle entering a restricted area",
    "multiple people fighting",
]


def classify_frame(frame):
    image = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    inputs = processor(
        text=texts,
        images=image,
        padding="max_length",
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    # SigLIP image-text logits
    logits = outputs.logits_per_image[0]

    # Convert logits into probabilities
    probabilities = torch.softmax(logits, dim=0)

    # Highest probability
    index = torch.argmax(probabilities)

    return texts[index], probabilities[index].item(), probabilities


cap = cv2.VideoCapture("face.mpg")

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    label, confidence, probabilities = classify_frame(frame)

    print(
        f"Frame {frame_number}: "
        f"{label} ({confidence:.3f})"
    )

    # Display
    cv2.putText(
        frame,
        f"{label}: {confidence:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("SigLIP Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_number += 1


cap.release()
cv2.destroyAllWindows()