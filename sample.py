import os
import cv2
import json
import time
import numpy as np
import torch

from PIL import Image
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from rapidocr_onnxruntime import RapidOCR
from transformers import AutoProcessor, AutoModel

    
# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "sample.mp4"

VIRTUAL_FENCE_ENABLED = False

# 4-point virtual fence polygon in image coordinates.
# Points are ordered clockwise or counter-clockwise.
VIRTUAL_FENCE_POINTS = [
    (100, 100),
    (500, 100),
    (500, 400),
    (100, 400),
]

YOLO_MODEL = "yolo11n.pt"

SIGLIP_MODEL = "google/siglip-base-patch16-224"

KNOWN_FACES_DIR = "known_faces"

EVENT_DIR = "events"

DISPLAY_SIZE = (960, 540)

YOLO_CONF = 0.50

# Run expensive models periodically
FACE_INTERVAL = 5
OCR_INTERVAL = 10
SIGLIP_INTERVAL = 15

# Face recognition threshold
FACE_THRESHOLD = 0.45

# SigLIP event threshold
SIGLIP_THRESHOLD = 0.40


# ============================================================
# SETUP
# ============================================================

os.makedirs(EVENT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Device: {device}")


# ============================================================
# YOLO + BYTE TRACK
# ============================================================

print("[INFO] Loading YOLO...")

yolo = YOLO(YOLO_MODEL)


# ============================================================
# ARCFACE
# ============================================================

print("[INFO] Loading ArcFace...")

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# ============================================================
# RAPIDOCR
# ============================================================

print("[INFO] Loading OCR...")

ocr = RapidOCR()


# ============================================================
# SIGLIP
# ============================================================

print("[INFO] Loading SigLIP...")

processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)

siglip = AutoModel.from_pretrained(SIGLIP_MODEL)

siglip = siglip.to(device)

siglip.eval()


# ============================================================
# SIGLIP CLASSES
# ============================================================

SIGLIP_TEXTS = [
    "a normal person walking",
    "a person crossing a restricted area",
    "a person loitering",
    "a person carrying a suspicious bag",
    "an abandoned bag",
    "a vehicle entering a restricted area",
    "multiple people fighting",
    "a normal vehicle movement",
]


# ============================================================
# KNOWN FACE DATABASE
# ============================================================

known_faces = {}


def load_known_faces():

    if not os.path.exists(KNOWN_FACES_DIR):
        print("[INFO] No known_faces directory.")
        return

    for filename in os.listdir(KNOWN_FACES_DIR):

        path = os.path.join(
            KNOWN_FACES_DIR,
            filename
        )

        image = cv2.imread(path)

        if image is None:
            continue

        faces = face_app.get(image)

        if not faces:
            print(f"[WARNING] No face found: {filename}")
            continue

        # Use largest face
        face = max(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0])
            * (x.bbox[3] - x.bbox[1])
        )

        embedding = face.embedding

        embedding = embedding / np.linalg.norm(embedding)

        name = os.path.splitext(filename)[0]

        known_faces[name] = embedding

        print(f"[FACE] Registered: {name}")


load_known_faces()


# ============================================================
# FACE MATCHING
# ============================================================

def recognize_face(embedding):

    if not known_faces:
        return "Unknown", 0.0

    embedding = embedding / np.linalg.norm(embedding)

    best_name = "Unknown"
    best_score = -1

    for name, known_embedding in known_faces.items():

        score = np.dot(
            embedding,
            known_embedding
        )

        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= FACE_THRESHOLD:
        return best_name, float(best_score)

    return "Unknown", float(best_score)


# ============================================================
# SIGLIP CLASSIFICATION
# ============================================================

def classify_frame(frame):

    image = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    inputs = processor(
        text=SIGLIP_TEXTS,
        images=image,
        padding="max_length",
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = siglip(**inputs)

    logits = outputs.logits_per_image[0]

    # Custom softmax ranking
    probabilities = torch.softmax(
        logits,
        dim=0
    )

    index = torch.argmax(probabilities)

    label = SIGLIP_TEXTS[index]

    confidence = probabilities[index].item()

    return label, confidence, probabilities


# ============================================================
# OCR
# ============================================================

def read_text(image):

    if image is None:
        return []

    result, _ = ocr(image)

    texts = []

    if result:

        for line in result:

            try:

                text = line[1]
                score = float(line[2])

                texts.append(
                    (text, score)
                )

            except Exception:
                pass

    return texts


# ============================================================
# EVENT LOGGER
# ============================================================

def log_event(
    event_type,
    frame_number,
    confidence,
    details=None,
    frame=None
):

    timestamp = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    event = {
        "timestamp": timestamp,
        "frame": frame_number,
        "event": event_type,
        "confidence": round(
            float(confidence),
            3
        ),
        "details": details or {}
    }

    print(
        f"[ALERT] {event_type} "
        f"| confidence={confidence:.2f}"
    )

    # JSON event log
    log_path = os.path.join(
        EVENT_DIR,
        "events.jsonl"
    )

    with open(
        log_path,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(event)
            + "\n"
        )

    # Evidence image
    if frame is not None:

        filename = (
            f"{frame_number}_"
            f"{event_type.replace(' ', '_')}.jpg"
        )

        path = os.path.join(
            EVENT_DIR,
            filename
        )

        cv2.imwrite(
            path,
            frame
        )


# ============================================================
# VIRTUAL FENCE
# ============================================================

def point_in_polygon(x, y, points):
    """Ray-casting algorithm for a point inside a polygon."""
    inside = False
    n = len(points)

    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]

        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
        ):
            inside = not inside

    return inside


def segment_intersects_segment(p1, p2, p3, p4):
    """Returns True if two line segments intersect."""

    def orientation(a, b, c):
        value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(value) < 1e-9:
            return 0
        return 1 if value > 0 else -1

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0:
        return False

    if o1 * o2 <= 0 and o3 * o4 <= 0:
        return True

    return False


def box_overlaps_restricted_zone(x1, y1, x2, y2):
    """Return True when a box intersects the configured 4-point fence."""
    if not VIRTUAL_FENCE_ENABLED:
        return False

    points = VIRTUAL_FENCE_POINTS
    box_corners = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    ]

    if any(point_in_polygon(px, py, points) for px, py in box_corners):
        return True

    if any(point_in_polygon(px, py, box_corners) for px, py in points):
        return True

    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]

        for j in range(len(box_corners)):
            q1 = box_corners[j]
            q2 = box_corners[(j + 1) % len(box_corners)]

            if segment_intersects_segment(p1, p2, q1, q2):
                return True

    return False


def inside_restricted_zone(x, y):
    if not VIRTUAL_FENCE_ENABLED:
        return False
    return point_in_polygon(x, y, VIRTUAL_FENCE_POINTS)


# ============================================================
# VIDEO
# ============================================================

print("[INFO] Opening video...")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open video: {VIDEO_PATH}"
    )


frame_number = 0

last_siglip_label = ""
last_siglip_confidence = 0.0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    display = frame.copy()


    # ========================================================
    # YOLO + BYTE TRACK
    # ========================================================

    results = yolo.track(
        frame,
        tracker="bytetrack.yaml",
        persist=True,
        conf=YOLO_CONF,
        verbose=False
    )

    result = results[0]

    person_boxes = []
    vehicle_boxes = []


    if result.boxes is not None:

        boxes = result.boxes

        for i in range(len(boxes)):

            cls = int(
                boxes.cls[i].item()
            )

            confidence = float(
                boxes.conf[i].item()
            )

            x1, y1, x2, y2 = map(
                int,
                boxes.xyxy[i].tolist()
            )

            # Track ID
            track_id = None

            if boxes.id is not None:

                track_id = int(
                    boxes.id[i].item()
                )


            # COCO classes
            #
            # 0 = person
            # 2 = car
            # 3 = motorcycle
            # 5 = bus
            # 7 = truck

            if cls == 0:

                person_boxes.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        track_id
                    )
                )

            elif cls in [2, 3, 5, 7]:

                vehicle_boxes.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        track_id,
                        cls
                    )
                )


    # ========================================================
    # DRAW RESTRICTED AREA
    # ========================================================

    if VIRTUAL_FENCE_ENABLED:
        fence_points = np.array(
            VIRTUAL_FENCE_POINTS,
            dtype=np.int32
        )

        cv2.polylines(
            display,
            [fence_points],
            True,
            (0, 0, 255),
            2
        )

        cv2.putText(
            display,
            "RESTRICTED ZONE",
            (VIRTUAL_FENCE_POINTS[0][0] + 10, VIRTUAL_FENCE_POINTS[0][1] + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )


    # ========================================================
    # PERSON PROCESSING
    # ========================================================

    for (
        x1,
        y1,
        x2,
        y2,
        track_id
    ) in person_boxes:

        # -----------------------------------------------
        # Virtual fence
        # -----------------------------------------------

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        intrusion = (
            box_overlaps_restricted_zone(x1, y1, x2, y2)
            or inside_restricted_zone(cx, cy)
        )

        if intrusion:

            log_event(
                "Intrusion",
                frame_number,
                1.0,
                {
                    "track_id": track_id,
                    "object": "person"
                },
                frame
            )


        # -----------------------------------------------
        # Face recognition
        # -----------------------------------------------

        if frame_number % FACE_INTERVAL == 0:

            crop = frame[
                max(0, y1):min(frame.shape[0], y2),
                max(0, x1):min(frame.shape[1], x2)
            ]

            if crop.size > 0:

                faces = face_app.get(crop)

                for face in faces:

                    name, score = recognize_face(
                        face.embedding
                    )

                    fx1, fy1, fx2, fy2 = (
                        face.bbox.astype(int)
                    )

                    # Convert face coordinates
                    # to full-frame coordinates

                    fx1 += x1
                    fx2 += x1
                    fy1 += y1
                    fy2 += y1

                    cv2.rectangle(
                        display,
                        (fx1, fy1),
                        (fx2, fy2),
                        (255, 0, 255),
                        2
                    )

                    cv2.putText(
                        display,
                        f"{name} {score:.2f}",
                        (fx1, max(20, fy1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        2
                    )


        # Person bounding box

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = "Person"

        if track_id is not None:

            label += f" ID:{track_id}"

        cv2.putText(
            display,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


    # ========================================================
    # VEHICLE PROCESSING + OCR
    # ========================================================

    for (
        x1,
        y1,
        x2,
        y2,
        track_id,
        cls
    ) in vehicle_boxes:

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        vehicle_intrusion = (
            box_overlaps_restricted_zone(x1, y1, x2, y2)
            or inside_restricted_zone(cx, cy)
        )

        # -----------------------------------------------
        # Vehicle intrusion
        # -----------------------------------------------

        if vehicle_intrusion:

            log_event(
                "Vehicle Intrusion",
                frame_number,
                1.0,
                {
                    "track_id": track_id,
                    "object": "vehicle"
                },
                frame
            )


        # -----------------------------------------------
        # OCR
        # -----------------------------------------------

        if frame_number % OCR_INTERVAL == 0:

            vehicle_crop = frame[
                max(0, y1):min(frame.shape[0], y2),
                max(0, x1):min(frame.shape[1], x2)
            ]

            if vehicle_crop.size > 0:

                texts_found = read_text(
                    vehicle_crop
                )

                for text, score in texts_found:

                    if score > 0.40:

                        plate_text = text.strip()

                        print(
                            f"[ANPR] "
                            f"Track {track_id}: "
                            f"{plate_text} "
                            f"({score:.2f})"
                        )

                        log_event(
                            "Number Plate Read",
                            frame_number,
                            score,
                            {
                                "track_id": track_id,
                                "object": "vehicle",
                                "plate_number": plate_text,
                                "ocr_score": round(float(score), 3)
                            },
                            frame
                        )

                        cv2.putText(
                            display,
                            plate_text,
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2
                        )


        # Vehicle box

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (255, 165, 0),
            2
        )

        label = "Vehicle"

        if track_id is not None:

            label += f" ID:{track_id}"

        cv2.putText(
            display,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 165, 0),
            2
        )


    # ========================================================
    # SIGLIP
    # ========================================================

    if frame_number % SIGLIP_INTERVAL == 0:

        try:

            (
                last_siglip_label,
                last_siglip_confidence,
                _
            ) = classify_frame(frame)

            print(
                f"[SIGLIP] "
                f"{last_siglip_label} "
                f"{last_siglip_confidence:.3f}"
            )

            # Only generate an alert for suspicious
            # semantic categories.

            suspicious_words = [
                "restricted",
                "loitering",
                "suspicious",
                "abandoned",
                "fighting"
            ]

            is_suspicious = any(
                word in last_siglip_label.lower()
                for word in suspicious_words
            )

            if (
                is_suspicious
                and
                last_siglip_confidence >= SIGLIP_THRESHOLD
            ):

                log_event(
                    "Suspicious Activity",
                    frame_number,
                    last_siglip_confidence,
                    {
                        "description":
                            last_siglip_label
                    },
                    frame
                )

        except Exception as e:

            print(
                f"[SIGLIP ERROR] {e}"
            )


    # ========================================================
    # NIGHT DETECTION
    # ========================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    brightness = float(
        np.mean(gray)
    )

    is_night = brightness < 45

    if is_night:

        cv2.putText(
            display,
            "NIGHT MODE",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    # ========================================================
    # SIGLIP STATUS
    # ========================================================

    cv2.putText(
        display,
        (
            f"AI: "
            f"{last_siglip_label[:45]} "
            f"{last_siglip_confidence:.2f}"
        ),
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


    # ========================================================
    # FRAME INFO
    # ========================================================

    cv2.putText(
        display,
        f"Frame: {frame_number}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Persons: {len(person_boxes)}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Vehicles: {len(vehicle_boxes)}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    display = cv2.resize(
        display,
        DISPLAY_SIZE
    )

    cv2.imshow(
        "AI Border Surveillance",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("[INFO] Surveillance stopped.")