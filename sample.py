import os
import cv2
import json
import queue
import re
import threading
import time
from collections import Counter
import numpy as np
from insightface.app import FaceAnalysis
from rapidocr_onnxruntime import RapidOCR
from shapely.geometry import Polygon
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("SCOPE_CONFIG_PATH", os.path.join(ROOT_DIR, "scope_config.json"))


def resolve_path(path_value):
    if not path_value:
        return "car.mp4"
    if os.path.isabs(path_value):
        return path_value
    candidate = os.path.join(ROOT_DIR, path_value)
    return candidate if os.path.exists(candidate) else path_value


def load_runtime_config():
    default_points = [
        [220, 365],
        [532, 290],
        [588, 454],
        [355, 526],
    ]
    config = {
        "video_path": "car.mp4",
        "enable_virtual_fence": False,
        "fence_points": default_points,
    }

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                if isinstance(loaded.get("video_path"), str):
                    config["video_path"] = loaded["video_path"]
                if isinstance(loaded.get("enable_virtual_fence"), bool):
                    config["enable_virtual_fence"] = loaded["enable_virtual_fence"]
                if isinstance(loaded.get("fence_points"), list):
                    normalized = []
                    try:
                        for item in loaded["fence_points"]:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                normalized.append([int(float(item[0])), int(float(item[1]))])
                        if normalized:
                            config["fence_points"] = normalized
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass

    return config


CONFIG = load_runtime_config()
VIDEO_PATH = resolve_path(CONFIG["video_path"])  # Override via dashboard or set SCOPE_VIDEO=yourfile.mp4
DATABASE_PATH = os.path.join(ROOT_DIR, "face_database.npz")  # Pre-saved face embeddings file
JSON_LOG_PATH = os.path.join(ROOT_DIR, "live_surveillance_log.json")
LATEST_FRAME_PATH = os.path.join(ROOT_DIR, "latest_frame.jpg")

YOLO_MODEL = "yolo11n.pt"
PLATE_MODEL = "license-plate-finetune-v1n.pt"

DETECTION_CONF = 0.50
PLATE_CONF = 0.25
FACE_SIMILARITY_THRESHOLD = 0.40

# --- VIRTUAL FENCE CONFIGURATION ---
ENABLE_VIRTUAL_FENCE = bool(CONFIG["enable_virtual_fence"])  # Toggle Virtual Fence feature on/off
OVERLAP_THRESHOLD = 0.30     # Minimum area fraction overlap required to trigger intrusion (0.0 to 1.0)

# Define 4-point polygon coordinates (x, y) relative to raw video frame size
VIRTUAL_FENCE_PTS = np.array(CONFIG["fence_points"], np.int32)

TRACK_IMG_SIZE = 640
PLATE_IMG_SIZE = 320

OCR_COOLDOWN = 1.0
FACE_COOLDOWN = 0.5
BUFFER_FRAMES = 30
FRAME_SKIP = 1

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

# Class maps
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
PERSON_CLASS = 0
TRACK_CLASSES = [0, 2, 3, 5, 7]  # person + vehicles

# Pre-build Shapely polygon for geometric calculations
FENCE_POLYGON = Polygon(VIRTUAL_FENCE_PTS) if ENABLE_VIRTUAL_FENCE else None

# --- HARDWARE ACCELERATION SETUP ---
device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
face_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
print(f"[INFO] Using main device: {device}")

# --- MODEL INITIALIZATION ---
yolo_model = YOLO(YOLO_MODEL).to(device)
plate_model = YOLO(PLATE_MODEL).to(device)
ocr = RapidOCR()

face_app = FaceAnalysis(name="buffalo_l", providers=face_providers)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load face database
if os.path.exists(DATABASE_PATH):
    face_data = np.load(DATABASE_PATH)
    known_embeddings = face_data["embeddings"]
    known_names = face_data["names"]
    print(f"[INFO] Loaded {len(known_names)} faces from '{DATABASE_PATH}'")
else:
    known_embeddings, known_names = np.array([]), np.array([])
    print(f"[WARN] Face database '{DATABASE_PATH}' not found. Faces will log as 'Unknown'.")

# --- STATE MANAGEMENT ---
active_tracks = {}
logged_events = []
pending_ocr = set()
pending_face = set()
state_lock = threading.Lock()

ocr_queue = queue.Queue(maxsize=30)
face_queue = queue.Queue(maxsize=30)
shutdown_event = threading.Event()

# Clear live log at startup
with open(JSON_LOG_PATH, "w") as f:
    json.dump([], f)

# --- HELPER FUNCTIONS ---
def preprocess_plate(crop):
    if crop is None or crop.shape[1] < 50 or crop.shape[0] < 15:
        return None
    crop = cv2.resize(crop, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def clean_plate_text(text):
    if not text:
        return None
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    return cleaned if len(cleaned) >= 3 else None

def check_intrusion(bbox):
    """Calculates Area Overlap fraction between bounding box and Virtual Fence polygon."""
    if not ENABLE_VIRTUAL_FENCE or FENCE_POLYGON is None or not FENCE_POLYGON.is_valid:
        return False, 0.0

    x1, y1, x2, y2 = bbox
    box_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    
    if not box_polygon.is_valid or box_polygon.area == 0:
        return False, 0.0

    intersection_area = FENCE_POLYGON.intersection(box_polygon).area
    overlap_ratio = intersection_area / box_polygon.area
    
    return overlap_ratio >= OVERLAP_THRESHOLD, round(overlap_ratio, 2)

def flush_live_log_unlocked():
    """Writes current logged events directly to JSON file."""
    with open(JSON_LOG_PATH, "w") as f:
        json.dump(logged_events, f, indent=4)

def emit_logged_event(event):
    """Append to in-memory event log, print to console, and flush file."""
    logged_events.append(event)
    print(json.dumps(event, ensure_ascii=False))
    flush_live_log_unlocked()

def publish_frame(frame):
    """Publish a complete JPEG atomically so the API never reads a partial frame."""
    encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if encoded:
        temporary_path = f"{LATEST_FRAME_PATH}.tmp"
        with open(temporary_path, "wb") as output:
            output.write(buffer.tobytes())
        for attempt in range(5):
            try:
                os.replace(temporary_path, LATEST_FRAME_PATH)
                break
            except PermissionError:
                if attempt == 4:
                    try:
                        os.remove(temporary_path)
                    except OSError:
                        pass
                else:
                    time.sleep(0.01)

# --- THREAD WORKERS ---
def ocr_worker():
    while not shutdown_event.is_set():
        try:
            task = ocr_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if task is None:
            ocr_queue.task_done()
            break

        track_id, vehicle_crop = task
        try:
            res = plate_model(vehicle_crop, conf=PLATE_CONF, imgsz=PLATE_IMG_SIZE, verbose=False)[0]
            if res.boxes and len(res.boxes) > 0:
                best_idx = res.boxes.conf.argmax().item()
                px1, py1, px2, py2 = res.boxes.xyxy[best_idx].int().cpu().tolist()

                plate_crop = vehicle_crop[py1:py2, px1:px2]
                proc = preprocess_plate(plate_crop)

                if proc is not None:
                    ocr_res, _ = ocr(proc)
                    if ocr_res:
                        raw = "".join([line[1] for line in ocr_res if len(line) >= 2])
                        valid = clean_plate_text(raw)
                        if valid:
                            with state_lock:
                                if track_id in active_tracks:
                                    t = active_tracks[track_id]
                                    t["plate_history"].append(valid)
                                    if len(t["plate_history"]) > 10:
                                        t["plate_history"].pop(0)
                                    t["license_plate"] = Counter(t["plate_history"]).most_common(1)[0][0]
        except Exception:
            pass
        finally:
            with state_lock:
                pending_ocr.discard(track_id)
            ocr_queue.task_done()

def face_worker():
    while not shutdown_event.is_set():
        try:
            task = face_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if task is None:
            face_queue.task_done()
            break

        track_id, person_crop = task
        try:
            faces = face_app.get(person_crop)
            if faces:
                face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                embedding = face.embedding
                embedding = embedding / np.linalg.norm(embedding)

                matched_name = "Unknown"
                matched_score = 0.0

                if len(known_embeddings) > 0:
                    similarities = np.dot(known_embeddings, embedding)
                    best_idx = np.argmax(similarities)
                    best_score = similarities[best_idx]

                    if best_score >= FACE_SIMILARITY_THRESHOLD:
                        matched_name = known_names[best_idx]
                        matched_score = float(best_score)

                with state_lock:
                    if track_id in active_tracks:
                        t = active_tracks[track_id]
                        if matched_name != "Unknown":
                            t["face_history"].append(matched_name)
                            t["person_name"] = Counter(t["face_history"]).most_common(1)[0][0]
                            t["is_known_person"] = True
                        elif t["person_name"] is None:
                            t["person_name"] = "Unknown"
                            t["is_known_person"] = False
                        t["face_confidence"] = round(matched_score, 2)
                        if t["person_name"] not in (None, "Unknown"):
                            t["intrusion"] = False
        except Exception:
            pass
        finally:
            with state_lock:
                pending_face.discard(track_id)
            face_queue.task_done()

# Start background workers
threading.Thread(target=ocr_worker, daemon=True).start()
threading.Thread(target=face_worker, daemon=True).start()

# --- MAIN STREAM ENGINE ---
# Determine source: file or webcam
_use_webcam = False
if not os.path.exists(VIDEO_PATH):
    print(f"[WARN] Video file '{VIDEO_PATH}' not found. Trying webcam (device 0)...")
    cap = cv2.VideoCapture(0)
    _use_webcam = True
else:
    cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"[ERROR] Could not open video source. Provide a valid video file:")
    print(f"  Windows: set SCOPE_VIDEO=C:\\path\\to\\your_video.mp4 && uv run python api.py")
    print(f"  Or place a video file named 'car.mp4' in the project directory.")
    raise SystemExit(1)

frame_count, fps_start, fps_counter, current_fps = 0, time.time(), 0, 0.0

print("[INFO] Processing stream with vehicle plate, face recognition, & virtual fence...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        if _use_webcam:
            break  # Webcam disconnected
        # Loop video file back to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    now_sec = time.time()
    frame_h, frame_w = frame.shape[:2]

    results = yolo_model.track(
        frame, tracker="bytetrack.yaml", persist=True,
        conf=DETECTION_CONF, imgsz=TRACK_IMG_SIZE, classes=TRACK_CLASSES, verbose=False
    )[0]

    if results.boxes is not None and results.boxes.id is not None:
        tids = results.boxes.id.int().cpu().tolist()
        cls_idxs = results.boxes.cls.int().cpu().tolist()
        confs = results.boxes.conf.cpu().tolist()
        boxes = results.boxes.xyxy.int().cpu().tolist()

        with state_lock:
            for tid, c_idx, conf, box in zip(tids, cls_idxs, confs, boxes):
                obj_cls = results.names[c_idx]

                # Virtual Fence Intersection Check
                is_intruding, overlap_pct = check_intrusion(box)

                # Init new track entry
                if tid not in active_tracks:
                    active_tracks[tid] = {
                        "class": obj_cls, "entry_time": now_str, "last_seen_time": now_str,
                        "last_seen_frame": frame_count, "license_plate": None,
                        "plate_history": [], "last_ocr_time": 0.0,
                        "person_name": None, "face_confidence": 0.0,
                        "face_history": [], "last_face_time": 0.0,
                        "intrusion": False,
                        "max_overlap_ratio": overlap_pct,
                        "is_known_person": False
                    }
                else:
                    active_tracks[tid]["last_seen_time"] = now_str
                    active_tracks[tid]["last_seen_frame"] = frame_count
                    # Latch intrusion flag if triggered at any point during track lifecycle
                    if is_intruding and not active_tracks[tid].get("is_known_person", False):
                        active_tracks[tid]["intrusion"] = True
                    active_tracks[tid]["max_overlap_ratio"] = max(active_tracks[tid]["max_overlap_ratio"], overlap_pct)

                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(frame_w, box[2]), min(frame_h, box[3])
                crop = frame[y1:y2, x1:x2]

                # Vehicle -> Process License Plate
                if c_idx in VEHICLE_CLASSES and conf >= 0.70 and crop.size > 0:
                    track = active_tracks[tid]
                    if (not track["license_plate"] and tid not in pending_ocr and 
                            (now_sec - track["last_ocr_time"] >= OCR_COOLDOWN)):
                        pending_ocr.add(tid)
                        track["last_ocr_time"] = now_sec
                        try:
                            ocr_queue.put_nowait((tid, crop.copy()))
                        except queue.Full:
                            pending_ocr.discard(tid)

                # Person -> Process Face Recognition
                elif c_idx == PERSON_CLASS and crop.size > 0:
                    track = active_tracks[tid]
                    if (
                        track.get("person_name") in (None, "Unknown")
                        and tid not in pending_face
                        and (now_sec - track["last_face_time"] >= FACE_COOLDOWN)
                    ):
                        pending_face.add(tid)
                        track["last_face_time"] = now_sec
                        try:
                            face_queue.put_nowait((tid, crop.copy()))
                        except queue.Full:
                            pending_face.discard(tid)

    # Log & Evict Exited Tracks + Live Write to File
    with state_lock:
        missing_ids = [tid for tid, data in active_tracks.items() if frame_count - data["last_seen_frame"] > BUFFER_FRAMES]
        if missing_ids:
            for tid in missing_ids:
                info = active_tracks.pop(tid)
                pending_ocr.discard(tid)
                pending_face.discard(tid)

                plate = Counter(info["plate_history"]).most_common(1)[0][0] if info["plate_history"] else info["license_plate"]
                person = Counter(info["face_history"]).most_common(1)[0][0] if info["face_history"] else info["person_name"]

                if info["class"] == "person" and person not in (None, "Unknown"):
                    info["intrusion"] = False

                emit_logged_event({
                    "track_id": tid,
                    "class": info["class"],
                    "entry_time": info["entry_time"],
                    "exit_time": info["last_seen_time"],
                    "license_plate": plate,
                    "person_name": person,
                    "face_confidence": info["face_confidence"],
                    "intrusion": info["intrusion"],
                    "overlap_ratio": info["max_overlap_ratio"]
                })

    # Performance Monitoring & Drawing Annotations
    fps_counter += 1
    if (time.time() - fps_start) >= 1.0:
        current_fps = fps_counter / (time.time() - fps_start)
        fps_counter, fps_start = 0, time.time()

    annotated = results.plot()

    # Draw Virtual Fence Polygon on frame
    if ENABLE_VIRTUAL_FENCE:
        cv2.polylines(annotated, [VIRTUAL_FENCE_PTS], isClosed=True, color=(0, 0, 255), thickness=2)

    annotated_resized = cv2.resize(annotated, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.putText(annotated_resized, f"FPS: {current_fps:.1f} | Active Tracks: {len(active_tracks)} | Logged Events: {len(logged_events)}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    publish_frame(annotated_resized)

    cv2.imshow("Surveillance Analytics (Vehicle + Face)", annotated_resized)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- SHUTDOWN & CLEANUP ---
cap.release()
cv2.destroyAllWindows()
shutdown_event.set()

try: ocr_queue.put_nowait(None)
except queue.Full: pass
try: face_queue.put_nowait(None)
except queue.Full: pass

# Final flush of active tracks
with state_lock:
    for tid, info in active_tracks.items():
        plate = Counter(info["plate_history"]).most_common(1)[0][0] if info["plate_history"] else info["license_plate"]
        person = Counter(info["face_history"]).most_common(1)[0][0] if info["face_history"] else info["person_name"]

        if info["class"] == "person" and person not in (None, "Unknown"):
            info["intrusion"] = False

        emit_logged_event({
            "track_id": tid,
            "class": info["class"],
            "entry_time": info["entry_time"],
            "exit_time": info["last_seen_time"],
            "license_plate": plate,
            "person_name": person,
            "face_confidence": info["face_confidence"],
            "intrusion": info["intrusion"],
            "overlap_ratio": info["max_overlap_ratio"]
        })

print(f"[INFO] Processing complete. Total {len(logged_events)} events saved live to '{JSON_LOG_PATH}'.")