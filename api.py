import asyncio
import base64
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
import mimetypes
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse


ROOT = Path(__file__).resolve().parent
FRAME_PATH = ROOT / "latest_frame.jpg"
LOG_PATH = ROOT / "live_surveillance_log.json"
FACE_DATABASE_PATH = ROOT / "face_database.npz"
KNOWN_FACES_PATH = ROOT / "known_faces"
PROCESS = None


def read_frame_base64():
    if not FRAME_PATH.exists():
        raise HTTPException(status_code=404, detail="No frame has been published yet")
    return base64.b64encode(FRAME_PATH.read_bytes()).decode("ascii")


def read_logs():
    if not LOG_PATH.exists():
        return []
    try:
        return json.loads(LOG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def read_face_records():
    if not FACE_DATABASE_PATH.exists():
        raise HTTPException(status_code=404, detail="Face database not found")

    with np.load(FACE_DATABASE_PATH, allow_pickle=False) as database:
        names = [str(name) for name in database["names"].tolist()]

    records = []
    for name in names:
        image_path = next(
            (
                path
                for path in KNOWN_FACES_PATH.glob(f"{name}.*")
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ),
            None,
        )
        image_base64 = None
        content_type = None
        filename = None
        if image_path:
            image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
            content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            filename = image_path.name

        records.append({
            "name": name,
            "image_filename": filename,
            "content_type": content_type,
            "image_base64": image_base64,
        })
    return records


@asynccontextmanager
async def lifespan(app):
    global PROCESS
    PROCESS = subprocess.Popen([sys.executable, str(ROOT / "sample.py")], cwd=ROOT)
    yield
    if PROCESS and PROCESS.poll() is None:
        PROCESS.terminate()
        try:
            PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            PROCESS.kill()


app = FastAPI(title="Scope Live Surveillance API", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def client():
    return FileResponse(ROOT / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "processing": PROCESS is not None and PROCESS.poll() is None}


@app.get("/frame")
def frame():
    return {"content_type": "image/jpeg", "frame_base64": read_frame_base64()}


@app.get("/logs")
def logs():
    return {"events": read_logs()}


@app.get("/faces")
def faces():
    return {"faces": read_face_records()}


async def live_events():
    last_frame_mtime = 0
    last_log_mtime = 0
    while True:
        frame_mtime = FRAME_PATH.stat().st_mtime_ns if FRAME_PATH.exists() else 0
        log_mtime = LOG_PATH.stat().st_mtime_ns if LOG_PATH.exists() else 0
        if frame_mtime != last_frame_mtime or log_mtime != last_log_mtime:
            payload = {
                "frame_base64": read_frame_base64() if frame_mtime else None,
                "events": read_logs(),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            last_frame_mtime = frame_mtime
            last_log_mtime = log_mtime
        await asyncio.sleep(0.2)


@app.get("/live")
async def live():
    return StreamingResponse(
        live_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))