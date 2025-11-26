"""FastAPI backend for the mask detection system (YOLOv11 primary pipeline)."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Ensure project root is on PYTHONPATH for absolute imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.api.detection import get_detection_api, initialize_detection_api  # noqa: E402


class Base64ImagePayload(BaseModel):
    image: str


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mask Detection FastAPI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = PROJECT_ROOT / "frontend" / "static"
TEMPLATE_DIR = PROJECT_ROOT / "frontend" / "templates"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR)) if TEMPLATE_DIR.exists() else None

UPLOAD_FOLDER = CURRENT_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

detection_api = None


def _ensure_detection_api():
    global detection_api
    if detection_api is not None:
        return detection_api

    api = get_detection_api()
    if api is not None:
        detection_api = api
        return detection_api

    if initialize_detection_api():
        detection_api = get_detection_api()
    else:
        detection_api = None
        logger.error("Failed to initialise detection API")
    return detection_api


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb_image = image.convert("RGB")
    return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)


def _decode_base64_image(image_data: str) -> np.ndarray:
    if image_data.startswith("data:image"):
        image_data = image_data.split(",", 1)[1]

    image_bytes = base64.b64decode(image_data)
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解析图像数据")
    return image


async def _run_detection(api, frame: np.ndarray) -> dict:
    return await run_in_threadpool(lambda: api.detect_image(frame, return_annotated=False))


@app.on_event("startup")
async def _startup_event():  # noqa: D401
    """Load detection API once the application boots."""
    api = _ensure_detection_api()
    if api:
        logger.info("Detection API initialised during startup")
    else:
        logger.error("Detection API failed to initialise during startup")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    return HTMLResponse("<h1>Mask detection backend is running.</h1>")


@app.get("/api/health")
async def health_check():
    api = _ensure_detection_api()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": bool(api and api.model_loader.is_loaded()),
    }


@app.post("/api/detect")
async def detect_mask(image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="没有选择文件")
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="上传的文件为空")

    try:
        pil_image = Image.open(io.BytesIO(contents))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"无法读取图像: {exc}") from exc

    api = _ensure_detection_api()
    if api is None:
        raise HTTPException(status_code=503, detail="检测服务尚未初始化")

    result = await _run_detection(api, _pil_to_bgr(pil_image))
    if "inference_time" in result:
        result["processing_time"] = result["inference_time"]
    return result


@app.post("/api/detect_base64")
async def detect_mask_base64(payload: Base64ImagePayload):
    api = _ensure_detection_api()
    if api is None:
        raise HTTPException(status_code=503, detail="检测服务尚未初始化")

    try:
        frame = _decode_base64_image(payload.image)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"无法解析Base64图像: {exc}") from exc

    result = await _run_detection(api, frame)
    if "inference_time" in result:
        result["processing_time"] = result["inference_time"]
    return result


@app.post("/api/batch_detect")
async def batch_detect(images: List[UploadFile] = File(...)):
    if not images:
        raise HTTPException(status_code=400, detail="没有上传图像文件")

    api = _ensure_detection_api()
    if api is None:
        raise HTTPException(status_code=503, detail="检测服务尚未初始化")

    results = []
    for idx, upload in enumerate(images):
        if not upload.filename:
            continue
        try:
            contents = await upload.read()
            pil_image = Image.open(io.BytesIO(contents))
            detection_result = await _run_detection(api, _pil_to_bgr(pil_image))
            if "inference_time" in detection_result:
                detection_result["processing_time"] = detection_result["inference_time"]
            detection_result["filename"] = upload.filename
            detection_result["index"] = idx
            results.append(detection_result)
        except Exception as exc:  # noqa: BLE001
            logger.error("处理文件 %s 时发生错误: %s", upload.filename, exc)
            results.append(
                {
                    "filename": upload.filename,
                    "index": idx,
                    "error": str(exc),
                }
            )

    return {
        "total_images": len(images),
        "processed_images": len(results),
        "results": results,
    }


@app.get("/api/model/info")
async def model_info():
    api = _ensure_detection_api()
    if api is None:
        raise HTTPException(status_code=503, detail="检测服务尚未初始化")
    try:
        return api.model_loader.get_model_info()
    except Exception as exc:  # noqa: BLE001
        logger.error("获取模型信息时发生错误: %s", exc)
        raise HTTPException(status_code=500, detail="获取模型信息失败") from exc


@app.websocket("/ws")
async def websocket_inference(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")
    api = _ensure_detection_api()
    if api is None:
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": "Detection API initialisation failed",
        }, ensure_ascii=False))
        await websocket.close(code=1011)
        return
    try:
        while True:
            try:
                raw_message = await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "error": "INVALID_JSON"}, ensure_ascii=False))
                continue
            msg_type = message.get("type")
            if msg_type == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False))
                continue
            if msg_type != "frame":
                await websocket.send_text(json.dumps({"type": "error", "error": "UNSUPPORTED_MESSAGE"}, ensure_ascii=False))
                continue
            frame_payload = message.get("payload")
            if not frame_payload:
                await websocket.send_text(json.dumps({"type": "error", "error": "EMPTY_PAYLOAD"}, ensure_ascii=False))
                continue
            try:
                frame = _decode_base64_image(frame_payload)
            except Exception as exc:
                await websocket.send_text(json.dumps({"type": "error", "error": f"INVALID_IMAGE: {exc}"}, ensure_ascii=False))
                continue
            start = time.perf_counter()
            try:
                result = await _run_detection(api, frame)
            except Exception as exc:
                logger.exception("WebSocket detection failed: %s", exc)
                await websocket.send_text(json.dumps({"type": "error", "error": f"DETECTION_FAILED: {exc}"}, ensure_ascii=False))
                continue
            inference_time = result.get("inference_time", time.perf_counter() - start)
            detections = result.get("detections", [])
            detection_count = result.get("detection_count", len(detections))
            response = {
                "type": "result",
                "success": result.get("success", True),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "detections": detections,
                "detection_count": detection_count,
                "inference_time": inference_time,
                "fps": (1.0 / inference_time) if inference_time else None,
            }
            response.pop("annotated_image", None)
            await websocket.send_text(json.dumps(response, ensure_ascii=False))
    except Exception as exc:
        logger.error(f"WebSocket main loop error: {exc}")
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=5000, reload=True)
