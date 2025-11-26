"""Simplified FastAPI WebSocket test server (no model loading)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mask Detection Test Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_test(websocket: WebSocket):
    await websocket.accept()
    logger.info("✅ WebSocket客户端已连接")
    try:
        while True:
            raw_message = await websocket.receive_text()
            logger.info("📥 收到消息: %s...", raw_message[:100])

            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "error": "INVALID_JSON"}))
                logger.error("❌ JSON解析失败")
                continue

            msg_type = message.get("type")
            if msg_type == "ping":
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "pong",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                )
                logger.info("🏓 发送pong")
                continue

            if msg_type == "frame":
                response = {
                    "type": "result",
                    "success": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "detections": [
                        {
                            "bbox": [100, 100, 300, 300],
                            "confidence": 0.95,
                            "class_id": 1,
                            "class_name": "mask",
                            "center": [200, 200],
                            "area": 40000,
                        }
                    ],
                    "detection_count": 1,
                    "inference_time": 0.05,
                    "fps": 20.0,
                }
                await websocket.send_text(json.dumps(response, ensure_ascii=False))
                logger.info("📤 发送模拟检测结果")
                continue

            await websocket.send_text(json.dumps({"type": "error", "error": "UNSUPPORTED_MESSAGE"}))
    except Exception as exc:  # noqa: BLE001
        logger.error("❌ WebSocket错误: %s", exc)
    finally:
        logger.info("👋 WebSocket客户端断开")
        await websocket.close()


@app.get("/api/health")
async def health():
    return {"status": "healthy", "message": "测试服务器运行中"}


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("🚀 WebSocket测试服务器启动")
    print("📡 WebSocket端点: ws://127.0.0.1:5000/ws")
    print("🏥 健康检查: http://127.0.0.1:5000/api/health")
    print("=" * 60 + "\n")

    uvicorn.run("backend.test_websocket_server:app", host="0.0.0.0", port=5000, reload=False)
