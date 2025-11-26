from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import base64

app = FastAPI()

# 允许跨域（便于小程序/前端调试）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载ONNX模型（单例）
session = ort.InferenceSession(r"runs/yolov11_mask_custom_test/custom/weights/best.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# 图像预处理（根据训练时的输入尺寸和归一化方式调整）
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((640, 640))
    img_np = np.array(img).astype(np.float32)
    img_np = img_np / 255.0  # 归一化
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC->CHW
    img_np = np.expand_dims(img_np, axis=0)  # batch
    return img_np

@app.post("/api/detect_base64")
async def detect_base64(payload: dict):
    try:
        img_b64 = payload.get("image")
        if not img_b64:
            return JSONResponse({"error": "No image provided"}, status_code=400)
        image_bytes = base64.b64decode(img_b64)
        img_np = preprocess_image(image_bytes)
        outputs = session.run(None, {input_name: img_np})
        # TODO: 后处理outputs，解析检测框、类别、置信度
        # 这里只返回原始输出，实际部署需按YOLO后处理格式化
        return {"outputs": np.array(outputs).tolist()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/health")
def health():
    return {"status": "ok"}
