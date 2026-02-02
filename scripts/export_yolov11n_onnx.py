import sys
import os
import warnings

# 1. 抑制无关警告
warnings.filterwarnings("ignore")

# 2. 确保能找到项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

try:
    from ultralytics import YOLO
    import ultralytics.nn.modules.block
    import ultralytics.nn.tasks
    
    # === [关键] 注册自定义模块 ===
    # 必须导入并注册 GAMAttention，否则加载权重时会报错
    from models.modules.attention import GAMAttention
    setattr(ultralytics.nn.modules.block, 'GAMAttention', GAMAttention)
    setattr(ultralytics.nn.tasks, 'GAMAttention', GAMAttention)
    print("✅ [System] Custom module 'GAMAttention' registered successfully.")
    
except ImportError as e:
    print(f"❌ [Error] Failed to register custom modules: {e}")
    sys.exit(1)

def export_model():
    # 指向你训练好的最佳权重路径 (根据你的 project/name 参数调整)
    # 假设你现在的 project='runs/yolov11_mask_detection' name='custom_v2_accum'
    model_path = os.path.join(project_root, 'runs/yolov11_mask_detection/custom_v2_accum/weights/best.pt')
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found at {model_path}, please check the path.")
        return

    print(f"🚀 Loading model from {model_path}...")
    # 加载模型 (会自动使用上面注册的 GAMAttention)
    model = YOLO(model_path)
    
    # 导出为 ONNX (针对微信小程序优化)
    print("📦 Starting ONNX export...")
    success = model.export(
        format='onnx',
        imgsz=640,
        opset=12,       # 微信/移动端兼容性最佳
        simplify=True,  # 简化图结构
        dynamic=False   # 静态输入对小程序更友好
    )
    print(f"🎉 Export Success: {success}")

if __name__ == "__main__":
    export_model()
