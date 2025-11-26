"""
简化的口罩检测模型训练脚本
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(data_config: str, epochs: int = 10, batch_size: int = 8, img_size: int = 640):
    """训练模型"""
    try:
        # 导入ultralytics
        from ultralytics import YOLO
        import torch
        
        logger.info("开始训练口罩检测模型...")
        logger.info(f"数据配置: {data_config}")
        logger.info(f"训练轮数: {epochs}")
        logger.info(f"批次大小: {batch_size}")
        logger.info(f"图像尺寸: {img_size}")
        
        # 检查CUDA
        device = '0' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {device}")
        
        # 创建输出目录
        output_dir = Path('models/runs')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 尝试加载预训练模型
        try:
            logger.info("尝试加载YOLOv5s预训练模型...")
            model = YOLO('yolov5s.pt')
        except Exception as e:
            logger.warning(f"加载预训练模型失败: {e}")
            logger.info("使用随机初始化的模型...")
            model = YOLO('yolov5s.yaml')
        
        # 开始训练
        logger.info("开始训练...")
        results = model.train(
            data=data_config,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=str(output_dir),
            name='mask_detection',
            exist_ok=True,
            verbose=True,
            patience=20,
            save=True,
            cache=False,  # 不缓存以节省内存
            workers=4,    # 减少工作进程
            amp=False,    # 关闭混合精度以避免问题
        )
        
        logger.info("训练完成！")
        
        # 获取模型路径
        best_model = output_dir / 'mask_detection' / 'weights' / 'best.pt'
        last_model = output_dir / 'mask_detection' / 'weights' / 'last.pt'
        
        logger.info(f"最佳模型: {best_model}")
        logger.info(f"最后模型: {last_model}")
        
        return str(best_model), str(last_model)
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def validate_model(model_path: str, data_config: str):
    """验证模型"""
    try:
        from ultralytics import YOLO
        import torch
        
        logger.info(f"验证模型: {model_path}")
        
        device = '0' if torch.cuda.is_available() else 'cpu'
        model = YOLO(model_path)
        
        results = model.val(
            data=data_config,
            device=device,
            verbose=True
        )
        
        logger.info("验证完成！")
        logger.info(f"mAP50: {results.box.map50:.4f}")
        logger.info(f"mAP50-95: {results.box.map:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"验证失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='简化口罩检测模型训练')
    parser.add_argument('--data', required=True, help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--validate', action='store_true', help='训练后进行验证')
    
    args = parser.parse_args()
    
    # 检查数据配置文件
    if not os.path.exists(args.data):
        logger.error(f"数据配置文件不存在: {args.data}")
        return
    
    # 训练模型
    best_model, last_model = train_model(
        args.data, 
        args.epochs, 
        args.batch_size, 
        args.img_size
    )
    
    if best_model and args.validate:
        validate_model(best_model, args.data)
    
    print(f"\n训练完成！")
    if best_model:
        print(f"最佳模型: {best_model}")
        print(f"最后模型: {last_model}")

if __name__ == "__main__":
    main()