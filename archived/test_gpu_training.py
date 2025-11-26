"""
GPU训练性能和稳定性测试脚本
"""

import os
import sys
import time
import logging
from pathlib import Path
import torch
from ultralytics import YOLO

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_environment():
    """检查GPU环境"""
    logger.info("=== GPU环境检查 ===")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}")
            logger.info(f"  显存: {props.total_memory / 1024**3:.1f}GB")
            logger.info(f"  计算能力: {props.major}.{props.minor}")
            logger.info(f"  多处理器数量: {props.multi_processor_count}")
    else:
        logger.warning("CUDA不可用，将使用CPU训练")
    
    return torch.cuda.is_available()

def test_gpu_memory_management():
    """测试GPU内存管理"""
    if not torch.cuda.is_available():
        logger.warning("跳过GPU内存测试 - CUDA不可用")
        return
    
    logger.info("=== GPU内存管理测试 ===")
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    # 获取初始内存状态
    initial_allocated = torch.cuda.memory_allocated() / 1024**3
    initial_reserved = torch.cuda.memory_reserved() / 1024**3
    logger.info(f"初始GPU内存: {initial_allocated:.2f}GB / {initial_reserved:.2f}GB")
    
    # 创建一些张量测试内存分配
    logger.info("创建测试张量...")
    test_tensors = []
    for i in range(5):
        tensor = torch.randn(1000, 1000, device='cuda')
        test_tensors.append(tensor)
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"张量 {i+1}: {allocated:.2f}GB / {reserved:.2f}GB")
    
    # 清理张量
    del test_tensors
    torch.cuda.empty_cache()
    
    final_allocated = torch.cuda.memory_allocated() / 1024**3
    final_reserved = torch.cuda.memory_reserved() / 1024**3
    logger.info(f"清理后GPU内存: {final_allocated:.2f}GB / {final_reserved:.2f}GB")

def test_model_loading():
    """测试模型加载"""
    logger.info("=== 模型加载测试 ===")
    
    try:
        # 测试加载不同大小的模型
        models = ['yolov5n.pt', 'yolov5s.pt']
        
        for model_name in models:
            logger.info(f"测试加载 {model_name}...")
            start_time = time.time()
            
            model = YOLO(model_name)
            load_time = time.time() - start_time
            
            logger.info(f"{model_name} 加载时间: {load_time:.2f}秒")
            
            # 检查模型是否在GPU上
            if torch.cuda.is_available():
                device = next(model.model.parameters()).device
                logger.info(f"模型设备: {device}")
            
            # 显示GPU内存使用
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"加载后GPU内存: {allocated:.2f}GB / {reserved:.2f}GB")
            
            del model
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"模型加载测试失败: {e}")

def test_quick_training():
    """测试快速训练"""
    logger.info("=== 快速训练测试 ===")
    
    # 检查数据集是否存在
    dataset_config = Path('data/quick_dataset/dataset.yaml')
    if not dataset_config.exists():
        logger.warning(f"数据集配置文件不存在: {dataset_config}")
        logger.info("请先运行快速训练脚本创建小数据集")
        return
    
    try:
        logger.info("开始快速训练测试...")
        
        # 使用最小的模型进行测试
        model = YOLO('yolov5n.pt')
        
        # 记录开始时间
        start_time = time.time()
        
        # 显示训练前GPU内存
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"训练前GPU内存: {allocated:.2f}GB / {reserved:.2f}GB")
        
        # 开始训练（只训练1个epoch用于测试）
        results = model.train(
            data=str(dataset_config.absolute()),
            epochs=1,
            batch=4,  # 使用小批次大小
            imgsz=320,
            device='0' if torch.cuda.is_available() else 'cpu',
            workers=2,
            patience=10,
            save=True,
            cache=False,
            project='models/test_runs',
            name='gpu_test',
            exist_ok=True,
            verbose=True,
            amp=True,
        )
        
        training_time = time.time() - start_time
        logger.info(f"训练完成，耗时: {training_time:.2f}秒")
        
        # 显示训练后GPU内存
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"训练后GPU内存: {allocated:.2f}GB / {reserved:.2f}GB")
        
        # 检查模型文件是否生成
        model_path = Path('models/test_runs/gpu_test/weights/last.pt')
        if model_path.exists():
            logger.info(f"训练成功，模型已保存: {model_path}")
            logger.info(f"模型文件大小: {model_path.stat().st_size / 1024**2:.1f}MB")
        else:
            logger.error("训练失败，未找到模型文件")
        
    except Exception as e:
        logger.error(f"快速训练测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def test_inference_performance():
    """测试推理性能"""
    logger.info("=== 推理性能测试 ===")
    
    try:
        # 检查是否有训练好的模型
        model_path = Path('models/test_runs/gpu_test/weights/last.pt')
        if not model_path.exists():
            logger.warning("未找到测试模型，使用预训练模型")
            model_path = 'yolov5n.pt'
        
        logger.info(f"加载模型: {model_path}")
        model = YOLO(str(model_path))
        
        # 创建测试图像
        import numpy as np
        from PIL import Image
        
        # 创建随机测试图像
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_image)
        
        # 预热
        logger.info("预热推理...")
        for _ in range(5):
            _ = model(test_image, verbose=False)
        
        # 性能测试
        logger.info("开始性能测试...")
        num_tests = 20
        times = []
        
        for i in range(num_tests):
            start_time = time.time()
            results = model(test_image, verbose=False)
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            times.append(inference_time)
            
            if i % 5 == 0:
                logger.info(f"推理 {i+1}/{num_tests}: {inference_time:.2f}ms")
        
        # 计算统计信息
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        fps = 1000 / avg_time
        
        logger.info("=== 推理性能统计 ===")
        logger.info(f"平均推理时间: {avg_time:.2f}ms")
        logger.info(f"最小推理时间: {min_time:.2f}ms")
        logger.info(f"最大推理时间: {max_time:.2f}ms")
        logger.info(f"标准差: {std_time:.2f}ms")
        logger.info(f"平均FPS: {fps:.1f}")
        
        # 显示GPU内存使用
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"推理时GPU内存: {allocated:.2f}GB / {reserved:.2f}GB")
        
    except Exception as e:
        logger.error(f"推理性能测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    logger.info("开始GPU训练性能和稳定性测试")
    
    # 检查GPU环境
    gpu_available = check_gpu_environment()
    
    # 测试GPU内存管理
    test_gpu_memory_management()
    
    # 测试模型加载
    test_model_loading()
    
    # 测试快速训练
    test_quick_training()
    
    # 测试推理性能
    test_inference_performance()
    
    logger.info("GPU训练测试完成！")
    
    if gpu_available:
        logger.info("✅ GPU环境正常，可以进行GPU加速训练")
    else:
        logger.warning("⚠️ GPU不可用，只能使用CPU训练")

if __name__ == "__main__":
    main()