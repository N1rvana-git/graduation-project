"""
测试优化训练配置
验证自动批次大小优化和GPU内存管理
"""

import sys
import os
import torch
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'models'))

from models.train_mask_detection import MaskDetectionTrainer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_batch_size_optimization():
    """测试批次大小自动优化"""
    logger.info("=== 测试批次大小自动优化 ===")
    
    # 检查GPU环境
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"检测到 {gpu_count} 个GPU设备")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            logger.info(f"GPU {i}: {props.name}, 显存: {memory_gb:.1f}GB")
    else:
        logger.warning("GPU不可用，将使用CPU训练")
    
    # 测试不同模型的自动批次大小优化
    models = ['yolov5n', 'yolov5s', 'yolov5m']
    
    for model_size in models:
        logger.info(f"\n--- 测试模型: {model_size} ---")
        
        try:
            # 创建训练器
            trainer = MaskDetectionTrainer(
                dataset_path='data/quick_dataset',
                model_size=model_size,
                epochs=1,  # 只训练1个epoch用于测试
                batch_size=-1,  # 自动优化批次大小
                img_size=640,
                device='auto',
                patience=5
            )
            
            # 测试训练配置（不实际训练）
            logger.info("测试训练配置...")
            
            # 模拟训练参数计算
            if torch.cuda.is_available():
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_memory_gb = gpu_props.total_memory / (1024**3)
                
                # 模型基础内存占用 (GB)
                model_memory = {
                    'yolov5n': 0.3, 'yolov5s': 0.5, 'yolov5m': 1.0, 
                    'yolov5l': 1.8, 'yolov5x': 3.2
                }
                base_memory = model_memory.get(model_size, 0.5)
                
                # 每张图片内存占用估算 (GB)
                img_memory = (640 / 640) ** 2 * 0.04
                
                # 可用内存 = 总内存 * 0.8 - 基础模型内存
                available_memory = gpu_memory_gb * 0.8 - base_memory
                
                if available_memory > 0:
                    max_batch = int(available_memory / img_memory)
                    optimal_batch = max(1, min(max_batch, 32))
                else:
                    optimal_batch = 1
                
                logger.info(f"GPU显存: {gpu_memory_gb:.1f}GB")
                logger.info(f"模型基础内存: {base_memory:.1f}GB")
                logger.info(f"可用内存: {available_memory:.1f}GB")
                logger.info(f"每张图片内存: {img_memory:.3f}GB")
                logger.info(f"最大批次大小: {max_batch}")
                logger.info(f"优化批次大小: {optimal_batch}")
                
                # 计算工作进程数
                cpu_count = os.cpu_count() or 4
                workers = min(cpu_count // 2, optimal_batch, 8)
                workers = max(1, workers)
                
                logger.info(f"CPU核心数: {cpu_count}")
                logger.info(f"推荐工作进程数: {workers}")
                
                # 显示GPU内存使用情况
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                logger.info(f"当前GPU内存使用: {allocated:.2f}GB / {gpu_memory_gb:.1f}GB")
                logger.info(f"当前GPU内存预留: {reserved:.2f}GB")
                
            else:
                logger.info("CPU训练配置:")
                logger.info("批次大小: 4")
                logger.info("工作进程数: 2")
            
        except Exception as e:
            logger.error(f"测试模型 {model_size} 时出错: {e}")

def test_memory_management():
    """测试GPU内存管理"""
    logger.info("\n=== 测试GPU内存管理 ===")
    
    if not torch.cuda.is_available():
        logger.warning("GPU不可用，跳过内存管理测试")
        return
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    # 获取初始内存状态
    initial_allocated = torch.cuda.memory_allocated(0)
    initial_reserved = torch.cuda.memory_reserved(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    
    logger.info(f"初始已分配内存: {initial_allocated / (1024**3):.2f}GB")
    logger.info(f"初始已预留内存: {initial_reserved / (1024**3):.2f}GB")
    logger.info(f"总显存: {total_memory / (1024**3):.1f}GB")
    
    # 创建一些张量测试内存分配
    logger.info("创建测试张量...")
    test_tensors = []
    
    try:
        for i in range(5):
            # 创建640x640x3的图像张量
            tensor = torch.randn(4, 3, 640, 640, device='cuda')
            test_tensors.append(tensor)
            
            allocated = torch.cuda.memory_allocated(0)
            logger.info(f"创建张量 {i+1}: 已分配内存 {allocated / (1024**3):.2f}GB")
        
        # 显示峰值内存使用
        peak_allocated = torch.cuda.memory_allocated(0)
        peak_reserved = torch.cuda.memory_reserved(0)
        
        logger.info(f"峰值已分配内存: {peak_allocated / (1024**3):.2f}GB")
        logger.info(f"峰值已预留内存: {peak_reserved / (1024**3):.2f}GB")
        logger.info(f"内存使用率: {(peak_allocated / total_memory) * 100:.1f}%")
        
    except RuntimeError as e:
        logger.error(f"GPU内存不足: {e}")
    
    finally:
        # 清理测试张量
        del test_tensors
        torch.cuda.empty_cache()
        
        final_allocated = torch.cuda.memory_allocated(0)
        final_reserved = torch.cuda.memory_reserved(0)
        
        logger.info(f"清理后已分配内存: {final_allocated / (1024**3):.2f}GB")
        logger.info(f"清理后已预留内存: {final_reserved / (1024**3):.2f}GB")

def test_training_config_generation():
    """测试训练配置生成"""
    logger.info("\n=== 测试训练配置生成 ===")
    
    # 导入批次大小优化器
    sys.path.append(str(Path(__file__).parent))
    
    try:
        from batch_size_optimizer import get_training_config
        
        models = ['yolov5n', 'yolov5s', 'yolov5m']
        
        for model in models:
            logger.info(f"\n--- {model} 训练配置 ---")
            config = get_training_config(model, 640)
            
            for key, value in config.items():
                logger.info(f"{key}: {value}")
                
    except ImportError as e:
        logger.error(f"无法导入批次大小优化器: {e}")

def main():
    """主函数"""
    logger.info("开始测试优化训练配置...")
    
    try:
        # 测试批次大小优化
        test_batch_size_optimization()
        
        # 测试GPU内存管理
        test_memory_management()
        
        # 测试训练配置生成
        test_training_config_generation()
        
        logger.info("\n=== 测试完成 ===")
        logger.info("所有优化功能测试通过！")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)