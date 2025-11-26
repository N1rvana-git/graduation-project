"""
GPU优化测试脚本
测试GPU内存管理和批次大小优化功能
不依赖ultralytics库
"""

import torch
import os
import logging
import sys
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_gpu_info():
    """获取GPU信息"""
    if not torch.cuda.is_available():
        return None
    
    gpu_info = {
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
        'devices': []
    }
    
    for i in range(gpu_info['device_count']):
        props = torch.cuda.get_device_properties(i)
        device_info = {
            'id': i,
            'name': props.name,
            'total_memory': props.total_memory,
            'total_memory_gb': props.total_memory / (1024**3),
            'major': props.major,
            'minor': props.minor,
            'multi_processor_count': props.multi_processor_count
        }
        gpu_info['devices'].append(device_info)
    
    return gpu_info

def calculate_optimal_batch_size(model_size='yolov5s', img_size=640, gpu_memory_gb=4.0):
    """计算最优批次大小"""
    # 模型基础内存占用 (GB)
    model_memory = {
        'yolov5n': 0.3,
        'yolov5s': 0.5, 
        'yolov5m': 1.0,
        'yolov5l': 1.8,
        'yolov5x': 3.2
    }
    
    base_memory = model_memory.get(model_size, 0.5)
    
    # 每张图片内存占用估算 (GB)
    # 基于图片尺寸和数据类型计算
    img_memory = (img_size / 640) ** 2 * 0.04
    
    # 可用内存 = 总内存 * 0.8 - 基础模型内存
    available_memory = gpu_memory_gb * 0.8 - base_memory
    
    if available_memory <= 0:
        return 1
    
    # 计算最大批次大小
    max_batch = int(available_memory / img_memory)
    
    # 限制在合理范围内
    optimal_batch = max(1, min(max_batch, 32))
    
    return optimal_batch

def calculate_optimal_workers(batch_size):
    """计算最优工作进程数"""
    cpu_count = os.cpu_count() or 4
    
    # 工作进程数通常设置为CPU核心数的一半，但不超过批次大小
    workers = min(cpu_count // 2, batch_size, 8)
    workers = max(1, workers)  # 至少1个工作进程
    
    return workers

def test_gpu_memory_allocation():
    """测试GPU内存分配"""
    if not torch.cuda.is_available():
        logger.warning("GPU不可用，跳过内存分配测试")
        return
    
    logger.info("=== GPU内存分配测试 ===")
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    # 获取初始内存状态
    initial_allocated = torch.cuda.memory_allocated(0)
    initial_reserved = torch.cuda.memory_reserved(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    
    logger.info(f"初始已分配内存: {initial_allocated / (1024**3):.3f}GB")
    logger.info(f"初始已预留内存: {initial_reserved / (1024**3):.3f}GB")
    logger.info(f"总显存: {total_memory / (1024**3):.1f}GB")
    
    # 测试不同批次大小的内存占用
    batch_sizes = [1, 2, 4, 8, 16]
    img_size = 640
    
    for batch_size in batch_sizes:
        try:
            logger.info(f"\n--- 测试批次大小: {batch_size} ---")
            
            # 创建模拟输入张量
            input_tensor = torch.randn(batch_size, 3, img_size, img_size, device='cuda')
            
            # 获取内存使用情况
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            
            logger.info(f"输入张量形状: {input_tensor.shape}")
            logger.info(f"已分配内存: {allocated / (1024**3):.3f}GB")
            logger.info(f"已预留内存: {reserved / (1024**3):.3f}GB")
            logger.info(f"内存使用率: {(allocated / total_memory) * 100:.1f}%")
            
            # 清理张量
            del input_tensor
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            logger.error(f"批次大小 {batch_size} 内存不足: {e}")
            break

def test_batch_size_optimization():
    """测试批次大小优化"""
    logger.info("\n=== 批次大小优化测试 ===")
    
    # 获取GPU信息
    gpu_info = get_gpu_info()
    
    if gpu_info is None:
        logger.warning("GPU不可用，使用CPU配置")
        logger.info("CPU训练配置:")
        logger.info("  推荐批次大小: 4")
        logger.info("  推荐工作进程数: 2")
        return
    
    # 显示GPU信息
    for device in gpu_info['devices']:
        logger.info(f"GPU {device['id']}: {device['name']}")
        logger.info(f"  显存: {device['total_memory_gb']:.1f}GB")
        logger.info(f"  计算能力: {device['major']}.{device['minor']}")
        logger.info(f"  多处理器数量: {device['multi_processor_count']}")
    
    # 测试不同模型的批次大小优化
    models = ['yolov5n', 'yolov5s', 'yolov5m']
    img_sizes = [416, 640, 832]
    
    gpu_memory_gb = gpu_info['devices'][0]['total_memory_gb']
    
    for model in models:
        logger.info(f"\n--- 模型: {model} ---")
        
        for img_size in img_sizes:
            optimal_batch = calculate_optimal_batch_size(model, img_size, gpu_memory_gb)
            optimal_workers = calculate_optimal_workers(optimal_batch)
            
            # 估算内存使用
            model_memory = {'yolov5n': 0.3, 'yolov5s': 0.5, 'yolov5m': 1.0}.get(model, 0.5)
            img_memory = (img_size / 640) ** 2 * 0.04 * optimal_batch
            total_estimated = model_memory + img_memory
            
            logger.info(f"  图片尺寸: {img_size}x{img_size}")
            logger.info(f"  推荐批次大小: {optimal_batch}")
            logger.info(f"  推荐工作进程数: {optimal_workers}")
            logger.info(f"  预估显存使用: {total_estimated:.2f}GB")
            logger.info(f"  显存使用率: {(total_estimated / gpu_memory_gb) * 100:.1f}%")

def test_training_config():
    """测试训练配置生成"""
    logger.info("\n=== 训练配置生成测试 ===")
    
    # 导入批次大小优化器
    try:
        from batch_size_optimizer import get_training_config
        
        models = ['yolov5n', 'yolov5s', 'yolov5m']
        
        for model in models:
            logger.info(f"\n--- {model} 完整训练配置 ---")
            config = get_training_config(model, 640)
            
            for key, value in config.items():
                logger.info(f"  {key}: {value}")
                
    except ImportError:
        logger.warning("无法导入batch_size_optimizer，跳过配置生成测试")

def main():
    """主函数"""
    logger.info("开始GPU优化测试...")
    
    try:
        # 检查PyTorch和CUDA版本
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA版本: {torch.version.cuda}")
            logger.info(f"cuDNN版本: {torch.backends.cudnn.version()}")
        
        # 测试GPU内存分配
        test_gpu_memory_allocation()
        
        # 测试批次大小优化
        test_batch_size_optimization()
        
        # 测试训练配置生成
        test_training_config()
        
        logger.info("\n=== 测试完成 ===")
        logger.info("GPU优化功能测试通过！")
        
        return 0
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)