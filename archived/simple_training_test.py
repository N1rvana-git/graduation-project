"""
简化训练测试脚本
测试GPU优化后的训练配置，不依赖ultralytics
使用PyTorch直接实现简单的训练流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import logging
import os
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    """简单的CNN模型用于测试"""
    def __init__(self, num_classes=2):
        super(SimpleModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_optimal_config():
    """获取优化的训练配置"""
    try:
        from batch_size_optimizer import get_training_config
        config = get_training_config('yolov5s', 640)
        return config
    except ImportError:
        # 如果无法导入优化器，使用默认配置
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = gpu_props.total_memory / (1024**3)
            
            if gpu_memory_gb >= 4.0:
                batch_size = 16
                workers = 4
            else:
                batch_size = 8
                workers = 2
        else:
            batch_size = 4
            workers = 2
        
        return {
            'batch_size': batch_size,
            'workers': workers,
            'img_size': 640,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'amp': torch.cuda.is_available()
        }

def create_dummy_dataset(batch_size, img_size=640, num_samples=100):
    """创建虚拟数据集用于测试"""
    # 创建随机图像数据
    images = torch.randn(num_samples, 3, img_size, img_size)
    # 创建随机标签（二分类：戴口罩/不戴口罩）
    labels = torch.randint(0, 2, (num_samples,))
    
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # 在Windows上设置为0避免多进程问题
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader

def test_training_with_optimization():
    """测试优化后的训练配置"""
    logger.info("=== 测试优化训练配置 ===")
    
    # 获取优化配置
    config = get_optimal_config()
    logger.info("训练配置:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # 设置设备
    if torch.cuda.is_available() and config['device'] != 'cpu':
        if isinstance(config['device'], int):
            device = torch.device(f'cuda:{config["device"]}')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {gpu_props.name}")
        logger.info(f"显存: {gpu_props.total_memory / (1024**3):.1f}GB")
    
    # 创建模型
    model = SimpleModel(num_classes=2).to(device)
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建数据加载器
    dataloader = create_dummy_dataset(
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        num_samples=200
    )
    logger.info(f"数据批次大小: {config['batch_size']}")
    logger.info(f"数据批次数量: {len(dataloader)}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if config['amp'] and torch.cuda.is_available() else None
    
    # 训练循环
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # 记录GPU内存使用
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(0) / (1024**3)
        logger.info(f"训练前GPU内存: {initial_memory:.3f}GB")
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            # 混合精度训练
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准训练
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # 统计
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        total_correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += target.size(0)
        
        # 记录内存使用
        if torch.cuda.is_available() and batch_idx % 5 == 0:
            current_memory = torch.cuda.memory_allocated(0) / (1024**3)
            logger.info(f"批次 {batch_idx+1}/{len(dataloader)}, "
                       f"损失: {loss.item():.4f}, "
                       f"GPU内存: {current_memory:.3f}GB")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # 计算最终指标
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * total_correct / total_samples
    
    logger.info("\n=== 训练结果 ===")
    logger.info(f"平均损失: {avg_loss:.4f}")
    logger.info(f"准确率: {accuracy:.2f}%")
    logger.info(f"训练时间: {training_time:.2f}秒")
    logger.info(f"每批次平均时间: {training_time/len(dataloader):.3f}秒")
    
    if torch.cuda.is_available():
        final_memory = torch.cuda.memory_allocated(0) / (1024**3)
        peak_memory = torch.cuda.max_memory_allocated(0) / (1024**3)
        logger.info(f"最终GPU内存: {final_memory:.3f}GB")
        logger.info(f"峰值GPU内存: {peak_memory:.3f}GB")
        
        # 重置峰值内存统计
        torch.cuda.reset_peak_memory_stats(0)

def test_different_batch_sizes():
    """测试不同批次大小的性能"""
    logger.info("\n=== 测试不同批次大小性能 ===")
    
    if not torch.cuda.is_available():
        logger.warning("GPU不可用，跳过批次大小性能测试")
        return
    
    device = torch.device('cuda')
    model = SimpleModel(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    
    batch_sizes = [4, 8, 16, 32]
    results = []
    
    for batch_size in batch_sizes:
        logger.info(f"\n--- 测试批次大小: {batch_size} ---")
        
        try:
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            # 创建数据
            dataloader = create_dummy_dataset(batch_size=batch_size, num_samples=64)
            
            # 记录初始内存
            initial_memory = torch.cuda.memory_allocated(0) / (1024**3)
            
            # 测试一个批次的前向传播
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                
                for data, target in dataloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    break  # 只测试一个批次
                
                end_time = time.time()
            
            # 记录内存使用
            peak_memory = torch.cuda.max_memory_allocated(0) / (1024**3)
            inference_time = (end_time - start_time) * 1000  # 转换为毫秒
            
            results.append({
                'batch_size': batch_size,
                'inference_time_ms': inference_time,
                'peak_memory_gb': peak_memory,
                'memory_per_sample_mb': (peak_memory - initial_memory) * 1024 / batch_size
            })
            
            logger.info(f"推理时间: {inference_time:.2f}ms")
            logger.info(f"峰值内存: {peak_memory:.3f}GB")
            logger.info(f"每样本内存: {(peak_memory - initial_memory) * 1024 / batch_size:.1f}MB")
            
            # 重置内存统计
            torch.cuda.reset_peak_memory_stats(0)
            
        except RuntimeError as e:
            logger.error(f"批次大小 {batch_size} 测试失败: {e}")
            break
    
    # 显示性能对比
    logger.info("\n=== 批次大小性能对比 ===")
    logger.info("批次大小 | 推理时间(ms) | 峰值内存(GB) | 每样本内存(MB)")
    logger.info("-" * 60)
    
    for result in results:
        logger.info(f"{result['batch_size']:8d} | {result['inference_time_ms']:11.2f} | "
                   f"{result['peak_memory_gb']:11.3f} | {result['memory_per_sample_mb']:13.1f}")

def main():
    """主函数"""
    logger.info("开始简化训练测试...")
    
    try:
        # 检查环境
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA版本: {torch.version.cuda}")
            logger.info(f"GPU数量: {torch.cuda.device_count()}")
        
        # 测试优化训练配置
        test_training_with_optimization()
        
        # 测试不同批次大小性能
        test_different_batch_sizes()
        
        logger.info("\n=== 测试完成 ===")
        logger.info("简化训练测试通过！")
        
        return 0
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)