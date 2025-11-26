"""
GPU内存优化工具
用于优化GPU内存使用和批处理大小
"""

import torch
import logging
import os
import multiprocessing
from pathlib import Path
from typing import Dict, Tuple, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUMemoryOptimizer:
    """GPU内存优化器"""
    
    def __init__(self):
        self.gpu_info = self._get_gpu_info()
        self.system_info = self._get_system_info()
    
    def _get_gpu_info(self) -> Dict:
        """获取GPU信息"""
        gpu_info = {
            'available': torch.cuda.is_available(),
            'count': 0,
            'devices': []
        }
        
        if torch.cuda.is_available():
            gpu_info['count'] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    'id': i,
                    'name': props.name,
                    'total_memory': props.total_memory,
                    'total_memory_gb': props.total_memory / 1024**3,
                    'major': props.major,
                    'minor': props.minor,
                    'multi_processor_count': props.multi_processor_count
                }
                gpu_info['devices'].append(device_info)
        
        return gpu_info
    
    def _get_system_info(self) -> Dict:
        """获取系统信息"""
        try:
            # 获取CPU核心数
            cpu_count = multiprocessing.cpu_count()
            
            # 尝试获取内存信息（Windows）
            try:
                import subprocess
                result = subprocess.run(['wmic', 'computersystem', 'get', 'TotalPhysicalMemory'], 
                                      capture_output=True, text=True, shell=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip() and line.strip() != 'TotalPhysicalMemory':
                            memory_total = int(line.strip())
                            break
                    else:
                        memory_total = 8 * 1024**3  # 默认8GB
                else:
                    memory_total = 8 * 1024**3  # 默认8GB
            except:
                memory_total = 8 * 1024**3  # 默认8GB
            
            return {
                'cpu_count': cpu_count,
                'memory_total': memory_total,
                'memory_total_gb': memory_total / 1024**3,
                'memory_available': memory_total * 0.8,  # 估算80%可用
                'memory_available_gb': (memory_total * 0.8) / 1024**3
            }
        except Exception as e:
            logger.warning(f"获取系统信息失败: {e}")
            return {
                'cpu_count': 4,
                'memory_total': 8 * 1024**3,
                'memory_total_gb': 8.0,
                'memory_available': 6.4 * 1024**3,
                'memory_available_gb': 6.4
            }
    
    def get_optimal_batch_size(self, model_size: str = 'yolov5s', img_size: int = 640, 
                              safety_factor: float = 0.8) -> int:
        """
        根据GPU显存计算最优批次大小
        
        Args:
            model_size: 模型大小 (yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
            img_size: 输入图像尺寸
            safety_factor: 安全系数，预留部分显存
        
        Returns:
            推荐的批次大小
        """
        if not self.gpu_info['available']:
            logger.warning("GPU不可用，返回CPU推荐批次大小")
            return 4
        
        # 选择显存最大的GPU
        best_gpu = max(self.gpu_info['devices'], key=lambda x: x['total_memory'])
        gpu_memory_gb = best_gpu['total_memory_gb']
        
        logger.info(f"使用GPU: {best_gpu['name']}, 显存: {gpu_memory_gb:.1f}GB")
        
        # 模型内存占用估算 (GB)
        model_memory = {
            'yolov5n': 0.3,
            'yolov5s': 0.5,
            'yolov5m': 1.0,
            'yolov5l': 1.8,
            'yolov5x': 3.2
        }
        
        base_memory = model_memory.get(model_size, 0.5)
        
        # 图像内存占用估算 (每张图片约占用的显存，单位GB)
        # 考虑输入图像、特征图、梯度等
        img_memory_per_batch = (img_size * img_size * 3 * 4) / 1024**3  # 输入图像
        img_memory_per_batch += img_memory_per_batch * 8  # 特征图和梯度的估算倍数
        
        # 可用显存 = 总显存 * 安全系数 - 基础模型内存
        available_memory = gpu_memory_gb * safety_factor - base_memory
        
        if available_memory <= 0:
            logger.warning(f"GPU显存不足以加载{model_size}模型")
            return 1
        
        # 计算最大批次大小
        max_batch_size = int(available_memory / img_memory_per_batch)
        
        # 限制批次大小范围
        batch_size = max(1, min(max_batch_size, 32))
        
        logger.info(f"推荐批次大小: {batch_size}")
        logger.info(f"预估显存使用: {base_memory + batch_size * img_memory_per_batch:.1f}GB")
        
        return batch_size
    
    def get_optimal_workers(self, batch_size: int) -> int:
        """
        根据系统配置计算最优工作进程数
        
        Args:
            batch_size: 批次大小
        
        Returns:
            推荐的工作进程数
        """
        cpu_count = self.system_info['cpu_count']
        
        # GPU训练时，工作进程数不宜过多
        if self.gpu_info['available']:
            # GPU训练：工作进程数 = min(CPU核心数/2, 批次大小, 8)
            workers = min(cpu_count // 2, batch_size, 8)
        else:
            # CPU训练：工作进程数 = min(CPU核心数, 批次大小)
            workers = min(cpu_count, batch_size)
        
        workers = max(1, workers)  # 至少1个工作进程
        
        logger.info(f"推荐工作进程数: {workers}")
        return workers
    
    def optimize_training_config(self, model_size: str = 'yolov5s', 
                                img_size: int = 640) -> Dict:
        """
        生成优化的训练配置
        
        Args:
            model_size: 模型大小
            img_size: 输入图像尺寸
        
        Returns:
            优化的训练配置字典
        """
        batch_size = self.get_optimal_batch_size(model_size, img_size)
        workers = self.get_optimal_workers(batch_size)
        
        config = {
            'batch_size': batch_size,
            'workers': workers,
            'img_size': img_size,
            'device': '0' if self.gpu_info['available'] else 'cpu',
            'amp': self.gpu_info['available'],  # 只在GPU上启用混合精度
            'cache': batch_size <= 16,  # 小批次时启用缓存
            'multi_scale': batch_size >= 8,  # 大批次时启用多尺度训练
        }
        
        # 根据GPU显存调整其他参数
        if self.gpu_info['available']:
            best_gpu = max(self.gpu_info['devices'], key=lambda x: x['total_memory'])
            gpu_memory_gb = best_gpu['total_memory_gb']
            
            if gpu_memory_gb < 4:
                # 小显存GPU优化
                config.update({
                    'patience': 20,  # 减少早停耐心值
                    'save_period': 20,  # 减少保存频率
                    'close_mosaic': 5,  # 更早关闭马赛克增强
                    'mosaic': 0.5,  # 减少马赛克增强
                    'mixup': 0.0,  # 关闭mixup
                    'copy_paste': 0.0,  # 关闭复制粘贴
                })
            elif gpu_memory_gb < 8:
                # 中等显存GPU优化
                config.update({
                    'patience': 30,
                    'save_period': 15,
                    'close_mosaic': 10,
                    'mosaic': 0.8,
                    'mixup': 0.1,
                    'copy_paste': 0.0,
                })
            else:
                # 大显存GPU优化
                config.update({
                    'patience': 50,
                    'save_period': 10,
                    'close_mosaic': 15,
                    'mosaic': 1.0,
                    'mixup': 0.15,
                    'copy_paste': 0.1,
                })
        
        return config
    
    def monitor_gpu_memory(self, device_id: int = 0) -> Dict:
        """
        监控GPU内存使用情况
        
        Args:
            device_id: GPU设备ID
        
        Returns:
            GPU内存使用信息
        """
        if not self.gpu_info['available']:
            return {'error': 'GPU不可用'}
        
        if device_id >= len(self.gpu_info['devices']):
            return {'error': f'GPU设备{device_id}不存在'}
        
        torch.cuda.set_device(device_id)
        
        memory_info = {
            'device_id': device_id,
            'device_name': self.gpu_info['devices'][device_id]['name'],
            'allocated': torch.cuda.memory_allocated(device_id),
            'allocated_gb': torch.cuda.memory_allocated(device_id) / 1024**3,
            'reserved': torch.cuda.memory_reserved(device_id),
            'reserved_gb': torch.cuda.memory_reserved(device_id) / 1024**3,
            'total': self.gpu_info['devices'][device_id]['total_memory'],
            'total_gb': self.gpu_info['devices'][device_id]['total_memory_gb'],
        }
        
        memory_info['usage_percent'] = (memory_info['allocated'] / memory_info['total']) * 100
        memory_info['available_gb'] = memory_info['total_gb'] - memory_info['allocated_gb']
        
        return memory_info
    
    def clear_gpu_cache(self):
        """清理GPU缓存"""
        if self.gpu_info['available']:
            torch.cuda.empty_cache()
            logger.info("GPU缓存已清理")
        else:
            logger.warning("GPU不可用，无需清理缓存")
    
    def print_optimization_report(self, model_size: str = 'yolov5s', 
                                 img_size: int = 640):
        """
        打印优化报告
        
        Args:
            model_size: 模型大小
            img_size: 输入图像尺寸
        """
        logger.info("=== GPU内存优化报告 ===")
        
        # 系统信息
        logger.info(f"系统CPU核心数: {self.system_info['cpu_count']}")
        logger.info(f"系统内存: {self.system_info['memory_total_gb']:.1f}GB")
        logger.info(f"可用内存: {self.system_info['memory_available_gb']:.1f}GB")
        
        # GPU信息
        if self.gpu_info['available']:
            logger.info(f"GPU数量: {self.gpu_info['count']}")
            for device in self.gpu_info['devices']:
                logger.info(f"GPU {device['id']}: {device['name']}, "
                          f"显存: {device['total_memory_gb']:.1f}GB, "
                          f"计算能力: {device['major']}.{device['minor']}")
        else:
            logger.warning("GPU不可用")
        
        # 优化配置
        config = self.optimize_training_config(model_size, img_size)
        logger.info("=== 推荐训练配置 ===")
        for key, value in config.items():
            logger.info(f"{key}: {value}")
        
        # GPU内存监控
        if self.gpu_info['available']:
            memory_info = self.monitor_gpu_memory()
            if 'error' not in memory_info:
                logger.info("=== GPU内存状态 ===")
                logger.info(f"已分配: {memory_info['allocated_gb']:.2f}GB")
                logger.info(f"已预留: {memory_info['reserved_gb']:.2f}GB")
                logger.info(f"总显存: {memory_info['total_gb']:.1f}GB")
                logger.info(f"使用率: {memory_info['usage_percent']:.1f}%")
                logger.info(f"可用显存: {memory_info['available_gb']:.2f}GB")

def main():
    """主函数"""
    optimizer = GPUMemoryOptimizer()
    
    # 测试不同模型大小的优化配置
    models = ['yolov5n', 'yolov5s', 'yolov5m']
    
    for model in models:
        logger.info(f"\n{'='*50}")
        logger.info(f"模型: {model}")
        optimizer.print_optimization_report(model)

if __name__ == "__main__":
    main()