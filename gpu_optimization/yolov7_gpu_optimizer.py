"""
YOLOv7 GPU优化器
专门为YOLOv7模型提供GPU优化功能
"""

import torch
import psutil
import GPUtil
import numpy as np
from typing import Dict, List, Tuple, Any
import json
import os
from pathlib import Path


class YOLOv7GPUOptimizer:
    """YOLOv7 GPU优化器"""
    
    def __init__(self):
        """初始化优化器"""
        self.gpu_info = self._get_gpu_info()
        self.system_info = self._get_system_info()
        
        # YOLOv7模型配置
        self.model_configs = {
            'yolov7-tiny': {
                'base_memory': 1.2,  # GB
                'params': 6.2,       # M
                'gflops': 13.8,
                'recommended_batch': 32
            },
            'yolov7': {
                'base_memory': 2.8,
                'params': 37.2,
                'gflops': 104.7,
                'recommended_batch': 16
            },
            'yolov7x': {
                'base_memory': 4.2,
                'params': 71.3,
                'gflops': 189.9,
                'recommended_batch': 8
            },
            'yolov7-w6': {
                'base_memory': 6.8,
                'params': 70.4,
                'gflops': 360.0,
                'recommended_batch': 6
            },
            'yolov7-e6': {
                'base_memory': 8.5,
                'params': 97.2,
                'gflops': 515.2,
                'recommended_batch': 4
            },
            'yolov7-d6': {
                'base_memory': 10.2,
                'params': 133.8,
                'gflops': 702.6,
                'recommended_batch': 3
            },
            'yolov7-e6e': {
                'base_memory': 12.8,
                'params': 151.7,
                'gflops': 843.2,
                'recommended_batch': 2
            }
        }
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """获取GPU信息"""
        gpu_info = {
            'available': torch.cuda.is_available(),
            'count': 0,
            'devices': []
        }
        
        if gpu_info['available']:
            gpu_info['count'] = torch.cuda.device_count()
            
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    device_info = {
                        'id': i,
                        'name': gpu.name,
                        'memory_total': gpu.memoryTotal,
                        'memory_used': gpu.memoryUsed,
                        'memory_free': gpu.memoryFree,
                        'memory_util': gpu.memoryUtil,
                        'gpu_util': gpu.load,
                        'temperature': gpu.temperature
                    }
                    gpu_info['devices'].append(device_info)
            except:
                # 如果GPUtil失败，使用PyTorch获取基本信息
                for i in range(gpu_info['count']):
                    props = torch.cuda.get_device_properties(i)
                    device_info = {
                        'id': i,
                        'name': props.name,
                        'memory_total': props.total_memory // (1024**2),  # MB
                        'memory_used': 0,
                        'memory_free': props.total_memory // (1024**2),
                        'memory_util': 0.0,
                        'gpu_util': 0.0,
                        'temperature': 0
                    }
                    gpu_info['devices'].append(device_info)
        
        return gpu_info
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total // (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available // (1024**3)
        }
    
    def calculate_optimal_batch_size(self, 
                                   model_name: str = 'yolov7',
                                   img_size: int = 640,
                                   gpu_id: int = 0,
                                   safety_factor: float = 0.8) -> Dict[str, Any]:
        """
        计算最优批次大小
        
        Args:
            model_name: 模型名称
            img_size: 输入图像尺寸
            gpu_id: GPU设备ID
            safety_factor: 安全系数
            
        Returns:
            优化结果
        """
        if not self.gpu_info['available'] or gpu_id >= len(self.gpu_info['devices']):
            return {
                'recommended_batch_size': 1,
                'max_batch_size': 1,
                'memory_usage': 0,
                'warning': 'GPU不可用或设备ID无效，使用CPU模式'
            }
        
        gpu_device = self.gpu_info['devices'][gpu_id]
        available_memory = gpu_device['memory_free']  # MB
        
        # 获取模型配置
        model_config = self.model_configs.get(model_name, self.model_configs['yolov7'])
        
        # 计算内存需求
        # 基础内存 + 图像内存 + 梯度内存
        base_memory = model_config['base_memory'] * 1024  # MB
        
        # 图像内存计算 (batch_size * 3 * H * W * 4 bytes * 2 for forward+backward)
        img_memory_per_batch = (3 * img_size * img_size * 4 * 2) / (1024**2)  # MB
        
        # 计算最大批次大小
        available_for_batch = (available_memory * safety_factor) - base_memory
        max_batch_size = max(1, int(available_for_batch / img_memory_per_batch))
        
        # 推荐批次大小（考虑性能平衡）
        recommended_batch = min(max_batch_size, model_config['recommended_batch'])
        
        # 内存使用估算
        estimated_memory = base_memory + (recommended_batch * img_memory_per_batch)
        
        return {
            'model_name': model_name,
            'img_size': img_size,
            'gpu_id': gpu_id,
            'gpu_name': gpu_device['name'],
            'available_memory_mb': available_memory,
            'recommended_batch_size': recommended_batch,
            'max_batch_size': max_batch_size,
            'estimated_memory_mb': estimated_memory,
            'memory_utilization': estimated_memory / gpu_device['memory_total'],
            'model_params_m': model_config['params'],
            'model_gflops': model_config['gflops']
        }
    
    def optimize_training_config(self, 
                               model_name: str = 'yolov7',
                               img_size: int = 640,
                               epochs: int = 100,
                               gpu_id: int = 0) -> Dict[str, Any]:
        """
        优化训练配置
        
        Args:
            model_name: 模型名称
            img_size: 输入图像尺寸
            epochs: 训练轮数
            gpu_id: GPU设备ID
            
        Returns:
            优化的训练配置
        """
        # 计算最优批次大小
        batch_result = self.calculate_optimal_batch_size(model_name, img_size, gpu_id)
        
        # 基础配置
        config = {
            'model': model_name,
            'img_size': img_size,
            'epochs': epochs,
            'batch_size': batch_result['recommended_batch_size'],
            'device': gpu_id if self.gpu_info['available'] else 'cpu',
            'workers': min(8, self.system_info['cpu_count']),
            'amp': True,  # 自动混合精度
            'multi_scale': False,
            'rect': False,
            'cache_images': False,
            'image_weights': False,
            'single_cls': False,
            'adam': False,
            'sync_bn': False,
            'linear_lr': False,
            'label_smoothing': 0.0,
            'patience': 100,
            'freeze': [0],
            'save_period': -1
        }
        
        # 根据GPU内存调整配置
        if self.gpu_info['available'] and gpu_id < len(self.gpu_info['devices']):
            gpu_memory = self.gpu_info['devices'][gpu_id]['memory_total']
            
            if gpu_memory >= 16000:  # 16GB+
                config.update({
                    'cache_images': True,
                    'multi_scale': True,
                    'workers': min(16, self.system_info['cpu_count'])
                })
            elif gpu_memory >= 8000:  # 8GB+
                config.update({
                    'multi_scale': True,
                    'workers': min(12, self.system_info['cpu_count'])
                })
            elif gpu_memory < 4000:  # <4GB
                config.update({
                    'amp': True,
                    'workers': min(4, self.system_info['cpu_count']),
                    'batch_size': min(config['batch_size'], 8)
                })
        
        # 添加优化信息
        config['optimization_info'] = {
            'gpu_info': batch_result,
            'system_info': self.system_info,
            'recommendations': self._get_training_recommendations(config)
        }
        
        return config
    
    def _get_training_recommendations(self, config: Dict[str, Any]) -> List[str]:
        """获取训练建议"""
        recommendations = []
        
        if config['device'] == 'cpu':
            recommendations.append("建议使用GPU加速训练")
        
        if config['batch_size'] < 8:
            recommendations.append("批次大小较小，可能影响训练稳定性")
        
        if config['amp']:
            recommendations.append("已启用自动混合精度，可提升训练速度")
        
        if config['cache_images']:
            recommendations.append("已启用图像缓存，首次加载较慢但后续训练更快")
        
        if config['multi_scale']:
            recommendations.append("已启用多尺度训练，可提升模型鲁棒性")
        
        return recommendations
    
    def benchmark_models(self, 
                        models: List[str] = None,
                        img_size: int = 640,
                        gpu_id: int = 0) -> Dict[str, Any]:
        """
        模型性能基准测试
        
        Args:
            models: 要测试的模型列表
            img_size: 输入图像尺寸
            gpu_id: GPU设备ID
            
        Returns:
            基准测试结果
        """
        if models is None:
            models = ['yolov7-tiny', 'yolov7', 'yolov7x']
        
        results = {}
        
        for model_name in models:
            if model_name in self.model_configs:
                batch_result = self.calculate_optimal_batch_size(model_name, img_size, gpu_id)
                model_config = self.model_configs[model_name]
                
                results[model_name] = {
                    'parameters_m': model_config['params'],
                    'gflops': model_config['gflops'],
                    'base_memory_gb': model_config['base_memory'],
                    'recommended_batch': batch_result['recommended_batch_size'],
                    'max_batch': batch_result['max_batch_size'],
                    'estimated_memory_mb': batch_result.get('estimated_memory_mb', 0),
                    'memory_efficiency': batch_result.get('memory_utilization', 0)
                }
        
        return {
            'benchmark_results': results,
            'gpu_info': self.gpu_info,
            'system_info': self.system_info,
            'test_config': {
                'img_size': img_size,
                'gpu_id': gpu_id
            }
        }
    
    def save_optimization_report(self, 
                               config: Dict[str, Any],
                               save_path: str = "gpu_optimization_report.json"):
        """保存优化报告"""
        report = {
            'timestamp': torch.cuda.Event(enable_timing=True).record(),
            'optimization_config': config,
            'gpu_info': self.gpu_info,
            'system_info': self.system_info
        }
        
        # 确保保存目录存在
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"优化报告已保存到: {save_path}")
    
    def get_memory_usage(self, gpu_id: int = 0) -> Dict[str, Any]:
        """获取GPU内存使用情况"""
        if not self.gpu_info['available'] or gpu_id >= len(self.gpu_info['devices']):
            return {'error': 'GPU不可用或设备ID无效'}
        
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**2)  # MB
            cached = torch.cuda.memory_reserved(gpu_id) / (1024**2)  # MB
            
            return {
                'gpu_id': gpu_id,
                'allocated_mb': allocated,
                'cached_mb': cached,
                'total_mb': self.gpu_info['devices'][gpu_id]['memory_total'],
                'free_mb': self.gpu_info['devices'][gpu_id]['memory_free'],
                'utilization': allocated / self.gpu_info['devices'][gpu_id]['memory_total']
            }
        
        return {'error': 'PyTorch CUDA不可用'}


def main():
    """测试函数"""
    print("YOLOv7 GPU优化器测试")
    print("=" * 50)
    
    # 创建优化器
    optimizer = YOLOv7GPUOptimizer()
    
    # 显示GPU信息
    print("\nGPU信息:")
    if optimizer.gpu_info['available']:
        for device in optimizer.gpu_info['devices']:
            print(f"  GPU {device['id']}: {device['name']}")
            print(f"    内存: {device['memory_total']}MB (可用: {device['memory_free']}MB)")
            print(f"    利用率: GPU {device['gpu_util']:.1f}%, 内存 {device['memory_util']:.1f}%")
    else:
        print("  无可用GPU")
    
    # 测试不同模型的优化配置
    models = ['yolov7-tiny', 'yolov7', 'yolov7x']
    
    print(f"\n模型优化配置:")
    for model in models:
        print(f"\n{model}:")
        config = optimizer.optimize_training_config(model_name=model)
        print(f"  推荐批次大小: {config['batch_size']}")
        print(f"  预估内存使用: {config['optimization_info']['gpu_info'].get('estimated_memory_mb', 0):.1f}MB")
        print(f"  工作进程数: {config['workers']}")
        print(f"  自动混合精度: {config['amp']}")
    
    # 基准测试
    print(f"\n模型基准测试:")
    benchmark = optimizer.benchmark_models()
    for model, result in benchmark['benchmark_results'].items():
        print(f"\n{model}:")
        print(f"  参数量: {result['parameters_m']:.1f}M")
        print(f"  计算量: {result['gflops']:.1f} GFLOPs")
        print(f"  推荐批次: {result['recommended_batch']}")
        print(f"  最大批次: {result['max_batch']}")
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()