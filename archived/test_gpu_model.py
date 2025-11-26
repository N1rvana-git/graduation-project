#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU模型测试脚本
测试模型在GPU上的加载和推理性能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from utils.model_loader import ModelLoader
import torch
import time
import numpy as np
from PIL import Image
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gpu_model():
    """测试GPU模型加载和推理"""
    
    print("=" * 60)
    print("GPU模型测试开始")
    print("=" * 60)
    
    # 1. 检查CUDA环境
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} (显存: {props.total_memory/1024**3:.1f}GB)")
    
    print("\n" + "=" * 60)
    
    # 2. 测试自动设备选择
    print("测试自动设备选择...")
    loader_auto = ModelLoader(device='auto')
    success = loader_auto.load_model()
    
    if success:
        print("✓ 模型加载成功")
        model_info = loader_auto.get_model_info()
        device_info = loader_auto.get_device_info()
        
        print(f"设备: {model_info['device']}")
        print(f"模型类型: {model_info['model_type']}")
        print(f"类别数: {model_info['num_classes']}")
        print(f"类别名称: {model_info['class_names']}")
        print(f"参数数量: {model_info['parameters']:,}")
        
        if device_info['device_type'] == 'cuda':
            print(f"GPU名称: {device_info['gpu_name']}")
            print(f"GPU总显存: {device_info['gpu_memory_total']/1024**3:.1f}GB")
            print(f"已分配显存: {device_info['gpu_memory_allocated']/1024**2:.1f}MB")
            print(f"已缓存显存: {device_info['gpu_memory_cached']/1024**2:.1f}MB")
    else:
        print("✗ 模型加载失败")
        return False
    
    print("\n" + "=" * 60)
    
    # 3. 性能基准测试
    print("进行性能基准测试...")
    try:
        benchmark_results = loader_auto.benchmark(num_runs=20)
        print(f"平均推理时间: {benchmark_results['mean_inference_time']*1000:.2f}ms")
        print(f"标准差: {benchmark_results['std_inference_time']*1000:.2f}ms")
        print(f"最小推理时间: {benchmark_results['min_inference_time']*1000:.2f}ms")
        print(f"最大推理时间: {benchmark_results['max_inference_time']*1000:.2f}ms")
        print(f"平均FPS: {benchmark_results['fps']:.1f}")
    except Exception as e:
        print(f"基准测试失败: {e}")
    
    print("\n" + "=" * 60)
    
    # 4. 测试实际推理
    print("测试实际推理...")
    try:
        # 创建测试图像 (使用torch tensor而不是PIL Image)
        test_tensor = torch.randn(1, 3, 640, 640).to(loader_auto.device)
        
        # 进行推理
        start_time = time.time()
        with torch.no_grad():
            results = loader_auto.get_model()(test_tensor)
        inference_time = time.time() - start_time
        
        print(f"推理时间: {inference_time*1000:.2f}ms")
        
        # 解析结果
        if hasattr(results, 'pred') and len(results.pred) > 0:
            detections = results.pred[0]  # 获取第一张图片的检测结果
            print(f"检测结果数量: {len(detections)}")
            
            if len(detections) > 0:
                print("检测到的对象:")
                for detection in detections:
                    conf = detection[4].item()
                    cls = int(detection[5].item())
                    class_name = loader_auto.get_model().names.get(cls, f'class_{cls}')
                    print(f"  - 类别: {class_name}, 置信度: {conf:.3f}")
            else:
                print("未检测到对象")
        else:
            print("推理完成，未检测到对象")
            
    except Exception as e:
        print(f"推理测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    
    # 5. 内存使用情况
    if torch.cuda.is_available():
        print("GPU内存使用情况:")
        print(f"已分配: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        print(f"已缓存: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
        print(f"最大已分配: {torch.cuda.max_memory_allocated()/1024**2:.1f}MB")
    
    # 6. 清理资源
    loader_auto.unload_model()
    print("\n模型已卸载，资源已清理")
    
    print("\n" + "=" * 60)
    print("GPU模型测试完成")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_gpu_model()