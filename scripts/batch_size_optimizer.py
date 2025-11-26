"""
批次大小优化工具
用于根据GPU显存自动计算最优批次大小
"""

import torch

def get_gpu_memory_info():
    """获取GPU显存信息"""
    if not torch.cuda.is_available():
        print("GPU不可用")
        return None
    
    device_count = torch.cuda.device_count()
    print(f"检测到 {device_count} 个GPU设备:")
    
    gpu_info = []
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        gpu_info.append({
            'id': i,
            'name': props.name,
            'memory_gb': memory_gb
        })
        print(f"GPU {i}: {props.name}, 显存: {memory_gb:.1f}GB")
    
    return gpu_info

def calculate_optimal_batch_size(model_size='yolov5s', img_size=640):
    """计算最优批次大小"""
    gpu_info = get_gpu_memory_info()
    
    if not gpu_info:
        print("使用CPU训练，推荐批次大小: 4")
        return 4
    
    # 选择显存最大的GPU
    best_gpu = max(gpu_info, key=lambda x: x['memory_gb'])
    memory_gb = best_gpu['memory_gb']
    
    print(f"\n使用GPU: {best_gpu['name']}")
    print(f"显存: {memory_gb:.1f}GB")
    
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
    # 640x640图片约占用: 输入(0.005GB) + 特征图和梯度(约8倍) = 0.04GB
    img_memory = (img_size / 640) ** 2 * 0.04
    
    # 可用内存 = 总内存 * 0.8 - 基础模型内存
    available_memory = memory_gb * 0.8 - base_memory
    
    if available_memory <= 0:
        print(f"警告: GPU显存不足以加载{model_size}模型")
        return 1
    
    # 计算最大批次大小
    max_batch = int(available_memory / img_memory)
    batch_size = max(1, min(max_batch, 32))  # 限制在1-32之间
    
    print(f"\n优化结果:")
    print(f"模型: {model_size}")
    print(f"图片尺寸: {img_size}x{img_size}")
    print(f"推荐批次大小: {batch_size}")
    print(f"预估显存使用: {base_memory + batch_size * img_memory:.1f}GB")
    
    return batch_size

def get_training_config(model_size='yolov5s', img_size=640):
    """获取完整的训练配置"""
    batch_size = calculate_optimal_batch_size(model_size, img_size)
    
    # 工作进程数
    import os
    cpu_count = os.cpu_count() or 4
    workers = min(cpu_count // 2, batch_size, 8) if torch.cuda.is_available() else min(cpu_count, batch_size)
    workers = max(1, workers)
    
    config = {
        'batch_size': batch_size,
        'workers': workers,
        'img_size': img_size,
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'amp': torch.cuda.is_available(),  # 混合精度
    }
    
    print(f"\n完整训练配置:")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    return config

if __name__ == "__main__":
    print("=== GPU内存优化工具 ===\n")
    
    # 测试不同模型
    models = ['yolov5n', 'yolov5s', 'yolov5m']
    
    for model in models:
        print(f"\n{'='*50}")
        print(f"测试模型: {model}")
        get_training_config(model)
        print()