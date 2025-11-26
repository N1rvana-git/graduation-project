"""
快速训练脚本 - 使用小数据集子集进行快速验证
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
import random

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_small_dataset(original_data_dir: str, output_dir: str, sample_size: int = 1000):
    """创建小数据集用于快速训练"""
    try:
        original_path = Path(original_data_dir)
        output_path = Path(output_dir)
        
        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        for split in ['train', 'val', 'test']:
            (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # 复制训练集的一小部分
        train_images = list((original_path / 'images' / 'train').glob('*.jpg'))
        if len(train_images) > sample_size:
            selected_images = random.sample(train_images, sample_size)
        else:
            selected_images = train_images
        
        logger.info(f"从 {len(train_images)} 张图片中选择 {len(selected_images)} 张用于快速训练")
        
        # 复制选中的图片和标签
        for img_path in selected_images:
            # 复制图片
            dst_img = output_path / 'images' / 'train' / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # 复制对应的标签文件
            label_name = img_path.stem + '.txt'
            src_label = original_path / 'labels' / 'train' / label_name
            if src_label.exists():
                dst_label = output_path / 'labels' / 'train' / label_name
                shutil.copy2(src_label, dst_label)
        
        # 复制验证集的一小部分
        val_images = list((original_path / 'images' / 'val').glob('*.jpg'))
        val_sample_size = min(200, len(val_images))
        if val_images:
            selected_val_images = random.sample(val_images, val_sample_size)
            
            for img_path in selected_val_images:
                # 复制图片
                dst_img = output_path / 'images' / 'val' / img_path.name
                shutil.copy2(img_path, dst_img)
                
                # 复制对应的标签文件
                label_name = img_path.stem + '.txt'
                src_label = original_path / 'labels' / 'val' / label_name
                if src_label.exists():
                    dst_label = output_path / 'labels' / 'val' / label_name
                    shutil.copy2(src_label, dst_label)
        
        # 创建数据集配置文件
        config_content = f"""# 快速训练数据集配置
path: {output_dir}
train: images/train
val: images/val
test: images/test

# 类别数量
nc: 2

# 类别名称
names:
  0: no_mask
  1: mask
"""
        
        config_path = output_path / 'dataset.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"小数据集创建完成: {output_dir}")
        logger.info(f"训练样本: {len(selected_images)}")
        logger.info(f"验证样本: {len(selected_val_images) if val_images else 0}")
        
        return str(config_path)
        
    except Exception as e:
        logger.error(f"创建小数据集失败: {e}")
        return None

def quick_train(data_config: str, epochs: int = 3, batch_size: int = 8):
    """快速训练"""
    try:
        from ultralytics import YOLO
        import torch
        
        logger.info("开始快速训练...")
        logger.info(f"数据配置: {data_config}")
        logger.info(f"训练轮数: {epochs}")
        logger.info(f"批次大小: {batch_size}")
        
        # 优化GPU设备选择
        if torch.cuda.is_available():
            # 选择显存最大的GPU
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                max_memory = 0
                best_gpu = 0
                for i in range(gpu_count):
                    memory = torch.cuda.get_device_properties(i).total_memory
                    if memory > max_memory:
                        max_memory = memory
                        best_gpu = i
                device = str(best_gpu)
                logger.info(f"选择GPU {best_gpu}，显存: {max_memory / 1024**3:.1f}GB")
            else:
                device = '0'
                
            # 显示GPU信息
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}, 显存: {props.total_memory / 1024**3:.1f}GB")
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            # 根据GPU显存自动调整批次大小
            gpu_id = int(device) if device.isdigit() else 0
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            
            # 根据显存大小建议批次大小
            if gpu_memory < 4:
                suggested_batch = min(batch_size, 4)
                logger.warning(f"GPU显存较小({gpu_memory:.1f}GB)，调整批次大小为: {suggested_batch}")
                batch_size = suggested_batch
            elif gpu_memory < 8:
                suggested_batch = min(batch_size, 8)
                if batch_size > suggested_batch:
                    logger.warning(f"GPU显存适中({gpu_memory:.1f}GB)，调整批次大小为: {suggested_batch}")
                    batch_size = suggested_batch
            else:
                logger.info(f"GPU显存充足({gpu_memory:.1f}GB)，使用原批次大小: {batch_size}")
        else:
            device = 'cpu'
            logger.warning("CUDA不可用，使用CPU训练")
        
        logger.info(f"使用设备: {device}")
        
        # 创建输出目录
        output_dir = Path('models/quick_runs')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用随机初始化的模型
        logger.info("使用随机初始化的YOLOv5n模型...")
        model = YOLO('yolov5n.yaml')  # 使用更小的模型
        
        # 开始训练
        logger.info("开始训练...")
        
        # 显示GPU内存使用情况
        if device != 'cpu' and torch.cuda.is_available():
            gpu_id = int(device) if device.isdigit() else 0
            torch.cuda.set_device(gpu_id)
            memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
            logger.info(f"训练前GPU内存使用: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
        
        results = model.train(
            data=data_config,
            epochs=epochs,
            batch=batch_size,
            imgsz=320,  # 使用更小的图像尺寸
            device=device,
            project=str(output_dir),
            name='quick_mask_detection',
            exist_ok=True,
            verbose=True,
            patience=10,
            save=True,
            cache=False,
            workers=2,  # 减少工作进程
            amp=True,   # 启用混合精度训练
            lr0=0.01,
            warmup_epochs=1,
            # GPU优化参数
            optimizer='AdamW',
            cos_lr=True,
            close_mosaic=5,  # 更早关闭马赛克增强
            resume=False,
            pretrained=True,
            seed=42,
            deterministic=True,
            # 数据增强参数（快速训练使用较少增强）
            hsv_h=0.01,
            hsv_s=0.5,
            hsv_v=0.3,
            degrees=0.0,
            translate=0.05,
            scale=0.3,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.8,  # 减少马赛克增强
            mixup=0.0,
            copy_paste=0.0,
        )
        
        logger.info("快速训练完成！")
        
        # 获取模型路径
        best_model = output_dir / 'quick_mask_detection' / 'weights' / 'best.pt'
        last_model = output_dir / 'quick_mask_detection' / 'weights' / 'last.pt'
        
        logger.info(f"最佳模型: {best_model}")
        logger.info(f"最后模型: {last_model}")
        
        return str(best_model), str(last_model)
        
    except Exception as e:
        logger.error(f"快速训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    parser = argparse.ArgumentParser(description='快速口罩检测模型训练')
    parser.add_argument('--original-data', required=True, help='原始数据集目录')
    parser.add_argument('--sample-size', type=int, default=1000, help='训练样本数量')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(42)
    
    # 创建小数据集
    small_dataset_dir = 'data/quick_dataset'
    config_path = create_small_dataset(
        args.original_data, 
        small_dataset_dir, 
        args.sample_size
    )
    
    if not config_path:
        logger.error("创建小数据集失败")
        return
    
    # 快速训练
    best_model, last_model = quick_train(
        config_path,
        args.epochs,
        args.batch_size
    )
    
    print(f"\n快速训练完成！")
    if best_model:
        print(f"最佳模型: {best_model}")
        print(f"最后模型: {last_model}")

if __name__ == "__main__":
    main()