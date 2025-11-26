"""
简化版数据集准备脚本
快速处理AFDB数据集并转换为YOLO格式
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import random

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDatasetPreparer:
    """简化版数据集准备器"""
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        
        # YOLO类别定义
        self.classes = {
            'no_mask': 0,      # 未戴口罩
            'mask': 1,         # 戴口罩
        }
        
        # 创建输出目录
        self.setup_directories()
    
    def setup_directories(self):
        """创建必要的目录结构"""
        directories = [
            'images/train',
            'images/val', 
            'images/test',
            'labels/train',
            'labels/val',
            'labels/test'
        ]
        
        for directory in directories:
            (self.output_path / directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"创建输出目录: {self.output_path}")
    
    def process_datasets(self, face_dataset_path: str, masked_dataset_path: str):
        """处理两个数据集"""
        all_samples = []
        
        # 处理无口罩数据集
        logger.info("处理无口罩数据集...")
        face_samples = self.process_directory(face_dataset_path, 'no_mask')
        all_samples.extend(face_samples)
        logger.info(f"无口罩样本数量: {len(face_samples)}")
        
        # 处理戴口罩数据集
        logger.info("处理戴口罩数据集...")
        masked_samples = self.process_directory(masked_dataset_path, 'mask')
        all_samples.extend(masked_samples)
        logger.info(f"戴口罩样本数量: {len(masked_samples)}")
        
        # 随机打乱样本
        random.shuffle(all_samples)
        
        # 分割数据集
        self.split_dataset(all_samples)
        
        # 创建配置文件
        self.create_dataset_config()
        
        logger.info(f"数据集准备完成！总样本数: {len(all_samples)}")
        
        return all_samples
    
    def process_directory(self, dataset_path: str, label: str) -> list:
        """处理数据集目录"""
        dataset_path = Path(dataset_path)
        samples = []
        
        if not dataset_path.exists():
            logger.warning(f"数据集路径不存在: {dataset_path}")
            return samples
        
        # 遍历所有子目录（人物目录）
        person_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        for person_dir in tqdm(person_dirs, desc=f"处理{label}数据"):
            # 获取该人物目录下的所有图片
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(person_dir.glob(ext))
            
            for img_path in image_files:
                try:
                    # 读取图像
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                    
                    h, w = image.shape[:2]
                    
                    # 使用简单的人脸检测（假设整个图像就是人脸区域）
                    # 为了简化，我们假设人脸占据图像的中心80%区域
                    face_w = int(w * 0.8)
                    face_h = int(h * 0.8)
                    face_x = (w - face_w) // 2
                    face_y = (h - face_h) // 2
                    
                    # 转换为YOLO格式 (中心点坐标和宽高，都是相对值)
                    center_x = (face_x + face_w / 2) / w
                    center_y = (face_y + face_h / 2) / h
                    norm_w = face_w / w
                    norm_h = face_h / h
                    
                    sample = {
                        'image_path': img_path,
                        'label': label,
                        'class_id': self.classes[label],
                        'bbox': [center_x, center_y, norm_w, norm_h],
                        'image_size': (w, h)
                    }
                    
                    samples.append(sample)
                    
                except Exception as e:
                    logger.warning(f"处理图像失败 {img_path}: {e}")
                    continue
        
        return samples
    
    def split_dataset(self, samples: list, train_ratio: float = 0.7, val_ratio: float = 0.2):
        """分割数据集"""
        total = len(samples)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        logger.info(f"数据集分割: 训练集={len(train_samples)}, 验证集={len(val_samples)}, 测试集={len(test_samples)}")
        
        # 处理各个分割
        self.process_split('train', train_samples)
        self.process_split('val', val_samples)
        self.process_split('test', test_samples)
    
    def process_split(self, split_name: str, samples: list):
        """处理数据集分割"""
        images_dir = self.output_path / 'images' / split_name
        labels_dir = self.output_path / 'labels' / split_name
        
        for i, sample in enumerate(tqdm(samples, desc=f"处理{split_name}集")):
            # 复制图像文件
            src_path = sample['image_path']
            dst_name = f"{split_name}_{i:06d}.jpg"
            dst_path = images_dir / dst_name
            
            try:
                shutil.copy2(src_path, dst_path)
                
                # 创建标签文件
                label_path = labels_dir / f"{split_name}_{i:06d}.txt"
                self.create_yolo_label(label_path, sample)
                
            except Exception as e:
                logger.warning(f"处理样本失败: {e}")
    
    def create_yolo_label(self, label_path: Path, sample: dict):
        """创建YOLO格式标签文件"""
        with open(label_path, 'w') as f:
            class_id = sample['class_id']
            bbox = sample['bbox']
            f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    
    def create_dataset_config(self):
        """创建数据集配置文件"""
        config = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.classes),
            'names': list(self.classes.keys())
        }
        
        config_path = self.output_path / 'dataset.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(f"# 口罩检测数据集配置\n")
            f.write(f"path: {config['path']}\n")
            f.write(f"train: {config['train']}\n")
            f.write(f"val: {config['val']}\n")
            f.write(f"test: {config['test']}\n")
            f.write(f"\n")
            f.write(f"# 类别数量\n")
            f.write(f"nc: {config['nc']}\n")
            f.write(f"\n")
            f.write(f"# 类别名称\n")
            f.write(f"names:\n")
            for i, name in enumerate(config['names']):
                f.write(f"  {i}: {name}\n")
        
        logger.info(f"数据集配置文件已创建: {config_path}")

def main():
    parser = argparse.ArgumentParser(description='简化版数据集准备工具')
    parser.add_argument('--face_dataset', required=True, help='无口罩人脸数据集路径')
    parser.add_argument('--masked_dataset', required=True, help='戴口罩人脸数据集路径')
    parser.add_argument('--output', default='data/yolo_dataset', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建数据集准备器
    preparer = SimpleDatasetPreparer(args.output)
    
    # 处理数据集
    samples = preparer.process_datasets(args.face_dataset, args.masked_dataset)
    
    print(f"\n数据集准备完成！")
    print(f"输出目录: {args.output}")
    print(f"总样本数: {len(samples)}")

if __name__ == "__main__":
    main()