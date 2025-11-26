"""
数据集准备脚本
将AFDB人脸数据集转换为YOLO格式，用于口罩检测训练
"""

import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import argparse
import logging
from tqdm import tqdm
import random

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreparer:
    """数据集准备器"""

    def __init__(self, source_path: str, output_path: str, seed: int = 42):
        """初始化数据集准备器"""

        self.source_path = Path(source_path)
        self.output_path = Path(output_path)

        # 固定随机种子，增强实验可复现性
        random.seed(seed)

        # YOLO类别定义（按开题报告要求仅保留戴口罩/未戴口罩两类）
        self.classes = {
            'no_mask': 0,
            'mask': 1,
        }

        # 提前加载人脸检测器，避免重复初始化
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

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
        
        for dir_name in directories:
            (self.output_path / dir_name).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"创建输出目录: {self.output_path}")
    
    def process_afdb_dataset(self, face_dataset_path: str, masked_dataset_path: str):
        """
        处理AFDB数据集
        
        Args:
            face_dataset_path: 普通人脸数据集路径
            masked_dataset_path: 戴口罩人脸数据集路径
        """
        logger.info("开始处理AFDB数据集...")
        
        all_samples = []
        
        # 处理普通人脸数据集（标记为no_mask）
        if os.path.exists(face_dataset_path):
            face_samples = self.process_face_images(face_dataset_path, 'no_mask')
            all_samples.extend(face_samples)
            logger.info(f"处理普通人脸图像: {len(face_samples)} 张")
        
        # 处理戴口罩数据集（标记为mask）
        if os.path.exists(masked_dataset_path):
            mask_samples = self.process_face_images(masked_dataset_path, 'mask')
            all_samples.extend(mask_samples)
            logger.info(f"处理戴口罩图像: {len(mask_samples)} 张")
        
        # 随机打乱数据
        random.shuffle(all_samples)
        
        # 分割数据集
        self.split_dataset(all_samples)
        
        # 创建数据集配置文件
        self.create_dataset_config()
        
        logger.info(f"数据集处理完成，总计 {len(all_samples)} 张图像")
    
    def process_face_images(self, dataset_path: str, label: str) -> list:
        """
        处理人脸图像文件夹
        
        Args:
            dataset_path: 数据集路径
            label: 标签类型
            
        Returns:
            处理后的样本列表
        """
        samples = []
        dataset_path = Path(dataset_path)
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 遍历所有图像文件
        for img_path in dataset_path.rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                try:
                    # 验证图像是否可以正常读取
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        
                        # 创建样本信息
                        sample = {
                            'image_path': str(img_path),
                            'label': label,
                            'class_id': self.classes[label],
                            'width': w,
                            'height': h,
                            'bbox': self.detect_face_bbox(img)  # 检测人脸边界框
                        }
                        samples.append(sample)
                        
                except Exception as e:
                    logger.warning(f"跳过无效图像: {img_path}, 错误: {str(e)}")
        
        return samples
    
    def detect_face_bbox(self, image: np.ndarray) -> tuple:
        """
        使用OpenCV检测人脸边界框
        
        Args:
            image: 输入图像
            
        Returns:
            边界框坐标 (x, y, w, h)，归一化到[0,1]
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4) if not self.face_cascade.empty() else []
            
            h, w = image.shape[:2]
            
            if len(faces) > 0:
                # 选择最大的人脸
                face = max(faces, key=lambda x: x[2] * x[3])
                x, y, fw, fh = face
                
                # 归一化坐标
                center_x = (x + fw/2) / w
                center_y = (y + fh/2) / h
                norm_w = fw / w
                norm_h = fh / h
                
                return (center_x, center_y, norm_w, norm_h)
            else:
                # 如果没有检测到人脸，使用整个图像
                return (0.5, 0.5, 1.0, 1.0)
                
        except Exception as e:
            logger.warning(f"人脸检测失败: {str(e)}")
            return (0.5, 0.5, 1.0, 1.0)
    
    def split_dataset(self, samples: list, train_ratio: float = 0.7, val_ratio: float = 0.2):
        """
        分割数据集为训练集、验证集和测试集
        
        Args:
            samples: 样本列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        total = len(samples)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        logger.info(f"数据集分割: 训练集 {len(train_samples)}, 验证集 {len(val_samples)}, 测试集 {len(test_samples)}")
        
        # 处理各个数据集
        self.process_split('train', train_samples)
        self.process_split('val', val_samples)
        self.process_split('test', test_samples)
    
    def process_split(self, split_name: str, samples: list):
        """
        处理数据集分割
        
        Args:
            split_name: 分割名称 (train/val/test)
            samples: 样本列表
        """
        logger.info(f"处理 {split_name} 数据集...")
        
        for i, sample in enumerate(tqdm(samples, desc=f"Processing {split_name}")):
            try:
                # 生成新的文件名
                new_name = f"{split_name}_{i:06d}"
                
                # 复制图像文件
                src_path = sample['image_path']
                dst_img_path = self.output_path / 'images' / split_name / f"{new_name}.jpg"
                
                # 读取并保存图像（统一为JPG格式）
                img = cv2.imread(src_path)
                cv2.imwrite(str(dst_img_path), img)
                
                # 创建YOLO格式标签文件
                dst_label_path = self.output_path / 'labels' / split_name / f"{new_name}.txt"
                self.create_yolo_label(dst_label_path, sample)
                
            except Exception as e:
                logger.error(f"处理样本失败: {sample['image_path']}, 错误: {str(e)}")
    
    def create_yolo_label(self, label_path: Path, sample: dict):
        """
        创建YOLO格式标签文件
        
        Args:
            label_path: 标签文件路径
            sample: 样本信息
        """
        with open(label_path, 'w') as f:
            class_id = sample['class_id']
            bbox = sample['bbox']
            
            # YOLO格式: class_id center_x center_y width height
            f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    
    def create_dataset_config(self):
        """创建数据集配置文件"""
        config = {
            'path': str(self.output_path.resolve()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.classes),
            'names': list(self.classes.keys())
        }
        
        config_path = self.output_path / 'dataset.yaml'
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("# 口罩检测数据集配置\n")
            f.write(f"path: {config['path']}\n")
            f.write(f"train: {config['train']}\n")
            f.write(f"val: {config['val']}\n")
            f.write(f"test: {config['test']}\n")
            f.write(f"\n")
            f.write(f"nc: {config['nc']}\n")
            f.write(f"names: {config['names']}\n")
        
        logger.info(f"数据集配置文件已创建: {config_path}")
    
    def visualize_samples(self, num_samples: int = 10):
        """
        可视化数据集样本
        
        Args:
            num_samples: 可视化样本数量
        """
        logger.info("创建数据集可视化...")
        
        vis_dir = self.output_path / 'visualization'
        vis_dir.mkdir(exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            img_dir = self.output_path / 'images' / split
            label_dir = self.output_path / 'labels' / split
            
            img_files = list(img_dir.glob('*.jpg'))[:num_samples]
            
            for img_file in img_files:
                label_file = label_dir / f"{img_file.stem}.txt"
                
                if label_file.exists():
                    self.draw_sample(img_file, label_file, vis_dir / f"{split}_{img_file.name}")
    
    def draw_sample(self, img_path: Path, label_path: Path, output_path: Path):
        """
        绘制带标注的样本图像
        
        Args:
            img_path: 图像路径
            label_path: 标签路径
            output_path: 输出路径
        """
        try:
            # 读取图像
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            # 读取标签
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # 绘制边界框
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    center_x, center_y, width, height = map(float, parts[1:])
                    
                    # 转换为像素坐标
                    x1 = int((center_x - width/2) * w)
                    y1 = int((center_y - height/2) * h)
                    x2 = int((center_x + width/2) * w)
                    y2 = int((center_y + height/2) * h)
                    
                    # 选择颜色
                    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # 红、绿、蓝
                    color = colors[class_id % len(colors)]
                    
                    # 绘制边界框
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # 添加标签
                    class_names = list(self.classes.keys())
                    label_text = class_names[class_id]
                    cv2.putText(img, label_text, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 保存图像
            cv2.imwrite(str(output_path), img)
            
        except Exception as e:
            logger.error(f"绘制样本失败: {str(e)}")
    
    def get_dataset_statistics(self):
        """获取数据集统计信息"""
        stats = {}
        
        for split in ['train', 'val', 'test']:
            label_dir = self.output_path / 'labels' / split
            
            class_counts = {name: 0 for name in self.classes.keys()}
            total_images = 0
            
            for label_file in label_dir.glob('*.txt'):
                total_images += 1
                
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        class_name = list(self.classes.keys())[class_id]
                        class_counts[class_name] += 1
            
            stats[split] = {
                'total_images': total_images,
                'class_counts': class_counts
            }
        
        return stats
    
    def print_statistics(self):
        """打印数据集统计信息"""
        stats = self.get_dataset_statistics()
        
        print("\n" + "="*60)
        print("数据集统计信息")
        print("="*60)
        
        for split, data in stats.items():
            print(f"\n{split.upper()} 数据集:")
            print(f"  总图像数: {data['total_images']}")
            print(f"  类别分布:")
            
            for class_name, count in data['class_counts'].items():
                percentage = (count / data['total_images'] * 100) if data['total_images'] > 0 else 0
                print(f"    {class_name}: {count} ({percentage:.1f}%)")
        
        # 总计
        total_images = sum(data['total_images'] for data in stats.values())
        total_classes = {}
        
        for data in stats.values():
            for class_name, count in data['class_counts'].items():
                total_classes[class_name] = total_classes.get(class_name, 0) + count
        
        print(f"\n总计:")
        print(f"  总图像数: {total_images}")
        print(f"  类别分布:")
        
        for class_name, count in total_classes.items():
            percentage = (count / total_images * 100) if total_images > 0 else 0
            print(f"    {class_name}: {count} ({percentage:.1f}%)")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AFDB数据集准备工具（生成YOLO检测格式）')
    parser.add_argument('--dataset_root', default=None,
                        help='包含 AFDB_face_dataset 与 AFDB_masked_face_dataset 的目录')
    parser.add_argument('--face_dataset', default=None,
                        help='普通人脸数据集路径（未指定时自动从 dataset_root 推断）')
    parser.add_argument('--masked_dataset', default=None,
                        help='戴口罩人脸数据集路径（未指定时自动从 dataset_root 推断）')
    parser.add_argument('--output', default='./yolo_dataset',
                        help='输出目录路径')
    parser.add_argument('--visualize', action='store_true',
                        help='生成可视化样本')

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root) if args.dataset_root else None

    face_dataset = Path(args.face_dataset) if args.face_dataset else None
    masked_dataset = Path(args.masked_dataset) if args.masked_dataset else None

    if dataset_root:
        if not dataset_root.exists():
            raise FileNotFoundError(f"数据集根目录不存在: {dataset_root}")
        face_dataset = face_dataset or dataset_root / 'AFDB_face_dataset'
        masked_dataset = masked_dataset or dataset_root / 'AFDB_masked_face_dataset'

    if face_dataset is None or not face_dataset.exists():
        raise FileNotFoundError('未找到普通人脸数据集，请通过 --face_dataset 或 --dataset_root 指定有效路径')

    if masked_dataset is None or not masked_dataset.exists():
        raise FileNotFoundError('未找到戴口罩数据集，请通过 --masked_dataset 或 --dataset_root 指定有效路径')

    output_dir = Path(args.output)

    # 创建数据集准备器
    preparer = DatasetPreparer(str(face_dataset.parent), str(output_dir))

    # 处理数据集
    preparer.process_afdb_dataset(str(face_dataset), str(masked_dataset))

    # 生成统计信息
    preparer.print_statistics()

    # 可视化（可选）
    if args.visualize:
        preparer.visualize_samples()

    logger.info("数据集准备完成！")

if __name__ == "__main__":
    main()