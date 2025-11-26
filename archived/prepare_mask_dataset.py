#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备YOLOv7格式的口罩检测数据集
"""

import os
import shutil
import random
from pathlib import Path
import json

def create_directory_structure():
    """创建YOLOv7数据集目录结构"""
    base_dir = Path("data/yolo_dataset")
    
    # 创建目录结构
    dirs = [
        "images/train",
        "images/val", 
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test"
    ]
    
    for dir_path in dirs:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
    print("✓ 创建YOLOv7数据集目录结构")
    return base_dir

def convert_annotations_to_yolo(source_dir, target_dir, class_mapping):
    """将标注转换为YOLO格式"""
    print(f"转换标注格式: {source_dir} -> {target_dir}")
    
    # 这里需要根据实际的标注格式进行转换
    # 示例：假设有JSON格式的标注文件
    
    converted_count = 0
    
    # 遍历源目录中的标注文件
    for annotation_file in Path(source_dir).glob("*.json"):
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换为YOLO格式
            yolo_annotations = []
            
            # 假设JSON格式包含边界框信息
            if 'annotations' in data:
                img_width = data.get('image_width', 640)
                img_height = data.get('image_height', 640)
                
                for ann in data['annotations']:
                    class_name = ann.get('class', 'unknown')
                    if class_name in class_mapping:
                        class_id = class_mapping[class_name]
                        
                        # 获取边界框 (假设格式为 [x, y, width, height])
                        bbox = ann.get('bbox', [0, 0, 0, 0])
                        x, y, w, h = bbox
                        
                        # 转换为YOLO格式 (归一化的中心点坐标和宽高)
                        x_center = (x + w/2) / img_width
                        y_center = (y + h/2) / img_height
                        norm_width = w / img_width
                        norm_height = h / img_height
                        
                        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
            
            # 保存YOLO格式标注
            output_file = target_dir / f"{annotation_file.stem}.txt"
            with open(output_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
                
            converted_count += 1
            
        except Exception as e:
            print(f"转换失败 {annotation_file}: {e}")
    
    print(f"✓ 转换完成: {converted_count} 个标注文件")
    return converted_count

def create_sample_dataset():
    """创建示例数据集（用于测试）"""
    print("创建示例口罩检测数据集...")
    
    base_dir = create_directory_structure()
    
    # 创建示例标注文件
    sample_annotations = [
        # 训练集示例
        ("train", "sample_001.txt", "1 0.5 0.3 0.2 0.4"),  # 戴口罩
        ("train", "sample_002.txt", "0 0.4 0.5 0.3 0.3"),  # 未戴口罩
        ("train", "sample_003.txt", "1 0.6 0.4 0.25 0.35"), # 戴口罩
        
        # 验证集示例
        ("val", "sample_004.txt", "0 0.45 0.4 0.28 0.32"),  # 未戴口罩
        ("val", "sample_005.txt", "1 0.55 0.35 0.22 0.38"), # 戴口罩
    ]
    
    for split, filename, annotation in sample_annotations:
        label_file = base_dir / "labels" / split / filename
        with open(label_file, 'w') as f:
            f.write(annotation)
    
    # 创建对应的图像占位符文件
    for split, filename, _ in sample_annotations:
        img_filename = filename.replace('.txt', '.jpg')
        img_file = base_dir / "images" / split / img_filename
        
        # 创建空的图像文件作为占位符
        img_file.touch()
    
    print("✓ 创建示例数据集完成")
    
    # 创建数据集统计信息
    stats = {
        "total_images": len(sample_annotations),
        "train_images": len([x for x in sample_annotations if x[0] == "train"]),
        "val_images": len([x for x in sample_annotations if x[0] == "val"]),
        "classes": {
            "0": "no_mask",
            "1": "mask"
        }
    }
    
    with open(base_dir / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    return base_dir

def validate_dataset(dataset_dir):
    """验证数据集完整性"""
    print("验证数据集完整性...")
    
    dataset_dir = Path(dataset_dir)
    issues = []
    
    # 检查目录结构
    required_dirs = [
        "images/train", "images/val",
        "labels/train", "labels/val"
    ]
    
    for dir_path in required_dirs:
        if not (dataset_dir / dir_path).exists():
            issues.append(f"缺少目录: {dir_path}")
    
    # 检查图像和标注文件匹配
    for split in ["train", "val"]:
        img_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        
        if img_dir.exists() and label_dir.exists():
            img_files = set(f.stem for f in img_dir.glob("*"))
            label_files = set(f.stem for f in label_dir.glob("*.txt"))
            
            # 检查孤立文件
            orphan_images = img_files - label_files
            orphan_labels = label_files - img_files
            
            if orphan_images:
                issues.append(f"{split}集中有 {len(orphan_images)} 个图像缺少标注")
            if orphan_labels:
                issues.append(f"{split}集中有 {len(orphan_labels)} 个标注缺少图像")
    
    # 输出验证结果
    if issues:
        print("❌ 数据集验证发现问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ 数据集验证通过")
        return True

def main():
    """主函数"""
    print("=== YOLOv7口罩检测数据集准备工具 ===")
    
    # 检查是否已有数据集
    dataset_dir = Path("data/yolo_dataset")
    
    if dataset_dir.exists():
        print(f"数据集目录已存在: {dataset_dir}")
        
        # 验证现有数据集
        if validate_dataset(dataset_dir):
            print("✅ 现有数据集验证通过")
        else:
            print("⚠️ 现有数据集存在问题")
    else:
        print("创建新的数据集...")
        
        # 创建示例数据集
        dataset_dir = create_sample_dataset()
        
        # 验证创建的数据集
        validate_dataset(dataset_dir)
    
    # 显示数据集信息
    print(f"\n=== 数据集信息 ===")
    print(f"数据集路径: {dataset_dir.absolute()}")
    
    # 统计文件数量
    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        
        if img_dir.exists():
            img_count = len(list(img_dir.glob("*")))
            label_count = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0
            print(f"{split}集: {img_count} 图像, {label_count} 标注")
    
    # 检查配置文件
    config_file = Path("yolov7/data/mask_detection.yaml")
    if config_file.exists():
        print(f"✅ 配置文件: {config_file}")
    else:
        print(f"❌ 配置文件缺失: {config_file}")
    
    print("\n🎉 数据集准备完成！")
    print("\n下一步:")
    print("1. 将真实的图像和标注文件放入对应目录")
    print("2. 运行训练脚本开始训练")
    print("3. 使用 py models/train_yolov7_mask_detection.py 开始训练")

if __name__ == "__main__":
    main()