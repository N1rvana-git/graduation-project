#!/usr/bin/env python3
"""
手动安装YOLOv7依赖脚本
使用更简单直接的方法安装依赖包
"""

import os
import sys

def main():
    print("=== 手动安装YOLOv7依赖 ===")
    print(f"Python版本: {sys.version}")
    print(f"当前目录: {os.getcwd()}")
    
    # 检查现有包
    print("\n检查现有包...")
    packages_to_check = [
        'torch', 'torchvision', 'numpy', 'opencv-python', 
        'matplotlib', 'pillow', 'pyyaml', 'requests', 
        'scipy', 'tqdm', 'seaborn'
    ]
    
    installed_packages = []
    missing_packages = []
    
    for package in packages_to_check:
        try:
            if package == 'opencv-python':
                import cv2
                print(f"✓ {package} (cv2) - 版本: {cv2.__version__}")
                installed_packages.append(package)
            elif package == 'pillow':
                from PIL import Image
                print(f"✓ {package} (PIL) - 已安装")
                installed_packages.append(package)
            elif package == 'pyyaml':
                import yaml
                print(f"✓ {package} (yaml) - 已安装")
                installed_packages.append(package)
            else:
                module = __import__(package)
                if hasattr(module, '__version__'):
                    print(f"✓ {package} - 版本: {module.__version__}")
                else:
                    print(f"✓ {package} - 已安装")
                installed_packages.append(package)
        except ImportError:
            print(f"✗ {package} - 未安装")
            missing_packages.append(package)
    
    print(f"\n已安装: {len(installed_packages)} 个包")
    print(f"缺失: {len(missing_packages)} 个包")
    
    if missing_packages:
        print(f"缺失的包: {missing_packages}")
        print("\n请手动安装缺失的包:")
        for package in missing_packages:
            print(f"pip install {package}")
    else:
        print("\n所有必需的包都已安装!")
    
    # 测试YOLOv7相关功能
    print("\n=== 测试YOLOv7相关功能 ===")
    
    # 测试torch和GPU
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA版本: {torch.version.cuda}")
            print(f"✓ GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"✗ PyTorch测试失败: {e}")
    
    # 测试OpenCV
    try:
        import cv2
        print(f"✓ OpenCV版本: {cv2.__version__}")
        # 测试基本功能
        import numpy as np
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        print("✓ OpenCV基本功能正常")
    except Exception as e:
        print(f"✗ OpenCV测试失败: {e}")
    
    # 测试matplotlib
    try:
        import matplotlib
        print(f"✓ Matplotlib版本: {matplotlib.__version__}")
    except Exception as e:
        print(f"✗ Matplotlib测试失败: {e}")
    
    # 检查YOLOv7目录
    yolov7_path = os.path.join(os.getcwd(), 'yolov7')
    if os.path.exists(yolov7_path):
        print(f"✓ YOLOv7目录存在: {yolov7_path}")
        
        # 检查关键文件
        key_files = ['train.py', 'detect.py', 'models/yolo.py', 'utils/general.py']
        for file in key_files:
            file_path = os.path.join(yolov7_path, file)
            if os.path.exists(file_path):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file}")
    else:
        print(f"✗ YOLOv7目录不存在: {yolov7_path}")
    
    print("\n=== 安装检查完成 ===")

if __name__ == "__main__":
    main()