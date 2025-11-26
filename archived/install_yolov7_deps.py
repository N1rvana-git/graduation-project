#!/usr/bin/env python3
"""
YOLOv7依赖安装脚本
解决命令行环境问题，直接通过Python安装所需依赖
"""

import subprocess
import sys
import os

def install_package(package, index_url="https://mirrors.aliyun.com/pypi/simple/"):
    """安装单个包"""
    try:
        print(f"正在安装 {package}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package, 
            "-i", index_url, "--trusted-host", "mirrors.aliyun.com"
        ], capture_output=True, text=True, check=True)
        print(f"✓ {package} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {package} 安装失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def main():
    """主函数"""
    print("开始安装YOLOv7依赖包...")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    # YOLOv7核心依赖包
    packages = [
        "matplotlib>=3.2.2",
        "numpy>=1.18.5",
        "opencv-python>=4.1.1",
        "Pillow>=7.1.2",
        "PyYAML>=5.3.1",
        "requests>=2.23.0",
        "scipy>=1.4.1",
        "torch>=1.7.0",
        "torchvision>=0.8.1",
        "tqdm>=4.41.0",
        "protobuf<4.21.3",
        "seaborn>=0.11.0",
        "thop",
        "tensorboard>=2.4.1",
        "pandas>=1.1.4"
    ]
    
    # 升级pip
    print("升级pip...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip",
            "-i", "https://mirrors.aliyun.com/pypi/simple/",
            "--trusted-host", "mirrors.aliyun.com"
        ], check=True)
        print("✓ pip升级成功")
    except subprocess.CalledProcessError as e:
        print(f"✗ pip升级失败: {e}")
    
    # 安装包
    success_count = 0
    failed_packages = []
    
    for package in packages:
        if install_package(package):
            success_count += 1
        else:
            failed_packages.append(package)
    
    # 总结
    print(f"\n安装完成!")
    print(f"成功安装: {success_count}/{len(packages)} 个包")
    
    if failed_packages:
        print(f"失败的包: {failed_packages}")
    else:
        print("所有依赖包安装成功!")
    
    # 验证关键包
    print("\n验证关键包...")
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA版本: {torch.version.cuda}")
            print(f"✓ GPU数量: {torch.cuda.device_count()}")
    except ImportError:
        print("✗ PyTorch导入失败")
    
    try:
        import cv2
        print(f"✓ OpenCV版本: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV导入失败")
    
    try:
        import numpy as np
        print(f"✓ NumPy版本: {np.__version__}")
    except ImportError:
        print("✗ NumPy导入失败")

if __name__ == "__main__":
    main()