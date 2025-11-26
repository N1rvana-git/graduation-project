#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载YOLOv7预训练权重文件
"""

import os
import urllib.request
import hashlib
from pathlib import Path

def download_file(url, filename, expected_size=None):
    """下载文件并显示进度"""
    print(f"正在下载: {filename}")
    print(f"URL: {url}")
    
    try:
        # 创建目录
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 下载文件
        urllib.request.urlretrieve(url, filename)
        
        # 检查文件大小
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"✓ 下载完成: {filename}")
            print(f"  文件大小: {file_size / (1024*1024):.1f} MB")
            
            if expected_size and abs(file_size - expected_size) > 1024*1024:  # 允许1MB误差
                print(f"⚠️ 警告: 文件大小不匹配 (期望: {expected_size / (1024*1024):.1f} MB)")
            
            return True
        else:
            print(f"✗ 下载失败: {filename}")
            return False
            
    except Exception as e:
        print(f"✗ 下载错误: {e}")
        return False

def main():
    """主函数"""
    print("=== YOLOv7预训练权重下载器 ===")
    
    # 权重文件配置
    weights_config = [
        {
            "name": "yolov7.pt",
            "url": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
            "path": "yolov7/weights/yolov7.pt",
            "size": 75628875,  # 约72MB
            "description": "YOLOv7基础模型"
        },
        {
            "name": "yolov7-tiny.pt", 
            "url": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt",
            "path": "yolov7/weights/yolov7-tiny.pt",
            "size": 12103595,  # 约11.5MB
            "description": "YOLOv7轻量级模型"
        },
        {
            "name": "yolov7x.pt",
            "url": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt", 
            "path": "yolov7/weights/yolov7x.pt",
            "size": 143201229,  # 约136MB
            "description": "YOLOv7扩展模型"
        }
    ]
    
    # 检查现有文件
    print("\n=== 检查现有权重文件 ===")
    for config in weights_config:
        if os.path.exists(config["path"]):
            size = os.path.getsize(config["path"])
            print(f"✓ {config['name']} 已存在 ({size / (1024*1024):.1f} MB)")
        else:
            print(f"✗ {config['name']} 不存在")
    
    # 下载缺失的文件
    print("\n=== 开始下载缺失的权重文件 ===")
    success_count = 0
    
    for config in weights_config:
        if not os.path.exists(config["path"]):
            print(f"\n下载 {config['description']}...")
            if download_file(config["url"], config["path"], config["size"]):
                success_count += 1
        else:
            print(f"\n跳过 {config['name']} (已存在)")
            success_count += 1
    
    # 总结
    print(f"\n=== 下载完成 ===")
    print(f"成功: {success_count}/{len(weights_config)} 个权重文件")
    
    if success_count == len(weights_config):
        print("🎉 所有权重文件准备就绪！")
        
        # 验证权重文件
        print("\n=== 验证权重文件 ===")
        try:
            import torch
            for config in weights_config:
                if os.path.exists(config["path"]):
                    try:
                        # 尝试加载权重文件
                        checkpoint = torch.load(config["path"], map_location='cpu')
                        print(f"✓ {config['name']} - 权重文件有效")
                        
                        # 显示模型信息
                        if 'model' in checkpoint:
                            print(f"  包含模型结构")
                        if 'epoch' in checkpoint:
                            print(f"  训练轮次: {checkpoint['epoch']}")
                            
                    except Exception as e:
                        print(f"✗ {config['name']} - 权重文件损坏: {e}")
                        
        except ImportError:
            print("PyTorch未安装，跳过权重文件验证")
    else:
        print("❌ 部分权重文件下载失败")
        
    return success_count == len(weights_config)

if __name__ == "__main__":
    main()