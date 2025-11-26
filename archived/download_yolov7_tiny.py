#!/usr/bin/env python3
"""
下载YOLOv7-tiny预训练权重文件
"""

import requests
import os
from pathlib import Path

def download_yolov7_tiny():
    """下载YOLOv7-tiny权重文件"""
    url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt"
    filename = "yolov7-tiny.pt"
    
    print(f"开始下载 {filename}...")
    print(f"下载地址: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r下载进度: {progress:.1f}% ({downloaded_size}/{total_size} bytes)", end='')
        
        print(f"\n下载完成! 文件大小: {os.path.getsize(filename)} bytes")
        return True
        
    except Exception as e:
        print(f"下载失败: {e}")
        return False

if __name__ == "__main__":
    success = download_yolov7_tiny()
    if success:
        print("YOLOv7-tiny权重文件下载成功!")
    else:
        print("下载失败，请检查网络连接或手动下载")