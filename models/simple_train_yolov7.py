#!/usr/bin/env python3
"""
简化的YOLOv7口罩检测训练脚本
直接调用YOLOv7原生训练脚本
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # 设置路径
    project_root = Path(__file__).parent.parent
    yolov7_root = project_root / "yolov7"
    data_config = project_root / "data" / "mask_detection.yaml"
    
    # 检查YOLOv7目录
    if not yolov7_root.exists():
        print(f"错误: YOLOv7目录不存在: {yolov7_root}")
        return
    
    # 检查数据配置文件
    if not data_config.exists():
        print(f"错误: 数据配置文件不存在: {data_config}")
        return
    
    # 切换到YOLOv7目录
    os.chdir(yolov7_root)
    
    # 构建训练命令
    cmd = [
        "py", "train.py",
        "--data", str(data_config),
        "--cfg", "cfg/training/yolov7-tiny.yaml",
        "--weights", "",  # 使用空权重从头开始训练
        "--batch-size", "2",
        "--epochs", "1",
        "--img-size", "320", "320",
        "--device", "0",
        "--name", "mask_detection_test",
        "--project", "runs/train",
        "--workers", "0"  # 设置为0避免多进程问题
    ]
    
    print("开始YOLOv7口罩检测训练...")
    print(f"命令: {' '.join(cmd)}")
    print(f"工作目录: {os.getcwd()}")
    
    try:
        # 执行训练命令
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("训练完成!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        return e.returncode
    except Exception as e:
        print(f"执行错误: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)