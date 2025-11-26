#!/usr/bin/env python3
"""
简单的Python环境测试脚本
"""

import sys
import os

def main():
    print("=== Python环境测试 ===")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"系统平台: {sys.platform}")
    
    # 测试基础模块
    try:
        import json
        print("✓ json模块正常")
    except ImportError as e:
        print(f"✗ json模块导入失败: {e}")
    
    try:
        import os
        print("✓ os模块正常")
    except ImportError as e:
        print(f"✗ os模块导入失败: {e}")
    
    try:
        import subprocess
        print("✓ subprocess模块正常")
    except ImportError as e:
        print(f"✗ subprocess模块导入失败: {e}")
    
    print("=== 测试完成 ===")

if __name__ == "__main__":
    main()