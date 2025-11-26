#!/usr/bin/env python3
"""
基础YOLOv7测试脚本
直接测试YOLOv7的核心功能和依赖
"""

import os
import sys

def test_basic_imports():
    """测试基础导入"""
    print("=== 测试基础导入 ===")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU数量: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"✗ PyTorch导入失败: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy导入失败: {e}")
        return False
    
    return True

def test_yolov7_structure():
    """测试YOLOv7目录结构"""
    print("\n=== 测试YOLOv7目录结构 ===")
    
    yolov7_path = os.path.join(os.getcwd(), 'yolov7')
    if not os.path.exists(yolov7_path):
        print(f"✗ YOLOv7目录不存在: {yolov7_path}")
        return False
    
    print(f"✓ YOLOv7目录存在: {yolov7_path}")
    
    # 检查关键文件
    key_files = [
        'detect.py',
        'train.py', 
        'models/__init__.py',
        'models/yolo.py',
        'utils/__init__.py',
        'utils/general.py',
        'utils/torch_utils.py'
    ]
    
    missing_files = []
    for file in key_files:
        file_path = os.path.join(yolov7_path, file)
        if os.path.exists(file_path):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"缺失文件: {missing_files}")
        return False
    
    return True

def test_yolov7_imports():
    """测试YOLOv7模块导入"""
    print("\n=== 测试YOLOv7模块导入 ===")
    
    # 添加YOLOv7路径到sys.path
    yolov7_path = os.path.join(os.getcwd(), 'yolov7')
    if yolov7_path not in sys.path:
        sys.path.insert(0, yolov7_path)
    
    try:
        from models.experimental import attempt_load
        print("✓ models.experimental.attempt_load")
    except ImportError as e:
        print(f"✗ models.experimental.attempt_load: {e}")
        return False
    
    try:
        from utils.general import check_img_size
        print("✓ utils.general.check_img_size")
    except ImportError as e:
        print(f"✗ utils.general.check_img_size: {e}")
        return False
    
    try:
        from utils.torch_utils import select_device
        print("✓ utils.torch_utils.select_device")
    except ImportError as e:
        print(f"✗ utils.torch_utils.select_device: {e}")
        return False
    
    return True

def test_device_selection():
    """测试设备选择"""
    print("\n=== 测试设备选择 ===")
    
    try:
        # 添加YOLOv7路径
        yolov7_path = os.path.join(os.getcwd(), 'yolov7')
        if yolov7_path not in sys.path:
            sys.path.insert(0, yolov7_path)
        
        from utils.torch_utils import select_device
        
        # 测试CPU设备
        device = select_device('cpu')
        print(f"✓ CPU设备选择成功: {device}")
        
        # 测试GPU设备（如果可用）
        import torch
        if torch.cuda.is_available():
            device = select_device('0')
            print(f"✓ GPU设备选择成功: {device}")
        else:
            print("ℹ GPU不可用，跳过GPU测试")
        
        return True
    except Exception as e:
        print(f"✗ 设备选择测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=== YOLOv7基础功能测试 ===")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    
    tests = [
        ("基础导入", test_basic_imports),
        ("YOLOv7目录结构", test_yolov7_structure),
        ("YOLOv7模块导入", test_yolov7_imports),
        ("设备选择", test_device_selection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"运行测试: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                print(f"✓ {test_name} - 通过")
                passed += 1
            else:
                print(f"✗ {test_name} - 失败")
        except Exception as e:
            print(f"✗ {test_name} - 异常: {e}")
    
    print(f"\n{'='*50}")
    print(f"测试结果: {passed}/{total} 通过")
    print('='*50)
    
    if passed == total:
        print("🎉 所有测试通过！YOLOv7环境配置正确。")
        return True
    else:
        print("❌ 部分测试失败，需要修复环境配置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)