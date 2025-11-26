#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查PyTorch和CUDA环境
"""

try:
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA不可用")
        
except ImportError:
    print("PyTorch未安装")
    
try:
    import torchvision
    print(f"TorchVision版本: {torchvision.__version__}")
except ImportError:
    print("TorchVision未安装")
    
try:
    import ultralytics
    print(f"Ultralytics版本: {ultralytics.__version__}")
except ImportError:
    print("Ultralytics未安装")