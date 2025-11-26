"""
YOLOv7环境测试脚本
测试YOLOv7相关功能是否正常工作
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

def test_basic_imports():
    """测试基础导入"""
    print("1. 测试基础导入...")
    
    try:
        import torch
        print(f"   ✓ PyTorch版本: {torch.__version__}")
        
        import torchvision
        print(f"   ✓ TorchVision版本: {torchvision.__version__}")
        
        import cv2
        print(f"   ✓ OpenCV版本: {cv2.__version__}")
        
        import numpy as np
        print(f"   ✓ NumPy版本: {np.__version__}")
        
        import yaml
        print(f"   ✓ PyYAML可用")
        
        return True
    except ImportError as e:
        print(f"   ✗ 导入失败: {e}")
        return False

def test_yolov7_imports():
    """测试YOLOv7模块导入"""
    print("\n2. 测试YOLOv7模块导入...")
    
    # 添加YOLOv7路径
    yolov7_path = Path(__file__).parent / "yolov7"
    sys.path.append(str(yolov7_path))
    
    try:
        from yolov7.utils.general import check_img_size
        print("   ✓ YOLOv7 utils.general 导入成功")
        
        from yolov7.utils.torch_utils import select_device
        print("   ✓ YOLOv7 utils.torch_utils 导入成功")
        
        from yolov7.models.experimental import attempt_load
        print("   ✓ YOLOv7 models.experimental 导入成功")
        
        return True
    except ImportError as e:
        print(f"   ✗ YOLOv7模块导入失败: {e}")
        print(f"   YOLOv7路径: {yolov7_path}")
        print(f"   路径存在: {yolov7_path.exists()}")
        return False

def test_gpu_availability():
    """测试GPU可用性"""
    print("\n3. 测试GPU可用性...")
    
    try:
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA可用: {cuda_available}")
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            print(f"   GPU数量: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        return cuda_available
    except Exception as e:
        print(f"   ✗ GPU测试失败: {e}")
        return False

def test_yolov7_training_script():
    """测试YOLOv7训练脚本"""
    print("\n4. 测试YOLOv7训练脚本...")
    
    try:
        from models.train_yolov7_mask_detection import YOLOv7MaskDetectionTrainer
        print("   ✓ YOLOv7训练脚本导入成功")
        
        # 创建训练器实例（不实际训练）
        trainer = YOLOv7MaskDetectionTrainer(
            model_size="yolov7-tiny",
            batch_size=1,
            epochs=1,
            device="cpu"
        )
        print("   ✓ 训练器实例创建成功")
        
        return True
    except Exception as e:
        print(f"   ✗ 训练脚本测试失败: {e}")
        return False

def test_yolov7_model_loader():
    """测试YOLOv7模型加载器"""
    print("\n5. 测试YOLOv7模型加载器...")
    
    try:
        from models.yolov7_model_loader import YOLOv7ModelLoader
        print("   ✓ YOLOv7模型加载器导入成功")
        
        # 注意：这里不实际加载模型，因为可能没有权重文件
        print("   ✓ 模型加载器类可用")
        
        return True
    except Exception as e:
        print(f"   ✗ 模型加载器测试失败: {e}")
        return False

def test_yolov7_gpu_optimizer():
    """测试YOLOv7 GPU优化器"""
    print("\n6. 测试YOLOv7 GPU优化器...")
    
    try:
        from gpu_optimization.yolov7_gpu_optimizer import YOLOv7GPUOptimizer
        print("   ✓ YOLOv7 GPU优化器导入成功")
        
        optimizer = YOLOv7GPUOptimizer()
        print("   ✓ GPU优化器实例创建成功")
        
        # 测试批次大小计算
        result = optimizer.calculate_optimal_batch_size('yolov7-tiny')
        print(f"   ✓ 批次大小计算: {result['recommended_batch_size']}")
        
        # 测试训练配置优化
        config = optimizer.optimize_training_config('yolov7-tiny')
        print(f"   ✓ 训练配置优化: batch_size={config['batch_size']}")
        
        return True
    except Exception as e:
        print(f"   ✗ GPU优化器测试失败: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("\n7. 测试文件结构...")
    
    required_paths = [
        "yolov7",
        "models",
        "gpu_optimization",
        "data",
        "weights",
        "runs"
    ]
    
    all_exist = True
    for path in required_paths:
        path_obj = Path(path)
        exists = path_obj.exists()
        print(f"   {path}: {'✓' if exists else '✗'}")
        if not exists and path not in ['weights', 'runs']:  # weights和runs目录可能不存在
            all_exist = False
    
    return all_exist

def create_sample_data_config():
    """创建示例数据配置"""
    print("\n8. 创建示例数据配置...")
    
    try:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        config = {
            'train': 'datasets/mask_detection/train',
            'val': 'datasets/mask_detection/val',
            'test': 'datasets/mask_detection/test',
            'nc': 2,
            'names': ['no_mask', 'mask']
        }
        
        import yaml
        config_path = data_dir / "mask_detection.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"   ✓ 数据配置文件已创建: {config_path}")
        return True
    except Exception as e:
        print(f"   ✗ 创建数据配置失败: {e}")
        return False

def main():
    """主测试函数"""
    print("YOLOv7环境测试")
    print("=" * 50)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(("基础导入", test_basic_imports()))
    test_results.append(("YOLOv7模块导入", test_yolov7_imports()))
    test_results.append(("GPU可用性", test_gpu_availability()))
    test_results.append(("训练脚本", test_yolov7_training_script()))
    test_results.append(("模型加载器", test_yolov7_model_loader()))
    test_results.append(("GPU优化器", test_yolov7_gpu_optimizer()))
    test_results.append(("文件结构", test_file_structure()))
    test_results.append(("数据配置", create_sample_data_config()))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！YOLOv7环境配置完成。")
        print("\n下一步:")
        print("1. 下载YOLOv7预训练权重")
        print("2. 准备口罩检测数据集")
        print("3. 开始训练模型")
    else:
        print("⚠️  部分测试失败，请检查相关配置。")
        
        # 提供修复建议
        print("\n修复建议:")
        for test_name, result in test_results:
            if not result:
                if "导入" in test_name:
                    print(f"- {test_name}: 检查相关库是否正确安装")
                elif "GPU" in test_name:
                    print(f"- {test_name}: 检查CUDA和PyTorch GPU版本")
                elif "文件结构" in test_name:
                    print(f"- {test_name}: 确保所有必要目录存在")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)