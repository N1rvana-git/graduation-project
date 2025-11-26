#!/usr/bin/env python3
"""
GPU环境配置脚本
检查并配置CUDA环境，安装支持GPU的PyTorch
"""

import subprocess
import sys
import os
import platform

def run_command(cmd, shell=True):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, encoding='utf-8')
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_python_env():
    """检查Python环境"""
    print("=== Python环境检查 ===")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    # 检查pip
    success, stdout, stderr = run_command([sys.executable, "-m", "pip", "--version"], shell=False)
    if success:
        print(f"pip版本: {stdout.strip()}")
    else:
        print("pip未正确安装")
        return False
    return True

def check_cuda_env():
    """检查CUDA环境"""
    print("\n=== CUDA环境检查 ===")
    
    # 检查nvidia-smi
    success, stdout, stderr = run_command("nvidia-smi")
    if success:
        lines = stdout.split('\n')
        for line in lines:
            if 'CUDA Version:' in line:
                cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                print(f"CUDA驱动版本: {cuda_version}")
                return cuda_version
    else:
        print("未检测到NVIDIA GPU或CUDA驱动")
        return None

def check_current_torch():
    """检查当前PyTorch安装情况"""
    print("\n=== PyTorch环境检查 ===")
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    except ImportError:
        print("PyTorch未安装")
        return False

def install_pytorch_gpu():
    """安装支持GPU的PyTorch"""
    print("\n=== 安装PyTorch GPU版本 ===")
    
    # 根据CUDA版本选择合适的PyTorch版本
    # CUDA 12.x 使用 cu121
    install_cmd = [
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]
    
    print("正在安装PyTorch GPU版本...")
    print(f"命令: {' '.join(install_cmd)}")
    
    success, stdout, stderr = run_command(install_cmd, shell=False)
    if success:
        print("PyTorch GPU版本安装成功!")
        return True
    else:
        print(f"安装失败: {stderr}")
        return False

def install_other_dependencies():
    """安装其他GPU相关依赖"""
    print("\n=== 安装其他依赖 ===")
    
    dependencies = [
        "ultralytics",
        "opencv-python",
        "numpy",
        "matplotlib",
        "Pillow",
        "PyYAML",
        "tqdm"
    ]
    
    for dep in dependencies:
        print(f"安装 {dep}...")
        success, stdout, stderr = run_command([sys.executable, "-m", "pip", "install", dep], shell=False)
        if success:
            print(f"✅ {dep} 安装成功")
        else:
            print(f"❌ {dep} 安装失败: {stderr}")

def test_gpu_setup():
    """测试GPU配置"""
    print("\n=== GPU配置测试 ===")
    
    test_code = '''
import torch
import time

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} 显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # 简单的GPU计算测试
    device = torch.device("cuda:0")
    print(f"\\n使用设备: {device}")
    
    # 创建测试张量
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    # 计时GPU计算
    start_time = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()  # 等待GPU计算完成
    gpu_time = time.time() - start_time
    
    print(f"GPU矩阵乘法耗时: {gpu_time:.4f}秒")
    print("✅ GPU配置测试成功!")
else:
    print("❌ CUDA不可用，请检查安装")
'''
    
    success, stdout, stderr = run_command([sys.executable, "-c", test_code], shell=False)
    if success:
        print(stdout)
    else:
        print(f"测试失败: {stderr}")

def main():
    """主函数"""
    print("🚀 开始配置GPU环境...")
    
    # 1. 检查Python环境
    if not check_python_env():
        print("❌ Python环境有问题，请先解决Python安装问题")
        return
    
    # 2. 检查CUDA环境
    cuda_version = check_cuda_env()
    if not cuda_version:
        print("❌ 未检测到CUDA环境")
        return
    
    # 3. 检查当前PyTorch
    torch_installed = check_current_torch()
    
    # 4. 安装PyTorch GPU版本
    if not torch_installed or input("是否重新安装PyTorch GPU版本? (y/n): ").lower() == 'y':
        if install_pytorch_gpu():
            print("✅ PyTorch GPU版本安装完成")
        else:
            print("❌ PyTorch安装失败")
            return
    
    # 5. 安装其他依赖
    install_other_dependencies()
    
    # 6. 测试GPU配置
    test_gpu_setup()
    
    print("\n🎉 GPU环境配置完成!")
    print("现在您可以使用GPU进行模型训练和推理了。")

if __name__ == "__main__":
    main()