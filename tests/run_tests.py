"""
测试运行脚本
提供便捷的测试执行接口
"""

import os
import sys
import subprocess
import argparse
import time
import requests
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 服务配置
BACKEND_URL = "http://localhost:5000"
FRONTEND_URL = "http://localhost:8080"

class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.tests_dir = Path(__file__).parent
    
    def check_service_availability(self, url, service_name, timeout=30):
        """检查服务可用性"""
        print(f"检查{service_name}服务可用性...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if service_name == "后端":
                    response = requests.get(f"{url}/api/health", timeout=5)
                else:
                    response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    print(f"✅ {service_name}服务可用")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        print(f"❌ {service_name}服务不可用")
        return False
    
    def run_api_tests(self, verbose=False):
        """运行API测试"""
        print("\n" + "="*50)
        print("运行API测试")
        print("="*50)
        
        # 检查后端服务
        if not self.check_service_availability(BACKEND_URL, "后端"):
            print("⚠️  后端服务不可用，跳过API测试")
            return False
        
        # 运行API测试
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "test_api.py"),
            "-v" if verbose else "-q",
            "--tb=short",
            "-x"  # 遇到第一个失败就停止
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ API测试通过")
                if verbose:
                    print(result.stdout)
                return True
            else:
                print("❌ API测试失败")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 运行API测试时出错: {str(e)}")
            return False
    
    def run_integration_tests(self, verbose=False):
        """运行集成测试"""
        print("\n" + "="*50)
        print("运行集成测试")
        print("="*50)
        
        # 检查服务可用性
        backend_available = self.check_service_availability(BACKEND_URL, "后端")
        frontend_available = self.check_service_availability(FRONTEND_URL, "前端")
        
        if not backend_available:
            print("⚠️  后端服务不可用，跳过集成测试")
            return False
        
        # 运行集成测试
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "test_integration.py"),
            "-v" if verbose else "-q",
            "--tb=short"
        ]
        
        # 如果前端不可用，跳过前端相关测试
        if not frontend_available:
            cmd.extend(["-k", "not frontend"])
            print("⚠️  前端服务不可用，跳过前端相关测试")
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 集成测试通过")
                if verbose:
                    print(result.stdout)
                return True
            else:
                print("❌ 集成测试失败")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 运行集成测试时出错: {str(e)}")
            return False
    
    def run_performance_tests(self, verbose=False):
        """运行性能测试"""
        print("\n" + "="*50)
        print("运行性能测试")
        print("="*50)
        
        # 检查后端服务
        if not self.check_service_availability(BACKEND_URL, "后端"):
            print("⚠️  后端服务不可用，跳过性能测试")
            return False
        
        # 运行性能测试
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir),
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "performance"
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 性能测试通过")
                if verbose:
                    print(result.stdout)
                return True
            else:
                print("❌ 性能测试失败")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 运行性能测试时出错: {str(e)}")
            return False
    
    def run_all_tests(self, verbose=False, generate_report=False):
        """运行所有测试"""
        print("\n" + "="*60)
        print("运行完整测试套件")
        print("="*60)
        
        # 构建pytest命令
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir),
            "-v" if verbose else "-q",
            "--tb=short"
        ]
        
        # 添加HTML报告生成
        if generate_report:
            report_path = self.project_root / "test_report.html"
            cmd.extend(["--html", str(report_path), "--self-contained-html"])
            print(f"测试报告将生成到: {report_path}")
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            
            if result.returncode == 0:
                print("\n🎉 所有测试通过！")
                return True
            else:
                print("\n❌ 部分测试失败")
                return False
                
        except Exception as e:
            print(f"❌ 运行测试时出错: {str(e)}")
            return False
    
    def run_quick_check(self):
        """快速检查"""
        print("\n" + "="*50)
        print("快速系统检查")
        print("="*50)
        
        # 检查服务状态
        backend_ok = self.check_service_availability(BACKEND_URL, "后端", timeout=10)
        frontend_ok = self.check_service_availability(FRONTEND_URL, "前端", timeout=10)
        
        # 检查关键文件
        print("\n检查关键文件...")
        files_to_check = [
            "backend/app.py",
            "requirements.txt",
            "Dockerfile",
            "docker-compose.yml",
            "frontend/index.html",
            "deployment/nginx/nginx.conf"
        ]
        
        missing_files = []
        for file_path in files_to_check:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path}")
                missing_files.append(file_path)
        
        # 运行基本API测试
        if backend_ok:
            print("\n运行基本API测试...")
            try:
                response = requests.get(f"{BACKEND_URL}/api/health", timeout=10)
                if response.status_code == 200:
                    print("✅ API健康检查通过")
                else:
                    print(f"❌ API健康检查失败: {response.status_code}")
            except Exception as e:
                print(f"❌ API测试失败: {str(e)}")
        
        # 总结
        print("\n" + "="*50)
        print("检查结果总结")
        print("="*50)
        print(f"后端服务: {'✅ 正常' if backend_ok else '❌ 异常'}")
        print(f"前端服务: {'✅ 正常' if frontend_ok else '❌ 异常'}")
        print(f"关键文件: {'✅ 完整' if not missing_files else f'❌ 缺少 {len(missing_files)} 个文件'}")
        
        if missing_files:
            print("缺少的文件:")
            for file_path in missing_files:
                print(f"  - {file_path}")
        
        return backend_ok and frontend_ok and not missing_files

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='口罩检测系统测试运行器')
    parser.add_argument('--type', '-t', 
                       choices=['api', 'integration', 'performance', 'all', 'quick'],
                       default='all',
                       help='测试类型 (默认: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    parser.add_argument('--report', '-r', action='store_true',
                       help='生成HTML测试报告')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # 根据参数运行相应测试
    if args.type == 'quick':
        success = runner.run_quick_check()
    elif args.type == 'api':
        success = runner.run_api_tests(args.verbose)
    elif args.type == 'integration':
        success = runner.run_integration_tests(args.verbose)
    elif args.type == 'performance':
        success = runner.run_performance_tests(args.verbose)
    elif args.type == 'all':
        success = runner.run_all_tests(args.verbose, args.report)
    else:
        print(f"未知的测试类型: {args.type}")
        return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())