#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前端集成测试
测试前端页面与后端API的完整交互流程
"""

import os
import sys
import time
import requests
import json
from io import BytesIO
from PIL import Image
import base64

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FrontendIntegrationTest:
    """前端集成测试类"""
    
    def __init__(self):
        self.backend_url = "http://localhost:5000"
        self.frontend_url = "http://localhost:8080"
        self.api_base = f"{self.backend_url}/api"
        self.session = requests.Session()
        
        # 设置CORS头
        self.session.headers.update({
            'Origin': self.frontend_url,
            'Referer': f"{self.frontend_url}/templates/index.html"
        })
    
    def test_frontend_accessibility(self):
        """测试前端页面可访问性"""
        print("=== 测试前端页面可访问性 ===")
        
        try:
            # 测试主页面
            response = self.session.get(f"{self.frontend_url}/templates/index.html")
            assert response.status_code == 200, f"前端页面不可访问: {response.status_code}"
            
            content = response.text
            # 检查页面是否包含口罩相关内容（可能是繁体或简体）
            has_mask_content = ('口罩' in content or 'mask' in content.lower() or 
                              '检测' in content or 'detection' in content.lower())
            assert has_mask_content, "页面未包含口罩检测相关内容"
            assert 'class MaskDetectionApp' in content or 'MaskDetectionApp' in content, "JavaScript应用类未找到"
            assert 'api' in content.lower(), "页面中未找到API相关内容"
            
            print("✓ 前端页面可正常访问")
            print("✓ 页面内容完整")
            
            return True
            
        except Exception as e:
            print(f"✗ 前端页面访问失败: {e}")
            return False
    
    def test_api_cors_configuration(self):
        """测试API CORS配置"""
        print("\n=== 测试API CORS配置 ===")
        
        try:
            # 测试预检请求
            headers = {
                'Origin': self.frontend_url,
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'Content-Type'
            }
            
            response = self.session.options(f"{self.api_base}/detect", headers=headers)
            assert response.status_code == 200, f"CORS预检请求失败: {response.status_code}"
            
            # 检查CORS头
            cors_headers = response.headers
            assert 'Access-Control-Allow-Origin' in cors_headers, "缺少Access-Control-Allow-Origin头"
            assert 'Access-Control-Allow-Methods' in cors_headers, "缺少Access-Control-Allow-Methods头"
            
            print("✓ CORS预检请求成功")
            print(f"✓ 允许的源: {cors_headers.get('Access-Control-Allow-Origin')}")
            print(f"✓ 允许的方法: {cors_headers.get('Access-Control-Allow-Methods')}")
            
            return True
            
        except Exception as e:
            print(f"✗ CORS配置测试失败: {e}")
            return False
    
    def test_frontend_api_integration(self):
        """测试前端与API的集成"""
        print("\n=== 测试前端与API的集成 ===")
        
        try:
            # 创建测试图像
            test_image = Image.new('RGB', (640, 480), color='blue')
            img_buffer = BytesIO()
            test_image.save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            
            # 模拟前端请求
            files = {'image': ('test.jpg', img_buffer, 'image/jpeg')}
            headers = {
                'Origin': self.frontend_url,
                'Referer': f"{self.frontend_url}/templates/index.html"
            }
            
            response = self.session.post(f"{self.api_base}/detect", files=files, headers=headers)
            assert response.status_code == 200, f"API请求失败: {response.status_code}"
            
            # 检查响应格式
            data = response.json()
            assert 'success' in data, "响应缺少success字段"
            assert 'detections' in data, "响应缺少detections字段"
            assert 'inference_time' in data, "响应缺少inference_time字段"
            assert 'processing_time' in data, "响应缺少processing_time字段"
            assert 'image_shape' in data, "响应缺少image_shape字段"
            
            print("✓ API请求成功")
            print(f"✓ 检测结果数量: {len(data['detections'])}")
            print(f"✓ 推理时间: {data['inference_time']:.3f}s")
            print(f"✓ 处理时间: {data['processing_time']:.3f}s")
            
            return True
            
        except Exception as e:
            print(f"✗ 前端API集成测试失败: {e}")
            return False
    
    def test_batch_upload_integration(self):
        """测试批量上传集成"""
        print("\n=== 测试批量上传集成 ===")
        
        try:
            # 创建多个测试图像
            images = []
            for i, color in enumerate(['red', 'green', 'blue']):
                test_image = Image.new('RGB', (640, 480), color=color)
                img_buffer = BytesIO()
                test_image.save(img_buffer, format='JPEG')
                img_buffer.seek(0)
                images.append(('images', (f'test_{i}.jpg', img_buffer, 'image/jpeg')))
            
            # 模拟前端批量请求
            headers = {
                'Origin': self.frontend_url,
                'Referer': f"{self.frontend_url}/templates/index.html"
            }
            
            response = self.session.post(f"{self.api_base}/batch_detect", files=images, headers=headers)
            assert response.status_code == 200, f"批量API请求失败: {response.status_code}"
            
            # 检查响应格式
            data = response.json()
            assert 'total_images' in data, "批量响应缺少total_images字段"
            assert 'results' in data, "批量响应缺少results字段"
            assert len(data['results']) == 3, f"批量结果数量不正确: {len(data['results'])}"
            
            # 检查每个结果
            for i, result in enumerate(data['results']):
                assert 'detections' in result, f"结果{i}缺少detections字段"
                assert 'inference_time' in result, f"结果{i}缺少inference_time字段"
                assert 'image_shape' in result, f"结果{i}缺少image_shape字段"
            
            print("✓ 批量API请求成功")
            print(f"✓ 处理图像数量: {len(data['results'])}")
            
            total_inference_time = sum(r['inference_time'] for r in data['results'])
            print(f"✓ 总推理时间: {total_inference_time:.3f}s")
            
            return True
            
        except Exception as e:
            print(f"✗ 批量上传集成测试失败: {e}")
            return False
    
    def test_error_handling_integration(self):
        """测试错误处理集成"""
        print("\n=== 测试错误处理集成 ===")
        
        try:
            # 测试无文件上传
            headers = {
                'Origin': self.frontend_url,
                'Referer': f"{self.frontend_url}/templates/index.html"
            }
            
            response = self.session.post(f"{self.api_base}/detect", headers=headers)
            assert response.status_code == 400, f"错误处理不正确: {response.status_code}"
            
            data = response.json()
            assert 'success' in data or 'error' in data, "错误响应缺少success或error字段"
            if 'success' in data:
                assert not data['success'], "错误响应success字段应为false"
            if 'error' in data:
                print(f"✓ 错误信息: {data['error']}")
            else:
                print("✓ 无文件上传错误处理正确")
            
            # 测试无效文件上传
            invalid_data = BytesIO(b"invalid image data")
            files = {'image': ('invalid.txt', invalid_data, 'text/plain')}
            
            response = self.session.post(f"{self.api_base}/detect", files=files, headers=headers)
            # 可能返回400或500，取决于具体实现
            assert response.status_code in [400, 500], f"无效文件错误处理不正确: {response.status_code}"
            
            data = response.json()
            assert 'success' in data or 'error' in data, "无效文件错误响应缺少success或error字段"
            if 'success' in data:
                assert not data['success'], "无效文件错误响应success字段应为false"
            
            print("✓ 无效文件上传错误处理正确")
            
            return True
            
        except Exception as e:
            print(f"✗ 错误处理集成测试失败: {e}")
            return False
    
    def test_health_check_integration(self):
        """测试健康检查集成"""
        print("\n=== 测试健康检查集成 ===")
        
        try:
            headers = {
                'Origin': self.frontend_url,
                'Referer': f"{self.frontend_url}/templates/index.html"
            }
            
            response = self.session.get(f"{self.api_base}/health", headers=headers)
            assert response.status_code == 200, f"健康检查失败: {response.status_code}"
            
            data = response.json()
            assert 'status' in data, "健康检查响应缺少status字段"
            assert data['status'] == 'healthy', f"服务状态不健康: {data['status']}"
            assert 'timestamp' in data, "健康检查响应缺少timestamp字段"
            
            print("✓ 健康检查成功")
            print(f"✓ 服务状态: {data['status']}")
            print(f"✓ 模型加载状态: {data.get('model_loaded', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"✗ 健康检查集成测试失败: {e}")
            return False
    
    def test_response_format_consistency(self):
        """测试响应格式一致性"""
        print("\n=== 测试响应格式一致性 ===")
        
        try:
            # 创建测试图像
            test_image = Image.new('RGB', (640, 480), color='yellow')
            img_buffer = BytesIO()
            test_image.save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            
            files = {'image': ('test.jpg', img_buffer, 'image/jpeg')}
            headers = {
                'Origin': self.frontend_url,
                'Referer': f"{self.frontend_url}/templates/index.html"
            }
            
            response = self.session.post(f"{self.api_base}/detect", files=files, headers=headers)
            assert response.status_code == 200, f"API请求失败: {response.status_code}"
            
            data = response.json()
            
            # 检查响应头
            assert 'Content-Type' in response.headers, "响应缺少Content-Type头"
            assert 'application/json' in response.headers['Content-Type'], "响应Content-Type不正确"
            
            # 检查数据类型
            assert isinstance(data['success'], bool), "success字段类型不正确"
            assert isinstance(data['detections'], list), "detections字段类型不正确"
            assert isinstance(data['inference_time'], (int, float)), "inference_time字段类型不正确"
            assert isinstance(data['processing_time'], (int, float)), "processing_time字段类型不正确"
            assert isinstance(data['image_shape'], (list, tuple)), "image_shape字段类型不正确"
            
            # 检查图像信息
            image_shape = data['image_shape']
            assert len(image_shape) >= 2, "图像形状信息不完整"
            
            print("✓ 响应格式一致性检查通过")
            print(f"✓ 响应Content-Type: {response.headers['Content-Type']}")
            print(f"✓ 图像尺寸: {image_shape[1]}x{image_shape[0]}")
            
            return True
            
        except Exception as e:
            print(f"✗ 响应格式一致性测试失败: {e}")
            return False

def run_frontend_integration_tests():
    """运行前端集成测试"""
    print("=" * 60)
    print("开始运行前端集成测试")
    print("=" * 60)
    
    # 检查服务是否可用
    try:
        backend_response = requests.get("http://localhost:5000/api/health", timeout=5)
        frontend_response = requests.get("http://localhost:8080/templates/index.html", timeout=5)
        
        if backend_response.status_code != 200:
            print("错误: 后端服务不可用，请先启动后端服务")
            return False
            
        if frontend_response.status_code != 200:
            print("错误: 前端服务不可用，请先启动前端服务")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"错误: 无法连接到服务: {e}")
        return False
    
    # 创建测试实例
    test = FrontendIntegrationTest()
    
    # 运行测试
    tests = [
        test.test_frontend_accessibility,
        test.test_api_cors_configuration,
        test.test_frontend_api_integration,
        test.test_batch_upload_integration,
        test.test_error_handling_integration,
        test.test_health_check_integration,
        test.test_response_format_consistency
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ 测试执行异常: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("前端集成测试完成")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"总计: {passed + failed}")
    print("=" * 60)
    
    return failed == 0

if __name__ == '__main__':
    success = run_frontend_integration_tests()
    sys.exit(0 if success else 1)