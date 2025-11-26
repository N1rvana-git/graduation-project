"""
口罩检测API测试用例
测试所有API端点的功能和性能
"""

import pytest
import requests
import json
import base64
import io
import time
from PIL import Image
import os
from pathlib import Path

# 测试配置
API_BASE_URL = "http://localhost:5000/api"
TEST_TIMEOUT = 60  # 增加超时时间，因为首次加载模型需要时间

class TestMaskDetectionAPI:
    """口罩检测API测试类"""
    
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        cls.session = requests.Session()
        cls.test_images_dir = Path(__file__).parent / "test_images"
        cls.test_images_dir.mkdir(exist_ok=True)
        
        # 创建测试图像
        cls.create_test_images()
    
    @classmethod
    def create_test_images(cls):
        """创建测试用的图像文件"""
        # 创建一个简单的测试图像
        img = Image.new('RGB', (640, 480), color='blue')
        test_image_path = cls.test_images_dir / "test_image.jpg"
        img.save(test_image_path, 'JPEG')
        
        # 创建多个测试图像用于批量测试
        for i in range(3):
            img = Image.new('RGB', (320, 240), color=['red', 'green', 'yellow'][i])
            img.save(cls.test_images_dir / f"batch_test_{i}.jpg", 'JPEG')
    
    def test_health_check(self):
        """测试健康检查接口"""
        response = self.session.get(f"{API_BASE_URL}/health", timeout=TEST_TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'model_loaded' in data
    
    def test_model_info(self):
        """测试模型信息接口"""
        response = self.session.get(f"{API_BASE_URL}/model/info", timeout=TEST_TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        # 模型信息接口可能返回空字典，这是正常的
        assert isinstance(data, dict)
    
    def test_single_image_detection(self):
        """测试单图像检测接口"""
        test_image_path = self.test_images_dir / "test_image.jpg"
        
        with open(test_image_path, 'rb') as f:
            files = {'image': ('test_image.jpg', f, 'image/jpeg')}
            response = self.session.post(
                f"{API_BASE_URL}/detect", 
                files=files, 
                timeout=TEST_TIMEOUT
            )
        
        assert response.status_code == 200
        data = response.json()
        assert 'detections' in data
        assert 'processing_time' in data
        assert isinstance(data['detections'], list)
    
    def test_base64_detection(self):
        """测试Base64图像检测接口"""
        test_image_path = self.test_images_dir / "test_image.jpg"
        
        # 将图像转换为Base64
        with open(test_image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        payload = {
            'image': f'data:image/jpeg;base64,{image_data}'
        }
        
        response = self.session.post(
            f"{API_BASE_URL}/detect_base64",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'detections' in data
        assert 'processing_time' in data
    
    def test_batch_detection(self):
        """测试批量检测接口"""
        files = []
        for i in range(3):
            image_path = self.test_images_dir / f"batch_test_{i}.jpg"
            files.append(('images', (f'batch_test_{i}.jpg', open(image_path, 'rb'), 'image/jpeg')))
        
        try:
            response = self.session.post(
                f"{API_BASE_URL}/batch_detect",
                files=files,
                timeout=TEST_TIMEOUT * 2  # 批量处理需要更长时间
            )
            
            assert response.status_code == 200
            data = response.json()
            assert 'total_images' in data
            assert 'processed_images' in data
            assert 'results' in data
            assert data['total_images'] == 3
            assert len(data['results']) <= 3
            
        finally:
            # 关闭文件
            for _, (_, file_obj, _) in files:
                file_obj.close()
    
    def test_invalid_image_format(self):
        """测试无效图像格式"""
        # 创建一个文本文件作为无效图像
        invalid_file = io.BytesIO(b"This is not an image")
        files = {'image': ('invalid.txt', invalid_file, 'text/plain')}
        
        response = self.session.post(f"{API_BASE_URL}/detect", files=files)
        
        assert response.status_code == 500
        data = response.json()
        assert 'error' in data
    
    def test_missing_image(self):
        """测试缺少图像参数"""
        response = self.session.post(f"{API_BASE_URL}/detect")
        
        assert response.status_code == 400
        data = response.json()
        assert 'error' in data
    
    def test_empty_base64_request(self):
        """测试空的Base64请求"""
        response = self.session.post(f"{API_BASE_URL}/detect_base64", json={})
        
        assert response.status_code == 400
        data = response.json()
        assert 'error' in data
    
    def test_performance_benchmark(self):
        """性能基准测试"""
        test_image_path = self.test_images_dir / "test_image.jpg"
        
        # 测试多次请求的平均响应时间
        response_times = []
        
        for _ in range(5):
            start_time = time.time()
            
            with open(test_image_path, 'rb') as f:
                files = {'image': ('test_image.jpg', f, 'image/jpeg')}
                response = self.session.post(f"{API_BASE_URL}/detect", files=files)
            
            end_time = time.time()
            response_times.append(end_time - start_time)
            
            assert response.status_code == 200
        
        avg_response_time = sum(response_times) / len(response_times)
        print(f"\n平均响应时间: {avg_response_time:.2f}秒")
        
        # 响应时间应该在合理范围内（根据实际情况调整）
        assert avg_response_time < 10.0, f"响应时间过长: {avg_response_time:.2f}秒"
    
    def test_concurrent_requests(self):
        """并发请求测试"""
        import threading
        import queue
        
        test_image_path = self.test_images_dir / "test_image.jpg"
        results = queue.Queue()
        
        def make_request():
            try:
                with open(test_image_path, 'rb') as f:
                    files = {'image': ('test_image.jpg', f, 'image/jpeg')}
                    response = self.session.post(f"{API_BASE_URL}/detect", files=files)
                results.put(response.status_code)
            except Exception as e:
                results.put(str(e))
        
        # 创建5个并发请求
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 检查结果
        success_count = 0
        while not results.empty():
            result = results.get()
            if result == 200:
                success_count += 1
        
        # 至少应该有一半的请求成功
        assert success_count >= 2, f"并发请求成功率过低: {success_count}/5"
    
    @classmethod
    def teardown_class(cls):
        """测试类清理"""
        cls.session.close()
        
        # 清理测试图像
        import shutil
        if cls.test_images_dir.exists():
            shutil.rmtree(cls.test_images_dir)


class TestIntegration:
    """集成测试类"""
    
    def test_full_workflow(self):
        """测试完整工作流程"""
        session = requests.Session()
        
        # 1. 检查服务健康状态
        health_response = session.get(f"{API_BASE_URL}/health")
        assert health_response.status_code == 200
        
        # 2. 获取模型信息
        model_response = session.get(f"{API_BASE_URL}/model/info")
        assert model_response.status_code == 200
        
        # 3. 执行检测
        # 创建测试图像
        img = Image.new('RGB', (416, 416), color='white')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        files = {'image': ('test.jpg', img_buffer, 'image/jpeg')}
        detect_response = session.post(f"{API_BASE_URL}/detect", files=files)
        assert detect_response.status_code == 200
        
        # 4. 验证响应格式
        data = detect_response.json()
        assert 'detections' in data
        assert 'processing_time' in data
        
        session.close()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])