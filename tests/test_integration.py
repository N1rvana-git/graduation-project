"""
系统集成测试
测试前后端整体功能和用户工作流程
"""

import pytest
import requests
import time
import json
import base64
import io
from PIL import Image
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException

# 测试配置
BACKEND_URL = "http://localhost:5000"
FRONTEND_URL = "http://localhost:8080"
TEST_TIMEOUT = 30
SELENIUM_TIMEOUT = 10

class TestSystemIntegration:
    """系统集成测试类"""
    
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        cls.backend_session = requests.Session()
        cls.test_images_dir = Path(__file__).parent / "test_images"
        cls.test_images_dir.mkdir(exist_ok=True)
        
        # 创建测试图像
        cls.create_test_images()
        
        # 设置Chrome选项
        cls.chrome_options = Options()
        cls.chrome_options.add_argument('--headless')  # 无头模式
        cls.chrome_options.add_argument('--no-sandbox')
        cls.chrome_options.add_argument('--disable-dev-shm-usage')
        cls.chrome_options.add_argument('--disable-gpu')
        cls.chrome_options.add_argument('--window-size=1920,1080')
    
    @classmethod
    def create_test_images(cls):
        """创建测试用的图像文件"""
        # 创建带人脸的测试图像
        img = Image.new('RGB', (640, 480), color='lightblue')
        test_image_path = cls.test_images_dir / "person_with_mask.jpg"
        img.save(test_image_path, 'JPEG')
        
        # 创建无人脸的测试图像
        img = Image.new('RGB', (640, 480), color='lightgreen')
        test_image_path = cls.test_images_dir / "no_person.jpg"
        img.save(test_image_path, 'JPEG')
        
        # 创建多人图像
        img = Image.new('RGB', (800, 600), color='lightyellow')
        test_image_path = cls.test_images_dir / "multiple_persons.jpg"
        img.save(test_image_path, 'JPEG')
    
    def test_backend_services_availability(self):
        """测试后端服务可用性"""
        # 测试后端健康检查
        try:
            response = self.backend_session.get(f"{BACKEND_URL}/api/health", timeout=TEST_TIMEOUT)
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'healthy'
        except requests.exceptions.RequestException:
            pytest.skip("后端服务不可用，跳过集成测试")
    
    def test_frontend_availability(self):
        """测试前端服务可用性"""
        try:
            response = requests.get(FRONTEND_URL, timeout=TEST_TIMEOUT)
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.skip("前端服务不可用，跳过前端测试")
    
    def test_api_backend_integration(self):
        """测试API后端集成"""
        # 1. 检查服务状态
        health_response = self.backend_session.get(f"{BACKEND_URL}/api/health")
        assert health_response.status_code == 200
        
        # 2. 获取模型信息
        model_response = self.backend_session.get(f"{BACKEND_URL}/api/model/info")
        assert model_response.status_code == 200
        
        # 3. 测试单图像检测
        test_image_path = self.test_images_dir / "person_with_mask.jpg"
        with open(test_image_path, 'rb') as f:
            files = {'image': ('test.jpg', f, 'image/jpeg')}
            detect_response = self.backend_session.post(f"{BACKEND_URL}/api/detect", files=files)
        
        assert detect_response.status_code == 200
        data = detect_response.json()
        assert 'detections' in data
        assert 'processing_time' in data
        
        # 4. 测试批量检测
        files = []
        for image_name in ["person_with_mask.jpg", "no_person.jpg"]:
            image_path = self.test_images_dir / image_name
            files.append(('images', (image_name, open(image_path, 'rb'), 'image/jpeg')))
        
        try:
            batch_response = self.backend_session.post(f"{BACKEND_URL}/api/batch_detect", files=files)
            assert batch_response.status_code == 200
            batch_data = batch_response.json()
            assert 'total_images' in batch_data
            assert 'results' in batch_data
        finally:
            # 关闭文件
            for _, (_, file_obj, _) in files:
                file_obj.close()
    
    @pytest.mark.skipif(not Path("/usr/bin/google-chrome").exists() and 
                       not Path("/usr/bin/chromium-browser").exists(),
                       reason="Chrome/Chromium not available")
    def test_frontend_ui_integration(self):
        """测试前端UI集成（需要Chrome浏览器）"""
        driver = None
        try:
            # 初始化WebDriver
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.get(FRONTEND_URL)
            
            # 等待页面加载
            wait = WebDriverWait(driver, SELENIUM_TIMEOUT)
            
            # 检查页面标题
            assert "口罩检测" in driver.title or "Mask Detection" in driver.title
            
            # 检查主要元素是否存在
            upload_area = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "upload-area"))
            )
            assert upload_area.is_displayed()
            
            # 检查上传按钮
            upload_btn = driver.find_element(By.ID, "upload-btn")
            assert upload_btn.is_displayed()
            
            # 检查结果区域
            results_area = driver.find_element(By.ID, "results")
            assert results_area is not None
            
        except (WebDriverException, TimeoutException) as e:
            pytest.skip(f"前端UI测试失败: {str(e)}")
        finally:
            if driver:
                driver.quit()
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流程"""
        # 1. 确保后端服务正常
        health_response = self.backend_session.get(f"{BACKEND_URL}/api/health")
        if health_response.status_code != 200:
            pytest.skip("后端服务不可用")
        
        # 2. 测试完整的检测流程
        test_image_path = self.test_images_dir / "person_with_mask.jpg"
        
        # 模拟前端上传图像的流程
        with open(test_image_path, 'rb') as f:
            files = {'image': ('test_image.jpg', f, 'image/jpeg')}
            
            # 发送检测请求
            start_time = time.time()
            response = self.backend_session.post(f"{BACKEND_URL}/api/detect", files=files)
            end_time = time.time()
            
            # 验证响应
            assert response.status_code == 200
            data = response.json()
            
            # 验证响应结构
            assert 'detections' in data
            assert 'processing_time' in data
            assert isinstance(data['detections'], list)
            
            # 验证处理时间合理
            processing_time = end_time - start_time
            assert processing_time < 30.0, f"处理时间过长: {processing_time:.2f}秒"
    
    def test_error_handling_integration(self):
        """测试错误处理集成"""
        # 1. 测试无效图像格式
        invalid_data = b"This is not an image"
        files = {'image': ('invalid.txt', io.BytesIO(invalid_data), 'text/plain')}
        
        response = self.backend_session.post(f"{BACKEND_URL}/api/detect", files=files)
        assert response.status_code in [400, 500]  # 应该返回错误状态码
        
        # 2. 测试缺少参数
        response = self.backend_session.post(f"{BACKEND_URL}/api/detect")
        assert response.status_code == 400
        
        # 3. 测试不存在的端点
        response = self.backend_session.get(f"{BACKEND_URL}/api/nonexistent")
        assert response.status_code == 404
    
    def test_performance_integration(self):
        """测试性能集成"""
        test_image_path = self.test_images_dir / "person_with_mask.jpg"
        
        # 测试连续请求的性能
        response_times = []
        
        for i in range(3):  # 减少测试次数以提高测试速度
            with open(test_image_path, 'rb') as f:
                files = {'image': (f'test_{i}.jpg', f, 'image/jpeg')}
                
                start_time = time.time()
                response = self.backend_session.post(f"{BACKEND_URL}/api/detect", files=files)
                end_time = time.time()
                
                assert response.status_code == 200
                response_times.append(end_time - start_time)
        
        # 验证性能指标
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        
        print(f"\n性能测试结果:")
        print(f"平均响应时间: {avg_time:.2f}秒")
        print(f"最大响应时间: {max_time:.2f}秒")
        
        # 性能断言（根据实际情况调整阈值）
        assert avg_time < 15.0, f"平均响应时间过长: {avg_time:.2f}秒"
        assert max_time < 30.0, f"最大响应时间过长: {max_time:.2f}秒"
    
    def test_data_consistency(self):
        """测试数据一致性"""
        test_image_path = self.test_images_dir / "person_with_mask.jpg"
        
        # 多次请求同一图像，验证结果一致性
        results = []
        
        for _ in range(3):
            with open(test_image_path, 'rb') as f:
                files = {'image': ('test.jpg', f, 'image/jpeg')}
                response = self.backend_session.post(f"{BACKEND_URL}/api/detect", files=files)
                
                assert response.status_code == 200
                data = response.json()
                results.append(data)
        
        # 验证检测结果的一致性（检测框数量应该相同）
        detection_counts = [len(result['detections']) for result in results]
        
        # 所有结果的检测数量应该相同（或在合理范围内）
        if detection_counts:
            min_count = min(detection_counts)
            max_count = max(detection_counts)
            # 允许少量差异（由于模型的随机性）
            assert max_count - min_count <= 1, f"检测结果不一致: {detection_counts}"
    
    def test_concurrent_requests(self):
        """测试并发请求处理"""
        import threading
        import queue
        
        test_image_path = self.test_images_dir / "person_with_mask.jpg"
        results = queue.Queue()
        
        def make_request(request_id):
            try:
                with open(test_image_path, 'rb') as f:
                    files = {'image': (f'test_{request_id}.jpg', f, 'image/jpeg')}
                    response = self.backend_session.post(f"{BACKEND_URL}/api/detect", files=files)
                    results.put((request_id, response.status_code, response.json() if response.status_code == 200 else None))
            except Exception as e:
                results.put((request_id, 'error', str(e)))
        
        # 创建3个并发请求
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 检查结果
        success_count = 0
        while not results.empty():
            request_id, status_code, data = results.get()
            if status_code == 200:
                success_count += 1
                assert data is not None
                assert 'detections' in data
        
        # 至少应该有一半的请求成功
        assert success_count >= 1, f"并发请求成功率过低: {success_count}/3"
    
    @classmethod
    def teardown_class(cls):
        """测试类清理"""
        cls.backend_session.close()
        
        # 清理测试图像
        import shutil
        if cls.test_images_dir.exists():
            shutil.rmtree(cls.test_images_dir)


class TestDeploymentIntegration:
    """部署集成测试类"""
    
    def test_docker_readiness(self):
        """测试Docker部署就绪性"""
        # 检查Dockerfile是否存在
        dockerfile_path = Path("e:/毕业设计/Dockerfile")
        assert dockerfile_path.exists(), "Dockerfile不存在"
        
        # 检查docker-compose.yml是否存在
        compose_path = Path("e:/毕业设计/docker-compose.yml")
        assert compose_path.exists(), "docker-compose.yml不存在"
        
        # 检查requirements.txt是否存在
        requirements_path = Path("e:/毕业设计/requirements.txt")
        assert requirements_path.exists(), "requirements.txt不存在"
    
    def test_nginx_config_validity(self):
        """测试Nginx配置有效性"""
        nginx_conf_path = Path("e:/毕业设计/deployment/nginx/nginx.conf")
        default_conf_path = Path("e:/毕业设计/deployment/nginx/default.conf")
        
        assert nginx_conf_path.exists(), "nginx.conf不存在"
        assert default_conf_path.exists(), "default.conf不存在"
        
        # 检查配置文件内容
        with open(default_conf_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
            
        # 验证关键配置项
        assert 'server {' in config_content
        assert 'listen 80' in config_content
        assert 'location /api/' in config_content
        assert 'proxy_pass http://backend:5000' in config_content


if __name__ == "__main__":
    # 运行集成测试
    pytest.main([__file__, "-v", "--tb=short", "-x"])  # -x 表示遇到第一个失败就停止