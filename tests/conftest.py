"""
pytest配置文件
设置测试环境和共享fixtures
"""

import pytest
import requests
import time
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 测试配置
BACKEND_URL = "http://localhost:5000"
FRONTEND_URL = "http://localhost:8080"
MAX_WAIT_TIME = 60  # 最大等待时间（秒）

@pytest.fixture(scope="session")
def backend_url():
    """后端服务URL fixture"""
    return BACKEND_URL

@pytest.fixture(scope="session")
def frontend_url():
    """前端服务URL fixture"""
    return FRONTEND_URL

@pytest.fixture(scope="session")
def wait_for_backend():
    """等待后端服务启动的fixture"""
    def _wait_for_service(url, timeout=MAX_WAIT_TIME):
        """等待服务启动"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{url}/api/health", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        return False
    
    # 等待后端服务启动
    if not _wait_for_service(BACKEND_URL):
        pytest.skip(f"后端服务在{MAX_WAIT_TIME}秒内未启动，跳过相关测试")
    
    return BACKEND_URL

@pytest.fixture(scope="session")
def wait_for_frontend():
    """等待前端服务启动的fixture"""
    def _wait_for_service(url, timeout=MAX_WAIT_TIME):
        """等待服务启动"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        return False
    
    # 等待前端服务启动
    if not _wait_for_service(FRONTEND_URL):
        pytest.skip(f"前端服务在{MAX_WAIT_TIME}秒内未启动，跳过相关测试")
    
    return FRONTEND_URL

@pytest.fixture(scope="function")
def api_client():
    """API客户端fixture"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'pytest-api-client/1.0'
    })
    yield session
    session.close()

@pytest.fixture(scope="session")
def test_images_dir():
    """测试图像目录fixture"""
    images_dir = Path(__file__).parent / "test_images"
    images_dir.mkdir(exist_ok=True)
    
    # 创建基本测试图像
    from PIL import Image
    
    # 创建测试图像
    test_image = Image.new('RGB', (640, 480), color='lightblue')
    test_image.save(images_dir / "test_image.jpg", 'JPEG')
    
    # 创建小尺寸图像
    small_image = Image.new('RGB', (320, 240), color='lightgreen')
    small_image.save(images_dir / "small_image.jpg", 'JPEG')
    
    # 创建大尺寸图像
    large_image = Image.new('RGB', (1920, 1080), color='lightcoral')
    large_image.save(images_dir / "large_image.jpg", 'JPEG')
    
    yield images_dir
    
    # 清理测试图像
    import shutil
    if images_dir.exists():
        shutil.rmtree(images_dir)

@pytest.fixture(scope="function")
def temp_image():
    """临时图像fixture"""
    from PIL import Image
    import tempfile
    
    # 创建临时图像
    image = Image.new('RGB', (416, 416), color='white')
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        image.save(tmp_file.name, 'JPEG')
        yield tmp_file.name
    
    # 清理临时文件
    try:
        os.unlink(tmp_file.name)
    except FileNotFoundError:
        pass

def pytest_configure(config):
    """pytest配置钩子"""
    # 添加自定义标记
    config.addinivalue_line(
        "markers", "integration: 标记集成测试"
    )
    config.addinivalue_line(
        "markers", "api: 标记API测试"
    )
    config.addinivalue_line(
        "markers", "frontend: 标记前端测试"
    )
    config.addinivalue_line(
        "markers", "performance: 标记性能测试"
    )
    config.addinivalue_line(
        "markers", "slow: 标记慢速测试"
    )

def pytest_collection_modifyitems(config, items):
    """修改测试收集项"""
    # 为慢速测试添加标记
    for item in items:
        if "performance" in item.nodeid or "concurrent" in item.nodeid:
            item.add_marker(pytest.mark.slow)

def pytest_runtest_setup(item):
    """测试运行前的设置"""
    # 检查是否需要跳过某些测试
    if "frontend" in item.keywords:
        # 检查前端服务是否可用
        try:
            response = requests.get(FRONTEND_URL, timeout=5)
            if response.status_code != 200:
                pytest.skip("前端服务不可用")
        except requests.exceptions.RequestException:
            pytest.skip("前端服务不可用")
    
    if "api" in item.keywords:
        # 检查后端API是否可用
        try:
            response = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("后端API服务不可用")
        except requests.exceptions.RequestException:
            pytest.skip("后端API服务不可用")

@pytest.fixture(autouse=True)
def setup_test_environment():
    """自动设置测试环境"""
    # 设置测试环境变量
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'INFO'
    
    yield
    
    # 清理环境变量
    os.environ.pop('TESTING', None)
    os.environ.pop('LOG_LEVEL', None)

# 测试报告钩子
def pytest_html_report_title(report):
    """自定义HTML报告标题"""
    report.title = "口罩检测系统测试报告"

def pytest_html_results_summary(prefix, summary, postfix):
    """自定义HTML报告摘要"""
    prefix.extend([
        "<h2>测试环境信息</h2>",
        f"<p>后端服务: {BACKEND_URL}</p>",
        f"<p>前端服务: {FRONTEND_URL}</p>",
        f"<p>测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>"
    ])

# 性能测试配置
@pytest.fixture(scope="session")
def performance_config():
    """性能测试配置"""
    return {
        'max_response_time': 10.0,  # 最大响应时间（秒）
        'concurrent_requests': 5,   # 并发请求数
        'test_iterations': 3,       # 测试迭代次数
    }