#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试
测试系统在不同负载下的性能表现
"""

import os
import sys
import time
import requests
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from PIL import Image
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PerformanceTest:
    """性能测试类"""
    
    def __init__(self, backend_url="http://localhost:5000"):
        self.backend_url = backend_url
        self.api_base = f"{backend_url}/api"
        self.session = requests.Session()
        
    def create_test_image(self, size=(640, 480), color='red'):
        """创建测试图像"""
        test_image = Image.new('RGB', size, color=color)
        img_buffer = BytesIO()
        test_image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        return img_buffer
    
    def single_detection_request(self, image_data=None):
        """单次检测请求"""
        if image_data is None:
            image_data = self.create_test_image()
        
        start_time = time.time()
        
        try:
            files = {'image': ('test.jpg', image_data, 'image/jpeg')}
            response = self.session.post(f"{self.api_base}/detect", files=files, timeout=30)
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'response_time': end_time - start_time,
                    'inference_time': data.get('inference_time', 0),
                    'processing_time': data.get('processing_time', 0),
                    'detections_count': len(data.get('detections', []))
                }
            else:
                return {
                    'success': False,
                    'response_time': end_time - start_time,
                    'error': f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            end_time = time.time()
            return {
                'success': False,
                'response_time': end_time - start_time,
                'error': str(e)
            }
    
    def test_single_request_performance(self, num_requests=10):
        """测试单请求性能"""
        print(f"\n=== 单请求性能测试 (请求数: {num_requests}) ===")
        
        results = []
        for i in range(num_requests):
            print(f"执行请求 {i+1}/{num_requests}...", end=' ')
            result = self.single_detection_request()
            results.append(result)
            
            if result['success']:
                print(f"成功 ({result['response_time']:.3f}s)")
            else:
                print(f"失败: {result['error']}")
        
        # 统计结果
        successful_results = [r for r in results if r['success']]
        success_rate = len(successful_results) / len(results) * 100
        
        if successful_results:
            response_times = [r['response_time'] for r in successful_results]
            inference_times = [r['inference_time'] for r in successful_results]
            
            print(f"\n性能统计:")
            print(f"成功率: {success_rate:.1f}%")
            print(f"平均响应时间: {statistics.mean(response_times):.3f}s")
            print(f"响应时间中位数: {statistics.median(response_times):.3f}s")
            print(f"最快响应时间: {min(response_times):.3f}s")
            print(f"最慢响应时间: {max(response_times):.3f}s")
            print(f"平均推理时间: {statistics.mean(inference_times):.3f}s")
        
        return results
    
    def test_concurrent_requests(self, num_threads=5, requests_per_thread=3):
        """测试并发请求性能"""
        print(f"\n=== 并发请求性能测试 (线程数: {num_threads}, 每线程请求数: {requests_per_thread}) ===")
        
        def worker_thread(thread_id):
            """工作线程函数"""
            thread_results = []
            for i in range(requests_per_thread):
                print(f"线程{thread_id} 请求{i+1}...", end=' ')
                result = self.single_detection_request()
                result['thread_id'] = thread_id
                thread_results.append(result)
                
                if result['success']:
                    print(f"成功 ({result['response_time']:.3f}s)")
                else:
                    print(f"失败: {result['error']}")
            
            return thread_results
        
        start_time = time.time()
        
        # 使用线程池执行并发请求
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
            
            all_results = []
            for future in as_completed(futures):
                thread_results = future.result()
                all_results.extend(thread_results)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 统计结果
        successful_results = [r for r in all_results if r['success']]
        success_rate = len(successful_results) / len(all_results) * 100
        total_requests = len(all_results)
        
        print(f"\n并发性能统计:")
        print(f"总请求数: {total_requests}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"总耗时: {total_time:.3f}s")
        print(f"吞吐量: {total_requests/total_time:.2f} 请求/秒")
        
        if successful_results:
            response_times = [r['response_time'] for r in successful_results]
            inference_times = [r['inference_time'] for r in successful_results]
            
            print(f"平均响应时间: {statistics.mean(response_times):.3f}s")
            print(f"响应时间中位数: {statistics.median(response_times):.3f}s")
            print(f"最快响应时间: {min(response_times):.3f}s")
            print(f"最慢响应时间: {max(response_times):.3f}s")
            print(f"平均推理时间: {statistics.mean(inference_times):.3f}s")
        
        return all_results
    
    def test_load_performance(self, duration_seconds=30, max_concurrent=10):
        """测试负载性能"""
        print(f"\n=== 负载性能测试 (持续时间: {duration_seconds}s, 最大并发: {max_concurrent}) ===")
        
        results = []
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        def continuous_requests():
            """持续发送请求"""
            thread_results = []
            while time.time() < end_time:
                result = self.single_detection_request()
                result['timestamp'] = time.time() - start_time
                thread_results.append(result)
                
                # 短暂休息避免过度负载
                time.sleep(0.1)
            
            return thread_results
        
        # 启动多个线程持续发送请求
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(continuous_requests) for _ in range(max_concurrent)]
            
            for future in as_completed(futures):
                thread_results = future.result()
                results.extend(thread_results)
        
        actual_duration = time.time() - start_time
        
        # 统计结果
        successful_results = [r for r in results if r['success']]
        success_rate = len(successful_results) / len(results) * 100 if results else 0
        total_requests = len(results)
        
        print(f"\n负载性能统计:")
        print(f"实际测试时间: {actual_duration:.1f}s")
        print(f"总请求数: {total_requests}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"平均吞吐量: {total_requests/actual_duration:.2f} 请求/秒")
        
        if successful_results:
            response_times = [r['response_time'] for r in successful_results]
            inference_times = [r['inference_time'] for r in successful_results]
            
            print(f"平均响应时间: {statistics.mean(response_times):.3f}s")
            print(f"响应时间标准差: {statistics.stdev(response_times):.3f}s")
            print(f"95%响应时间: {sorted(response_times)[int(len(response_times)*0.95)]:.3f}s")
            print(f"99%响应时间: {sorted(response_times)[int(len(response_times)*0.99)]:.3f}s")
            print(f"平均推理时间: {statistics.mean(inference_times):.3f}s")
        
        return results
    
    def test_memory_usage(self, num_requests=20):
        """测试内存使用情况"""
        print(f"\n=== 内存使用测试 (请求数: {num_requests}) ===")
        
        # 获取初始内存状态
        try:
            health_response = self.session.get(f"{self.api_base}/health")
            if health_response.status_code == 200:
                initial_health = health_response.json()
                print(f"初始健康状态: {initial_health}")
        except Exception as e:
            print(f"无法获取初始健康状态: {e}")
        
        # 执行多次请求
        results = []
        for i in range(num_requests):
            print(f"内存测试请求 {i+1}/{num_requests}...", end=' ')
            
            # 创建较大的测试图像
            large_image = self.create_test_image(size=(1920, 1080))
            result = self.single_detection_request(large_image)
            results.append(result)
            
            if result['success']:
                print(f"成功 ({result['response_time']:.3f}s)")
            else:
                print(f"失败: {result['error']}")
        
        # 获取最终内存状态
        try:
            health_response = self.session.get(f"{self.api_base}/health")
            if health_response.status_code == 200:
                final_health = health_response.json()
                print(f"最终健康状态: {final_health}")
        except Exception as e:
            print(f"无法获取最终健康状态: {e}")
        
        successful_results = [r for r in results if r['success']]
        success_rate = len(successful_results) / len(results) * 100
        
        print(f"\n内存测试统计:")
        print(f"成功率: {success_rate:.1f}%")
        
        if successful_results:
            response_times = [r['response_time'] for r in successful_results]
            print(f"平均响应时间: {statistics.mean(response_times):.3f}s")
        
        return results

def run_performance_tests():
    """运行所有性能测试"""
    print("=" * 60)
    print("开始运行性能测试")
    print("=" * 60)
    
    # 检查后端服务是否可用
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code != 200:
            print("错误: 后端服务不可用，请先启动后端服务")
            return False
    except requests.exceptions.RequestException:
        print("错误: 无法连接到后端服务，请确保服务正在运行")
        return False
    
    # 创建性能测试实例
    perf_test = PerformanceTest()
    
    try:
        # 1. 单请求性能测试
        perf_test.test_single_request_performance(num_requests=5)
        
        # 2. 并发请求性能测试
        perf_test.test_concurrent_requests(num_threads=3, requests_per_thread=2)
        
        # 3. 负载性能测试
        perf_test.test_load_performance(duration_seconds=15, max_concurrent=3)
        
        # 4. 内存使用测试
        perf_test.test_memory_usage(num_requests=5)
        
        print("\n" + "=" * 60)
        print("性能测试完成")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n性能测试过程中发生错误: {e}")
        return False

if __name__ == '__main__':
    success = run_performance_tests()
    sys.exit(0 if success else 1)