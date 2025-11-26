// 口罩检测系统前端JavaScript

class MaskDetectionApp {
    constructor() {
        this.apiBaseUrl = 'http://127.0.0.1:5000/api';
        this.stats = {
            totalDetections: 0,
            maskedCount: 0,
            unmaskedCount: 0,
            totalConfidence: 0
        };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.checkServerStatus();
        this.loadStats();
        
        // 定期检查服务器状态
        setInterval(() => this.checkServerStatus(), 30000);
    }
    
    setupEventListeners() {
        const uploadArea = document.getElementById('uploadArea');
        const imageInput = document.getElementById('imageInput');
        
        // 文件选择
        imageInput.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
        });
        
        // 拖拽上传
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            this.handleFiles(e.dataTransfer.files);
        });
        
        // 点击上传区域
        uploadArea.addEventListener('click', () => {
            imageInput.click();
        });
    }
    
    async checkServerStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();
            
            if (response.ok && data.status === 'healthy') {
                this.updateStatus(true, '服务正常运行');
            } else {
                this.updateStatus(false, '服务异常');
            }
        } catch (error) {
            this.updateStatus(false, '无法连接到服务器');
        }
    }
    
    updateStatus(isOnline, message) {
        const indicator = document.getElementById('statusIndicator');
        const text = document.getElementById('statusText');
        
        indicator.className = `status-indicator ${isOnline ? 'status-online' : 'status-offline'}`;
        text.textContent = message;
    }
    
    handleFiles(files) {
        if (files.length === 0) return;
        
        // 清空之前的预览和结果
        document.getElementById('imagePreview').innerHTML = '';
        document.getElementById('detectionResult').style.display = 'none';
        
        // 显示加载动画
        this.showLoading(true);
        
        // 处理多个文件
        if (files.length === 1) {
            this.detectSingleImage(files[0]);
        } else {
            this.detectMultipleImages(files);
        }
    }
    
    async detectSingleImage(file) {
        try {
            // 显示图片预览
            this.showImagePreview(file);
            
            // 创建FormData
            const formData = new FormData();
            formData.append('image', file);
            
            // 发送检测请求
            const response = await fetch(`${this.apiBaseUrl}/detect`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayDetectionResult(result);
                this.updateStats(result);
            } else {
                this.showError(result.error || '检测失败');
            }
        } catch (error) {
            this.showError('网络错误: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }
    
    async detectMultipleImages(files) {
        try {
            // 创建FormData
            const formData = new FormData();
            Array.from(files).forEach(file => {
                formData.append('images', file);
            });
            
            // 发送批量检测请求
            const response = await fetch(`${this.apiBaseUrl}/batch_detect`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayBatchResults(result);
                this.updateBatchStats(result);
            } else {
                this.showError(result.error || '批量检测失败');
            }
        } catch (error) {
            this.showError('网络错误: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }
    
    showImagePreview(file) {
        const preview = document.getElementById('imagePreview');
        const reader = new FileReader();
        
        reader.onload = (e) => {
            preview.innerHTML = `
                <div class="preview-container">
                    <img src="${e.target.result}" alt="预览图片" class="preview-image">
                    <div class="mt-2">
                        <small class="text-muted">${file.name}</small>
                    </div>
                </div>
            `;
        };
        
        reader.readAsDataURL(file);
    }
    
    displayDetectionResult(result) {
        const resultDiv = document.getElementById('detectionResult');
        const contentDiv = document.getElementById('resultContent');
        
        if (!result.detections || result.detections.length === 0) {
            contentDiv.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle"></i> 未检测到人脸
                </div>
            `;
        } else {
            let html = `
                <div class="mb-3">
                    <strong>检测到 ${result.detections.length} 个目标</strong>
                    <small class="text-muted ms-2">处理时间: ${result.processing_time}ms</small>
                </div>
            `;
            
            result.detections.forEach((detection, index) => {
                const confidence = Math.round(detection.confidence * 100);
                const className = detection.class === 'with_mask' ? 'with_mask' : 'without_mask';
                const label = detection.class === 'with_mask' ? '戴口罩' : '未戴口罩';
                const iconClass = detection.class === 'with_mask' ? 'fas fa-check-circle text-success' : 'fas fa-exclamation-triangle text-warning';
                
                html += `
                    <div class="result-item">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <i class="${iconClass}"></i>
                                <strong>${label}</strong>
                                <small class="text-muted ms-2">目标 ${index + 1}</small>
                            </div>
                            <span class="badge bg-primary">${confidence}%</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence}%"></div>
                        </div>
                        <small class="text-muted">
                            位置: (${Math.round(detection.bbox[0])}, ${Math.round(detection.bbox[1])}) - 
                            (${Math.round(detection.bbox[2])}, ${Math.round(detection.bbox[3])})
                        </small>
                    </div>
                `;
            });
            
            contentDiv.innerHTML = html;
        }
        
        resultDiv.style.display = 'block';
    }
    
    displayBatchResults(result) {
        const resultDiv = document.getElementById('detectionResult');
        const contentDiv = document.getElementById('resultContent');
        
        let html = `
            <div class="mb-3">
                <strong>批量检测完成</strong>
                <div class="text-muted">
                    处理了 ${result.processed_images} / ${result.total_images} 张图片
                </div>
            </div>
        `;
        
        result.results.forEach((item, index) => {
            if (item.error) {
                html += `
                    <div class="result-item border-danger">
                        <div class="text-danger">
                            <i class="fas fa-exclamation-circle"></i>
                            ${item.filename}: ${item.error}
                        </div>
                    </div>
                `;
            } else {
                const detectionCount = item.detections ? item.detections.length : 0;
                html += `
                    <div class="result-item">
                        <div class="d-flex justify-content-between">
                            <strong>${item.filename}</strong>
                            <span class="badge bg-info">${detectionCount} 个目标</span>
                        </div>
                        <small class="text-muted">处理时间: ${item.processing_time}ms</small>
                    </div>
                `;
            }
        });
        
        contentDiv.innerHTML = html;
        resultDiv.style.display = 'block';
    }
    
    updateStats(result) {
        if (result.detections) {
            this.stats.totalDetections++;
            
            result.detections.forEach(detection => {
                if (detection.class === 'with_mask') {
                    this.stats.maskedCount++;
                } else {
                    this.stats.unmaskedCount++;
                }
                this.stats.totalConfidence += detection.confidence;
            });
        }
        
        this.renderStats();
        this.saveStats();
    }
    
    updateBatchStats(result) {
        result.results.forEach(item => {
            if (item.detections) {
                this.stats.totalDetections++;
                
                item.detections.forEach(detection => {
                    if (detection.class === 'with_mask') {
                        this.stats.maskedCount++;
                    } else {
                        this.stats.unmaskedCount++;
                    }
                    this.stats.totalConfidence += detection.confidence;
                });
            }
        });
        
        this.renderStats();
        this.saveStats();
    }
    
    renderStats() {
        document.getElementById('totalDetections').textContent = this.stats.totalDetections;
        document.getElementById('maskedCount').textContent = this.stats.maskedCount;
        document.getElementById('unmaskedCount').textContent = this.stats.unmaskedCount;
        
        const totalObjects = this.stats.maskedCount + this.stats.unmaskedCount;
        const avgConfidence = totalObjects > 0 ? 
            Math.round((this.stats.totalConfidence / totalObjects) * 100) : 0;
        document.getElementById('avgConfidence').textContent = avgConfidence + '%';
    }
    
    saveStats() {
        localStorage.setItem('maskDetectionStats', JSON.stringify(this.stats));
    }
    
    loadStats() {
        const saved = localStorage.getItem('maskDetectionStats');
        if (saved) {
            this.stats = JSON.parse(saved);
            this.renderStats();
        }
    }
    
    showLoading(show) {
        const spinner = document.getElementById('loadingSpinner');
        const uploadArea = document.getElementById('uploadArea');
        
        if (show) {
            spinner.style.display = 'block';
            uploadArea.style.display = 'none';
        } else {
            spinner.style.display = 'none';
            uploadArea.style.display = 'block';
        }
    }
    
    showError(message) {
        const resultDiv = document.getElementById('detectionResult');
        const contentDiv = document.getElementById('resultContent');
        
        contentDiv.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle"></i>
                <strong>错误:</strong> ${message}
            </div>
        `;
        
        resultDiv.style.display = 'block';
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new MaskDetectionApp();
});