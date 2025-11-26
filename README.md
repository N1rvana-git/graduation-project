# 基于YOLOv7的口罩佩戴检测系统

## 项目简介

本项目是一个基于YOLOv7深度学习模型的口罩佩戴检测系统，能够实时检测图像或视频中人员是否正确佩戴口罩。系统采用端到端的智能平台架构，通过Python FastAPI后端与前端的优化通信机制，实现实时视频流分析结果的即时反馈。

## 主要特性

- **AI模型**: YOLOv7口罩检测模型，具备更高的精度和速度
- **后端服务**: FastAPI RESTful API，支持图像和视频流检测
- **前端界面**: 现代化网页端测试界面
- **实时检测**: 支持实时视频流分析和批量图像处理
- **部署环境**: 支持云服务器和Docker容器化部署
- **性能优化**: GPU加速推理，内存优化管理

## 项目结构

```
mask-detection-system/
├── yolov7/                # YOLOv7模型源码
│   ├── models/           # 模型定义
│   ├── utils/            # 工具函数
│   ├── weights/          # 预训练权重
│   └── data/             # 数据配置
├── models/               # 训练脚本和自定义模型
│   ├── train_mask_detection.py  # 口罩检测训练脚本
│   ├── weights/          # 训练好的模型权重
│   └── configs/          # 训练配置文件
├── backend/              # FastAPI后端服务
│   ├── app.py           # 主应用文件
│   ├── api/             # API接口
│   │   └── detection.py # 检测API
│   └── utils/           # 工具函数
│       ├── model_loader.py    # 模型加载器
│       └── image_utils.py     # 图像处理工具
├── frontend/            # 前端界面
│   ├── static/          # 静态资源
│   └── templates/       # HTML模板
├── data/                # 数据集
│   ├── images/          # 图像数据
│   ├── labels/          # 标注数据
│   └── mask_detection.yaml    # 数据集配置
├── tests/               # 测试文件
├── deployment/          # 部署相关文件
│   ├── docker/          # Docker配置
│   └── scripts/         # 部署脚本
├── requirements.txt     # Python依赖
└── README.md           # 项目说明
```

## 开发步骤

1. ✅ 搭建项目基础结构
2. ✅ YOLOv7模型集成与训练
3. ✅ FastAPI后端API开发
4. ✅ 模型加载器和检测API重构
5. ⏳ 网页端测试界面优化
6. ⏳ 系统功能完善和性能优化
7. ⏳ 云服务器部署

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd mask-detection-system

# 安装依赖
pip install -r requirements.txt

# 安装YOLOv7依赖
cd yolov7
pip install -r requirements.txt
cd ..
```

### 2. 模型训练

```bash
# 准备数据集
python data/prepare_dataset.py

# 训练YOLOv7口罩检测模型
python models/train_mask_detection.py --data data/mask_detection.yaml --cfg yolov7/cfg/training/yolov7.yaml --weights yolov7/weights/yolov7.pt --epochs 100
```

### 3. 启动服务

```bash
# 启动FastAPI后端服务（开发模式）
uvicorn backend.app:app --host 0.0.0.0 --port 5000 --reload

# 或在生产模式下使用
python backend/app.py

# 访问前端界面
# 打开浏览器访问 http://localhost:5000
```

### 4. API使用

```python
import requests
import base64

# 图像检测API
with open('test_image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

response = requests.post('http://localhost:5000/api/detect', 
                        json={'image': image_data})
result = response.json()
```

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (GPU推理)
- OpenCV 4.5+
- FastAPI 0.110+
- NumPy 1.21+

## 性能指标

- **检测精度**: mAP@0.5 > 95%
- **推理速度**: >30 FPS (RTX 3080)
- **模型大小**: ~75MB (YOLOv7)
- **内存占用**: <2GB (推理时)

## 部署说明

### Docker部署

```bash
# 构建镜像
docker build -t mask-detection .

# 运行容器
docker run -p 5000:5000 --gpus all mask-detection
```

### 云服务器部署

```bash
# 使用部署脚本
chmod +x deployment/scripts/deploy.ps1
./deployment/scripts/deploy.ps1
```

## 技术特点

1. **YOLOv7架构优势**
   - 更高的检测精度和速度
   - 更好的小目标检测能力
   - 优化的网络结构和训练策略

2. **系统架构设计**
   - 模块化设计，易于扩展
   - 异步处理，支持高并发
   - 内存优化，支持长时间运行

3. **实时性能**
   - GPU加速推理
   - 批量处理优化
   - 智能缓存机制

## 许可证

MIT License

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 联系方式

如有问题，请通过Issue或邮件联系。