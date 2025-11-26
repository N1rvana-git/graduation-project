"""
YOLOv7口罩检测训练脚本
基于YOLOv7官方代码实现的口罩检测模型训练
"""

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path

# 添加YOLOv7路径到系统路径
YOLOV7_PATH = Path(__file__).parent.parent / "yolov7"
sys.path.append(str(YOLOV7_PATH))

try:
    # 直接导入YOLOv7模块
    from train import train as yolov7_train
    from utils.general import check_img_size, colorstr
    from utils.torch_utils import select_device
    import yaml
except ImportError as e:
    print(f"无法导入YOLOv7模块: {e}")
    print(f"请确保YOLOv7代码库已正确安装在: {YOLOV7_PATH}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path}")
    sys.exit(1)


class YOLOv7MaskDetectionTrainer:
    """YOLOv7口罩检测训练器"""
    
    def __init__(self, 
                 data_path="data/mask_detection.yaml",
                 model_size="yolov7",
                 img_size=640,
                 batch_size=16,
                 epochs=100,
                 device="0"):
        """
        初始化训练器
        
        Args:
            data_path: 数据集配置文件路径
            model_size: 模型大小 (yolov7, yolov7x, yolov7-tiny等)
            img_size: 输入图像尺寸
            batch_size: 批次大小
            epochs: 训练轮数
            device: 设备 (cpu, 0, 1等)
        """
        self.data_path = data_path
        self.model_size = model_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        
        # 设置路径
        self.project_root = Path(__file__).parent.parent
        self.yolov7_root = self.project_root / "yolov7"
        self.weights_dir = self.project_root / "weights"
        self.runs_dir = self.project_root / "runs"
        
        # 创建必要目录
        self.weights_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        
    def prepare_data_config(self):
        """准备数据集配置文件"""
        data_config = {
            'train': str(self.project_root / "datasets" / "mask_detection" / "train"),
            'val': str(self.project_root / "datasets" / "mask_detection" / "val"),
            'test': str(self.project_root / "datasets" / "mask_detection" / "test"),
            'nc': 2,  # 类别数量
            'names': ['no_mask', 'mask']  # 类别名称
        }
        
        # 保存配置文件
        config_path = self.project_root / "data" / "mask_detection.yaml"
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"数据配置文件已保存到: {config_path}")
        return str(config_path)
    
    def get_model_config(self):
        """获取模型配置文件路径"""
        model_configs = {
            'yolov7': 'cfg/training/yolov7.yaml',
            'yolov7x': 'cfg/training/yolov7x.yaml',
            'yolov7-tiny': 'cfg/training/yolov7-tiny.yaml',
            'yolov7-w6': 'cfg/training/yolov7-w6.yaml',
            'yolov7-e6': 'cfg/training/yolov7-e6.yaml',
            'yolov7-d6': 'cfg/training/yolov7-d6.yaml',
            'yolov7-e6e': 'cfg/training/yolov7-e6e.yaml'
        }
        
        config_file = model_configs.get(self.model_size, 'cfg/training/yolov7.yaml')
        config_path = self.yolov7_root / config_file
        
        if not config_path.exists():
            print(f"警告: 模型配置文件不存在: {config_path}")
            print(f"使用默认配置: cfg/training/yolov7.yaml")
            config_path = self.yolov7_root / "cfg/training/yolov7.yaml"
        
        return str(config_path)
    
    def get_pretrained_weights(self):
        """获取预训练权重文件路径"""
        weights_urls = {
            'yolov7': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt',
            'yolov7x': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt',
            'yolov7-tiny': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt',
            'yolov7-w6': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt',
            'yolov7-e6': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt',
            'yolov7-d6': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt',
            'yolov7-e6e': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt'
        }
        
        weight_file = f"{self.model_size}.pt"
        weight_path = self.weights_dir / weight_file
        
        if not weight_path.exists():
            print(f"预训练权重文件不存在: {weight_path}")
            print(f"请从以下链接下载: {weights_urls.get(self.model_size, '')}")
            print(f"或使用空权重进行训练")
            return ""
        
        return str(weight_path)
    
    def train(self):
        """开始训练"""
        print(f"{colorstr('YOLOv7 口罩检测训练开始')}")
        print(f"模型: {self.model_size}")
        print(f"图像尺寸: {self.img_size}")
        print(f"批次大小: {self.batch_size}")
        print(f"训练轮数: {self.epochs}")
        print(f"设备: {self.device}")
        
        # 准备配置
        data_config = self.prepare_data_config()
        model_config = self.get_model_config()
        weights = self.get_pretrained_weights()
        
        # 构建训练参数
        train_args = argparse.Namespace(
            weights=weights,
            cfg=model_config,
            data=data_config,
            hyp=str(self.yolov7_root / "data/hyp.scratch.p5.yaml"),
            epochs=self.epochs,
            batch_size=self.batch_size,
            total_batch_size=self.batch_size,
            imgsz=self.img_size,
            img_size=[self.img_size, self.img_size],
            rect=False,
            resume=False,
            nosave=False,
            notest=False,
            noautoanchor=False,
            evolve=False,
            bucket='',
            cache_images=False,
            image_weights=False,
            device=self.device,
            multi_scale=False,
            single_cls=False,
            adam=False,
            sync_bn=False,
            workers=8,
            project=str(self.runs_dir),
            entity=None,
            name='yolov7_mask_detection',
            exist_ok=False,
            quad=False,
            linear_lr=False,
            label_smoothing=0.0,
            patience=100,
            freeze=[0],
            save_period=-1,
            local_rank=-1,
            global_rank=-1,
            world_size=1,
            bbox_interval=-1,
            save_conf=False,
            save_dir=str(self.runs_dir / 'yolov7_mask_detection'),
            upload_dataset=False,
            artifact_alias='latest',
            v5_metric=False
        )
        
        # 切换到YOLOv7目录
        original_cwd = os.getcwd()
        os.chdir(self.yolov7_root)
        
        try:
            # 准备训练参数
            device = select_device(self.device)
            
            # 加载超参数
            with open(train_args.hyp) as f:
                hyp = yaml.load(f, Loader=yaml.SafeLoader)
            
            # 开始训练
            yolov7_train(hyp, train_args, device)
            print(f"{colorstr('训练完成!')}")
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            raise
        finally:
            # 恢复原始工作目录
            os.chdir(original_cwd)
    
    def optimize_for_gpu(self):
        """GPU优化配置"""
        device = select_device(self.device)
        
        if device.type == 'cuda':
            # 获取GPU信息
            gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
            print(f"GPU内存: {gpu_memory:.1f}GB")
            
            # 根据GPU内存调整批次大小
            if gpu_memory < 4:
                recommended_batch = 8
            elif gpu_memory < 8:
                recommended_batch = 16
            elif gpu_memory < 12:
                recommended_batch = 24
            else:
                recommended_batch = 32
            
            if self.batch_size > recommended_batch:
                print(f"建议将批次大小从 {self.batch_size} 调整为 {recommended_batch}")
                self.batch_size = recommended_batch
        
        return device


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv7口罩检测训练')
    parser.add_argument('--data', type=str, default='data/mask_detection.yaml', help='数据集配置文件')
    parser.add_argument('--model', type=str, default='yolov7', help='模型大小')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--device', type=str, default='0', help='设备')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = YOLOv7MaskDetectionTrainer(
        data_path=args.data,
        model_size=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device
    )
    
    # GPU优化
    trainer.optimize_for_gpu()
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()