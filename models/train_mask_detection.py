"""
使用YOLOv7训练口罩检测模型
基于官方YOLOv7实现，针对口罩检测场景优化
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
import torch
import yaml

# 添加YOLOv7路径
sys.path.append(str(Path(__file__).parent.parent / 'yolov7'))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaskDetectionTrainer:
    """基于YOLOv7的口罩检测模型训练器"""
    
    def __init__(self, data_config: str, model_config: str = 'yolov7.yaml'):
        """
        初始化训练器
        
        Args:
            data_config: 数据集配置文件路径
            model_config: YOLOv7模型配置文件 (yolov7.yaml, yolov7-tiny.yaml, yolov7x.yaml)
        """
        self.data_config = Path(data_config)
        self.model_config = model_config
        self.yolov7_dir = Path(__file__).parent.parent / 'yolov7'
        
        # 检查数据配置文件
        if not self.data_config.exists():
            raise FileNotFoundError(f"数据配置文件不存在: {self.data_config}")
        
        # 检查YOLOv7目录
        if not self.yolov7_dir.exists():
            raise FileNotFoundError(f"YOLOv7目录不存在: {self.yolov7_dir}")
        
        # 创建输出目录
        self.output_dir = Path('models/runs/mask_detection')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"YOLOv7训练器初始化完成")
        logger.info(f"数据配置: {self.data_config}")
        logger.info(f"模型配置: {self.model_config}")
        logger.info(f"YOLOv7目录: {self.yolov7_dir}")
        logger.info(f"输出目录: {self.output_dir}")
    
    def train(self, epochs: int = 300, batch_size: int = 16, img_size: int = 640, 
              device: str = '0', workers: int = 8, weights: str = None):
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            img_size: 输入图像尺寸
            device: 设备 ('0', '1', 'cpu')
            workers: 数据加载进程数
            weights: 预训练权重路径
        """
        try:
            # 构建训练命令
            train_script = self.yolov7_dir / 'train.py'
            
            cmd = [
                'python', str(train_script),
                '--data', str(self.data_config),
                '--cfg', f'cfg/training/{self.model_config}',
                '--epochs', str(epochs),
                '--batch-size', str(batch_size),
                '--img-size', str(img_size),
                '--device', device,
                '--workers', str(workers),
                '--name', 'mask_detection',
                '--project', str(self.output_dir.parent),
                '--exist-ok'
            ]
            
            # 添加预训练权重
            if weights:
                cmd.extend(['--weights', weights])
            else:
                # 使用默认YOLOv7权重
                default_weights = self.yolov7_dir / 'weights' / 'yolov7.pt'
                if default_weights.exists():
                    cmd.extend(['--weights', str(default_weights)])
            
            logger.info(f"开始训练，命令: {' '.join(cmd)}")
            
            # 切换到YOLOv7目录执行训练
            result = subprocess.run(
                cmd, 
                cwd=self.yolov7_dir,
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("训练完成！")
                return True
            else:
                logger.error(f"训练失败，返回码: {result.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            return False
    
    def validate(self, weights_path: str, device: str = '0'):
        """
        验证模型
        
        Args:
            weights_path: 模型权重路径
            device: 设备
        """
        try:
            test_script = self.yolov7_dir / 'test.py'
            
            cmd = [
                'python', str(test_script),
                '--data', str(self.data_config),
                '--weights', weights_path,
                '--device', device,
                '--verbose'
            ]
            
            logger.info(f"开始验证，命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.yolov7_dir,
                capture_output=False,
                text=True
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"验证过程中出现错误: {e}")
            return False
    
    def detect(self, weights_path: str, source: str, output_dir: str = None):
        """
        使用训练好的模型进行检测
        
        Args:
            weights_path: 模型权重路径
            source: 输入源（图片/视频/摄像头）
            output_dir: 输出目录
        """
        try:
            detect_script = self.yolov7_dir / 'detect.py'
            
            cmd = [
                'python', str(detect_script),
                '--weights', weights_path,
                '--source', source,
                '--conf-thres', '0.25',
                '--iou-thres', '0.45'
            ]
            
            if output_dir:
                cmd.extend(['--project', output_dir])
            
            logger.info(f"开始检测，命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.yolov7_dir,
                capture_output=False,
                text=True
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"检测过程中出现错误: {e}")
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv7口罩检测模型训练')
    parser.add_argument('--data', type=str, required=True, help='数据集配置文件路径')
    parser.add_argument('--cfg', type=str, default='yolov7.yaml', help='模型配置文件')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='0', help='设备')
    parser.add_argument('--workers', type=int, default=8, help='数据加载进程数')
    parser.add_argument('--weights', type=str, help='预训练权重路径')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'detect'], help='运行模式')
    parser.add_argument('--source', type=str, help='检测源（用于detect模式）')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = MaskDetectionTrainer(args.data, args.cfg)
    
    if args.mode == 'train':
        success = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
            workers=args.workers,
            weights=args.weights
        )
        if success:
            logger.info("训练成功完成！")
        else:
            logger.error("训练失败！")
            
    elif args.mode == 'val':
        if not args.weights:
            logger.error("验证模式需要指定权重文件")
            return
        success = trainer.validate(args.weights, args.device)
        if success:
            logger.info("验证完成！")
        else:
            logger.error("验证失败！")
            
    elif args.mode == 'detect':
        if not args.weights or not args.source:
            logger.error("检测模式需要指定权重文件和输入源")
            return
        success = trainer.detect(args.weights, args.source)
        if success:
            logger.info("检测完成！")
        else:
            logger.error("检测失败！")

if __name__ == "__main__":
    main()