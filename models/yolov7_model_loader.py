"""
YOLOv7模型加载器
用于加载和管理YOLOv7模型
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

# 添加YOLOv7路径到系统路径
YOLOV7_PATH = Path(__file__).parent.parent / "yolov7"
sys.path.append(str(YOLOV7_PATH))

try:
    from yolov7.models.experimental import attempt_load
    from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords
    from yolov7.utils.plots import plot_one_box
    from yolov7.utils.torch_utils import select_device, time_synchronized
    from yolov7.utils.datasets import letterbox
except ImportError as e:
    print(f"无法导入YOLOv7模块: {e}")
    print(f"请确保YOLOv7代码库已正确安装在: {YOLOV7_PATH}")


class YOLOv7ModelLoader:
    """YOLOv7模型加载器"""
    
    def __init__(self, 
                 weights_path: str = "weights/yolov7.pt",
                 device: str = "auto",
                 img_size: int = 640,
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45):
        """
        初始化YOLOv7模型加载器
        
        Args:
            weights_path: 模型权重文件路径
            device: 设备 ('cpu', 'cuda', 'auto')
            img_size: 输入图像尺寸
            conf_thres: 置信度阈值
            iou_thres: IoU阈值
        """
        self.weights_path = weights_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 设备选择
        self.device = select_device(device)
        
        # 类别名称
        self.class_names = ['no_mask', 'mask']
        self.colors = [[255, 0, 0], [0, 255, 0]]  # 红色表示无口罩，绿色表示有口罩
        
        # 模型
        self.model = None
        self.stride = None
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """加载YOLOv7模型"""
        try:
            print(f"正在加载YOLOv7模型: {self.weights_path}")
            print(f"设备: {self.device}")
            
            # 检查权重文件是否存在
            if not os.path.exists(self.weights_path):
                raise FileNotFoundError(f"权重文件不存在: {self.weights_path}")
            
            # 加载模型
            self.model = attempt_load(self.weights_path, map_location=self.device)
            self.stride = int(self.model.stride.max())
            
            # 检查图像尺寸
            self.img_size = check_img_size(self.img_size, s=self.stride)
            
            # 设置为评估模式
            self.model.eval()
            
            # 预热
            if self.device.type != 'cpu':
                self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))
            
            print(f"模型加载成功!")
            print(f"模型步长: {self.stride}")
            print(f"输入尺寸: {self.img_size}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def preprocess_image(self, img: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        """
        预处理图像
        
        Args:
            img: 输入图像 (BGR格式)
            
        Returns:
            处理后的张量, 原始图像, 原始尺寸
        """
        # 保存原始图像和尺寸
        img0 = img.copy()
        h0, w0 = img.shape[:2]
        
        # Letterbox变换
        img = letterbox(img, self.img_size, stride=self.stride)[0]
        
        # 转换颜色空间 BGR -> RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        
        # 转换为张量
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        return img, img0, (h0, w0)
    
    def detect(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        检测图像中的口罩
        
        Args:
            img: 输入图像 (BGR格式)
            
        Returns:
            检测结果列表，每个结果包含 bbox, confidence, class_id, class_name
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        # 预处理
        img_tensor, img0, (h0, w0) = self.preprocess_image(img)
        
        # 推理
        with torch.no_grad():
            t1 = time_synchronized()
            pred = self.model(img_tensor, augment=False)[0]
            t2 = time_synchronized()
            
            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)
            t3 = time_synchronized()
        
        results = []
        
        # 处理检测结果
        for i, det in enumerate(pred):
            if len(det):
                # 将坐标缩放回原始图像尺寸
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()
                
                # 解析检测结果
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    confidence = float(conf)
                    class_id = int(cls)
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    })
        
        # 打印推理时间
        print(f"推理时间: {t2 - t1:.3f}s, NMS时间: {t3 - t2:.3f}s")
        
        return results
    
    def draw_results(self, img: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            img: 输入图像
            results: 检测结果
            
        Returns:
            绘制结果的图像
        """
        img_result = img.copy()
        
        for result in results:
            bbox = result['bbox']
            confidence = result['confidence']
            class_id = result['class_id']
            class_name = result['class_name']
            
            # 选择颜色
            color = self.colors[class_id] if class_id < len(self.colors) else [255, 255, 255]
            
            # 绘制边界框和标签
            label = f"{class_name}: {confidence:.2f}"
            plot_one_box(bbox, img_result, label=label, color=color, line_thickness=2)
        
        return img_result
    
    def predict_image(self, image_path: str, save_path: str = None) -> List[Dict[str, Any]]:
        """
        预测单张图像
        
        Args:
            image_path: 图像路径
            save_path: 保存结果图像的路径
            
        Returns:
            检测结果
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 检测
        results = self.detect(img)
        
        # 绘制结果
        if save_path:
            img_result = self.draw_results(img, results)
            cv2.imwrite(save_path, img_result)
            print(f"结果已保存到: {save_path}")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model is None:
            return {}
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'weights_path': self.weights_path,
            'device': str(self.device),
            'img_size': self.img_size,
            'stride': self.stride,
            'conf_thres': self.conf_thres,
            'iou_thres': self.iou_thres,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'class_names': self.class_names
        }


def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv7口罩检测测试')
    parser.add_argument('--weights', type=str, default='weights/yolov7.pt', help='权重文件路径')
    parser.add_argument('--source', type=str, default='data/images', help='测试图像路径')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU阈值')
    parser.add_argument('--device', type=str, default='auto', help='设备')
    parser.add_argument('--save-dir', type=str, default='runs/detect', help='保存目录')
    
    args = parser.parse_args()
    
    try:
        # 创建模型加载器
        model_loader = YOLOv7ModelLoader(
            weights_path=args.weights,
            device=args.device,
            img_size=args.img_size,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres
        )
        
        # 打印模型信息
        model_info = model_loader.get_model_info()
        print("\n模型信息:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # 测试检测
        if os.path.isfile(args.source):
            # 单张图像
            results = model_loader.predict_image(args.source, 
                                               os.path.join(args.save_dir, 'result.jpg'))
            print(f"\n检测结果: {len(results)} 个目标")
            for i, result in enumerate(results):
                print(f"  目标 {i+1}: {result}")
        
        print("\nYOLOv7模型测试完成!")
        
    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    main()