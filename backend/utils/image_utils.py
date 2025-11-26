"""
图像处理工具模块
处理图像的预处理、后处理和格式转换
"""

import cv2
import numpy as np
from PIL import Image
import torch
import base64
import io
import logging
from typing import Tuple, Union, List

logger = logging.getLogger(__name__)

class ImageProcessor:
    """图像处理器类"""
    
    def __init__(self, input_size: int = 640):
        """
        初始化图像处理器
        
        Args:
            input_size: 模型输入尺寸
        """
        self.input_size = input_size
        self.mean = [0.485, 0.456, 0.406]  # ImageNet标准化均值
        self.std = [0.229, 0.224, 0.225]   # ImageNet标准化标准差
    
    def preprocess(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        预处理图像用于模型推理
        
        Args:
            image: PIL图像对象
            
        Returns:
            处理后的张量和原始图像尺寸
        """
        try:
            # 保存原始尺寸
            original_shape = (image.height, image.width)
            
            # 转换为RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为numpy数组
            img_array = np.array(image)
            
            # 调整尺寸并保持宽高比
            img_resized = self._letterbox(img_array, self.input_size)
            
            # 归一化到[0,1]
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # 转换为张量并调整维度顺序 (H,W,C) -> (C,H,W)
            img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
            
            # 添加批次维度 (C,H,W) -> (1,C,H,W)
            img_tensor = img_tensor.unsqueeze(0)
            
            return img_tensor, original_shape
            
        except Exception as e:
            logger.error(f"图像预处理失败: {str(e)}")
            raise
    
    def _letterbox(self, img: np.ndarray, new_shape: int = 640, 
                   color: Tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
        """
        调整图像尺寸并保持宽高比（letterbox方法）
        
        Args:
            img: 输入图像数组
            new_shape: 目标尺寸
            color: 填充颜色
            
        Returns:
            调整后的图像数组
        """
        shape = img.shape[:2]  # 当前形状 [height, width]
        
        # 计算缩放比例
        r = min(new_shape / shape[0], new_shape / shape[1])
        
        # 计算新的未填充尺寸
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]  # 宽高差
        
        # 均匀分配填充
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:  # 如果需要调整尺寸
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=color)
        return img
    
    def postprocess_image(self, image: np.ndarray) -> Image.Image:
        """
        后处理图像，转换为PIL格式
        
        Args:
            image: numpy图像数组
            
        Returns:
            PIL图像对象
        """
        try:
            # 确保图像是uint8格式
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # 如果是BGR格式，转换为RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return Image.fromarray(image)
            
        except Exception as e:
            logger.error(f"图像后处理失败: {str(e)}")
            raise
    
    def image_to_base64(self, image: Image.Image, format: str = 'JPEG', 
                       quality: int = 85) -> str:
        """
        将PIL图像转换为Base64编码字符串
        
        Args:
            image: PIL图像对象
            format: 图像格式
            quality: JPEG质量 (1-100)
            
        Returns:
            Base64编码字符串
        """
        try:
            buffer = io.BytesIO()
            
            # 如果是JPEG格式且图像有透明通道，转换为RGB
            if format.upper() == 'JPEG' and image.mode in ('RGBA', 'LA'):
                # 创建白色背景
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            
            # 保存图像到缓冲区
            save_kwargs = {'format': format}
            if format.upper() == 'JPEG':
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
            
            image.save(buffer, **save_kwargs)
            
            # 编码为Base64
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/{format.lower()};base64,{img_str}"
            
        except Exception as e:
            logger.error(f"图像Base64编码失败: {str(e)}")
            raise
    
    def base64_to_image(self, base64_str: str) -> Image.Image:
        """
        将Base64编码字符串转换为PIL图像
        
        Args:
            base64_str: Base64编码字符串
            
        Returns:
            PIL图像对象
        """
        try:
            # 移除数据URL前缀
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',')[1]
            
            # 解码Base64
            image_data = base64.b64decode(base64_str)
            
            # 转换为PIL图像
            image = Image.open(io.BytesIO(image_data))
            
            return image
            
        except Exception as e:
            logger.error(f"Base64解码失败: {str(e)}")
            raise
    
    def resize_image(self, image: Image.Image, size: Tuple[int, int], 
                    keep_aspect_ratio: bool = True) -> Image.Image:
        """
        调整图像尺寸
        
        Args:
            image: PIL图像对象
            size: 目标尺寸 (width, height)
            keep_aspect_ratio: 是否保持宽高比
            
        Returns:
            调整后的图像
        """
        try:
            if keep_aspect_ratio:
                image.thumbnail(size, Image.Resampling.LANCZOS)
                
                # 创建新图像并居中粘贴
                new_image = Image.new('RGB', size, (255, 255, 255))
                paste_x = (size[0] - image.width) // 2
                paste_y = (size[1] - image.height) // 2
                new_image.paste(image, (paste_x, paste_y))
                
                return new_image
            else:
                return image.resize(size, Image.Resampling.LANCZOS)
                
        except Exception as e:
            logger.error(f"图像尺寸调整失败: {str(e)}")
            raise
    
    def enhance_image(self, image: Image.Image, brightness: float = 1.0, 
                     contrast: float = 1.0, saturation: float = 1.0) -> Image.Image:
        """
        图像增强
        
        Args:
            image: PIL图像对象
            brightness: 亮度调整因子
            contrast: 对比度调整因子
            saturation: 饱和度调整因子
            
        Returns:
            增强后的图像
        """
        try:
            from PIL import ImageEnhance
            
            # 亮度调整
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness)
            
            # 对比度调整
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast)
            
            # 饱和度调整
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(saturation)
            
            return image
            
        except Exception as e:
            logger.error(f"图像增强失败: {str(e)}")
            raise
    
    def crop_detections(self, image: Image.Image, 
                       detections: List[dict]) -> List[Image.Image]:
        """
        根据检测结果裁剪图像区域
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            
        Returns:
            裁剪后的图像列表
        """
        cropped_images = []
        
        try:
            for detection in detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                # 确保坐标在图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.width, x2)
                y2 = min(image.height, y2)
                
                # 裁剪图像
                cropped = image.crop((x1, y1, x2, y2))
                cropped_images.append(cropped)
                
        except Exception as e:
            logger.error(f"图像裁剪失败: {str(e)}")
            
        return cropped_images
    
    def create_mosaic(self, images: List[Image.Image], 
                     grid_size: Tuple[int, int] = None) -> Image.Image:
        """
        创建图像马赛克
        
        Args:
            images: 图像列表
            grid_size: 网格尺寸 (rows, cols)
            
        Returns:
            马赛克图像
        """
        try:
            if not images:
                raise ValueError("图像列表不能为空")
            
            # 自动计算网格尺寸
            if grid_size is None:
                num_images = len(images)
                cols = int(np.ceil(np.sqrt(num_images)))
                rows = int(np.ceil(num_images / cols))
                grid_size = (rows, cols)
            
            rows, cols = grid_size
            
            # 计算单个图像尺寸
            max_width = max(img.width for img in images)
            max_height = max(img.height for img in images)
            
            # 创建马赛克画布
            mosaic_width = cols * max_width
            mosaic_height = rows * max_height
            mosaic = Image.new('RGB', (mosaic_width, mosaic_height), (255, 255, 255))
            
            # 粘贴图像
            for i, image in enumerate(images):
                if i >= rows * cols:
                    break
                
                row = i // cols
                col = i % cols
                
                # 调整图像尺寸
                resized_image = image.resize((max_width, max_height), Image.Resampling.LANCZOS)
                
                # 计算粘贴位置
                x = col * max_width
                y = row * max_height
                
                mosaic.paste(resized_image, (x, y))
            
            return mosaic
            
        except Exception as e:
            logger.error(f"创建马赛克失败: {str(e)}")
            raise