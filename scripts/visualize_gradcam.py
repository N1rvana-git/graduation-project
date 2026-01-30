import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class _TensorOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


class _DetectionTarget:
    """将检测输出压缩为标量，便于Grad-CAM反向传播。"""
    def __call__(self, model_output):
        if isinstance(model_output, (list, tuple)):
            model_output = model_output[0]
        if isinstance(model_output, dict):
            model_output = model_output.get("preds", None) or next(iter(model_output.values()))
        # 常见检测输出形状: (B, N, 6/7/...)，第5列通常为objectness
        if model_output.ndim >= 3:
            return model_output[..., 4].max()
        return model_output.max()


def _auto_select_target_layer(torch_model):
    # 优先选择最后一个Conv层
    for module in reversed(list(torch_model.model)):
        if hasattr(module, "conv"):
            return module.conv
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise ValueError("未找到可用于Grad-CAM的Conv层。")


def generate_heatmap(model_path, img_path, save_path, target_layer_idx=None):
    model = YOLO(model_path)
    torch_model = model.model

    if not hasattr(torch_model, "model"):
        raise ValueError("无法定位内部模型结构，请检查 YOLO 版本。")

    if target_layer_idx is not None:
        target_layer = torch_model.model[target_layer_idx]
    else:
        target_layer = _auto_select_target_layer(torch_model)

    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_float = image_rgb.astype(np.float32) / 255.0

    input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
    input_tensor = input_tensor.to(next(torch_model.parameters()).device)
    input_tensor.requires_grad_(True)

    torch_model.eval()
    wrapped_model = _TensorOutputWrapper(torch_model)
    cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
    targets = [_DetectionTarget()]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    heatmap = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, heatmap_bgr)
    print(f"✅ Grad-CAM 已保存: {save_path}")


if __name__ == "__main__":
    # 示例用法
    # model_path = "runs/yolov11_mask_detection/final_gam_wiou_p2/weights/best.pt"
    # img_path = "data/combined_dataset/images/val/your_image.jpg"
    # save_path = "runs/gradcam/your_image_cam.jpg"
    # generate_heatmap(model_path, img_path, save_path)
    pass
