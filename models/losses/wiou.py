# WIoU Loss (Wise-IoU) 实现: Version 3
# 参考文献: Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism
import torch
import torch.nn as nn
import math

class WIoULoss(nn.Module):
    def __init__(self, mon=False):
        super(WIoULoss, self).__init__()
        self.mon = mon # Monotonicity flag

    def forward(self, pred_bboxes, true_bboxes):
        """
        计算 WIoU v3 Loss
        :param pred_bboxes: [x, y, w, h] (预测框)
        :param true_bboxes: [x, y, w, h] (真实框)
        """
        # 1. 计算基础 IoU 和坐标差
        tl = torch.max(pred_bboxes[:, :2] - pred_bboxes[:, 2:] / 2, true_bboxes[:, :2] - true_bboxes[:, 2:] / 2)
        br = torch.min(pred_bboxes[:, :2] + pred_bboxes[:, 2:] / 2, true_bboxes[:, :2] + true_bboxes[:, 2:] / 2)
        
        area_p = torch.prod(pred_bboxes[:, 2:], 1)
        area_g = torch.prod(true_bboxes[:, 2:], 1)
        
        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = area_i / (area_p + area_g - area_i + 1e-16)

        # 2. 距离关注度 (Distance Attention)
        # 获取最小包围框的宽和高 (c_w, c_h)
        c_tl = torch.min(pred_bboxes[:, :2] - pred_bboxes[:, 2:] / 2, true_bboxes[:, :2] - true_bboxes[:, 2:] / 2)
        c_br = torch.max(pred_bboxes[:, :2] + pred_bboxes[:, 2:] / 2, true_bboxes[:, :2] + true_bboxes[:, 2:] / 2)
        c_wh = c_br - c_tl  # [B, 2]
        
        # 中心点距离平方
        dist = torch.sum((pred_bboxes[:, :2] - true_bboxes[:, :2]) ** 2, dim=1)
        # 最小包围框对角线平方
        diag = torch.sum(c_wh ** 2, dim=1) + 1e-16
        
        R_WIoU = torch.exp(dist / diag)

        # 3. 动态非单调聚焦 (Dynamic Non-monotonic Focusing - v3)
        # 离群度 outlierness: beta = iou_loss / iou_avg (这里的实现简化为基于IoU的映射)
        # 论文 v3: L_WIoU = r * R_WIoU * L_IoU
        #          r = beta / (delta * alpha ^ (beta - delta))
        
        loss_iou = 1 - iou
        
        # 动态聚焦系数 r
        # 使用梯度增益归一化策略
        beta = loss_iou.detach() / (loss_iou.detach().mean() + 1e-16) 
        
        # 超参数 alpha, delta (论文经验值: alpha=1.9, delta=3)
        alpha = 1.9
        delta = 3.0
        r = beta / (delta * torch.pow(alpha, beta - delta))
        
        return (r * R_WIoU * loss_iou).mean()

