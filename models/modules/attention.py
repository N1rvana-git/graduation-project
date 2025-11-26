# GAM（Global Attention Mechanism）模块实现
# 可扩展为CA、CBAM等
import torch
import torch.nn as nn

class GAMAttention(nn.Module):
    def __init__(self, channels):
        super(GAMAttention, self).__init__()
        # 示例结构：可根据论文/需求自定义
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_att(x)
        x = x * ca
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        return x

# 可扩展：
# class CAAttention(nn.Module): ...
# class CBAMAttention(nn.Module): ...
