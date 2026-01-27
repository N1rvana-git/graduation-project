import torch
import torch.nn as nn

class GAMAttention(nn.Module):
    """
    GAM (Global Attention Mechanism) 模块
    
    论文: Global Attention Mechanism: Retain Information to Enhance Channel-Spatial Interactions
    核心特点: 
    1. 不使用 Pooling (如 AvgPool/MaxPool) 以避免信息丢失。
    2. 使用 3D 排列 (Permutation) 来保留跨维度的信息。
    """
    def __init__(self, c1, c2, group=True, rate=4):
        super(GAMAttention, self).__init__()
        
        # ==========================================
        # Channel Attention Sub-module
        # 区别于 CBAM/SE：这里不进行 Global Pooling
        # 流程: (N, C, H, W) -> Permute(N, W, H, C) -> MLP -> Permute(N, C, H, W)
        # ==========================================
        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1)
        )

        # ==========================================
        # Spatial Attention Sub-module
        # 使用 7x7 卷积提取空间信息
        # ==========================================
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c1, int(c1 / rate), kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(c1 / rate), c2, kernel_size=7, padding=3, bias=False),  # 输出通道 c2 (通常 c1==c2)
            nn.BatchNorm2d(c2),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        
        # ----------------------------------------
        # Channel Attention Path
        # ----------------------------------------
        # 1. Permute: (b, c, h, w) -> (b, w, h, c)
        #    这样 Linear 层作用于最后一个维度 c，实现了跨通道交互，但保留了每个空间位置 (w,h) 的独立性
        x_perm = x.permute(0, 3, 2, 1).contiguous()
        
        # 2. MLP 处理
        x_out = self.channel_attention(x_perm)
        
        # 3. Reverse Permute: (b, w, h, c) -> (b, c, h, w)
        x_out = x_out.permute(0, 3, 2, 1).contiguous()
        
        # 4. Element-wise Multiplication
        #    GAM 论文中 Channel attention 也是通过 sigmoid 激活后相乘
        x = x * x_out.sigmoid()

        # ----------------------------------------
        # Spatial Attention Path
        # ----------------------------------------
        # 直接通过卷积层处理，不进行压缩(Channel Pooling)
        x_spatial = self.spatial_attention(x)
        x = x * x_spatial
        
        return x
