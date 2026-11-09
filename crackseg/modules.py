"""针对裂缝"细长、蜿蜒、易断裂"拓扑特点设计的两个模块。

1. StripPooling（条状池化，Hou et al., CVPR 2020, SPNet）
   普通 3x3 卷积 / 方形池化的感受野是"团状"的，会把混凝土表面的气孔、斑点、
   水渍纳入上下文，对细长目标不友好。条状池化分别沿水平方向（1xW）和竖直方向
   （Hx1）做全局池化，一个像素可以直接"看到"整行、整列的信息：
     - 沿裂缝走向方向：同一条裂缝上相距很远的像素能互相提供证据，弥补中间
       因对比度低而出现的断点，增强长距离连续性；
     - 垂直裂缝走向方向：条带很窄，引入的无关背景少。
   计算量是 O(H+W)，远小于自注意力 / Criss-Cross Attention 的显存开销，
   适合高分辨率巡检影像。

2. DSConv（动态蛇形卷积，Qi et al., ICCV 2023, DSCNet）
   标准卷积的采样点固定在规则网格上；可变形卷积（DCN）的采样点自由偏移，
   容易"跑偏"到背景上。DSConv 把 9 个采样点约束成一条连续的"蛇形"链：
   沿 x 轴（或 y 轴）逐点前进，每一步只允许在垂直方向上做 [-1,1] 的累积偏移，
   这样卷积核能自适应地贴合蜿蜒的管状结构，又保持了采样点的连续性。
   用 x 方向和 y 方向两个 DSConv 加一个标准卷积并联，覆盖任意走向的裂缝。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(cin, cout, k=3, p=None):
    if p is None:
        p = (k // 2) if isinstance(k, int) else (k[0] // 2, k[1] // 2)
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=p, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class StripPooling(nn.Module):
    """条状池化注意力：x * sigmoid(conv1x1(relu(H-strip + W-strip)))，带残差。"""

    def __init__(self, channels, reduction=2):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.reduce = conv_bn_relu(channels, mid, 1)
        # 竖直条 (H x 1) 上沿 H 方向做 3x1 卷积，水平条 (1 x W) 上沿 W 方向做 1x3 卷积
        self.conv_h = nn.Sequential(nn.Conv2d(mid, mid, (3, 1), padding=(1, 0), bias=False), nn.BatchNorm2d(mid))
        self.conv_w = nn.Sequential(nn.Conv2d(mid, mid, (1, 3), padding=(0, 1), bias=False), nn.BatchNorm2d(mid))
        self.fuse = nn.Conv2d(mid, channels, 1)

    def forward(self, x):
        _, _, h, w = x.shape
        y = self.reduce(x)
        strip_h = self.conv_h(F.adaptive_avg_pool2d(y, (h, 1))).expand(-1, -1, h, w)
        strip_w = self.conv_w(F.adaptive_avg_pool2d(y, (1, w))).expand(-1, -1, h, w)
        attn = torch.sigmoid(self.fuse(F.relu(strip_h + strip_w)))
        return x + x * attn


class DSConv(nn.Module):
    """动态蛇形卷积。morph=0: 核沿 x 轴展开、在 y 方向蛇形偏移；morph=1 反之。"""

    def __init__(self, cin, cout, kernel_size=9, morph=0, extend_scope=1.0):
        super().__init__()
        assert kernel_size % 2 == 1
        self.k, self.morph, self.scope = kernel_size, morph, extend_scope
        self.offset_conv = nn.Sequential(nn.Conv2d(cin, kernel_size, 3, padding=1), nn.BatchNorm2d(kernel_size))
        nn.init.zeros_(self.offset_conv[0].weight)
        nn.init.zeros_(self.offset_conv[0].bias)
        # 对 K 个采样点的加权求和，等价于一个 Kx1 / 1xK 卷积核
        self.weight_conv = nn.Conv2d(cin * kernel_size, cout, 1, bias=False)
        self.norm = nn.GroupNorm(max(cout // 8, 1), cout)
        self.act = nn.ReLU(inplace=True)

    def _snake_offsets(self, raw):
        # raw: (B,K,H,W) -> 以中心点为起点向两侧累积，保证采样链连续
        o = torch.tanh(raw) * self.scope
        c = self.k // 2
        right = torch.cumsum(o[:, c + 1:], dim=1)
        left = torch.flip(torch.cumsum(torch.flip(o[:, :c], [1]), dim=1), [1])
        return torch.cat([left, torch.zeros_like(o[:, c:c + 1]), right], dim=1)

    def forward(self, x):
        b, ch, h, w = x.shape
        k, c = self.k, self.k // 2
        off = self._snake_offsets(self.offset_conv(x))                      # (B,K,H,W)
        ys = torch.arange(h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        xs = torch.arange(w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
        steps = (torch.arange(k, device=x.device, dtype=x.dtype) - c).view(1, k, 1, 1)
        if self.morph == 0:   # 沿 x 前进，y 方向蛇形摆动
            px = (xs + steps).expand(b, k, h, w)
            py = ys + off
        else:                 # 沿 y 前进，x 方向蛇形摆动
            py = (ys + steps).expand(b, k, h, w)
            px = xs + off
        gx = 2.0 * px / max(w - 1, 1) - 1.0
        gy = 2.0 * py / max(h - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).reshape(b, k * h, w, 2)
        sampled = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled = sampled.view(b, ch, k, h, w).reshape(b, ch * k, h, w)
        return self.act(self.norm(self.weight_conv(sampled)))


class DSCBlock(nn.Module):
    """标准 3x3 卷积 + x 向 DSConv + y 向 DSConv 三路并联，再 1x1 融合。"""

    def __init__(self, cin, cout, kernel_size=9):
        super().__init__()
        self.reduce = conv_bn_relu(cin, cout, 1)
        self.std = conv_bn_relu(cout, cout, 3)
        self.dsc_x = DSConv(cout, cout, kernel_size, morph=0)
        self.dsc_y = DSConv(cout, cout, kernel_size, morph=1)
        self.fuse = conv_bn_relu(cout * 3, cout, 1)

    def forward(self, x):
        x = self.reduce(x)
        return self.fuse(torch.cat([self.std(x), self.dsc_x(x), self.dsc_y(x)], dim=1))
