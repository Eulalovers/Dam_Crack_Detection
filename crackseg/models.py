"""U-Net（ResNet34 编码器）及其改进版本。

所有变体共用同一编码器和解码器骨架，只在以下位置插拔改进模块，保证消融实验公平：
  - sp : 在瓶颈层和 1/16、1/8、1/4 解码层后加入 StripPooling
  - dsc: 将 1/8、1/4、1/2 解码层的卷积替换为 DSCBlock（标准卷积 + 双向蛇形卷积）
另外所有变体都带一个全分辨率浅层分支，保留 1~2 像素宽细裂缝的边缘细节。
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import DSCBlock, StripPooling, conv_bn_relu

TORCHVISION_WEIGHTS = {
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
}


class DecoderBlock(nn.Module):
    def __init__(self, cin, skip, cout, use_dsc=False, use_sp=False):
        super().__init__()
        first = DSCBlock(cin + skip, cout) if use_dsc else conv_bn_relu(cin + skip, cout)
        self.body = nn.Sequential(first, conv_bn_relu(cout, cout))
        self.sp = StripPooling(cout) if use_sp else nn.Identity()

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.sp(self.body(torch.cat([x, skip], dim=1)))


class CrackUNet(nn.Module):
    def __init__(self, use_sp=False, use_dsc=False, encoder="resnet34", pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(encoder, features_only=True, pretrained=False)
        if pretrained:
            # 国内网络访问 HuggingFace 不稳定，直接从 PyTorch 官方加载 ImageNet 权重（参数名与 timm 一致）
            state = torch.hub.load_state_dict_from_url(TORCHVISION_WEIGHTS[encoder], progress=False)
            missing, _ = self.encoder.load_state_dict(state, strict=False)
            assert not missing, f"encoder weights missing: {missing[:5]}"
        c = self.encoder.feature_info.channels()          # resnet34: [64, 64, 128, 256, 512]
        self.center_sp = StripPooling(c[4]) if use_sp else nn.Identity()
        self.d4 = DecoderBlock(c[4], c[3], 256, use_sp=use_sp)                 # 1/16
        self.d3 = DecoderBlock(256, c[2], 128, use_dsc=use_dsc, use_sp=use_sp)  # 1/8
        self.d2 = DecoderBlock(128, c[1], 64, use_dsc=use_dsc, use_sp=use_sp)   # 1/4
        self.d1 = DecoderBlock(64, c[0], 32, use_dsc=use_dsc)                   # 1/2
        self.shallow = nn.Sequential(conv_bn_relu(3, 16), conv_bn_relu(16, 16))  # 1/1
        self.d0 = nn.Sequential(conv_bn_relu(32 + 16, 32), conv_bn_relu(32, 16))
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        f = self.encoder(x)
        y = self.center_sp(f[4])
        y = self.d4(y, f[3])
        y = self.d3(y, f[2])
        y = self.d2(y, f[1])
        y = self.d1(y, f[0])
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        y = self.d0(torch.cat([y, self.shallow(x)], dim=1))
        return self.head(y)


MODEL_VARIANTS = {
    "unet": dict(use_sp=False, use_dsc=False),
    "unet_sp": dict(use_sp=True, use_dsc=False),
    "unet_dsc": dict(use_sp=False, use_dsc=True),
    "unet_sp_dsc": dict(use_sp=True, use_dsc=True),
}


def build_model(name, pretrained=True):
    return CrackUNet(pretrained=pretrained, **MODEL_VARIANTS[name])
