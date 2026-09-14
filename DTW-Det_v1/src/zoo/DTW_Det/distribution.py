import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_wavelets import DWTForward, DWTInverse

from .utils import get_activation
from .attention import ChannelAttention
from .base_conv import PointConv, DeepConv, DWS_Conv


class PreGate(nn.Module):
    def __init__(self, in_ch, reduction=4):
        super(PreGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//reduction, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch//reduction),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch//reduction, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate(x)

class SpaceBranch(nn.Module):
    def __init__(self, in_ch=256):
        super(SpaceBranch, self).__init__()
        self.pconv1 = PointConv(in_ch * 2, in_ch, norm=True)
        self.dwsconv1 = DWS_Conv(in_ch, in_ch, kernel_size=3)
        # 多尺度
        self.dconv1 = DeepConv(in_ch // 2, kernel_size=3, dilation=2, padding=2, norm=False)
        self.dconv2 = DeepConv(in_ch // 2, kernel_size=3, dilation=3, padding=3, norm=False)
        self.pconv2 = PointConv(in_ch * 2, in_ch, norm=True)
        self.dwsconv2 = DWS_Conv(in_ch, in_ch, kernel_size=3)

    def forward(self, x1, x2):
        x_f = self.pconv1(torch.cat([x1, x2], dim=1))
        x_f = self.dwsconv1(x_f)
        # 多尺度
        xf_1, xf_2 = torch.split(x_f, x_f.shape[1] // 2, dim=1)
        xf_1 = self.dconv1(xf_1)
        xf_2 = self.dconv2(xf_2)
        x_f = self.pconv2(torch.cat([x_f, xf_1, xf_2], dim=1))
        x_f = self.dwsconv2(x_f)
        return x_f

class WaveletBranch(nn.Module):
    def __init__(self, in_ch=256):
        super(WaveletBranch, self).__init__()
        self.gate_pre = PreGate(in_ch)
        self.dconv = DeepConv(in_ch * 3, kernel_size=3, act=None, norm=False)

    def forward(self, xf, x_h):
        gate = self.gate_pre(xf)
        x_h = gate * x_h
        x_h = self.dconv(x_h)
        return x_h

class WSFusion(nn.Module):
    def __init__(self, in_ch=256, out_ch=256):
        super(WSFusion, self).__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        self.idwt = DWTInverse(wave='haar', mode='zero')
        self.pconv1 = PointConv(in_ch, out_ch, act=None, norm=True)
        self.pconv2 = nn.Sequential(
            PointConv(in_ch, out_ch, act=None, norm=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.sbranch = SpaceBranch(out_ch)
        self.wbranch = WaveletBranch(out_ch)
        self.ca = ChannelAttention(out_ch)
        self.dwsconv = DWS_Conv(in_ch, in_ch, kernel_size=3)

    def forward(self, x_shallow, x_deep):
        x_shallow = self.pconv1(x_shallow)
        x_deep = self.pconv2(x_deep)
        # 小波分解
        x_shallow_L, x_shallow_H = self.dwt(x_shallow)  # x_shallow_L=[B,C,H,W]  x_shallow_H[0]=[B,C,3,H,W]
        # 低频融合
        x_f = self.sbranch(x_shallow_L, x_deep)
        # 高频过滤
        B, C, _, H, W = x_shallow_H[0].shape
        x_shallow_H = x_shallow_H[0].transpose(1, 2).reshape(B, -1, H, W)  # [B,3C,H,W]
        x_shallow_H = self.wbranch(x_f, x_shallow_H)
        # 小波逆变换
        x_shallow_H = x_shallow_H.reshape(B, 3, C, H, W).transpose(1, 2)  # [B,C,3,H,W]
        x_recover = self.idwt((x_f, [x_shallow_H]))
        # 通道注意力
        x_recover = self.ca(x_recover) * x_recover
        x_recover = self.dwsconv(x_recover)
        return x_recover

class FADE(nn.Module):
    def __init__(self, in_ch=256, out_ch=256):
        super(FADE, self).__init__()
        self.wsfusion = WSFusion(in_ch, out_ch)
        self.distribution_head = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(out_ch // 2, 1, 1),  # 输出 [B,1,H,W]
            nn.Sigmoid()
        )
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x_shallow, x_deep):
        x = self.wsfusion(x_shallow, x_deep)
        # 生成分布图
        distribution = F.interpolate(
            self.distribution_head(x),
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )
        if distribution.max() > 0:
            distribution = distribution / distribution.max()
        reg_value = self.regression_head(x)
        return distribution, reg_value



if __name__ == '__main__':
    model = FADE()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)





