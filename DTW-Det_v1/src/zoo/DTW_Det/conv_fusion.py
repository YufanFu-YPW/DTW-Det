import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_conv import BaseConv, PointConv, DeepConv, DWS_Conv
from .utils import get_activation

# ================================== CDPC ==================================================
class ChannelDecreasingPartialConv(nn.Module):
    def __init__(self, in_ch=256, out_ch=256, n=3, act='silu'):
        super(ChannelDecreasingPartialConv, self).__init__()
        self.n = n
        self.pconv1 = PointConv(in_ch, out_ch, act=act)
        # 3*3卷积层
        self.conv_layers = nn.ModuleList()
        current_ch = out_ch
        for i in range(n):
            feat_ch = current_ch // 2
            self.conv_layers.append(BaseConv(feat_ch, feat_ch, kernel_size=3, padding=1, act=act))
            current_ch = feat_ch
        # 融合
        self.pconv2 = PointConv(out_ch, out_ch, act=act)

    def forward(self, x):
        x = self.pconv1(x)
        outputs = []
        current_feat = x
        for i in range(self.n):
            keep, process = torch.chunk(current_feat, 2, dim=1)
            outputs.append(keep)  # 保留一部分
            current_feat = self.conv_layers[i](process)
        outputs.append(current_feat)  # 把最后一次卷积的结果也加进去
        out = torch.cat(outputs, dim=1)
        out = self.pconv2(out)
        return out

class ChannelDecreasingPartialConv_v2(nn.Module):
    def __init__(self, in_ch=256, out_ch=256, n=3, act='silu'):
        super(ChannelDecreasingPartialConv_v2, self).__init__()
        self.n = n
        self.pconv1 = PointConv(in_ch, out_ch, act=act)
        # 3*3卷积层
        self.conv_layers = nn.ModuleList()
        self.conv_fusion_layers = nn.ModuleList()
        current_ch = out_ch
        for i in range(n):
            feat_ch = current_ch // 2
            self.conv_layers.append(BaseConv(feat_ch, feat_ch, kernel_size=3, padding=1, act=act))
            if i == n-1:
                self.conv_fusion_layers.append(BaseConv(feat_ch, feat_ch, kernel_size=3, padding=1, act=act))
            else:
                self.conv_fusion_layers.append(BaseConv(feat_ch // 2, feat_ch // 2, kernel_size=3, padding=1, act=act))
            current_ch = feat_ch
        # 融合
        self.pconv2 = PointConv(out_ch, out_ch, act=act)

    def forward(self, x):
        x = self.pconv1(x)
        outputs = []
        keep, process = torch.chunk(x, 2, dim=1)
        for i in range(self.n - 1):
            keep_1, keep_2 = torch.chunk(keep, 2, dim=1)
            outputs.append(keep_1)
            conv_out = self.conv_layers[i](process)
            keep, process = torch.chunk(conv_out, 2, dim=1)
            keep_fusion = self.conv_fusion_layers[i](keep_2 + keep)
            outputs.append(keep_fusion)
        outputs.append(keep)
        conv_out = self.conv_layers[self.n - 1](process)
        keep_fusion = self.conv_fusion_layers[self.n - 1](conv_out + keep)
        outputs.append(keep_fusion)
        out = torch.cat(outputs, dim=1)
        out = self.pconv2(out)
        return out

class ChannelDecreasingPartialConv_v3(nn.Module):
    def __init__(self, in_ch=256, out_ch=256, n=3, act='silu'):
        super(ChannelDecreasingPartialConv_v3, self).__init__()
        self.n = n
        self.pconv1 = PointConv(in_ch, out_ch, act=act)
        # 3*3卷积层
        self.conv_layers = nn.ModuleList()
        self.conv_fusion_layers = nn.ModuleList()
        current_ch = out_ch
        for i in range(n):
            feat_ch = current_ch // 2
            if i == n-1:
                self.conv_layers.append(BaseConv(feat_ch, feat_ch, kernel_size=3, padding=1, act=act))
                self.conv_fusion_layers.append(BaseConv(feat_ch, feat_ch, kernel_size=3, padding=1, act=act))
            else:
                self.conv_layers.append(BaseConv(feat_ch, feat_ch // 2, kernel_size=3, padding=1, act=act))
                self.conv_fusion_layers.append(BaseConv(feat_ch // 2, feat_ch // 2, kernel_size=3, padding=1, act=act))
            current_ch = feat_ch
        # 融合
        self.pconv2 = PointConv(out_ch, out_ch, act=act)

    def forward(self, x):
        x = self.pconv1(x)
        outputs = []
        keep, process = torch.chunk(x, 2, dim=1)
        for i in range(self.n - 1):
            keep_1, keep_2 = torch.chunk(keep, 2, dim=1)
            outputs.append(keep_1)
            keep = process = self.conv_layers[i](process)
            keep_fusion = self.conv_fusion_layers[i](keep_2 + keep)
            outputs.append(keep_fusion)
        outputs.append(keep)
        conv_out = self.conv_layers[self.n - 1](process)
        keep_fusion = self.conv_fusion_layers[self.n - 1](conv_out + keep)
        outputs.append(keep_fusion)
        out = torch.cat(outputs, dim=1)
        out = self.pconv2(out)
        return out

class CDPC_Fusion_Block(nn.Module):
    def __init__(self, in_ch=256, out_ch=256, n=3, act='silu'):
        super(CDPC_Fusion_Block, self).__init__()
        self.pconv1 = PointConv(in_ch, out_ch, act=act)
        self.CDPC1 = ChannelDecreasingPartialConv_v2(in_ch, out_ch, n=n, act=act)
        self.CDPC2 = ChannelDecreasingPartialConv_v2(in_ch, out_ch, n=n, act=act)
        self.pconv2 = PointConv(out_ch * 2, out_ch, act=act)

    def forward(self, x):
        x1 = self.CDPC1(x)
        x2 = self.CDPC2(x1)
        out = self.pconv2(torch.cat([x1, x2], dim=1)) + self.pconv1(x)
        return out




if __name__ == '__main__':
    model = CDPC_Fusion_Block()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)