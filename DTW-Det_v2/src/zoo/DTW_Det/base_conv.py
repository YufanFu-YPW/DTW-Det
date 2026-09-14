import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation


class PointConv(nn.Module):
    def __init__(self, in_ch, out_ch, act='silu', norm=True):
        super(PointConv, self).__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        if act is not None:
            layers.append(get_activation(act, inpace=True))
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)

class DeepConv(nn.Module):
    def __init__(self, in_ch, kernel_size=3, dilation=1, padding=1, stride=1, act='silu', norm=True):
        super(DeepConv, self).__init__()
        layers = [nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding,
                            dilation=dilation, bias=not norm, groups=in_ch)]
        if norm:
            layers.append(nn.BatchNorm2d(in_ch))
        if act is not None:
            layers.append(get_activation(act, inpace=True))
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)

class DWS_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, padding=1, stride=1, act='silu'):
        super(DWS_Conv, self).__init__()
        self.dconv = DeepConv(in_ch, kernel_size, dilation, padding, stride, act=None, norm=False)
        self.pconv = PointConv(in_ch, out_ch, act=act, norm=True)

    def forward(self, x):
        return self.pconv(self.dconv(x))

class BaseConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, padding=1, stride=1, act='silu', norm=True):
        super(BaseConv, self).__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        if act is not None:
            layers.append(get_activation(act, inpace=True))
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)








