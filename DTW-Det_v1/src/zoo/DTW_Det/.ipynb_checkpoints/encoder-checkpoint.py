import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from pytorch_wavelets import DWTForward

from src.core import register

from .distribution import FADE
from .utils import get_activation
from .base_conv import PointConv, DeepConv, DWS_Conv, BaseConv
from .attention import GlobalSelfAttention, CBAM, ChannelAttention, ECAChannelAttention
from .window_attention import TopologicalWindowAttention
from .conv_fusion import CDPC_Fusion_Block

# ===================================== Base =====================================
class ConvBnAct(nn.Module):
    def __init__(self,ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DWSConvBnAct(nn.Module):
    def __init__(self,ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super(DWSConvBnAct, self).__init__()
        self.dwconv = nn.Conv2d(
            ch_in,
            ch_in,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias,
            groups=ch_in)
        self.pwconv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.pwconv(self.dwconv(x))))
    
# ====================================== DGE ======================================
class DistributionGuidedEnhance(nn.Module):
    def __init__(self, in_ch=256, bias=False):
        super(DistributionGuidedEnhance, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=bias, groups=in_ch),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=5, padding=2, bias=bias, groups=in_ch),
            nn.SiLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch, kernel_size=1, bias=bias),
            nn.BatchNorm2d(in_ch),
            nn.SiLU()
        )

    def forward(self, x, dist_map):
        res = x
        b_dis_map = 1 - dist_map
        x = torch.cat([self.conv1(x * dist_map), self.conv2(x * b_dis_map)], dim=1)
        x = self.conv3(x) + res
        return x

# ====================================== SIF ======================================
class SemanticInjectionFusion(nn.Module):
    def __init__(self, in_ch=256, act='silu', win_size=10, n_head=8, dim_feedforward=1024, dropout=0.1, enc_act='gelu',
                 agent_pos_mode='deformable_offset', mask_mod='hard', hard_thrd=0.5, cdpc_n=3):
        super(SemanticInjectionFusion, self).__init__()
        self.pconv1 = PointConv(in_ch*2, in_ch, act=act)
        self.att = TopologicalWindowAttention(d_model=in_ch, window_size=win_size, num_heads=n_head,
                                              dim_feedforward=dim_feedforward, dropout=dropout, act=enc_act,
                                              agent_pos_mode=agent_pos_mode, mask_mod=mask_mod, hard_thrd=hard_thrd)
        self.conv_fusion = CDPC_Fusion_Block(in_ch=in_ch, out_ch=in_ch, n=cdpc_n, act=act)
        # self.pconv2 = PointConv(in_ch*3, in_ch, act=act)  # TODO 冗余未使用参数

    def forward(self, x_shallow, x_deep, dist_map_shallow):
        x_f = self.pconv1(torch.cat([x_shallow, x_deep], dim=1))
        x_f = self.att(x_f, dist_map_shallow) + x_f
        x_f = self.conv_fusion(x_f) + x_shallow  # TODO 考虑是否使用残差连接
        return x_f

# ====================================== DEF ======================================
class DetailEnhancementFusion(nn.Module):
    def __init__(self, in_ch=256, act='silu', cdpc_n=3, bias=False):
        super(DetailEnhancementFusion, self).__init__()
        # 小波分解
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        self.conv1 = ConvBnAct(in_ch * 3, in_ch, 1, 1, padding=0, act=act, bias=bias)
        self.conv2 = ConvBnAct(in_ch * 2, in_ch, 1, 1, padding=0, act=act, bias=bias)
        # 低频融合
        self.base_fusion = CDPC_Fusion_Block(in_ch=in_ch, out_ch=in_ch, n=cdpc_n, act=act)
        # 高频处理
        self.conv3 = nn.Sequential(
            DWSConvBnAct(in_ch, in_ch, 3, 1, padding=1, act=act, bias=bias),
            DWSConvBnAct(in_ch, in_ch, (3, 1), 1, padding=(1, 0), act=act, bias=bias),
            DWSConvBnAct(in_ch, in_ch, (1, 3), 1, padding=(0, 1), act=act, bias=bias)
        )
        self.conv4 = ConvBnAct(in_ch, in_ch, 1, 1, padding=0, bias=bias, act=act)
        # 门控
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch // 2),
            nn.SiLU(),
            nn.Conv2d(in_ch // 2, 1, kernel_size=1),
            nn.Tanh()
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))
        # 高低频融合
        self.conv5 = ConvBnAct(in_ch * 2, in_ch, 1, 1, padding=0, act=act, bias=bias)
        self.conv6 = DWSConvBnAct(in_ch, in_ch, 3, 1, padding=1, act=act, bias=bias)

    def forward(self, x_deep, x_shallow, dist_map_deep):
        # 小波分解
        x_shallow_L, x_shallow_H = self.dwt(x_shallow)
        # 高频部分
        b, c, _, h, w = x_shallow_H[0].shape
        x_shallow_H = x_shallow_H[0].view(b, c * 3, h, w)
        x_shallow_H = self.conv1(x_shallow_H)
        # 低频融合
        x_low= torch.cat([x_deep, x_shallow_L], dim=1)
        x_low = self.conv2(x_low)
        x_low = self.base_fusion(x_low) + x_deep
        # 高频处理
        x_shallow_H = self.conv3(x_shallow_H) + self.conv4(x_shallow_H)
        gate = self.gate_conv(x_low)
        gate = F.sigmoid(dist_map_deep + self.alpha * gate)
        x_shallow_H = gate * x_shallow_H
        # 高低频融合
        x_fusion = torch.cat([x_low, x_shallow_H], dim=1)
        x_fusion = self.conv6(self.conv5(x_fusion))
        return x_fusion

# ====================================== DFB ======================================
class DifferentialFeedbackBlock(nn.Module):
    def __init__(self, in_ch=256, act='silu', dis_map_filt=False):
        super(DifferentialFeedbackBlock, self).__init__()
        self.dis_map_filt = dis_map_filt
        # 深层特征下采样
        self.down_conv = BaseConv(in_ch, in_ch, kernel_size=3, padding=1, stride=2, act=act)
        self.ca = ChannelAttention(in_ch)  # 使用通道注意,减弱背景通道的影响
        # 差分门控
        self.diff_gate = nn.Sequential(
            BaseConv(in_ch, in_ch // 2, kernel_size=3, padding=1, act=act),
            PointConv(in_ch // 2, 1, act=None, norm=False),
            nn.Sigmoid()
        )

    def forward(self, x_deep, x_shallow, dist_map_shallow):
        x_deep = self.down_conv(x_deep)
        x_deep = F.interpolate(x_deep, size=x_shallow.shape[2:], mode='bilinear', align_corners=False)
        diff = F.relu(x_shallow - x_deep)
        channel_att = self.ca(x_shallow)  # 使用通道注意,减弱背景通道的影响
        diff = diff * channel_att
        diff_att = self.diff_gate(diff)
        if self.dis_map_filt:
            diff_att = diff_att * dist_map_shallow
        x_shallow = x_shallow * diff_att + x_shallow
        return x_shallow

# ====================================== Encoder ======================================
@register
class HybridEncoder(nn.Module):
    def __init__(self,
                 # 基本参数
                 in_channels=[256, 512, 1024],
                 feat_strides=[4, 8, 16],
                 hidden_dim=256,
                 act='silu',
                 eval_spatial_size=None,
                 # FADE参数
                 dist_detach=False,  # 是否截断分布图的梯度
                 # 基本注意力参数
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 pe_temperature=10000,
                 # DGE
                 DGE=True,  # 分布图引导增强
                 # CDPC
                 cdpc_n=3,
                 # SIF : agent-att 参数
                 windows_size=10,
                 agent_pos_mode='deformable_offset',
                 mask_mod='hard',
                 hard_thrd=0.5,
                 # DEF
                 dis_map_filt=False  # 差分反馈使用分布图过滤
                 ):
        super(HybridEncoder, self).__init__()
        # ================== 参数 ==================
        # 基础参数
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.eval_spatial_size = eval_spatial_size
        # attention参数
        self.pe_temperature = pe_temperature
        # agent-att 参数
        self.windows_size = windows_size
        # FADE
        self.dist_detach = dist_detach  # 是否截断分布图的梯度
        # DGE
        self.DGE = DGE

        # ================== 通道映射 ==================
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )
        # ================== 目标分布提取 ==================
        self.fade = FADE(in_ch=hidden_dim, out_ch=hidden_dim)
        # ================== DGE ==================
        if self.DGE:
            self.dge = nn.ModuleList()
            for _ in self.in_channels:
                self.dge.append(DistributionGuidedEnhance(in_ch=hidden_dim))
        # ================== Top-->Down ==================
        # 最深层增强
        self.global_att = GlobalSelfAttention(d_model=hidden_dim, n_head=nhead, dim_feedforward=dim_feedforward,
                                              dropout=dropout, activation=enc_act, use_flash=True)
        # TOP-->Down
        self.lateral_convs = nn.ModuleList()
        self.sif_blocks = nn.ModuleList()
        for i in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvBnAct(hidden_dim, hidden_dim, 1, 1, padding=0 ,act=act))
            self.sif_blocks.append(
                SemanticInjectionFusion(in_ch=hidden_dim, act=act, win_size=windows_size, n_head=nhead,
                                        dim_feedforward=dim_feedforward, dropout=dropout, enc_act=enc_act,
                                        agent_pos_mode=agent_pos_mode, mask_mod=mask_mod, hard_thrd=hard_thrd, cdpc_n=cdpc_n)
            )
        # ================== Down-->Top ==================
        # 最深层增强
        self.cbam = CBAM(hidden_dim)
        # Down-->Top
        self.def_blocks = nn.ModuleList()
        self.dfb_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.def_blocks.append(
                DetailEnhancementFusion(in_ch=hidden_dim, act=act, cdpc_n=cdpc_n)
            )
            self.dfb_blocks.append(
                DifferentialFeedbackBlock(in_ch=hidden_dim, act=act, dis_map_filt=dis_map_filt)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx, stride in enumerate(self.feat_strides):
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[0] // stride, self.eval_spatial_size[1] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    def dist_downsample(self, dist_map):
        multiscale_dist_maps = []
        for s in self.feat_strides:
            stride_dist_map = F.max_pool2d(dist_map, kernel_size=s, stride=s, ceil_mode=True)
            multiscale_dist_maps.append(stride_dist_map)
        return multiscale_dist_maps

    @staticmethod
    def build_2d_sincos_position_embedding(h, w, embed_dim=256, temperature=10000.):
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        out_h = grid_h.flatten()[..., None] @ omega[None]
        out_w = grid_w.flatten()[..., None] @ omega[None]
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats, gauss_gts=None):
        outs = {}
        assert len(feats) == len(self.in_channels)
        # ================== 输入特征统一通道 ==================
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # ================== 生成目标分布图 ==================
        dist_map, reg_value = self.fade(proj_feats[0], proj_feats[-1])
        outs["FADE"] = {"reg_value": reg_value, "distribution_map": dist_map}
        if gauss_gts is not None:
            outs["FADE"]["gt_distribution_map"] = gauss_gts
        if self.dist_detach:
            dist_map_detach = dist_map.detach()  # 截断分布图梯度, 只作为后续输入
        else:
            dist_map_detach = dist_map
        # 生成多尺度目标分布金字塔
        multiscale_dist_maps = self.dist_downsample(dist_map_detach)  # 最大/平均池化下采样
        outs["FADE"]["multiscale_dist_maps"] = multiscale_dist_maps
        # ================== DGE ==================
        if self.DGE:
            for i in range(len(self.in_channels)):
                proj_feats[i] = self.dge[i](proj_feats[i], multiscale_dist_maps[i])
        # ================== Top-->Down ==================
        # 最深层增强
        enc_ind = len(self.in_channels) - 1
        h, w = proj_feats[enc_ind].shape[2:]
        src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)  # [B,C,H,W] --> [B,H*W,C]
        if self.training or self.eval_spatial_size is None:
            pos_embed = self.build_2d_sincos_position_embedding(
                h, w, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
        else:
            pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
        memory = self.global_att(src_flatten, pos_embed=pos_embed) + src_flatten
        proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
        # Top-->Down
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            feat_low = proj_feats[idx - 1]
            distribution_map = multiscale_dist_maps[idx - 1]
            inner_out = self.sif_blocks[len(self.in_channels) - 1 - idx](feat_low, upsample_feat, distribution_map)
            inner_outs.insert(0, inner_out)
        # ================== Top-->Down ==================
        # 最浅层增强
        inner_outs[0] = self.cbam(inner_outs[0])
        # Down-->Top
        feat_outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_shallow = feat_outs[-1]
            feat_deep = inner_outs[idx + 1]
            distribution_map_deep = multiscale_dist_maps[idx + 1]
            distribution_map_shallow = multiscale_dist_maps[idx]
            # 处理
            feat_deep = self.def_blocks[idx](feat_deep, feat_shallow, distribution_map_deep)
            feat_shallow = self.dfb_blocks[idx](feat_deep, feat_shallow, distribution_map_shallow)
            # 保存
            feat_outs[-1] = feat_shallow
            feat_outs.append(feat_deep)

        outs["feats"] = feat_outs
        return outs

if __name__ == '__main__':
    model = SemanticInjectionFusion()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)