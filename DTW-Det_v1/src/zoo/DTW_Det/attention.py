import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation

# =================================== CA ===================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        att = self.sigmoid(avg_out + max_out)
        return att

# =================================== ECA ===================================
class ECAChannelAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(ECAChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 使用 1D 卷积进行跨通道信息交互，不降维
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # [B, C, H, W] -> [B, C, 1, 1] -> [B, 1, C]
        y = self.gap(x).squeeze(-1).transpose(-1, -2)

        # [B, 1, C] -> [B, 1, C] -> [B, C, 1, 1]
        att = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        att = self.sigmoid(att)

        return x * att.expand_as(x)

# =================================== CBAM ===================================
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        max_out = self.mlp(F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        channel_att_sum = avg_out + max_out

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

# =================================== FlashAtt ===================================
class FlashAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(FlashAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None):
        B, L_q, _ = query.shape
        _, L_k, _ = key.shape
        # 线性映射并变形为 [B, n_head, L, head_dim]
        q = self.q_proj(query).view(B, L_q, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, L_k, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, L_k, self.n_head, self.head_dim).transpose(1, 2)
        # 调用底层 C++ 的 FlashAttention 算子 (极省显存)
        dropout_rate = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_rate)
        # 恢复形状并输出
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        return self.out_proj(out), None

# =================================== GlobalSelfAtt ===================================
class GlobalSelfAttention(nn.Module):
    def __init__(self, d_model=256, n_head=8, dim_feedforward=2048, dropout=0.1, activation="gelu",
                 normalize_before=False, use_flash=True):
        super(GlobalSelfAttention, self).__init__()
        self.normalize_before = normalize_before
        if use_flash:
            self.self_attn = FlashAttention(d_model, n_head, dropout)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)
        # FFN
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src

