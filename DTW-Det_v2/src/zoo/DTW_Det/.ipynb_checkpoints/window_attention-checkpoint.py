import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation


class FFN(nn.Module):
    def __init__(self, d_model=256, dim_feedforward=1024, dropout=0.1, act='gelu'):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = get_activation(act)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class LocalAgentSelfAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8, dim_feedforward=1024, dropout=0.1, act='gelu', agent_pos_mode='deformable_offset'):
        super(LocalAgentSelfAttention, self).__init__()
        assert agent_pos_mode in ['attention_weighted', 'deformable_offset'], "不支持的 PE 模式"
        self.d_model = d_model
        self.num_heads = num_heads
        self.agent_pos_mode = agent_pos_mode
        # 初始化 Agent Token 和基础 PE
        self.agent_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.agent_pos_init = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.agent_token, std=0.02)
        nn.init.normal_(self.agent_pos_init, std=0.02)
        # self attention
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = FFN(d_model, dim_feedforward, dropout, act)
        # 偏移编码
        if self.agent_pos_mode == 'deformable_offset':
            self.offset_pred = nn.Linear(d_model, 2)

    @staticmethod
    def generate_continuous_2d_pe(coords, dim=256):
        x, y = coords[:, :, 0], coords[:, :, 1]
        pos_dim = dim // 4
        omega = torch.arange(pos_dim, device=coords.device) / pos_dim
        omega = 1.0 / (10000.0 ** omega)
        out_x = x.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)
        out_y = y.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)
        pe_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=-1)
        pe_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=-1)
        pe = torch.cat([pe_x, pe_y], dim=-1)
        return pe

    def forward(self, x_windows, x_windows_rel_pos, x_windows_global_pos, window_size, window_centers=None):
        B, M, N, C = x_windows.shape
        # Agent查询
        agent_query = self.agent_token.view(1, 1, 1, C).expand(B, M, -1, -1)
        agent_query_pos_init = self.agent_pos_init.view(1, 1, 1, C).expand(B, M, -1, -1)  # agent的初始化可学习位置编码
        # 拼接
        z = torch.cat([agent_query, x_windows], dim=2)  # (B, M, 1+N, C)
        z_pos = torch.cat([agent_query_pos_init, x_windows_rel_pos.expand(B, -1, -1, -1)], dim=2)  # (B, M, 1+N, C)
        z = z.view(B * M, 1 + N, C)
        z_pos = z_pos.view(B * M, 1 + N, C)
        # 自注意力
        query = key = z + z_pos
        att_out, att_weight = self.mha(query, key, z, need_weights=True, average_attn_weights=True)
        z = self.norm1(z + self.dropout1(att_out))
        # FFN
        z = self.norm2(z + self.dropout2(self.ffn(z)))
        z = z.view(B, M, 1 + N, C)
        # 分离特征
        agent_out = z[:, :, 0:1, :].squeeze(2)  # (B, M, C)
        x_windows_out = z[:, :, 1:, :]  # (B, M, N, C)
        # 生成Agent的全局编码
        if self.agent_pos_mode == 'attention_weighted':
            att_weight = att_weight.view(B, M, 1 + N, 1 + N)
            agent_pos_weight = att_weight[:, :, 0, 1:]  # (B, M, N)
            agent_pos_weight = agent_pos_weight / (agent_pos_weight.sum(dim=-1, keepdim=True) + 1e-6)  # 重新归一化
            agent_pos = torch.einsum('bmn,bmnc->bmc', agent_pos_weight, x_windows_global_pos.expand(B, -1, -1, -1))
        else:
            offsets = torch.tanh(self.offset_pred(agent_out))
            offsets = offsets * (window_size / 2.0)
            agent_pos = window_centers + offsets
            agent_pos = self.generate_continuous_2d_pe(agent_pos, self.d_model)
        return x_windows_out, agent_out, agent_pos

class GlobalTopologyScoring(nn.Module):
    def __init__(self, d_model=256, mask_mod='hard', hard_thrd=0.35):
        super(GlobalTopologyScoring, self).__init__()
        assert mask_mod in ['soft', 'hard', 'base'], "mask_mod 必须是 'soft', 'hard' 或 'base'"
        self.d_model = d_model
        self.scale = d_model ** -0.5
        self.mask_mod = mask_mod
        self.hard_thrd = hard_thrd
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        if mask_mod == 'soft':
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, agent_token, agent_pos, dist_map):
        B, M, C = agent_token.shape
        agent_token = agent_token + agent_pos  # 加入位置编码
        query = self.q_proj(agent_token)  # [B,M,C]
        key = self.k_proj(agent_token)  # [B,M,C]
        att_map = (query @ key.transpose(-2, -1)) * self.scale  # 基础注意力分数
        # 分布图前景背景分割
        p = dist_map.view(B, M)  # 将 [B, 1, num_win_h, num_win_w] 的分布图展平
        if self.mask_mod == 'soft':
            p_i = p.unsqueeze(2)  # (B, M, 1)
            p_j = p.unsqueeze(1)  # (B, 1, M)
            mask = self.gamma * torch.log(p_i * p_j + (1 - p_i) * (1 - p_j) + 1e-8)  # [B,M,M]
            att_map = att_map + mask
        elif self.mask_mod == 'hard':
            is_fg = p > self.hard_thrd  # 形状: [B, M], 布尔张量
            is_fg_i = is_fg.unsqueeze(2)  # [B, M, 1]
            is_fg_j = is_fg.unsqueeze(1)  # [B, 1, M]
            # 如果都是前景 (True==True)，或都是背景 (False==False)，则 same_class 为 True
            same_class = (is_fg_i == is_fg_j)  # 形状: [B, M, M], 布尔张量
            att_map = att_map.masked_fill(~same_class, float('-inf'))
        att_map = att_map.softmax(dim=-1)  # [B, M, M]
        return att_map

class TopologyGuidedCrossAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8, dim_feedforward=1024, dropout=0.1, act='gelu'):
        super(TopologyGuidedCrossAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FFN(d_model, dim_feedforward, dropout, act)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.lambda_gate = nn.Parameter(torch.zeros(1))  # 门控缩放因子

    def forward(self, x_windows, x_windows_global_pos, agent_token, agent_pos, att_map):
        B, M, N, C = x_windows.shape
        x_windows = x_windows.reshape(B, M * N, C)
        x_windows_global_pos = x_windows_global_pos.expand(B, -1, -1, -1).contiguous().view(B, M * N, C)
        query = x_windows + x_windows_global_pos
        key = agent_token + agent_pos
        # 根据att_map计算对数偏置
        bias = self.lambda_gate * torch.log(att_map + 1e-8)  # [B, M, M]
        bias = bias.unsqueeze(2).expand(B, M, N, M)  # [B, M, N, M]
        att_mask = bias.reshape(B, M * N, M)
        att_mask = att_mask.repeat_interleave(self.num_heads, dim=0)  # [B * num_heads, M*N, M]
        # 交叉注意
        cross_out, _ = self.cross_attn(query, key, value=agent_token, attn_mask=att_mask, need_weights=False)
        x_windows = self.norm1(x_windows + cross_out)
        # FFN
        x_windows = self.norm2(x_windows + self.ffn(x_windows))
        x_windows = x_windows.view(B, M, N, C)
        return x_windows

class TopologicalWindowAttention(nn.Module):
    def __init__(self, d_model=256, window_size=10, num_heads=8, dim_feedforward=1024, dropout=0.1,
                 act='gelu', agent_pos_mode='deformable_offset', mask_mod='soft', hard_thrd=0.5):
        """
        agent_pos_mode 可选:
          - 'attention_weighted': 使用注意力分布加权绝对位置编码 (推荐，无参数)
          - 'deformable_offset': 使用特征预测坐标偏移量生成位置编码 (可学习度高)
        """
        super(TopologicalWindowAttention, self).__init__()
        self.window_size = window_size
        self.d_model = d_model
        self.agent_pos_mode = agent_pos_mode  # agent token 编码方式
        # ===================== 结构 =====================
        # 可学习相对位置编码
        self.rel_pos_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
        self.stage1 = LocalAgentSelfAttention(d_model, num_heads, dim_feedforward, dropout, act, agent_pos_mode)
        self.stage2 = GlobalTopologyScoring(d_model, mask_mod, hard_thrd)
        self.stage3 = TopologyGuidedCrossAttention(d_model, num_heads, dim_feedforward, dropout, act)

    @staticmethod
    def build_2d_sincos_position_embedding(h, w, embed_dim=256, temperature=10000.):
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        out_h = grid_h.flatten()[..., None] @ omega[None]
        out_w = grid_w.flatten()[..., None] @ omega[None]
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)  # [H*W,C]

    def _get_rel_pos_emb(self, window_size, device):
        h = w = window_size
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        coords = torch.stack([grid_x / (w - 1), grid_y / (h - 1)], dim=-1)
        return self.rel_pos_proj(coords.to(device))

    def forward(self, x, dist_map):
        B, C, H, W = x.shape
        K = self.window_size
        assert H % K == 0 and W % K == 0, "特征图尺寸必须能被窗口大小整除"
        num_win_h, num_win_w = H // K, W // K
        M, N = num_win_h * num_win_w, K * K
        device = x.device
        # ===================== 预处理: 窗口划分，位置编码 =====================
        # 窗口相对编码
        x_windows_rel_pos = self._get_rel_pos_emb(K, device).view(-1, C)  # [K,K,C] --> [N,C]
        x_windows_rel_pos = x_windows_rel_pos.view(1, 1, N, C).expand(-1, M, -1, -1)  # [1,M,N,C]
        # 全局绝对编码
        global_pos_emb = self.build_2d_sincos_position_embedding(H, W, self.d_model).to(device)  # 绝对全局位置编码 [H*W,C]
        x_windows_global_pos = global_pos_emb.view(num_win_h, K, num_win_w, K, C)
        x_windows_global_pos = x_windows_global_pos.permute(0, 2, 1, 3, 4).contiguous().view(1, M, N, C)  # 变换到对应窗口
        # 特征划分窗口
        x_windows = x.permute(0, 2, 3, 1).view(B, num_win_h, K, num_win_w, K, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, M, N, C)  # 完成窗口划分
        # 准备窗口中心的绝对全局位置
        window_centers = None
        if self.agent_pos_mode == 'deformable_offset':
            grid_y, grid_x = (torch.meshgrid(torch.arange(num_win_h, device=device), torch.arange(num_win_w, device=device), indexing='ij'))
            center_y = (grid_y.flatten() + 0.5) * K
            center_x = (grid_x.flatten() + 0.5) * K
            window_centers = torch.stack([center_x, center_y], dim=-1).unsqueeze(0).expand(B, -1, -1)  # (B, M, 2)
        # 分布图池化
        dist_map_pooled = F.adaptive_max_pool2d(dist_map, (num_win_h, num_win_w))  # [B,1,H,W] --> [B,1,num_win_h,num_win_W]
        # ===================== 局部提取与动态Agent生成 =====================
        x_windows_out, agent_out, agent_pos = self.stage1(x_windows, x_windows_rel_pos, x_windows_global_pos, K, window_centers)
        # ===================== 拓扑建模 =====================
        topology_score = self.stage2(agent_out, agent_pos, dist_map_pooled)
        # ===================== 全局交叉注意 =====================
        x_windows_out = self.stage3(x_windows_out, x_windows_global_pos, agent_out, agent_pos, topology_score)
        # ===================== 重建特征 =====================
        final_out = x_windows_out.view(B, num_win_h, num_win_w, K, K, C)
        final_out = final_out.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
        return final_out


# FlashAttention



if __name__ == '__main__':
    model = TopologicalWindowAttention()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


