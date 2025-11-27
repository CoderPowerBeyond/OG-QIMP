import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, Set2Set, global_max_pool, global_add_pool, global_mean_pool
from torch_geometric.nn import GATv2Conv, GINConv, GCNConv, SAGEConv
from torch_geometric.utils import add_self_loops, softmax, degree
from torch.cuda.amp import autocast
import numpy as np

class EfficientOrbitalAttention(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super().__init__(aggr='add', node_dim=0, flow='source_to_target')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        # 修复: 确保heads能整除out_channels
        self.head_dim = out_channels // heads
        # 如果不能整除，调整out_channels
        self.actual_out_channels = self.head_dim * heads

        # 统一的QKV投影（更高效）
        self.qkv_sigma = nn.Linear(in_channels, self.actual_out_channels * 3, bias=False)
        self.qkv_pi = nn.Linear(in_channels, self.actual_out_channels * 3, bias=False)
        self.qkv_nb = nn.Linear(in_channels, self.actual_out_channels * 3, bias=False)

        # 轨道权重（可学习）
        self.orbital_weights = nn.Parameter(torch.tensor([0.6, 0.3, 0.1]))

        # 轨道特定的缩放因子
        self.sigma_scale = nn.Parameter(torch.ones(heads))
        self.pi_scale = nn.Parameter(torch.ones(heads))
        self.nb_scale = nn.Parameter(torch.ones(heads))

        # 高效的边特征投影
        self.edge_proj = nn.Linear(12, heads, bias=False)

        # 输出投影 - 确保输出维度正确
        self.out_proj = nn.Linear(self.actual_out_channels * 3, out_channels)

        # 归一化和dropout
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

        # 残差连接
        self.skip = nn.Linear(in_channels, out_channels, bias=False) if in_channels != out_channels else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_sigma.weight)
        nn.init.xavier_uniform_(self.qkv_pi.weight)
        nn.init.xavier_uniform_(self.qkv_nb.weight)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, edge_index, edge_attr, orbital_overlap, layer_idx, total_layers, return_intermediate=False):
        # 渐进权重
        lambda_prog = np.tanh(2.0 * layer_idx / total_layers)  # 使用tanh获得更平滑的过渡

        # 残差
        residual = self.skip(x)

        # 批量计算所有轨道的QKV
        N = x.size(0)

        # σ键处理
        qkv_sigma = self.qkv_sigma(x).view(N, 3, self.heads, self.head_dim)
        q_sigma, k_sigma, v_sigma = qkv_sigma.unbind(1)

        # π键处理
        qkv_pi = self.qkv_pi(x).view(N, 3, self.heads, self.head_dim)
        q_pi, k_pi, v_pi = qkv_pi.unbind(1)

        # 非键处理
        qkv_nb = self.qkv_nb(x).view(N, 3, self.heads, self.head_dim)
        q_nb, k_nb, v_nb = qkv_nb.unbind(1)

        # 并行消息传递
        out_sigma = self.propagate(edge_index, q=q_sigma, k=k_sigma, v=v_sigma,
                                   edge_attr=edge_attr, orbital_bias=orbital_overlap['sigma'],
                                   lambda_prog=lambda_prog, scale=self.sigma_scale, orbital_type='sigma')

        out_pi = self.propagate(edge_index, q=q_pi, k=k_pi, v=v_pi,
                                edge_attr=edge_attr, orbital_bias=orbital_overlap['pi'],
                                lambda_prog=lambda_prog, scale=self.pi_scale, orbital_type='pi')

        out_nb = self.propagate(edge_index, q=q_nb, k=k_nb, v=v_nb,
                               edge_attr=edge_attr, orbital_bias=orbital_overlap['nonbonding'],
                               lambda_prog=lambda_prog, scale=self.nb_scale, orbital_type='nb')

        # 动态轨道混合
        weights = F.softmax(self.orbital_weights * (1 + lambda_prog), dim=0)

        # 高效融合
        out = torch.cat([
            weights[0] * out_sigma.view(N, -1),
            weights[1] * out_pi.view(N, -1),
            weights[2] * out_nb.view(N, -1)
        ], dim=-1)

        # 输出投影
        out = self.out_proj(out)

        # 残差连接和归一化
        out = self.norm(out + residual)
        out = self.dropout(out)

        return out

    def message(self, q_i, k_j, v_j, edge_attr, orbital_bias, lambda_prog, scale, index, ptr, size_i, orbital_type=None):
        # 高效的注意力计算
        scores = (q_i * k_j).sum(dim=-1) / (self.head_dim ** 0.5)
        scores = scores * scale.unsqueeze(0)

        # 边特征融合
        if edge_attr is not None:
            edge_scores = self.edge_proj(edge_attr.float())
            scores = scores + edge_scores

        # 轨道引导（渐进式）
        if orbital_bias is not None:
            physics_weight = (1 - lambda_prog) * 2.0  # 增强物理约束
            orbital_guidance = physics_weight * orbital_bias.unsqueeze(-1)
            scores = scores + orbital_guidance

        # 保存softmax之前的scores供提取
        if not hasattr(self, '_scores_cache'):
            self._scores_cache = {}
        self._scores_cache[orbital_type] = scores
        
        # 稳定的softmax
        alpha = softmax(scores, index, ptr, size_i)
        alpha = F.dropout(alpha, p=0.1, training=self.training)
        
        # 保存attention权重供提取
        if not hasattr(self, '_alpha_cache'):
            self._alpha_cache = {}
        self._alpha_cache[orbital_type] = alpha

        return (alpha.unsqueeze(-1) * v_j).view(-1, self.heads * self.head_dim)

class OptimizedHOMOLUMOEncoder(nn.Module):
    """
    创新2: HOMO-LUMO前线分子轨道编码器 - 优化版
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        # 共享的特征提取器
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05)
        )

        # HOMO和LUMO特定头
        self.homo_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()  # HOMO能量为负
        )

        self.lumo_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()  # LUMO能量为正
        )

        # 能隙和化学硬度编码（合并计算）
        self.gap_hardness_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 最终融合（使用门控机制）
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )

        self.fusion_transform = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, x):
        # 共享编码
        shared_features = self.shared_encoder(x)

        # HOMO/LUMO特征
        homo = self.homo_head(shared_features)
        lumo = self.lumo_head(shared_features)

        # 能隙和化学硬度
        orbital_pair = torch.cat([homo, lumo], dim=-1)
        gap_hardness = self.gap_hardness_encoder(orbital_pair)

        # 门控融合
        combined = torch.cat([homo, lumo, gap_hardness], dim=-1)
        gate = self.fusion_gate(combined)
        transform = self.fusion_transform(combined)

        return gate * transform + (1 - gate) * shared_features

class EnhancedReactivityPooling(nn.Module):
    """
    创新3: 基于轨道的反应性池化 - 增强版
    """
    def __init__(self, hidden_dim):
        super().__init__()

        # 多尺度反应性预测
        self.local_reactivity = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 1)
        )

        self.global_reactivity = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 反应性聚合
        self.reactivity_agg = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, node_features, orbital_features, batch):
        combined = torch.cat([node_features, orbital_features], dim=-1)

        # 多尺度反应性
        local_react = self.local_reactivity(combined)
        global_react = self.global_reactivity(combined)

        # 融合反应性分数
        react_scores = self.reactivity_agg(torch.cat([local_react, global_react], dim=-1))

        # 加权池化
        weighted_features = node_features * react_scores

        # 多种聚合方式
        sum_pool = global_add_pool(weighted_features, batch)
        max_pool = global_max_pool(weighted_features, batch)
        mean_pool = global_mean_pool(weighted_features, batch)

        # 组合不同的池化结果
        combined_pool = torch.cat([sum_pool, max_pool, mean_pool], dim=-1)

        # 存储池化权重供提取
        self._pool_weights = react_scores

        return combined_pool, react_scores

class HybridGNNBlock(nn.Module):
    """高效的混合GNN块"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()

        # 使用不同的GNN捕获不同的图结构
        # 修复: 确保heads能整除hidden_dim
        num_heads = 8
        head_dim = hidden_dim // num_heads
        actual_hidden = head_dim * num_heads

        self.gat = GATv2Conv(hidden_dim, head_dim, heads=num_heads, dropout=dropout, concat=True)
        self.gcn = GCNConv(hidden_dim, hidden_dim)
        self.sage = SAGEConv(hidden_dim, hidden_dim)

        # 高效融合
        self.fusion = nn.Sequential(
            nn.Linear(actual_hidden + hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 残差和归一化
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index):
        # 并行计算
        x_gat = self.gat(x, edge_index)
        x_gcn = self.gcn(x, edge_index)
        x_sage = self.sage(x, edge_index)

        # 融合
        out = self.fusion(torch.cat([x_gat, x_gcn, x_sage], dim=-1))

        # 残差连接
        return self.norm(out + x)

class EnhancedOG_PGAT(nn.Module):
    """
    完整优化的OG-PGAT模型
    保留所有创新点，优化效率，目标达到0.93
    """

    def __init__(self, in_node_dim=78, in_edge_dim=12, hidden_dim=64, 
                 out_1=32, out_2=1, gride_size=16, num_layers=15, dropout=0.1):
        super().__init__()

        # 优化的参数设置 - 确保能被heads整除
        self.node_features = in_node_dim
        self.edge_features = in_edge_dim
        # 修复: 使用能被常见heads数整除的hidden_dim
        self.hidden_dim = hidden_dim * 8  # 512维，能被2,4,8,16整除
        self.out_1 = out_1
        self.out_2 = out_2
        self.num_layers = num_layers
        self.dropout = dropout

        # 高效的输入处理
        self.input_norm = nn.BatchNorm1d(self.node_features)
        self.input_projection = nn.Sequential(
            nn.Linear(self.node_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # 创新2：优化的HOMO-LUMO编码器
        self.homo_lumo_encoder = OptimizedHOMOLUMOEncoder(self.node_features, self.hidden_dim)

        # 边特征编码（如果有）
        if self.edge_features > 0:
            self.edge_embedding = nn.Sequential(
                nn.Linear(self.edge_features, 32),
                nn.GELU(),
                nn.Linear(32, 12)
            )

        # 创新1&4：轨道引导的渐进式注意力层（前40%）
        num_orbital_layers = max(2, int(self.num_layers * 0.4))
        self.orbital_guided_layers = nn.ModuleList([
            EfficientOrbitalAttention(
                self.hidden_dim, self.hidden_dim,
                heads=min(8, 4 + i),  # 限制最大heads数为8，确保能整除
                dropout=dropout * (1 + i * 0.01)
            ) for i in range(num_orbital_layers)
        ])

        # 混合层（中30%）
        num_mixed_layers = max(2, int(self.num_layers * 0.3))
        self.mixed_layers = nn.ModuleList([
            HybridGNNBlock(self.hidden_dim, dropout)
            for _ in range(num_mixed_layers)
        ])

        # 数据驱动层（后30%）
        num_data_layers = self.num_layers - num_orbital_layers - num_mixed_layers
        self.data_driven_layers = nn.ModuleList([
            GATv2Conv(
                self.hidden_dim, self.hidden_dim // 8,
                heads=8, dropout=dropout, concat=True
            ) if i < num_data_layers - 1 else
            GATv2Conv(
                self.hidden_dim, self.hidden_dim,
                heads=1, dropout=dropout, concat=False
            )
            for i in range(num_data_layers)
        ])

        # Virtual Node（全局信息）
        self.virtual_node = nn.Parameter(torch.zeros(1, self.hidden_dim))
        nn.init.xavier_uniform_(self.virtual_node)

        self.virtual_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        )

        # 优化的Jump Knowledge
        self.jump_weights = nn.Parameter(torch.ones(self.num_layers + 1))
        self.jump_norm = nn.LayerNorm(self.hidden_dim)

        # 创新3：增强的反应性池化
        self.reactivity_pooling = EnhancedReactivityPooling(self.hidden_dim)

        # Set2Set池化
        self.set2set = Set2Set(self.hidden_dim, processing_steps=6, num_layers=2)

        # 高效的注意力池化
        self.attention_pool = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Softmax(dim=0)
        )

        # 计算池化后的维度
        pool_dim = self.hidden_dim * 2  # set2set
        pool_dim += self.hidden_dim * 3  # reactivity pooling
        pool_dim += self.hidden_dim * 2  # max + mean

        # 优化的分类器
        self.classifier = nn.Sequential(
            # 第一阶段
            nn.Linear(pool_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout * 0.9),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),

            nn.Linear(512, self.out_1),
            nn.LayerNorm(self.out_1),
            nn.GELU(),

            # 第二阶段
            nn.Linear(self.out_1, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(64, self.out_2)
        )

        # 辅助分类器（深监督）
        self.aux_classifier = nn.Linear(self.hidden_dim, self.out_2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_orbital_overlap_features(self, x, edge_index):
        """高效计算轨道重叠特征"""
        num_edges = edge_index.size(1)
        device = x.device

        # 基于节点特征计算轨道重叠（更真实的模拟）
        row, col = edge_index

        # 使用节点特征差异来估计轨道重叠
        node_diff = (x[row] - x[col]).abs().mean(dim=-1)

        # σ键：短距离强相互作用
        sigma = torch.exp(-node_diff * 0.5).clamp(0.5, 1.0)

        # π键：中等距离
        pi = torch.exp(-node_diff * 1.5).clamp(0.1, 0.5)

        # 非键：长距离弱相互作用
        nonbonding = torch.exp(-node_diff * 3.0).clamp(0.0, 0.2)

        return {
            'sigma': sigma,
            'pi': pi,
            'nonbonding': nonbonding
        }

    def forward(self, data, return_intermediate=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)

        # 输入处理
        x = self.input_norm(x)
        x = self.input_projection(x)

        # HOMO-LUMO轨道特征
        orbital_features = self.homo_lumo_encoder(data.x)
        
        # 分别计算HOMO和LUMO特征用于返回
        shared_features = self.homo_lumo_encoder.shared_encoder(data.x)
        homo_features = self.homo_lumo_encoder.homo_head(shared_features)
        lumo_features = self.homo_lumo_encoder.lumo_head(shared_features)
        
        x = x + 0.15 * orbital_features  # 轻微融合

        # 边特征编码
        if edge_attr is not None and hasattr(self, 'edge_embedding'):
            edge_attr = self.edge_embedding(edge_attr)

        # 计算轨道重叠
        orbital_overlap = self.compute_orbital_overlap_features(x, edge_index)

        # Virtual node初始化
        batch_size = batch.max().item() + 1
        virtual_node = self.virtual_node.expand(batch_size, -1)

        # 存储Jump Knowledge
        layer_outputs = [x]

        # 存储注意力权重
        att_sigma_list = []
        att_pi_list = []
        att_nb_list = []

        # 阶段1：轨道引导的渐进式注意力
        for i, layer in enumerate(self.orbital_guided_layers):
            # Virtual node更新
            graph_repr = global_mean_pool(x, batch)
            gate = self.virtual_gate(torch.cat([virtual_node, graph_repr], dim=-1))
            virtual_node = gate * graph_repr + (1 - gate) * virtual_node

            # 添加全局信息
            x = x + 0.1 * virtual_node[batch]

            # 轨道引导的注意力
            x = layer(x, edge_index, edge_attr, orbital_overlap, i + 1, self.num_layers, return_intermediate)
            layer_outputs.append(x)

            # 收集注意力权重
            if return_intermediate and hasattr(layer, '_alpha_cache'):
                att_sigma_list.append(layer._alpha_cache.get('sigma', None))
                att_pi_list.append(layer._alpha_cache.get('pi', None))
                att_nb_list.append(layer._alpha_cache.get('nb', None))

        # 阶段2：混合层
        for layer in self.mixed_layers:
            x = layer(x, edge_index)
            layer_outputs.append(x)

        # 阶段3：数据驱动层
        for layer in self.data_driven_layers:
            x = F.elu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout * 0.5, training=self.training)
            layer_outputs.append(x)

        # Jump Knowledge聚合（高效版）
        jump_weights = F.softmax(self.jump_weights[:len(layer_outputs)], dim=0)
        x = sum(w * out for w, out in zip(jump_weights, layer_outputs))
        x = self.jump_norm(x)

        # 多尺度池化
        # 1. Set2Set
        s2s_pool = self.set2set(x, batch)

        # 2. 创新3：反应性池化
        react_pool, react_scores = self.reactivity_pooling(x, orbital_features, batch)

        # 3. 标准池化
        max_pool = global_max_pool(x, batch)
        mean_pool = global_mean_pool(x, batch)

        # 组合所有池化
        pooled = torch.cat([s2s_pool, react_pool, max_pool, mean_pool], dim=-1)

        # 分类
        output = self.classifier(pooled)

        if return_intermediate:
            # 返回中间结果和注意力权重
            result = {
                'prediction': output,
                'att_sigma': att_sigma_list,
                'att_pi': att_pi_list,
                'att_nb': att_nb_list,
                'homo_feats': homo_features,
                'lumo_feats': lumo_features,
                'pool_weights': self.reactivity_pooling._pool_weights if hasattr(self.reactivity_pooling, '_pool_weights') else None,
                'layer_embeddings': layer_outputs  # 返回所有层的embeddings
            }
            
            if self.training:
                # 辅助损失
                aux_output = self.aux_classifier(global_mean_pool(x, batch))
                result['auxiliary'] = aux_output
                result['reactivity_scores'] = react_scores
            
            return result

        if self.training:
            # 辅助损失
            aux_output = self.aux_classifier(global_mean_pool(x, batch))
            return {
                'prediction': output,
                'auxiliary': aux_output,
                'reactivity_scores': react_scores
            }

        return output