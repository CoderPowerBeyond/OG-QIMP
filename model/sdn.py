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
        self.head_dim = out_channels // heads
        self.actual_out_channels = self.head_dim * heads

        self.qkv_sigma = nn.Linear(in_channels, self.actual_out_channels * 3, bias=False)
        self.qkv_pi = nn.Linear(in_channels, self.actual_out_channels * 3, bias=False)
        self.qkv_nb = nn.Linear(in_channels, self.actual_out_channels * 3, bias=False)

        self.orbital_weights = nn.Parameter(torch.tensor([0.6, 0.3, 0.1]))

        self.sigma_scale = nn.Parameter(torch.ones(heads))
        self.pi_scale = nn.Parameter(torch.ones(heads))
        self.nb_scale = nn.Parameter(torch.ones(heads))

        self.edge_proj = nn.Linear(12, heads, bias=False)

        self.out_proj = nn.Linear(self.actual_out_channels * 3, out_channels)

        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.skip = nn.Linear(in_channels, out_channels, bias=False) if in_channels != out_channels else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_sigma.weight)
        nn.init.xavier_uniform_(self.qkv_pi.weight)
        nn.init.xavier_uniform_(self.qkv_nb.weight)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, edge_index, edge_attr, orbital_overlap, layer_idx, total_layers, return_intermediate=False):
        lambda_prog = np.tanh(2.0 * layer_idx / total_layers)

        residual = self.skip(x)

        N = x.size(0)

        qkv_sigma = self.qkv_sigma(x).view(N, 3, self.heads, self.head_dim)
        q_sigma, k_sigma, v_sigma = qkv_sigma.unbind(1)

        qkv_pi = self.qkv_pi(x).view(N, 3, self.heads, self.head_dim)
        q_pi, k_pi, v_pi = qkv_pi.unbind(1)

        qkv_nb = self.qkv_nb(x).view(N, 3, self.heads, self.head_dim)
        q_nb, k_nb, v_nb = qkv_nb.unbind(1)

        out_sigma = self.propagate(edge_index, q=q_sigma, k=k_sigma, v=v_sigma,
                                   edge_attr=edge_attr, orbital_bias=orbital_overlap['sigma'],
                                   lambda_prog=lambda_prog, scale=self.sigma_scale, orbital_type='sigma')

        out_pi = self.propagate(edge_index, q=q_pi, k=k_pi, v=v_pi,
                                edge_attr=edge_attr, orbital_bias=orbital_overlap['pi'],
                                lambda_prog=lambda_prog, scale=self.pi_scale, orbital_type='pi')

        out_nb = self.propagate(edge_index, q=q_nb, k=k_nb, v=v_nb,
                               edge_attr=edge_attr, orbital_bias=orbital_overlap['nonbonding'],
                               lambda_prog=lambda_prog, scale=self.nb_scale, orbital_type='nb')

        weights = F.softmax(self.orbital_weights * (1 + lambda_prog), dim=0)

        out = torch.cat([
            weights[0] * out_sigma.view(N, -1),
            weights[1] * out_pi.view(N, -1),
            weights[2] * out_nb.view(N, -1)
        ], dim=-1)

        out = self.out_proj(out)

        out = self.norm(out + residual)
        out = self.dropout(out)

        return out

    def message(self, q_i, k_j, v_j, edge_attr, orbital_bias, lambda_prog, scale, index, ptr, size_i, orbital_type=None):
        scores = (q_i * k_j).sum(dim=-1) / (self.head_dim ** 0.5)
        scores = scores * scale.unsqueeze(0)

        if edge_attr is not None:
            edge_scores = self.edge_proj(edge_attr.float())
            scores = scores + edge_scores

        if orbital_bias is not None:
            physics_weight = (1 - lambda_prog) * 2.0
            orbital_guidance = physics_weight * orbital_bias.unsqueeze(-1)
            scores = scores + orbital_guidance

        if not hasattr(self, '_scores_cache'):
            self._scores_cache = {}
        self._scores_cache[orbital_type] = scores

        alpha = softmax(scores, index, ptr, size_i)
        alpha = F.dropout(alpha, p=0.1, training=self.training)

        if not hasattr(self, '_alpha_cache'):
            self._alpha_cache = {}
        self._alpha_cache[orbital_type] = alpha

        return (alpha.unsqueeze(-1) * v_j).view(-1, self.heads * self.head_dim)


class OptimizedHOMOLUMOEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05)
        )

        self.homo_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

        self.lumo_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

        self.gap_hardness_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

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
        shared_features = self.shared_encoder(x)

        homo = self.homo_head(shared_features)
        lumo = self.lumo_head(shared_features)

        orbital_pair = torch.cat([homo, lumo], dim=-1)
        gap_hardness = self.gap_hardness_encoder(orbital_pair)

        combined = torch.cat([homo, lumo, gap_hardness], dim=-1)
        gate = self.fusion_gate(combined)
        transform = self.fusion_transform(combined)

        return gate * transform + (1 - gate) * shared_features


class EnhancedReactivityPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

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

        self.reactivity_agg = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, node_features, orbital_features, batch):
        combined = torch.cat([node_features, orbital_features], dim=-1)

        local_react = self.local_reactivity(combined)
        global_react = self.global_reactivity(combined)

        react_scores = self.reactivity_agg(torch.cat([local_react, global_react], dim=-1))

        weighted_features = node_features * react_scores

        sum_pool = global_add_pool(weighted_features, batch)
        max_pool = global_max_pool(weighted_features, batch)
        mean_pool = global_mean_pool(weighted_features, batch)

        combined_pool = torch.cat([sum_pool, max_pool, mean_pool], dim=-1)

        self._pool_weights = react_scores

        return combined_pool, react_scores


class HybridGNNBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()

        num_heads = 8
        head_dim = hidden_dim // num_heads
        actual_hidden = head_dim * num_heads

        self.gat = GATv2Conv(hidden_dim, head_dim, heads=num_heads, dropout=dropout, concat=True)
        self.gcn = GCNConv(hidden_dim, hidden_dim)
        self.sage = SAGEConv(hidden_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(actual_hidden + hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index):
        x_gat = self.gat(x, edge_index)
        x_gcn = self.gcn(x, edge_index)
        x_sage = self.sage(x, edge_index)

        out = self.fusion(torch.cat([x_gat, x_gcn, x_sage], dim=-1))

        return self.norm(out + x)


class EnhancedOG_PGAT(nn.Module):
    def __init__(self, in_node_dim=78, in_edge_dim=12, hidden_dim=64,
                 out_1=32, out_2=1, gride_size=16, num_layers=15, dropout=0.1,
                 overlap_mode='heuristic', overlap_channels='three'):
        super().__init__()

        # A/B 实验开关，默认值等于上游原行为（逐字节等价）
        self.overlap_mode = overlap_mode
        self.overlap_channels = overlap_channels

        self.node_features = in_node_dim
        self.edge_features = in_edge_dim
        self.hidden_dim = hidden_dim * 8
        self.out_1 = out_1
        self.out_2 = out_2
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_norm = nn.BatchNorm1d(self.node_features)
        self.input_projection = nn.Sequential(
            nn.Linear(self.node_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        self.homo_lumo_encoder = OptimizedHOMOLUMOEncoder(self.node_features, self.hidden_dim)

        if self.edge_features > 0:
            self.edge_embedding = nn.Sequential(
                nn.Linear(self.edge_features, 32),
                nn.GELU(),
                nn.Linear(32, 12)
            )

        num_orbital_layers = max(2, int(self.num_layers * 0.4))
        self.orbital_guided_layers = nn.ModuleList([
            EfficientOrbitalAttention(
                self.hidden_dim, self.hidden_dim,
                heads=min(8, 4 + i),
                dropout=dropout * (1 + i * 0.01)
            ) for i in range(num_orbital_layers)
        ])

        num_mixed_layers = max(2, int(self.num_layers * 0.3))
        self.mixed_layers = nn.ModuleList([
            HybridGNNBlock(self.hidden_dim, dropout)
            for _ in range(num_mixed_layers)
        ])

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

        self.virtual_node = nn.Parameter(torch.zeros(1, self.hidden_dim))
        nn.init.xavier_uniform_(self.virtual_node)

        self.virtual_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        )

        self.jump_weights = nn.Parameter(torch.ones(self.num_layers + 1))
        self.jump_norm = nn.LayerNorm(self.hidden_dim)

        self.reactivity_pooling = EnhancedReactivityPooling(self.hidden_dim)

        self.set2set = Set2Set(self.hidden_dim, processing_steps=6, num_layers=2)

        self.attention_pool = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Softmax(dim=0)
        )

        pool_dim = self.hidden_dim * 2
        pool_dim += self.hidden_dim * 3
        pool_dim += self.hidden_dim * 2

        self.classifier = nn.Sequential(
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

            nn.Linear(self.out_1, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(64, self.out_2)
        )

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

    def compute_orbital_overlap_features(self, x, edge_index, data=None, source=None):
        # source 显式指定时按它走，否则按 self.overlap_mode 推断；
        # 力矩匹配臂需要同时拿到两边的原始值，所以要能强制走启发式。
        if source is None:
            source = 'real' if (self.overlap_mode in ('real', 'real_matched')
                                and data is not None
                                and getattr(data, 'orb_sigma', None) is not None) else 'heuristic'

        # real 实验臂：图上带着 PySCF 算出的真实重叠，直接用它，不走下面的启发式
        if source == 'real' and data is not None and getattr(data, 'orb_sigma', None) is not None:
            device = x.device
            return {
                'sigma': data.orb_sigma.to(device).float(),
                'pi': data.orb_pi.to(device).float(),
                'nonbonding': data.orb_nonbonding.to(device).float(),
            }

        num_edges = edge_index.size(1)
        device = x.device

        row, col = edge_index

        node_diff = (x[row] - x[col]).abs().mean(dim=-1)

        sigma = torch.exp(-node_diff * 0.5).clamp(0.5, 1.0)

        pi = torch.exp(-node_diff * 1.5).clamp(0.1, 0.5)

        nonbonding = torch.exp(-node_diff * 3.0).clamp(0.0, 0.2)

        return {
            'sigma': sigma,
            'pi': pi,
            'nonbonding': nonbonding
        }

    def _log_overlap_stats(self, orbital_overlap, mode):
        # 两个实验臂的通道量级必须可比，否则比的就不只是重叠来源。只看第一批，只打一次。
        if getattr(self, '_overlap_logged', False):
            return
        self._overlap_logged = True
        parts = []
        n = 0
        for name in ('sigma', 'pi', 'nonbonding'):
            v = orbital_overlap[name].detach().float().flatten().cpu()
            n = max(n, v.numel())
            q = torch.quantile(v, torch.tensor([0.5, 0.9, 0.99]))
            parts.append(
                f"{name}: mean={v.mean():.4f} std={v.std():.4f} min={v.min():.4f} "
                f"max={v.max():.4f} p50={q[0]:.4f} p90={q[1]:.4f} p99={q[2]:.4f}")
        print(f"[overlap_stats] mode={mode} n={n} | " + " | ".join(parts), flush=True)

    @staticmethod
    def _match_moments(real, ref, eps=1e-6):
        """把 real 逐通道仿射映射到 ref 的均值和标准差。

        physics_weight 是裸加到 logits 上的，换来源会连偏置强度一起换掉，两臂就
        没法比。匹配前两阶矩后两臂偏置的分布相同，剩下的差别只有"哪条边拿到什么值"。
        没有方差的通道（实测真实重叠下 nonbonding 恒为 0）退化成常数，而常数加到
        logits 上被 softmax 抵消，等于该通道不携带逐边信息。
        """
        r_std = real.std()
        if r_std < eps:
            # full_like 的 fill_value 必须是 Python 数字，传张量会 TypeError
            return torch.full_like(real, ref.mean().item())
        return (real - real.mean()) * (ref.std() / r_std) + ref.mean()

    def _apply_overlap_mode(self, x, edge_index, data):
        """按 self.overlap_mode / self.overlap_channels 组装送入 attention 的三通道。"""
        mode = getattr(self, 'overlap_mode', 'heuristic')
        if mode in ('real', 'real_matched'):
            # 图上没有 orb_* 时 compute_...('real') 会静默落回启发式，
            # real_matched 就变成启发式对自己做力矩匹配（恒等），跑出一个看似正常
            # 其实全错的数。宁可直接报错。多半是 data/processed 里的缓存是别的模式建的。
            if data is None or getattr(data, 'orb_sigma', None) is None:
                raise RuntimeError(
                    f"overlap_mode={mode} 需要图上带 orb_sigma/orb_pi/orb_nonbonding，"
                    "但当前 batch 没有。缓存图可能是别的模式建的，删掉 data/processed 下的缓存重跑。")

        if mode == 'none':
            z = torch.zeros(edge_index.size(1), dtype=torch.float32, device=x.device)
            out = {'sigma': z, 'pi': z.clone(), 'nonbonding': z.clone()}
        elif mode == 'real_matched':
            real = self.compute_orbital_overlap_features(x, edge_index, data, source='real')
            ref = self.compute_orbital_overlap_features(x, edge_index, data, source='heuristic')
            out = {k: self._match_moments(real[k], ref[k])
                   for k in ('sigma', 'pi', 'nonbonding')}
        else:
            out = self.compute_orbital_overlap_features(x, edge_index, data)

        if getattr(self, 'overlap_channels', 'three') == 'sigma_pi':
            # 真实重叠下 nonbonding 恒为 0，拿它去比启发式里有信息的同通道没意义，
            # 所以两臂一起置零（该头退化为纯数据驱动），比较只在 sigma/pi 上进行
            out['nonbonding'] = torch.zeros_like(out['nonbonding'])
        return out

    def forward(self, data, return_intermediate=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)

        x = self.input_norm(x)
        x = self.input_projection(x)

        orbital_features = self.homo_lumo_encoder(data.x)

        shared_features = self.homo_lumo_encoder.shared_encoder(data.x)
        homo_features = self.homo_lumo_encoder.homo_head(shared_features)
        lumo_features = self.homo_lumo_encoder.lumo_head(shared_features)

        x = x + 0.15 * orbital_features

        if edge_attr is not None and hasattr(self, 'edge_embedding'):
            edge_attr = self.edge_embedding(edge_attr)

        orbital_overlap = self._apply_overlap_mode(x, edge_index, data)
        self._log_overlap_stats(
            orbital_overlap,
            f"{getattr(self, 'overlap_mode', 'heuristic')}"
            f"/{getattr(self, 'overlap_channels', 'three')}")

        batch_size = batch.max().item() + 1
        virtual_node = self.virtual_node.expand(batch_size, -1)

        layer_outputs = [x]

        att_sigma_list = []
        att_pi_list = []
        att_nb_list = []

        for i, layer in enumerate(self.orbital_guided_layers):
            graph_repr = global_mean_pool(x, batch)
            gate = self.virtual_gate(torch.cat([virtual_node, graph_repr], dim=-1))
            virtual_node = gate * graph_repr + (1 - gate) * virtual_node

            x = x + 0.1 * virtual_node[batch]

            x = layer(x, edge_index, edge_attr, orbital_overlap, i + 1, self.num_layers, return_intermediate)
            layer_outputs.append(x)

            if return_intermediate and hasattr(layer, '_alpha_cache'):
                att_sigma_list.append(layer._alpha_cache.get('sigma', None))
                att_pi_list.append(layer._alpha_cache.get('pi', None))
                att_nb_list.append(layer._alpha_cache.get('nb', None))

        for layer in self.mixed_layers:
            x = layer(x, edge_index)
            layer_outputs.append(x)

        for layer in self.data_driven_layers:
            x = F.elu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout * 0.5, training=self.training)
            layer_outputs.append(x)

        jump_weights = F.softmax(self.jump_weights[:len(layer_outputs)], dim=0)
        x = sum(w * out for w, out in zip(jump_weights, layer_outputs))
        x = self.jump_norm(x)

        s2s_pool = self.set2set(x, batch)

        react_pool, react_scores = self.reactivity_pooling(x, orbital_features, batch)

        max_pool = global_max_pool(x, batch)
        mean_pool = global_mean_pool(x, batch)

        pooled = torch.cat([s2s_pool, react_pool, max_pool, mean_pool], dim=-1)

        output = self.classifier(pooled)

        if return_intermediate:
            result = {
                'prediction': output,
                'att_sigma': att_sigma_list,
                'att_pi': att_pi_list,
                'att_nb': att_nb_list,
                'homo_feats': homo_features,
                'lumo_feats': lumo_features,
                'pool_weights': self.reactivity_pooling._pool_weights if hasattr(self.reactivity_pooling, '_pool_weights') else None,
                'layer_embeddings': layer_outputs
            }

            if self.training:
                aux_output = self.aux_classifier(global_mean_pool(x, batch))
                result['auxiliary'] = aux_output
                result['reactivity_scores'] = react_scores

            return result

        if self.training:
            aux_output = self.aux_classifier(global_mean_pool(x, batch))
            return {
                'prediction': output,
                'auxiliary': aux_output,
                'reactivity_scores': react_scores
            }

        return output
