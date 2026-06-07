import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
from transformers import RobertaModel, RobertaPreTrainedModel
from utils import hungarian_matcher, get_best_span, get_best_span_simple


# ---------------------------------------------------------------------------
# Dependency relation → edge weight mapping
# ---------------------------------------------------------------------------
DEPREL_WEIGHTS = {
    # Core arguments  ── high importance
    "nsubj":    1.0,  "nsubjpass": 1.0,
    "dobj":     1.0,  "iobj":      0.9,
    "csubj":    0.9,  "ccomp":     0.8,
    "xcomp":    0.8,
    # Predicates / root
    "root":     1.0,  "aux":       0.7,  "auxpass": 0.7,
    # Modifiers  ── medium importance
    "nmod":     0.5,  "amod":      0.4,  "advmod":  0.4,
    "nummod":   0.4,  "appos":     0.5,
    # Complements / adjuncts  ── low importance
    "advcl":    0.3,  "acl":       0.3,  "relcl":   0.3,
    "prep":     0.3,  "pobj":      0.4,
    # Others (default)
    "_default": 0.2,
}


def deprel_to_weight(rel: str) -> float:
    return DEPREL_WEIGHTS.get(rel.lower(), DEPREL_WEIGHTS["_default"])


# ---------------------------------------------------------------------------
# adj_w (N×N dense) → sparse edge list
# ---------------------------------------------------------------------------
def dense_adj_to_edge_list(adj_w: torch.Tensor):
    edge_idx = adj_w.nonzero(as_tuple=False)
    if edge_idx.numel() == 0:
        N   = adj_w.size(0)
        idx = torch.arange(N, device=adj_w.device)
        return idx, idx, torch.ones(N, device=adj_w.device)
    src     = edge_idx[:, 0]
    dst     = edge_idx[:, 1]
    weights = adj_w[src, dst]
    return src, dst, weights


# ---------------------------------------------------------------------------
# Edge-Weighted GAT Layer  ── sparse O(E·d)
# ---------------------------------------------------------------------------
class EdgeWeightedGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 dropout: float = 0.1, negative_slope: float = 0.2):
        super().__init__()
        self.out_dim = out_dim
        self.W_q     = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k     = nn.Linear(in_dim, out_dim, bias=False)
        self.attn    = nn.Linear(2 * out_dim, 1, bias=False)
        self.lrelu   = nn.LeakyReLU(negative_slope)
        self.drop    = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_w: torch.Tensor) -> torch.Tensor:
        N = h.size(0)
        q = self.W_q(h)
        k = self.W_k(h)

        src, dst, w = dense_adj_to_edge_list(adj_w)
        E = src.size(0)

        q_dst = q[dst]
        k_src = k[src]
        e     = self.lrelu(
            self.attn(torch.cat([q_dst, k_src], dim=-1)).squeeze(-1)
        )
        e = e * w

        # scatter softmax（数值稳定）
        e_max = torch.full((N,), float('-inf'), device=h.device)
        e_max.scatter_reduce_(0, dst, e, reduce='amax', include_self=True)
        e_exp = torch.exp(e - e_max[dst])
        e_sum = torch.zeros(N, device=h.device)
        e_sum.scatter_add_(0, dst, e_exp)
        e_sum = e_sum.clamp(min=1e-9)

        alpha = self.drop(e_exp / e_sum[dst])

        out = torch.zeros(N, self.out_dim, device=h.device)
        out.scatter_add_(
            0,
            dst.unsqueeze(1).expand(E, self.out_dim),
            alpha.unsqueeze(1) * k_src,
        )
        return out


# ---------------------------------------------------------------------------
# Multi-layer GAT stream
# ---------------------------------------------------------------------------
class GATStream(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EdgeWeightedGATLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_w: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + self.drop(layer(h, adj_w)))
        return h


# ---------------------------------------------------------------------------
# Dual-Stream GAT
# ---------------------------------------------------------------------------
class DualStreamGAT(nn.Module):
    """
    Stream-A → 图1（句法依存图 + 共指超级节点合并）
    Stream-B → 意群内触发词图
    """
    def __init__(self, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.stream_a = GATStream(hidden_dim, num_layers, dropout)
        self.stream_b = GATStream(hidden_dim, num_layers, dropout)

        self.gate_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.norm      = nn.LayerNorm(hidden_dim)

    def forward(self,
                h:       torch.Tensor,
                adj_dep: torch.Tensor,
                adj_trg: torch.Tensor) -> torch.Tensor:
        out_a = self.stream_a(h, adj_dep)
        out_b = self.stream_b(h, adj_trg)

        gate  = torch.sigmoid(self.gate_proj(torch.cat([out_a, out_b], dim=-1)))
        fused = gate * out_a + (1 - gate) * out_b
        return self.norm(h + self.out_proj(fused))


# ---------------------------------------------------------------------------
# 图1构建：句法依存图 + 共指超级节点合并
# ---------------------------------------------------------------------------
def build_dependency_graph(
        seq_len:           int,
        trigger_positions: list,
        dep_heads:         list,
        dep_rels:          list,
        device:            torch.device,
        coref_clusters:    list = None,
        coref_logits:      list = None,
    ) -> torch.Tensor:
        def logit_to_weight(logit, min_w=0.1, max_w=0.9):
            s = 1.0 / (1.0 + math.exp(-logit / 5.0))
            return min_w + (max_w - min_w) * s

        adj = torch.zeros(seq_len, seq_len, device=device)

        # ── 1. 句法依存边（保留原始结构，不做共指重定向）────────────────
        for tok_idx, (head, rel) in enumerate(zip(dep_heads, dep_rels)):
            if tok_idx >= seq_len or head >= seq_len:
                continue
            w = deprel_to_weight(rel)
            u, v = tok_idx, head
            if u == v:
                continue
            adj[u, v] = max(adj[u, v].item(), w)
            adj[v, u] = max(adj[v, u].item(), w)

        # ── 2. 共指边（置信度加权）───────────────────────────────────────
        if coref_clusters:
            for k, cluster in enumerate(coref_clusters):
                if coref_logits and k < len(coref_logits):
                    coref_w = logit_to_weight(coref_logits[k])
                else:
                    coref_w = 0.5
                for i in range(len(cluster)):
                    for j in range(i + 1, len(cluster)):
                        u, v = cluster[i], cluster[j]
                        if u >= seq_len or v >= seq_len:
                            continue
                        adj[u, v] = max(adj[u, v].item(), coref_w)
                        adj[v, u] = max(adj[v, u].item(), coref_w)

        # ── 3. 触发词邻域增强 ×1.2 ───────────────────────────────────────
        trigger_positions_valid = [t for t in trigger_positions if t < seq_len]
        for t in trigger_positions_valid:
            adj[t, :] = (adj[t, :] * 1.2).clamp(max=1.0)
            adj[:, t] = (adj[:, t] * 1.2).clamp(max=1.0)

        # ── 4. 自环 ───────────────────────────────────────────────────────
        idx = torch.arange(seq_len, device=device)
        adj[idx, idx] = 1.0

        return adj


# ---------------------------------------------------------------------------
# 意群内触发词图（原逻辑不变）
# ---------------------------------------------------------------------------
def build_trigger_graph(
    trigger_positions_list: list,
    event_groups:           list,
    seq_len:                int,
    device:                 torch.device,
) -> torch.Tensor:
    adj = torch.zeros(seq_len, seq_len, device=device)

    group_map = defaultdict(list)
    for positions, gid in zip(trigger_positions_list, event_groups):
        for p in positions:
            if p < seq_len:
                group_map[gid].append(p)

    for gid, positions in group_map.items():
        for pi in positions:
            for pj in positions:
                if pi != pj:
                    w = 1.0 / (1.0 + abs(pi - pj))
                    adj[pi, pj] = max(adj[pi, pj].item(), w)
            adj[pi, pi] = 1.0

    return adj


# ---------------------------------------------------------------------------
# DyGMA 主模型
# ---------------------------------------------------------------------------
class DyGMA(RobertaPreTrainedModel):
    """
    DyGMA + Dual-Stream GAT（含共指超级节点合并）。

    forward() 新增参数：
        dep_heads_batch     : list[list[int]]
        dep_rels_batch      : list[list[str]]
        event_groups_batch  : list[list[int]]
        coref_clusters_batch: list[list[list[int]]]  ← 新增
            每个样本的共指簇列表，格式：
            [ [sw_rep_0, sw_rep_1, ...],   # 簇0：各 mention 的代表 subword
              [sw_rep_0, sw_rep_1, ...],   # 簇1
              ... ]
            簇内第一个元素为超级节点（代表词），其余元素被重定向到它。
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.decode_layer_start = config.encoder_layers

        self.roberta = RobertaModel(config)

        self.w_prompt_start = nn.Parameter(torch.rand(config.hidden_size))
        self.w_prompt_end   = nn.Parameter(torch.rand(config.hidden_size))

        gat_layers  = getattr(config, 'gat_num_layers', 2)
        gat_dropout = getattr(config, 'gat_dropout',    0.1)
        self.ds_gat = DualStreamGAT(config.hidden_size, gat_layers, gat_dropout)

        self.memory_item_proj = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
        )
        self.memory_item_proj_with_dep = nn.Sequential(
            nn.Linear(3 * config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
        )

        self.memory_q_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.memory_k_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.memory_v_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.memory_out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.memory_gate = nn.Sequential(
            nn.Linear(4 * config.hidden_size, config.hidden_size),
            nn.Sigmoid(),
        )
        self.memory_query_norm = nn.LayerNorm(config.hidden_size)

        self.loss_fct = nn.CrossEntropyLoss(reduction='sum')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_trigger_positions(self, event_trigger: list) -> list:
        """
        支持多种触发词格式：
          - int
          - [start, end] 或 [start, end, "text"]  → range(start, end)
          - dict with 'tok_pos'
        """
        positions = []
        for t in event_trigger:
            if isinstance(t, int):
                positions.append(t)
            elif isinstance(t, (list, tuple)):
                int_vals = [v for v in t if isinstance(v, int)]
                if len(int_vals) >= 2:
                    positions.extend(range(int_vals[0], int_vals[1]))
                elif len(int_vals) == 1:
                    positions.append(int_vals[0])
            elif isinstance(t, dict):
                pos = t.get('tok_pos', t.get('position', None))
                if pos is not None:
                    if isinstance(pos, int):
                        positions.append(pos)
                    else:
                        positions.extend(pos)
        return positions

    def _enhance_query(
        self,
        prompt_query_sub:  torch.Tensor,
        gat_output:        torch.Tensor,
        trigger_positions: list,
        seq_len:           int,
    ) -> torch.Tensor:
        """
        Fuse GAT trigger topology into the role query with orthogonal projection.

        The base prompt query keeps its pretrained semantic direction, while the
        orthogonal component of the trigger representation contributes graph
        topology as a residual bias.
        """
        if not trigger_positions:
            return prompt_query_sub
        valid = [p for p in trigger_positions if 0 <= p < seq_len]
        if not valid:
            return prompt_query_sub

        trigger_repr = gat_output[valid].mean(0, keepdim=True)

        q_norm_sq = (
            torch.sum(prompt_query_sub * prompt_query_sub, dim=-1, keepdim=True)
            + 1e-8
        )
        projection_scalar = (
            torch.sum(trigger_repr * prompt_query_sub, dim=-1, keepdim=True)
            / q_norm_sq
        )
        trigger_orthogonal = trigger_repr - projection_scalar * prompt_query_sub

        fuse_weight = getattr(self.config, 'gat_query_fuse_weight', 0.2)
        return prompt_query_sub + fuse_weight * trigger_orthogonal

    def _dep_role_extractability(
        self,
        role_query_mean: torch.Tensor,
        trig_pos: list,
        context_output: torch.Tensor,
        dep_heads: list,
        seq_len: int,
        prompt_token_positions=None,
    ) -> float:
        """
        Static BFS score: shorter dependency paths from trigger to role-related
        tokens indicate better extractability. No learnable parameters involved.
        """
        if not dep_heads or not trig_pos or seq_len <= 0:
            return 0.0

        if prompt_token_positions is None:
            prompt_token_positions = set()

        with torch.no_grad():
            sims = F.cosine_similarity(
                role_query_mean.expand(seq_len, -1),
                context_output,
                dim=-1,
            )
            for p in prompt_token_positions:
                if 0 <= p < seq_len:
                    sims[p] = -1.0
            for p in trig_pos:
                if 0 <= p < seq_len:
                    sims[p] = -1.0
            topk_pos = set(sims.topk(min(3, seq_len)).indices.tolist())

        n = min(len(dep_heads), seq_len)
        adj = defaultdict(set)
        for idx, head in enumerate(dep_heads[:n]):
            try:
                head = int(head)
            except Exception:
                continue
            if 0 <= head < n and head != idx:
                adj[idx].add(head)
                adj[head].add(idx)

        min_depth = 999
        for trig in trig_pos:
            if not (0 <= trig < n):
                continue
            visited = {trig}
            queue = deque([(trig, 0)])
            while queue:
                node, depth = queue.popleft()
                if depth >= 5:
                    break
                if node in topk_pos:
                    min_depth = min(min_depth, depth)
                    break
                for nb in adj.get(node, ()):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, depth + 1))

        if min_depth == 999:
            return 0.0
        return max(0.0, 1.0 - (min_depth - 1) / 4.0)

    def _memory_coref_score(
        self,
        event_memory: list,
        coref_clusters: list,
        trig_pos: list,
        seq_len: int,
    ) -> float:
        """
        Optional planner signal for future memory-enabled variants. It is kept
        side-effect free so current version4 behavior is unchanged unless a
        memory loop is introduced.
        """
        if not event_memory or not coref_clusters:
            return 0.0

        memory_spans = set()
        for item in event_memory:
            span_pos = item.get('span_pos')
            if span_pos:
                s, e = span_pos
                for p in range(int(s), min(int(e) + 1, seq_len)):
                    if 0 <= p < seq_len:
                        memory_spans.add(p)

        if not memory_spans:
            return 0.0

        trig_nearby = set()
        for p in trig_pos:
            for offset in range(-3, 4):
                pos = p + offset
                if 0 <= pos < seq_len:
                    trig_nearby.add(pos)

        score = 0.0
        for cluster in coref_clusters:
            cluster_set = set()
            for x in cluster:
                try:
                    x = int(x)
                except Exception:
                    continue
                if 0 <= x < seq_len:
                    cluster_set.add(x)
            if (cluster_set & memory_spans) and (cluster_set & trig_nearby):
                score += 0.5
        return min(score, 1.0)

    def _memory_relation_score(
        self,
        role_query_mean: torch.Tensor,
        event_memory: list,
    ) -> torch.Tensor:
        """
        Soft relevance between current role query and memory bank.
        Returns a tensor to keep the existing call pattern (detach().item()).
        """
        if not event_memory:
            return role_query_mean.new_tensor(0.0)

        sims = []
        for item in event_memory:
            mem_q = item.get('role_query')
            if mem_q is None:
                continue
            if mem_q.dim() == 1:
                mem_q = mem_q.unsqueeze(0)
            sims.append(F.cosine_similarity(role_query_mean, mem_q, dim=-1).mean())
        if not sims:
            return role_query_mean.new_tensor(0.0)
        return torch.stack(sims).max()

    def _role_difficulty(self, role_name: str) -> float:
        role_name = str(role_name).lower()
        hard_keywords = ('place', 'destination', 'origin', 'instrument', 'artifact')
        medium_keywords = ('time', 'victim', 'target', 'participant')
        if any(k in role_name for k in hard_keywords):
            return 1.0
        if any(k in role_name for k in medium_keywords):
            return 0.5
        return 0.0

    def _pool_span_repr(
        self,
        token_repr: torch.Tensor,
        span,
        seq_len: int,
    ) -> torch.Tensor:
        s, e = int(span[0]), int(span[1])
        s = max(0, min(s, seq_len - 1))
        e = max(s, min(e, seq_len - 1))
        return token_repr[s:e + 1].mean(0, keepdim=True)

    def _dep_path_mean_repr(
        self,
        span,
        trig_pos: list,
        context_output: torch.Tensor,
        dep_heads: list,
        seq_len: int,
    ):
        if not dep_heads or not trig_pos or seq_len <= 0:
            return None

        s, _ = int(span[0]), int(span[1])
        span_tok = max(0, min(s, seq_len - 1))
        trig_tok = trig_pos[0] if trig_pos else None
        n = min(len(dep_heads), seq_len)
        if trig_tok is None or not (0 <= trig_tok < n):
            return None

        adj = defaultdict(set)
        for idx, head in enumerate(dep_heads[:n]):
            try:
                head = int(head)
            except Exception:
                continue
            if 0 <= head < n and head != idx:
                adj[idx].add(head)
                adj[head].add(idx)

        parent = {span_tok: None}
        queue = deque([span_tok])
        found = span_tok == trig_tok
        while queue and not found:
            node = queue.popleft()
            for nb in adj.get(node, ()):
                if nb not in parent:
                    parent[nb] = node
                    if nb == trig_tok:
                        found = True
                        break
                    queue.append(nb)

        if not found:
            return self._pool_span_repr(context_output, span, seq_len)

        path_nodes = []
        cur = trig_tok
        while cur is not None:
            path_nodes.append(cur)
            cur = parent.get(cur)

        valid = [p for p in path_nodes if 0 <= p < seq_len]
        if not valid:
            return self._pool_span_repr(context_output, span, seq_len)

        idx_tensor = torch.tensor(
            valid, dtype=torch.long, device=context_output.device
        )
        return context_output[idx_tensor].mean(0, keepdim=True)

    def _make_memory_item(
        self,
        role_query: torch.Tensor,
        token_repr: torch.Tensor,
        span,
        seq_len: int,
        dep_path_repr=None,
    ) -> torch.Tensor:
        span_repr = self._pool_span_repr(token_repr, span, seq_len)
        if dep_path_repr is not None and getattr(self.config, 'use_dep_memory', True):
            return self.memory_item_proj_with_dep(
                torch.cat([role_query, span_repr, dep_path_repr], dim=-1)
            )
        return self.memory_item_proj(torch.cat([role_query, span_repr], dim=-1))

    def _enhance_query_with_memory(
        self,
        prompt_query_sub: torch.Tensor,
        event_memory: list,
    ) -> torch.Tensor:
        """
        Attend to historical role memory and inject only the component that is
        orthogonal to the current query, preserving the role query direction.
        """
        if (
            not event_memory
            or not getattr(self.config, 'use_memory_query_enhancement', True)
        ):
            return prompt_query_sub

        mem_reprs = []
        mem_confs = []
        for item in event_memory:
            mem_repr = item.get('repr', None)
            if mem_repr is None:
                continue
            if mem_repr.dim() == 1:
                mem_repr = mem_repr.unsqueeze(0)
            mem_reprs.append(mem_repr.to(prompt_query_sub.device))

            conf = item.get('conf', None)
            if conf is None:
                conf = prompt_query_sub.new_tensor(1.0)
            elif not torch.is_tensor(conf):
                conf = prompt_query_sub.new_tensor(float(conf))
            else:
                conf = conf.to(prompt_query_sub.device)
            mem_confs.append(conf.view(1))

        if not mem_reprs:
            return prompt_query_sub

        memory_bank = torch.cat(mem_reprs, dim=0)
        memory_confs = torch.cat(mem_confs, dim=0)

        q = self.memory_q_proj(prompt_query_sub)
        k = self.memory_k_proj(memory_bank)
        v = self.memory_v_proj(memory_bank)

        scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(q.size(-1))
        scores = scores + torch.log(memory_confs.clamp(min=1e-6)).unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        mem_ctx = torch.matmul(attn, v)

        q_norm_sq = (
            torch.sum(prompt_query_sub * prompt_query_sub, dim=-1, keepdim=True)
            + 1e-8
        )
        projection_scalar = (
            torch.sum(mem_ctx * prompt_query_sub, dim=-1, keepdim=True)
            / q_norm_sq
        )
        mem_orthogonal = mem_ctx - projection_scalar * prompt_query_sub

        fusion_input = torch.cat(
            [
                prompt_query_sub,
                mem_orthogonal,
                prompt_query_sub * mem_orthogonal,
                torch.abs(prompt_query_sub - mem_orthogonal),
            ],
            dim=-1,
        )
        gate = self.memory_gate(fusion_input)
        mem_delta = gate * self.memory_out_proj(mem_orthogonal)

        fuse_weight = getattr(self.config, 'memory_query_fuse_weight', 0.2)
        return self.memory_query_norm(prompt_query_sub + fuse_weight * mem_delta)

    def _best_predicted_memory_span(
        self,
        s_logits_list: list,
        e_logits_list: list,
        old_tok_to_new_tok_index,
    ):
        best_span = None
        best_margin = None
        for s_l, e_l in zip(s_logits_list, e_logits_list):
            if self.config.matching_method_train == 'accurate':
                span = get_best_span(
                    s_l, e_l, old_tok_to_new_tok_index, self.config.max_span_length
                )
            else:
                span = get_best_span_simple(s_l, e_l)
            s_idx, e_idx = int(span[0]), int(span[1])

            s_sorted = torch.sort(s_l, descending=True).values
            e_sorted = torch.sort(e_l, descending=True).values
            s_gap = s_sorted[0] - s_sorted[1] if s_sorted.numel() > 1 else s_sorted[0]
            e_gap = e_sorted[0] - e_sorted[1] if e_sorted.numel() > 1 else e_sorted[0]
            margin = (s_gap + e_gap) / 2.0

            if best_margin is None or margin > best_margin:
                best_margin = margin
                best_span = (s_idx, e_idx)
        if best_span is None:
            return None, None
        return best_span, best_margin

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        all_ids=None,
        all_mask_ids=None,
        dec_prompt_ids=None,
        dec_prompt_mask_ids=None,
        arg_joint_prompts=None,
        target_info=None,
        old_tok_to_new_tok_indexs=None,
        event_triggers=None,
        dep_heads_batch=None,
        dep_rels_batch=None,
        event_groups_batch=None,
        coref_clusters_batch=None,   # ← 新增
        coref_logits_batch=None,
        enc_input_ids=None,
        enc_mask_ids=None,
        enc_attention_mask=None,
        arg_list=None,
        **kwargs,
    ):
        device = all_ids.device

        # ── 1. RoBERTa 编码上下文 ──────────────────────────────────────
        context_outputs_ = self.roberta(
            input_ids=all_ids,
            attention_mask=all_mask_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        enc_outputs     = context_outputs_.hidden_states
        decoder_context = enc_outputs[self.decode_layer_start]

        context_outputs = (enc_outputs[-1]
                           if self.config.context_representation == 'decoder'
                           else decoder_context)

        # ── 2. RoBERTa 编码 prompt ─────────────────────────────────────
        decoder_prompt_outputs = self.roberta(
            input_ids=dec_prompt_ids,
            attention_mask=dec_prompt_mask_ids,
            encoder_hidden_states=decoder_context,
            encoder_attention_mask=all_mask_ids,
        ).last_hidden_state

        logit_lists = []
        total_loss  = 0.0

        # ── 3. 逐样本处理 ──────────────────────────────────────────────
        for i, (context_output, decoder_prompt_output,
                arg_joint_prompt, old_tok_to_new_tok_index,
                event_trigger) in enumerate(zip(
            context_outputs, decoder_prompt_outputs,
            arg_joint_prompts, old_tok_to_new_tok_indexs,
            event_triggers,
        )):
            seq_len = context_output.size(0)

            # 触发词位置
            trigger_pos_list = [
                self._get_trigger_positions([ev]) for ev in event_trigger
            ]
            all_trigger_pos = [p for ps in trigger_pos_list for p in ps]

            # 依存信息
            dep_heads = dep_heads_batch[i] if dep_heads_batch else []
            dep_rels  = dep_rels_batch[i]  if dep_rels_batch  else []

            # 共指簇（subword 粒度，由 processor 离线预处理）
            coref_clusters = coref_clusters_batch[i] if coref_clusters_batch else []
            coref_logits   = coref_logits_batch[i]   if coref_logits_batch   else []

            # ── 图1：句法依存图 + 共指边 ─────────────────────────────
            adj_dep = build_dependency_graph(
                seq_len=seq_len,
                trigger_positions=all_trigger_pos,
                dep_heads=dep_heads,
                dep_rels=dep_rels,
                device=device,
                coref_clusters=coref_clusters,
                coref_logits=coref_logits,
            )

            # ── 意群内触发词图（Stream-B）────────────────────────────
            event_groups = (event_groups_batch[i] if event_groups_batch
                            else [0] * len(event_trigger))
            adj_trg = build_trigger_graph(
                trigger_pos_list, event_groups, seq_len, device
            )

            # ── 双流 GAT（只调用一次）─────────────────────────────────
            gat_output = self.ds_gat(context_output, adj_dep, adj_trg)

            # ── 4. 逐事件论元抽取 ──────────────────────────────────────
            batch_loss, cnt, list_output = [], 0, []

            for ii in range(len(event_trigger)):
                trig_pos = trigger_pos_list[ii]
                output   = {}

                prompt_token_positions = set()
                role_entries = []
                for order_idx, (arg_role, slots) in enumerate(arg_joint_prompt[ii].items()):
                    role_queries = []
                    for p_start, p_end, p_start_off, p_end_off in zip(
                        slots['tok_s'], slots['tok_e'],
                        slots['tok_s_off'], slots['tok_e_off'],
                    ):
                        for p in range(int(p_start_off), int(p_end_off)):
                            if 0 <= p < seq_len:
                                prompt_token_positions.add(p)
                        if self.config.dataset == 'wikievent':
                            raw_sub = decoder_prompt_output[p_start:p_end]
                        else:
                            raw_sub = context_output[p_start_off:p_end_off]
                        if raw_sub.shape[0] == 0:
                            raw_sub = context_output[0].unsqueeze(0)
                        role_queries.append(raw_sub.mean(0, keepdim=True))
                    role_query_mean = torch.cat(role_queries, dim=0).mean(0, keepdim=True)
                    dep_score = self._dep_role_extractability(
                        role_query_mean,
                        trig_pos,
                        context_output,
                        dep_heads,
                        seq_len,
                        prompt_token_positions=prompt_token_positions,
                    )
                    diff_score = self._role_difficulty(arg_role)
                    base_priority = getattr(self.config, 'dep_order_weight', 0.15) * dep_score
                    base_priority -= getattr(self.config, 'role_planner_diff_weight', 0.25) * diff_score
                    role_entries.append({
                        'order_idx': order_idx,
                        'role': arg_role,
                        'slots': slots,
                        'role_query_mean': role_query_mean,
                        'base_priority': base_priority,
                    })

                pending = role_entries[:]
                event_memory = []
                while pending:
                    best_idx, best_priority = 0, None
                    for jj, entry in enumerate(pending):
                        rel = self._memory_relation_score(
                            entry['role_query_mean'], event_memory
                        ).detach().item()
                        coref_score = self._memory_coref_score(
                            event_memory,
                            coref_clusters,
                            trig_pos,
                            seq_len,
                        )
                        priority = (
                            entry['base_priority']
                            + getattr(self.config, 'role_planner_prompt_weight', 0.10) * rel
                            + getattr(self.config, 'role_memory_coref_weight', 0.10) * coref_score
                        )
                        if best_priority is None or priority > best_priority:
                            best_priority = priority
                            best_idx = jj

                    entry = pending.pop(best_idx)
                    arg_role = entry['role']
                    slots = entry['slots']
                    s_logits_list = []
                    e_logits_list = []
                    memory_queries = []

                    for (p_start, p_end,
                         p_start_off, p_end_off) in zip(
                        slots['tok_s'],     slots['tok_e'],
                        slots['tok_s_off'], slots['tok_e_off'],
                    ):
                        if self.config.dataset == 'wikievent':
                            raw_sub = decoder_prompt_output[p_start:p_end]
                        else:
                            raw_sub = context_output[p_start_off:p_end_off]

                        if raw_sub.shape[0] == 0:
                            raw_sub = context_output[0].unsqueeze(0)

                        prompt_query_sub = raw_sub.mean(0).unsqueeze(0)
                        memory_queries.append(prompt_query_sub)

                        prompt_query_sub = self._enhance_query(
                            prompt_query_sub, gat_output, trig_pos, seq_len
                        )
                        prompt_query_sub = self._enhance_query_with_memory(
                            prompt_query_sub, event_memory
                        )

                        start_q = (prompt_query_sub * self.w_prompt_start).unsqueeze(-1)
                        end_q   = (prompt_query_sub * self.w_prompt_end).unsqueeze(-1)

                        # GAT only enhances the query; span logits are computed
                        # on the original context for stable token matching.
                        start_logits = torch.bmm(context_output.unsqueeze(0), start_q).squeeze()
                        end_logits   = torch.bmm(context_output.unsqueeze(0), end_q).squeeze()

                        s_logits_list.append(start_logits)
                        e_logits_list.append(end_logits)

                    output[arg_role] = [s_logits_list, e_logits_list]

                    memory_span = None
                    memory_conf = None
                    if self.training:
                        target = target_info[i][ii][arg_role]
                        predicted_spans = []
                        for s_l, e_l in zip(s_logits_list, e_logits_list):
                            if self.config.matching_method_train == 'accurate':
                                predicted_spans.append(
                                    get_best_span(s_l, e_l,
                                                  old_tok_to_new_tok_index,
                                                  self.config.max_span_length)
                                )
                            else:
                                predicted_spans.append(
                                    get_best_span_simple(s_l, e_l)
                                )

                        target_spans = [
                            [s, e] for s, e in zip(
                                target["span_s"], target["span_e"]
                            )
                        ]
                        if len(target_spans) < len(predicted_spans):
                            pad = len(predicted_spans) - len(target_spans)
                            target_spans     += [[0, 0]] * pad
                            target["span_s"] += [0] * pad
                            target["span_e"] += [0] * pad

                        if self.config.bipartite:
                            idx_preds, idx_targets = hungarian_matcher(
                                predicted_spans, target_spans
                            )
                        else:
                            idx_preds   = torch.arange(
                                len(predicted_spans), dtype=torch.int64
                            )
                            idx_targets = torch.arange(
                                min(len(target_spans), len(predicted_spans)),
                                dtype=torch.int64,
                            )

                        cnt    += len(idx_preds)
                        s_loss  = self.loss_fct(
                            torch.stack(s_logits_list)[idx_preds],
                            torch.LongTensor(
                                target["span_s"]
                            ).to(device)[idx_targets],
                        )
                        e_loss  = self.loss_fct(
                            torch.stack(e_logits_list)[idx_preds],
                            torch.LongTensor(
                                target["span_e"]
                            ).to(device)[idx_targets],
                        )
                        batch_loss.append((s_loss + e_loss) / 2)

                        for s_val, e_val in zip(target.get("span_s", []), target.get("span_e", [])):
                            if not (int(s_val) == 0 and int(e_val) == 0):
                                memory_span = (int(s_val), int(e_val))
                                memory_conf = context_output.new_tensor(1.0)
                                break
                    else:
                        best_span, best_margin = self._best_predicted_memory_span(
                            s_logits_list, e_logits_list, old_tok_to_new_tok_index
                        )
                        write_th = getattr(self.config, 'role_memory_write_threshold', 0.8)
                        if (
                            best_span is not None
                            and best_margin is not None
                            and best_span != (0, 0)
                            and float(best_margin.detach().item()) >= write_th
                        ):
                            memory_span = best_span
                            memory_conf = torch.sigmoid(best_margin.detach())

                    if memory_span is not None:
                        role_query_for_memory = (
                            torch.cat(memory_queries, dim=0).mean(0, keepdim=True)
                            if memory_queries
                            else entry['role_query_mean']
                        )
                        dep_path_repr = self._dep_path_mean_repr(
                            memory_span, trig_pos, context_output, dep_heads, seq_len
                        )
                        event_memory = [m for m in event_memory if m['role'] != arg_role]
                        event_memory.append({
                            'role': arg_role,
                            'role_query': role_query_for_memory,
                            'repr': self._make_memory_item(
                                role_query_for_memory,
                                context_output,
                                memory_span,
                                seq_len,
                                dep_path_repr=dep_path_repr,
                            ),
                            'span_pos': memory_span,
                            'conf': memory_conf,
                        })

                list_output.append(output)

            logit_lists.append(list_output)
            if self.training and batch_loss:
                total_loss += torch.stack(batch_loss).sum() / cnt

        if self.training:
            return total_loss / len(context_outputs), logit_lists
        else:
            return [], logit_lists
