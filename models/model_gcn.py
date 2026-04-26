# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
from torch.nn.utils import weight_norm
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_taihu import Config

# ==========================================
# 1) Dense GCN
# ==========================================
class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, adj_hat):
        xw = self.lin(x)
        return torch.einsum("ij,bjf->bif", adj_hat, xw)

class DenseGCNStack(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseGCNLayer(in_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj_hat):
        h = x
        for gcn in self.layers:
            h = torch.relu(gcn(h, adj_hat))
            h = self.drop(h)
        return h

# ==========================================
# 2) Adaptive adjacency
# ==========================================
class AdaptiveAdjacency(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int = 16):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.emb1 = nn.Embedding(self.num_nodes, emb_dim)
        self.emb2 = nn.Embedding(self.num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.emb1.weight)
        nn.init.xavier_uniform_(self.emb2.weight)

    def forward(self) -> torch.Tensor:
        a = torch.relu(self.emb1.weight @ self.emb2.weight.t())
        a = torch.softmax(a, dim=1)
        return a

# ==========================================
# 3) Temporal Encoders (CNN / LSTM / TCN)
# ==========================================

class DilatedResidualBlock1D(nn.Module):
    """Causal dilated residual 1D conv block.

    Why this matters in this repo:
    Many models (CNN baseline / fusion CNN branch) take the *last* time-step embedding
    (h[:, :, -1]) as the summary. If we use symmetric "same" padding, the last position
    is contaminated by fixed right-side zero padding, which can push the model toward
    mean-regression (nearly-constant predictions). This block uses TCN-style *causal*
    padding + right-side chomp to keep the last embedding purely history-dependent.

    Input/Output: x is [B, C, T] and the output has the same shape.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        if causal:
            pad = (kernel_size - 1) * dilation
            self._chomp_size = int(pad)
            padding = int(pad)
        else:
            pad = (kernel_size - 1) * dilation // 2
            self._chomp_size = 0
            padding = int(pad)

        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=int(kernel_size),
            padding=padding,
            dilation=int(dilation),
        )
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=int(kernel_size),
            padding=padding,
            dilation=int(dilation),
        )

    def _chomp(self, y: torch.Tensor) -> torch.Tensor:
        if self._chomp_size > 0:
            return y[:, :, :-self._chomp_size]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T]
        y = self.conv1(x)
        y = self._chomp(y)
        y = self.act(y)
        y = self.drop(y)

        y = self.conv2(y)
        y = self._chomp(y)
        y = self.act(y)
        y = self.drop(y)

        return x + y
class TemporalCNNEncoder(nn.Module):
    """CNN encoder used inside the fusion model.
    IMPORTANT: use dilated residual blocks and take the LAST time step embedding,
    instead of global mean pooling (which often causes mean-regression).
    Input : [B, T, C_in]
    Output: [B, C_out]
    """
    def __init__(self, in_dim: int, hidden_dim: int, kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        # fixed 3-layer dilation schedule: 1,2,4 (good for 4h sampling / daily patterns)
        dilations = [1, 2, 4]
        self.blocks = nn.Sequential(*[
            DilatedResidualBlock1D(hidden_dim, kernel_size=kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,C] -> [B,C,T]
        h = self.proj(x.transpose(1, 2))
        h = self.blocks(h)
        # take the last time step embedding
        return h[:, :, -1]


class TemporalLSTMEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden_dim, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.drop(out[:, -1, :])

# ===== TCN block reused =====
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.3):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalTCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 3, kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        num_channels = [hidden_dim] * int(num_layers)
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = in_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout
            )]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.network(x.transpose(1, 2))
        return y[:, :, -1]

# ==========================================
# 4) Spatial + Multi-branch temporal + Fusion
# ==========================================
class STGCN_MultiBranchFusion(nn.Module):
    adj_static: torch.Tensor
    adaptive_adj: Optional[AdaptiveAdjacency]
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int,
        adj_hat: torch.Tensor,
        num_nodes: int,
        gcn_hidden: int = 64,
        gcn_layers: int = 2,
        fusion_hidden: int = 64,
        tcn_layers: int = 3,
        tcn_kernel: int = 3,
        cnn_kernel: int = 3,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        use_adaptive_adj: bool = True,
        adapt_emb_dim: int = 16,
        adj_static_weight: float = 1.0,
        adj_adapt_weight: float = 0.5,
        temporal_branch_mode: str = "all",
        fusion_mode: str = "gate",
        # If provided, the model predicts a *delta* and adds it back to the
        # last observed target value in the input sequence. This helps mitigate
        # mean-regression (overly smooth predictions) for strongly periodic
        # variables (e.g., dissolved oxygen with diurnal cycles).
        target_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.register_buffer("adj_static", adj_hat)

        self.use_adaptive_adj = bool(use_adaptive_adj)
        self.adj_static_weight = float(adj_static_weight)
        self.adj_adapt_weight = float(adj_adapt_weight)

        self.output_dim = int(output_dim)
        self.pred_len = int(pred_len)
        self.target_indices = list(target_indices) if target_indices is not None else None
        self.temporal_branch_mode = str(temporal_branch_mode).lower()
        self.fusion_mode = str(fusion_mode).lower()

        self.adaptive_adj: Optional[AdaptiveAdjacency] = (
            AdaptiveAdjacency(self.num_nodes, emb_dim=adapt_emb_dim)
            if self.use_adaptive_adj else None
        )

        self.spatial = DenseGCNStack(input_dim, gcn_hidden, gcn_layers, dropout)
        self.spatial_ln = nn.LayerNorm(gcn_hidden)

        self.temporal_cnn = TemporalCNNEncoder(gcn_hidden, fusion_hidden, kernel_size=cnn_kernel, dropout=dropout)
        self.temporal_lstm = TemporalLSTMEncoder(gcn_hidden, fusion_hidden, num_layers=lstm_layers, dropout=dropout)
        self.temporal_tcn = TemporalTCNEncoder(gcn_hidden, fusion_hidden, num_layers=tcn_layers, kernel_size=tcn_kernel, dropout=dropout)

        if self.fusion_mode == "gate":
            self.gate = nn.Sequential(
                nn.Linear(fusion_hidden * 3, fusion_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, 3),
            )
        elif self.fusion_mode == "concat":
            self.concat_proj = nn.Sequential(
                nn.Linear(fusion_hidden * 3, fusion_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif self.fusion_mode == "avg":
            pass
        else:
            raise ValueError(f"Unsupported FUSION_MODE: {self.fusion_mode}")

        self.out = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, output_dim * self.pred_len),
        )

    @staticmethod
    def _row_norm_with_self_loop(a: torch.Tensor) -> torch.Tensor:
        n = a.shape[0]
        a = a + torch.eye(n, device=a.device, dtype=a.dtype)
        d = a.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return a / d

    def _effective_adj(self) -> torch.Tensor:
        """Build effective adjacency matrix.

        - Use static adjacency from data (self.adj_static)
        - Optionally add learnable adaptive adjacency
        - Row-normalize with self-loop

        Implementation detail:
        Use `Tensor.new_tensor(...)` to create scalar weights on the same
        dtype/device as the adjacency tensor. This avoids Pylance/Pyright
        complaints about `torch.as_tensor(..., dtype=...)` where `dtype`
        might be inferred as an invalid union type.
        """
        a_static = self.adj_static  # [N, N] Tensor buffer

        # scalar weights as tensors
        w_s = a_static.new_tensor(self.adj_static_weight)
        adj = a_static * w_s

        if self.use_adaptive_adj:
            assert self.adaptive_adj is not None
            a_adp = self.adaptive_adj()  # [N, N] Tensor
            w_a = a_static.new_tensor(self.adj_adapt_weight)
            adj = adj + a_adp * w_a

        return self._row_norm_with_self_loop(adj)




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, Fdim = x.shape
        if N != self.num_nodes:
            raise ValueError(f"num_nodes mismatch: model={self.num_nodes}, input={N}")

        adj = self._effective_adj()

        # Vectorized spatial GCN implementation without a Python loop.
        x_bt = x.reshape(B * T, N, Fdim)
        h_bt = self.spatial(x_bt, adj)
        h_bt = self.spatial_ln(h_bt)
        hseq = h_bt.view(B, T, N, -1)

        # Build an independent temporal sequence for each node.
        hseq_bn = hseq.permute(0, 2, 1, 3).contiguous().view(B * N, T, -1)

        e_cnn = self.temporal_cnn(hseq_bn)
        e_lstm = self.temporal_lstm(hseq_bn)
        e_tcn = self.temporal_tcn(hseq_bn)

        if self.temporal_branch_mode == "cnn":
            fused = e_cnn
        elif self.temporal_branch_mode == "lstm":
            fused = e_lstm
        elif self.temporal_branch_mode == "tcn":
            fused = e_tcn
        elif self.temporal_branch_mode == "all":
            if self.fusion_mode == "gate":
                gate_in = torch.cat([e_cnn, e_lstm, e_tcn], dim=-1)
                w = torch.softmax(self.gate(gate_in), dim=-1)
                fused = w[:, 0:1] * e_cnn + w[:, 1:2] * e_lstm + w[:, 2:3] * e_tcn
            elif self.fusion_mode == "avg":
                fused = (e_cnn + e_lstm + e_tcn) / 3.0
            elif self.fusion_mode == "concat":
                fused = self.concat_proj(torch.cat([e_cnn, e_lstm, e_tcn], dim=-1))
            else:
                raise ValueError(f"Unsupported FUSION_MODE: {self.fusion_mode}")
        else:
            raise ValueError(
                f"Unsupported TEMPORAL_BRANCH_MODE: {self.temporal_branch_mode}. "
                "Expected one of {'all','cnn','lstm','tcn'}."
            )

        # --- output head ---
        # Predict delta and add back the last observed target value (optional).
        # This usually makes the model much less prone to mean-regression.
        delta = self.out(fused).view(B, N, self.pred_len, self.output_dim).permute(0, 2, 1, 3).contiguous()  # [B,P,N,D]

        if self.target_indices is not None and len(self.target_indices) == delta.shape[-1]:
            last = x[:, -1, :, self.target_indices].unsqueeze(1)  # [B,1,N,D]
            return last + delta

        return delta

# ==========================================
# build_model
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, pred_len: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = int(output_dim)
        self.pred_len = int(pred_len)
        self.fc = nn.Linear(hidden_dim, self.output_dim * self.pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        y = self.fc(self.dropout(out[:, -1, :]))
        return y.view(x.shape[0], self.pred_len, self.output_dim)

class TCNModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, pred_len: int, num_channels: list, kernel_size: int, dropout: float):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.output_dim = int(output_dim)
        self.pred_len = int(pred_len)
        self.fc = nn.Linear(num_channels[-1], self.output_dim * self.pred_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.network(x)
        z = self.fc(y[:, :, -1])
        return z.view(x.shape[0], self.pred_len, self.output_dim)


# ==========================================
# 4) Baseline models (CNN / iTransformer) + STGCN baseline
# ==========================================

class CNNModel(nn.Module):
    """Dilated 1D-CNN baseline over the time axis (research-grade).

    Key fixes vs. the previous simple CNN:
      1) Dilated residual blocks enlarge receptive field (capture daily patterns).
      2) Predict DELTA and add back the last observed target value (residual forecasting),
         which strongly reduces mean-regression (straight-line predictions).
      3) Compatible with multi-target output (D = len(TARGET_FEATURES)).

    Input : [B, T, F]
    Output: [B, PRED_LEN, D]
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        output_dim: int,
        pred_len: int,
        dropout: float,
        target_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.pred_len = int(pred_len)
        self.target_indices = target_indices

        self.proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        L = max(1, int(num_layers))
        # dilation schedule: 1,2,4,8,... (cap at 2**(L-1))
        dilations = [2 ** i for i in range(L)]
        self.blocks = nn.Sequential(*[
            DilatedResidualBlock1D(hidden_dim, kernel_size=kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])

        # delta head
        self.fc = nn.Linear(hidden_dim, self.output_dim * self.pred_len)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,F]
        # backbone conv features
        h = self.proj(x.permute(0, 2, 1))  # [B,H,T]
        h = self.blocks(h)
        h_last = h[:, :, -1]              # [B,H]
        delta = self.fc(h_last).view(x.shape[0], self.pred_len, self.output_dim)  # [B,P,D]

        # residual forecasting: y_hat = y_last + delta
        if self.target_indices is not None and len(self.target_indices) == self.output_dim:
            last = x[:, -1, self.target_indices].unsqueeze(1)  # [B,1,D]
            return last + delta

        # fallback: no residual
        return delta


class ITransformerModel(nn.Module):
    """
    iTransformer-style baseline (variable-as-token, time-as-embedding).

    Input : [B, T, F]
    We treat each variable (feature) as a token: tokens=F, and each token carries a length-T series.
    We project each token's length-T vector into d_model, then apply TransformerEncoder over tokens.

    Output: [B, PRED_LEN, D]
    """
    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        output_dim: int,
        pred_len: int,
        dropout: float,
        target_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.pred_len = int(pred_len)
        self.target_indices = target_indices

        # Project a length-T series (per variable) into d_model
        self.token_proj = nn.Linear(self.seq_len, d_model)
        self.token_ln = nn.LayerNorm(d_model)

        # Use the default TransformerEncoderLayer signature (compat across older torch)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Heads
        self.head_pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim * self.pred_len),
        )
        # If predicting per target token, use shared token head -> scalar
        self.head_token = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,F] -> tokens: [B,F,T]
        if x.shape[1] != self.seq_len:
            raise ValueError(f"iTransformer expects fixed SEQ_LEN={self.seq_len}, got T={x.shape[1]}")

        xt = x.permute(0, 2, 1)  # [B,F,T]
        tok = self.token_proj(xt)  # [B,F,d_model]
        tok = self.token_ln(tok)

        # Transformer expects [S,B,E] by default: S=tokens=F
        tok = tok.permute(1, 0, 2)  # [F,B,d_model]
        enc = self.encoder(tok).permute(1, 0, 2)  # [B,F,d_model]

        if self.target_indices is not None and len(self.target_indices) == self.output_dim:
            # Gather target tokens and predict each with shared head_token
            sel = enc[:, self.target_indices, :]  # [B,D,d_model]
            out = self.head_token(sel)  # [B,D,P]
            return out.permute(0, 2, 1).contiguous()  # [B,P,D]

        # Otherwise: mean pool over tokens
        pooled = enc.mean(dim=1)  # [B,d_model]
        out = self.head_pool(pooled)
        return out.view(x.shape[0], self.pred_len, self.output_dim)


class STGCN_GRUBaseline(nn.Module):
    """
    STGCN baseline: Spatial DenseGCN over nodes for each time step + GRU over time per node.
    Input : [B,T,N,F]
    Output: [B,N,D]
    """
    adj_static: torch.Tensor
    adaptive_adj: Optional[AdaptiveAdjacency]

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int,
        adj_hat: torch.Tensor,
        num_nodes: int,
        gcn_hidden: int,
        gcn_layers: int,
        gru_layers: int,
        dropout: float,
        use_adaptive_adj: bool = True,
        adapt_emb_dim: int = 16,
        adj_static_weight: float = 1.0,
        adj_adapt_weight: float = 0.5,
        target_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.output_dim = int(output_dim)
        self.pred_len = int(pred_len)

        self.adj_static = nn.Parameter(adj_hat.clone().detach(), requires_grad=False)
        self.use_adaptive_adj = bool(use_adaptive_adj)
        self.adj_static_weight = float(adj_static_weight)
        self.adj_adapt_weight = float(adj_adapt_weight)
        self.target_indices = target_indices

        self.adaptive_adj = AdaptiveAdjacency(self.num_nodes, adapt_emb_dim) if self.use_adaptive_adj else None

        self.spatial = DenseGCNStack(input_dim, gcn_hidden, gcn_layers, dropout)
        self.spatial_ln = nn.LayerNorm(gcn_hidden)

        self.gru = nn.GRU(
            input_size=gcn_hidden,
            hidden_size=gcn_hidden,
            num_layers=max(1, gru_layers),
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(gcn_hidden, output_dim * self.pred_len)

    @staticmethod
    def _row_norm_with_self_loop(adj: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        n = adj.shape[0]
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
        a = adj + eye
        rs = a.sum(dim=1, keepdim=True).clamp_min(eps)
        return a / rs

    def _effective_adj(self) -> torch.Tensor:
        a_static = self.adj_static
        w_s = a_static.new_tensor(self.adj_static_weight)
        adj = a_static * w_s
        if self.use_adaptive_adj:
            assert self.adaptive_adj is not None
            a_adp = self.adaptive_adj()
            w_a = a_static.new_tensor(self.adj_adapt_weight)
            adj = adj + a_adp * w_a
        return self._row_norm_with_self_loop(adj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, Fdim = x.shape
        if N != self.num_nodes:
            raise ValueError(f"num_nodes mismatch: model={self.num_nodes}, input={N}")

        adj = self._effective_adj()

        x_bt = x.reshape(B * T, N, Fdim)      # [B*T,N,F]
        h_bt = self.spatial(x_bt, adj)        # [B*T,N,H]
        h_bt = self.spatial_ln(h_bt)
        hseq = h_bt.view(B, T, N, -1)         # [B,T,N,H]

        h_bn = hseq.permute(0, 2, 1, 3).contiguous().view(B * N, T, -1)  # [B*N,T,H]
        out_bn, _ = self.gru(h_bn)            # [B*N,T,H]
        last = self.drop(out_bn[:, -1, :])    # [B*N,H]
        delta = self.out(last).view(B, N, self.pred_len, self.output_dim).permute(0, 2, 1, 3).contiguous()  # [B,P,N,D]

        if self.target_indices is not None and len(self.target_indices) == delta.shape[-1]:
            last_y = x[:, -1, :, self.target_indices].unsqueeze(1)  # [B,1,N,D]
            return last_y + delta

        return delta


class PatchTSTLite(nn.Module):
    """Lightweight PatchTST-style baseline for long-horizon time series."""
    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        output_dim: int,
        pred_len: int,
        dropout: float,
        patch_len: int = 6,
        stride: int = 3,
        target_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.pred_len = int(pred_len)
        self.patch_len = int(max(2, patch_len))
        self.stride = int(max(1, stride))
        self.target_indices = target_indices

        self.patch_proj = nn.Linear(self.patch_len, d_model)
        self.in_ln = nn.LayerNorm(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(num_layers)))
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.pred_len * self.output_dim),
        )

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,F] -> [B,F,T]
        xf = x.transpose(1, 2).contiguous()
        # unfold: [B,F,NumPatch,PatchLen]
        p = xf.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.patch_len:
            raise ValueError(f"PatchTST requires T >= patch_len ({self.patch_len}), got T={x.shape[1]}")
        p = self._extract_patches(x)  # [B,F,P,L]
        b, f, npatch, _ = p.shape
        tok = self.patch_proj(p.reshape(b * f, npatch, self.patch_len))  # [B*F,P,d]
        tok = self.in_ln(tok)
        enc = self.encoder(tok)  # [B*F,P,d]
        pooled = enc.mean(dim=1).reshape(b, f, -1).mean(dim=1)  # [B,d]
        delta = self.out(pooled).view(b, self.pred_len, self.output_dim)
        if self.target_indices is not None and len(self.target_indices) == self.output_dim:
            last = x[:, -1, self.target_indices].unsqueeze(1)
            return last + delta
        return delta


class _DiffusionConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, k: int = 2):
        super().__init__()
        self.k = int(max(1, k))
        self.lin = nn.Linear(in_dim * (1 + 2 * self.k), out_dim)

    def forward(self, x: torch.Tensor, a_f: torch.Tensor, a_b: torch.Tensor) -> torch.Tensor:
        # x: [B,N,F], a_*: [N,N]
        outs = [x]
        xf = x
        xb = x
        for _ in range(self.k):
            xf = torch.einsum("ij,bjf->bif", a_f, xf)
            xb = torch.einsum("ij,bjf->bif", a_b, xb)
            outs.append(xf)
            outs.append(xb)
        h = torch.cat(outs, dim=-1)
        return self.lin(h)


class _DCRNNCell(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, k: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hid_dim = int(hid_dim)
        self.gate = _DiffusionConv(in_dim + hid_dim, 2 * hid_dim, k=k)
        self.cand = _DiffusionConv(in_dim + hid_dim, hid_dim, k=k)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, a_f: torch.Tensor, a_b: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_t, h], dim=-1)
        z_r = torch.sigmoid(self.gate(inp, a_f, a_b))
        z, r = torch.chunk(z_r, 2, dim=-1)
        cand_inp = torch.cat([x_t, r * h], dim=-1)
        h_tilde = torch.tanh(self.cand(cand_inp, a_f, a_b))
        h_new = (1.0 - z) * h + z * h_tilde
        return self.drop(h_new)


class DCRNNModel(nn.Module):
    """DCRNN baseline (diffusion-convolution graph spatiotemporal model)."""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int,
        adj_hat: torch.Tensor,
        num_nodes: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        target_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.output_dim = int(output_dim)
        self.pred_len = int(pred_len)
        self.hidden_dim = int(hidden_dim)
        self.target_indices = target_indices
        self.register_buffer("adj", adj_hat.clone().detach())

        layers = []
        for i in range(max(1, int(num_layers))):
            in_dim_i = input_dim if i == 0 else hidden_dim
            layers.append(_DCRNNCell(in_dim_i, hidden_dim, k=2, dropout=dropout))
        self.cells = nn.ModuleList(layers)
        self.out = nn.Linear(hidden_dim, self.pred_len * self.output_dim)

    @staticmethod
    def _row_norm(a: torch.Tensor) -> torch.Tensor:
        rs = a.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return a / rs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,N,F]
        b, t, n, _ = x.shape
        if n != self.num_nodes:
            raise ValueError(f"num_nodes mismatch: model={self.num_nodes}, input={n}")

        a_f = self._row_norm(self.adj)
        a_b = self._row_norm(self.adj.t())

        hs = [x.new_zeros((b, n, self.hidden_dim)) for _ in self.cells]
        for ti in range(t):
            h_in = x[:, ti, :, :]
            for li, cell in enumerate(self.cells):
                hs[li] = cell(h_in, hs[li], a_f, a_b)
                h_in = hs[li]

        delta = self.out(hs[-1]).view(b, n, self.pred_len, self.output_dim).permute(0, 2, 1, 3).contiguous()
        if self.target_indices is not None and len(self.target_indices) == delta.shape[-1]:
            last = x[:, -1, :, self.target_indices].unsqueeze(1)
            return last + delta
        return delta


def build_model(cfg: Config, input_dim: int, output_dim: int, graph: Optional[Dict] = None) -> nn.Module:
    """
    Factory for all models used in this project.

    Baselines:
      - cnn
      - tcn
      - lstm
      - itransformer
      - patchtst
      - stgcn
      - dcrnn

    Main model:
      - stgcn_fusion

    Note: "cnntransformer" baseline has been removed per the paper design.
    """
    name = cfg.MODEL_NAME.lower()
    pred_len = int(getattr(cfg, "PRED_LEN", 1))

    # ---- Graph (GNN/ST) models ----
    if name == "stgcn_fusion":
        if graph is None or "adj_hat" not in graph or "num_nodes" not in graph:
            raise ValueError("stgcn_fusion needs graph={'adj_hat':tensor[N,N],'num_nodes':N}")

        return STGCN_MultiBranchFusion(
            input_dim=input_dim,
            output_dim=output_dim,
            pred_len=pred_len,
            adj_hat=graph["adj_hat"],
            num_nodes=int(graph["num_nodes"]),
            gcn_hidden=cfg.GCN_HIDDEN_DIM,
            gcn_layers=cfg.GCN_LAYERS,
            fusion_hidden=cfg.FUSION_HIDDEN_DIM,
            tcn_layers=cfg.NUM_LAYERS,
            tcn_kernel=cfg.TCN_KERNEL_SIZE,
            cnn_kernel=cfg.TEMP_CNN_KERNEL,
            lstm_layers=cfg.NUM_LAYERS,
            dropout=cfg.DROPOUT_RATE,
            use_adaptive_adj=cfg.USE_ADAPTIVE_ADJ,
            adapt_emb_dim=cfg.ADAPT_EMB_DIM,
            adj_static_weight=cfg.ADJ_STATIC_WEIGHT,
            adj_adapt_weight=cfg.ADJ_ADAPT_WEIGHT,
            temporal_branch_mode=getattr(cfg, "TEMPORAL_BRANCH_MODE", "all"),
            fusion_mode=getattr(cfg, "FUSION_MODE", "gate"),
            target_indices=graph.get("target_indices"),
        )

    if name == "stgcn":
        if graph is None or "adj_hat" not in graph or "num_nodes" not in graph:
            raise ValueError("stgcn needs graph={'adj_hat':tensor[N,N],'num_nodes':N}")

        return STGCN_GRUBaseline(
            input_dim=input_dim,
            output_dim=output_dim,
            pred_len=pred_len,
            adj_hat=graph["adj_hat"],
            num_nodes=int(graph["num_nodes"]),
            gcn_hidden=cfg.GCN_HIDDEN_DIM,
            gcn_layers=cfg.GCN_LAYERS,
            gru_layers=cfg.NUM_LAYERS,
            dropout=cfg.DROPOUT_RATE,
            use_adaptive_adj=cfg.USE_ADAPTIVE_ADJ,
            adapt_emb_dim=cfg.ADAPT_EMB_DIM,
            adj_static_weight=cfg.ADJ_STATIC_WEIGHT,
            adj_adapt_weight=cfg.ADJ_ADAPT_WEIGHT,
            target_indices=graph.get("target_indices"),
        )

    if name == "dcrnn":
        if graph is None or "adj_hat" not in graph or "num_nodes" not in graph:
            raise ValueError("dcrnn needs graph={'adj_hat':tensor[N,N],'num_nodes':N}")
        return DCRNNModel(
            input_dim=input_dim,
            output_dim=output_dim,
            pred_len=pred_len,
            adj_hat=graph["adj_hat"],
            num_nodes=int(graph["num_nodes"]),
            hidden_dim=int(getattr(cfg, "GCN_HIDDEN_DIM", cfg.HIDDEN_DIM)),
            num_layers=max(1, int(cfg.NUM_LAYERS)),
            dropout=float(cfg.DROPOUT_RATE),
            target_indices=graph.get("target_indices"),
        )

    # ---- Non-graph baselines ----
    if name == "lstm":
        return LSTMModel(input_dim, cfg.HIDDEN_DIM, cfg.NUM_LAYERS, output_dim, pred_len, cfg.DROPOUT_RATE)

    if name == "tcn":
        return TCNModel(input_dim, output_dim, pred_len, [cfg.HIDDEN_DIM] * cfg.NUM_LAYERS, cfg.TCN_KERNEL_SIZE, cfg.DROPOUT_RATE)

    if name == "cnn":
        return CNNModel(input_dim, cfg.HIDDEN_DIM, cfg.NUM_LAYERS, cfg.TEMP_CNN_KERNEL, output_dim, pred_len, cfg.DROPOUT_RATE, target_indices=(graph.get('target_indices') if graph else None))

    if name in {"itransformer", "i_transformer", "i-transformer"}:
        # iTransformer: do NOT hardcode nhead; use cfg.NUM_HEADS.
        d_model = int(cfg.HIDDEN_DIM)
        nhead = int(getattr(cfg, "NUM_HEADS", 4))
        if nhead <= 0:
            raise ValueError(f"NUM_HEADS must be positive, got {nhead}")
        if d_model % nhead != 0:
            raise ValueError(
                f"Invalid iTransformer config: HIDDEN_DIM={d_model} is not divisible by NUM_HEADS={nhead}"
            )
        return ITransformerModel(
            seq_len=int(cfg.SEQ_LEN),
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=max(1, int(cfg.NUM_LAYERS)),
            output_dim=output_dim,
            pred_len=pred_len,
            dropout=float(cfg.DROPOUT_RATE),
            # Readout using target tokens (variable-as-token).
            target_indices=(graph.get("target_indices") if graph else None),
        )

    if name == "patchtst":
        d_model = int(cfg.HIDDEN_DIM)
        nhead = int(getattr(cfg, "NUM_HEADS", 4))
        if d_model % nhead != 0:
            raise ValueError(
                f"Invalid PatchTST config: HIDDEN_DIM={d_model} is not divisible by NUM_HEADS={nhead}"
            )
        return PatchTSTLite(
            seq_len=int(cfg.SEQ_LEN),
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=max(1, int(cfg.NUM_LAYERS)),
            output_dim=output_dim,
            pred_len=pred_len,
            dropout=float(cfg.DROPOUT_RATE),
            patch_len=int(max(2, min(int(cfg.SEQ_LEN), 6))),
            stride=int(max(1, min(3, int(cfg.SEQ_LEN) // 2))),
            target_indices=(graph.get("target_indices") if graph else None),
        )

    raise ValueError(f"Unknown MODEL_NAME: {cfg.MODEL_NAME}")
