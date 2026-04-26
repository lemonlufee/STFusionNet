r'''
代码用法：
脚本内置了5个消融：
full：完整模型（对照组）

w_o_adaptive_adj：去掉 adaptive adjacency，保留地理先验图

w_o_spatial_graph：移除整个空间图模块（运行时把 model.spatial 替换成“无图投影层”，不再做图卷积传播）

fusion_avg：把门控融合替换为三分支简单平均（通过把 gate 参数清零并冻结，使 softmax 恒为均匀 1/3）

w_o_cyclic：移除 hour/month 的 sin-cos 循环时间特征

w_o_delta_loss：去掉 delta regularization（把 DELTA_LOSS_WEIGHT=0）


1.默认输出：保存到项目目录下的 ablation_results/  并且跑全部消融
python ablation_runner.py --variants all
2.指定输出目录：会保存到位于同级目录的指定文件夹ExperimentOutputs下
python ablation_runner.py --variants all --results_root ./ExperimentOutputs
3.只跑指定消融：
python ablation_runner.py --variants full,w_o_adaptive_adj,fusion_avg
4.额外开启缺测鲁棒性评估（10/20/30% 遮蔽）
python ablation_runner.py --variants all --robustness --mask_rates 0.1,0.2,0.3
'''

from __future__ import annotations

import argparse
import copy
import math
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_taihu import Config
from evaluation.eval_metrics import calculate_nse  # reuse NSE
from data.data_pipeline import (
    add_time_features,
    build_graph_windows_from_df,
    choose_target_lakes,
    fit_scaler,
    impute_strict_per_lake,
    load_raw_data,
    physical_cleaning,
    split_by_time_per_lake_train_val_test,
)
from models.model_gcn import build_model
from utils.util_common import ensure_dir, now_str, save_json, set_seed, configure_stdio_for_server, collect_runtime_env


# -----------------------------
# Patch modules for ablations
# -----------------------------
class NoGraphSpatial(nn.Module):
    """
    Spatial ablation: remove graph propagation. Keep a simple MLP projection.
    Signature matches (x, adj_hat) -> Tensor.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, adj_hat: Optional[torch.Tensor] = None) -> torch.Tensor:  # noqa: ARG002
        return self.net(x)


def force_average_fusion(model: nn.Module) -> None:
    """
    Fusion ablation: gated fusion -> simple average.
    Achieved by zeroing and freezing gate params so softmax becomes uniform.
    """
    if not hasattr(model, "gate"):
        raise AttributeError("Model has no attribute `gate`; cannot force avg fusion.")
    gate = getattr(model, "gate")
    for p in gate.parameters():
        with torch.no_grad():
            p.zero_()
        p.requires_grad_(False)


# -----------------------------
# Helpers
# -----------------------------
def _clone_cfg(cfg: Config) -> Config:
    return copy.deepcopy(cfg)


def _set_torch_deterministic() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _default_input_features(cfg: Config) -> List[str]:
    return cfg.FEATURE_COLS + ["month_sin", "month_cos", "hour_sin", "hour_cos", "t_index"]


def _no_cyclic_input_features(cfg: Config) -> List[str]:
    return cfg.FEATURE_COLS + ["t_index"]


def _make_graph_dict(
    input_features: List[str],
    target_features: List[str],
    adj_hat: np.ndarray,
) -> Dict:
    target_indices = [input_features.index(t) for t in target_features if t in input_features]
    return {
        "adj_hat": torch.tensor(adj_hat, dtype=torch.float32),
        "num_nodes": int(adj_hat.shape[0]),
        "target_indices": target_indices,
    }


def _flatten_if_gnn(arr: np.ndarray) -> np.ndarray:
    """
    Flatten outputs for metric/scaler functions.

    Supported:
      - [S, P, N, D] -> [S*P*N, D]
      - [S, N, D] or [S, P, D] -> [S*?, D]
      - [M, D] -> [M, D]
    """
    if isinstance(arr, np.ndarray) and arr.ndim == 4:
        S, P, N, D = arr.shape
        return arr.reshape(S * P * N, D)
    if isinstance(arr, np.ndarray) and arr.ndim == 3:
        S, K, D = arr.shape
        return arr.reshape(S * K, D)
    return arr


def _metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true2 = _flatten_if_gnn(y_true)
    y_pred2 = _flatten_if_gnn(y_pred)
    err = y_true2 - y_pred2
    mse = float(mean_squared_error(y_true2, y_pred2))
    rmse = float(math.sqrt(mse if mse > 0.0 else 0.0))
    mae = float(np.mean(np.abs(err)))
    den = np.maximum(np.abs(y_true2), 1e-6)
    mape = float(np.mean(np.abs(err) / den) * 100.0)
    try:
        r2 = float(r2_score(y_true2, y_pred2))
    except Exception:
        r2 = float("nan")
    nse = float(calculate_nse(y_true2, y_pred2))
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "nse": nse}


def _inverse_transform_2d(scaler, y: np.ndarray) -> np.ndarray:
    """
    sklearn scaler expects 2D.
    Supports y shapes: [M,D] or [S,N,D].
    Returns 2D [M,D] in real scale.
    """
    y2 = _flatten_if_gnn(y)
    return scaler.inverse_transform(y2)


def _metrics_real(scaler_y, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_r = _inverse_transform_2d(scaler_y, y_true)
    y_pred_r = _inverse_transform_2d(scaler_y, y_pred)
    # NSE is invariant under affine scaling, but we compute anyway for clarity
    err = y_true_r - y_pred_r
    mse = float(mean_squared_error(y_true_r, y_pred_r))
    rmse = float(math.sqrt(mse if mse > 0.0 else 0.0))
    mae = float(np.mean(np.abs(err)))
    den = np.maximum(np.abs(y_true_r), 1e-6)
    mape = float(np.mean(np.abs(err) / den) * 100.0)
    try:
        r2 = float(r2_score(y_true_r, y_pred_r))
    except Exception:
        r2 = float("nan")
    nse = float(calculate_nse(y_true_r, y_pred_r))
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "nse": nse}


def _collect_model_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            out = model(bx)
            y_true_list.append(by.detach().cpu().numpy())
            y_pred_list.append(out.detach().cpu().numpy())
    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)
    return y_true, y_pred


def _collect_persistence_outputs(
    loader: DataLoader,
    target_feat_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Persistence baseline: y_hat = x_last[target_feature] (last time step).
    Supports:
      - GNN input bx: [B,T,N,F], label by: [B,N,D]
      - non-GNN input bx: [B,T,F], label by: [B,D]
    """
    y_true_list, y_pred_list = [], []

    for bx, by in loader:
        # bx, by are torch tensors
        if bx.dim() == 4:
            # [B,T,N,F] -> pick last step [B,N,F]
            last = bx[:, -1, :, :]  # [B,N,F]
            preds = []
            for fi in target_feat_indices:
                preds.append(last[:, :, fi].unsqueeze(-1))  # [B,N,1]
            y_hat = torch.cat(preds, dim=-1)  # [B,N,D]
            if by.dim() == 4:
                # match label shape [B,P,N,D]
                p = int(by.shape[1])
                y_hat = y_hat.unsqueeze(1).expand(-1, p, -1, -1).contiguous()
            elif by.dim() != 3:
                raise ValueError(f"Unsupported label shape for GNN persistence: {tuple(by.shape)}")
        elif bx.dim() == 3:
            # [B,T,F] -> [B,F]
            last = bx[:, -1, :]
            preds = []
            for fi in target_feat_indices:
                preds.append(last[:, fi].unsqueeze(-1))  # [B,1]
            y_hat = torch.cat(preds, dim=-1)  # [B,D]
            if by.dim() == 3:
                # match label shape [B,P,D]
                p = int(by.shape[1])
                y_hat = y_hat.unsqueeze(1).expand(-1, p, -1).contiguous()
            elif by.dim() != 2:
                raise ValueError(f"Unsupported label shape for baseline persistence: {tuple(by.shape)}")
        else:
            raise ValueError(f"Unsupported bx shape for persistence: {tuple(bx.shape)}")

        y_true_list.append(by.detach().cpu().numpy())
        y_pred_list.append(y_hat.detach().cpu().numpy())

    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)
    return y_true, y_pred


def _apply_random_mask(
    bx: torch.Tensor,
    feat_indices: List[int],
    mask_ratio: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    Randomly mask a fraction of bx values on selected feature indices.
    Mask value = 0.0 (in normalized space).
    bx shape:
      - GNN: [B, T, N, F]
      - non-GNN: [B, T, F]
    """
    if mask_ratio <= 0:
        return bx

    x = bx.clone()
    if x.dim() == 4:
        B, T, N, F = x.shape
        total = B * T * N * len(feat_indices)
        m = int(round(total * mask_ratio))
        if m <= 0:
            return x

        b_idx = rng.integers(0, B, size=m)
        t_idx = rng.integers(0, T, size=m)
        n_idx = rng.integers(0, N, size=m)
        f_sel = rng.integers(0, len(feat_indices), size=m)
        f_idx = np.array(feat_indices, dtype=np.int64)[f_sel]
        x[b_idx, t_idx, n_idx, f_idx] = 0.0
        return x

    if x.dim() == 3:
        B, T, F = x.shape
        total = B * T * len(feat_indices)
        m = int(round(total * mask_ratio))
        if m <= 0:
            return x

        b_idx = rng.integers(0, B, size=m)
        t_idx = rng.integers(0, T, size=m)
        f_sel = rng.integers(0, len(feat_indices), size=m)
        f_idx = np.array(feat_indices, dtype=np.int64)[f_sel]
        x[b_idx, t_idx, f_idx] = 0.0
        return x

    raise ValueError(f"Unsupported input shape for masking: {tuple(x.shape)}")


def _collect_masked_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_ratio: float,
    mask_feat_indices: List[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    rng = np.random.default_rng(seed)
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            bx_m = _apply_random_mask(bx, mask_feat_indices, mask_ratio, rng)
            out = model(bx_m)
            y_true_list.append(by.detach().cpu().numpy())
            y_pred_list.append(out.detach().cpu().numpy())
    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)
    return y_true, y_pred


# -----------------------------
# Data pipeline (strict Train/Val/Test)
# -----------------------------
def prepare_data_gnn(
    cfg: Config,
    input_features: List[str],
    run_dir: str,
) -> Tuple[
    DataLoader, DataLoader, DataLoader,
    Dict,  # graph_dict
    List[int],  # mask_feat_indices
    object,  # scaler_Y
]:
    """
    Prepare data for GNN/ST models using strict Train/Val/Test split.
    Returns scaler_Y for real-scale metrics.
    """
    df_raw = load_raw_data(cfg)
    df_selected = choose_target_lakes(df_raw, cfg)
    df_clean = physical_cleaning(df_selected, cfg)

    df_train, df_val, df_test, split_meta = split_by_time_per_lake_train_val_test(
        df_clean,
        cfg,
        train_ratio=getattr(cfg, "TRAIN_RATIO", 0.7),
        val_ratio=getattr(cfg, "VAL_RATIO", 0.1),
        overlap=min(int(getattr(cfg, "SPLIT_OVERLAP", cfg.SEQ_LEN)), int(cfg.SEQ_LEN)),
    )
    if df_train.empty or df_val.empty or df_test.empty:
        raise RuntimeError(
            f"Empty split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}. "
            "Adjust TRAIN_RATIO/VAL_RATIO/SPLIT_OVERLAP or check data length."
        )
    save_json(split_meta, os.path.join(run_dir, "splits.json"))

    # Impute per split (avoid leakage)
    df_train_imp, train_means = impute_strict_per_lake(df_train, cfg, return_train_means=True)
    train_means: Dict[str, float] = train_means
    df_val_imp   = impute_strict_per_lake(df_val, cfg, train_means=train_means)
    df_test_imp  = impute_strict_per_lake(df_test, cfg, train_means=train_means)
    save_json(train_means, os.path.join(run_dir, "train_feature_means.json"))
    df_train_fe = add_time_features(df_train_imp)
    df_val_fe = add_time_features(df_val_imp)
    df_test_fe = add_time_features(df_test_imp)

    scaler_X = fit_scaler(df_train_fe, input_features)
    scaler_Y = fit_scaler(df_train_fe, cfg.TARGET_FEATURES)

    # Apply scaling
    def _apply(df_fe):
        out = df_fe.copy()
        out[input_features] = scaler_X.transform(df_fe[input_features].values)
        out[cfg.TARGET_FEATURES] = scaler_Y.transform(df_fe[cfg.TARGET_FEATURES].values)
        return out

    df_train_norm = _apply(df_train_fe)
    df_val_norm = _apply(df_val_fe)
    df_test_norm = _apply(df_test_fe)

    # Use normalized train means for graph-window NaN fallback.
    train_fill_means_norm: Dict[str, float] = {}
    for c in list(cfg.FEATURE_COLS) + list(cfg.TARGET_FEATURES):
        if c in df_train_norm.columns:
            v = pd.to_numeric(df_train_norm[c], errors="coerce").mean(skipna=True)
            train_fill_means_norm[c] = float(v) if pd.notna(v) else 0.0

    mean_y = scaler_Y.mean_
    scale_y = scaler_Y.scale_
    assert mean_y is not None and scale_y is not None

    save_json(
        {"mean_": [float(v) for v in mean_y], "scale_": [float(v) for v in scale_y], "features": cfg.TARGET_FEATURES},
        os.path.join(run_dir, "scaler_Y.json"),
    )

    X_train, y_train, _, adj_hat, node_ids = build_graph_windows_from_df(
        df_train_norm, cfg, input_features, cfg.TARGET_FEATURES, train_fill_means=train_fill_means_norm
    )
    X_val, y_val, _, _, _ = build_graph_windows_from_df(
        df_val_norm, cfg, input_features, cfg.TARGET_FEATURES, node_ids=node_ids, adj_hat=adj_hat, train_fill_means=train_fill_means_norm)
    X_test, y_test, _, _, _ = build_graph_windows_from_df(
        df_test_norm, cfg, input_features, cfg.TARGET_FEATURES, node_ids=node_ids, adj_hat=adj_hat, train_fill_means=train_fill_means_norm)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    graph_dict = _make_graph_dict(input_features, cfg.TARGET_FEATURES, adj_hat)

    # For masking robustness, mask only real sensor features (exclude time encodings)
    mask_feat_indices = [i for i, f in enumerate(input_features) if f in cfg.FEATURE_COLS]

    save_json({"node_ids": node_ids}, os.path.join(run_dir, "graph_nodes.json"))
    return train_loader, val_loader, test_loader, graph_dict, mask_feat_indices, scaler_Y


# -----------------------------
# Training per variant
# -----------------------------
def train_one_variant_gnn(
    cfg: Config,
    variant_name: str,
    run_dir: str,
    input_features: List[str],
    graph_dict: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    scaler_Y,
    model_patches: Optional[List[str]] = None,
) -> Dict:
    """
    Train with Val selection; evaluate once on Test.
    Also compute persistence baseline and real-scale metrics.
    Returns a summary dict.
    """
    ensure_dir(run_dir)
    save_json({"variant": variant_name, "config": asdict(cfg)}, os.path.join(run_dir, "config.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()

    model = build_model(cfg, len(input_features), len(cfg.TARGET_FEATURES), graph=graph_dict).to(device)

    applied = []
    if model_patches:
        for p in model_patches:
            if p == "no_graph_spatial":
                model.spatial = NoGraphSpatial(len(input_features), cfg.GCN_HIDDEN_DIM, cfg.DROPOUT_RATE).to(device)
                applied.append(p)
            elif p == "fusion_avg":
                force_average_fusion(model)
                applied.append(p)
            else:
                raise ValueError(f"Unknown model patch: {p}")
    save_json({"applied_patches": applied}, os.path.join(run_dir, "patches.json"))

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val = float("inf")
    best_epoch = -1
    best_path = os.path.join(run_dir, "best_model.pth")

    history = {"train_loss": [], "val_mse": [], "val_nse": []}

    delta_w = float(getattr(cfg, "DELTA_LOSS_WEIGHT", 0.35))
    # first target index inside input features for delta loss & persistence
    target_feat_indices: List[int] = []
    for t in cfg.TARGET_FEATURES:
        if t not in input_features:
            raise RuntimeError(
                f"Persistence baseline needs target '{t}' included in input_features. "
                f"Current input_features={input_features}"
            )
        target_feat_indices.append(input_features.index(t))

    for epoch in range(cfg.MAX_EPOCHS):
        model.train()
        train_losses = []

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)

            if delta_w > 0:
                # bx: [B,T,N,F]
                # out/by can be [B,N,D] or [B,P,N,D]
                last_vals = []
                for fi in target_feat_indices:
                    last_vals.append(bx[:, -1, :, fi].unsqueeze(-1))
                last = torch.cat(last_vals, dim=-1)  # [B,N,D]
                if out.dim() == 4 and by.dim() == 4:
                    # expand to [B,P,N,D] for multi-horizon supervision
                    p = int(out.shape[1])
                    last = last.unsqueeze(1).expand(-1, p, -1, -1).contiguous()
                loss = loss + delta_w * criterion(out - last, by - last)

            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        # Validation (collect y for stable metrics)
        yv_true, yv_pred = _collect_model_outputs(model, val_loader, device)
        val_metrics = _metrics_np(yv_true, yv_pred)
        scheduler.step(val_metrics["mse"])

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_mse"].append(float(val_metrics["mse"]))
        history["val_nse"].append(float(val_metrics["nse"]))

        if val_metrics["mse"] < best_val:
            best_val = float(val_metrics["mse"])
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, best_path)

        if (epoch - best_epoch) > cfg.EARLY_STOP_PATIENCE:
            break

    save_json(history, os.path.join(run_dir, "train_history.json"))
    save_json({"best_epoch": best_epoch, "best_val_mse": best_val}, os.path.join(run_dir, "best_info.json"))

    # Load best and test once
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    yt_true, yt_pred = _collect_model_outputs(model, test_loader, device)
    test_metrics = _metrics_np(yt_true, yt_pred)
    test_metrics_real = _metrics_real(scaler_Y, yt_true, yt_pred)

    save_json(test_metrics, os.path.join(run_dir, "test_metrics.json"))
    save_json(test_metrics_real, os.path.join(run_dir, "test_metrics_real.json"))

    # Persistence baseline on the same test set
    yb_true, yb_pred = _collect_persistence_outputs(test_loader, target_feat_indices)
    base_metrics = _metrics_np(yb_true, yb_pred)
    base_metrics_real = _metrics_real(scaler_Y, yb_true, yb_pred)
    save_json(base_metrics, os.path.join(run_dir, "baseline_persistence.json"))
    save_json(base_metrics_real, os.path.join(run_dir, "baseline_persistence_real.json"))

    def _skill(model_v: float, base_v: float) -> float:
        if (not np.isfinite(model_v)) or (not np.isfinite(base_v)) or abs(base_v) < 1e-12:
            return float("nan")
        return float(1.0 - (model_v / base_v))

    skill_scores = {
        "rmse_skill": _skill(test_metrics["rmse"], base_metrics["rmse"]),
        "mae_skill": _skill(test_metrics["mae"], base_metrics["mae"]),
        "rmse_skill_real": _skill(test_metrics_real["rmse"], base_metrics_real["rmse"]),
        "mae_skill_real": _skill(test_metrics_real["mae"], base_metrics_real["mae"]),
    }
    save_json(skill_scores, os.path.join(run_dir, "skill_scores.json"))

    summary = {
        "variant": variant_name,
        "test": test_metrics,
        "test_real": test_metrics_real,
        "baseline_persistence": base_metrics,
        "baseline_persistence_real": base_metrics_real,
        "skill_scores": skill_scores,
        "best_epoch": int(best_epoch),
        "applied_patches": applied,
        "cfg_updates": {},
        "input_dim": len(input_features),
    }
    save_json(summary, os.path.join(run_dir, "summary.json"))
    return summary


# -----------------------------
# Variants (paper Sec. 2.6)
# -----------------------------
def build_variants(cfg: Config) -> Dict[str, Dict]:
    return {
        "full": {"cfg_updates": {}, "input_mode": "default", "patches": []},
        "w_o_adaptive_adj": {
            "cfg_updates": {"USE_ADAPTIVE_ADJ": False, "ADJ_ADAPT_WEIGHT": 0.0},
            "input_mode": "default",
            "patches": [],
        },
        "temporal_cnn_only": {
            "cfg_updates": {"TEMPORAL_BRANCH_MODE": "cnn", "FUSION_MODE": "gate"},
            "input_mode": "default",
            "patches": [],
        },
        "temporal_lstm_only": {
            "cfg_updates": {"TEMPORAL_BRANCH_MODE": "lstm", "FUSION_MODE": "gate"},
            "input_mode": "default",
            "patches": [],
        },
        "temporal_tcn_only": {
            "cfg_updates": {"TEMPORAL_BRANCH_MODE": "tcn", "FUSION_MODE": "gate"},
            "input_mode": "default",
            "patches": [],
        },
        "w_o_spatial_graph": {
            "cfg_updates": {"USE_ADAPTIVE_ADJ": False, "ADJ_ADAPT_WEIGHT": 0.0},
            "input_mode": "default",
            "patches": ["no_graph_spatial"],
        },
        "fusion_avg": {
            "cfg_updates": {"TEMPORAL_BRANCH_MODE": "all", "FUSION_MODE": "avg"},
            "input_mode": "default",
            "patches": [],
        },
        "fusion_concat": {
            "cfg_updates": {"TEMPORAL_BRANCH_MODE": "all", "FUSION_MODE": "concat"},
            "input_mode": "default",
            "patches": [],
        },
        "w_o_cyclic": {"cfg_updates": {}, "input_mode": "no_cyclic", "patches": []},
        "w_o_delta_loss": {"cfg_updates": {"DELTA_LOSS_WEIGHT": 0.0}, "input_mode": "default", "patches": []},
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    configure_stdio_for_server()
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=str, default="all",
                        help="Comma-separated variants (e.g., full,w_o_cyclic) or 'all'.")
    parser.add_argument("--robustness", action="store_true",
                        help="If set, also evaluate robustness by masking inputs at mask_rates.")
    parser.add_argument("--mask_rates", type=str, default="0.1,0.2,0.3",
                        help="Comma-separated mask rates for robustness evaluation.")
    parser.add_argument("--raw_data_file", type=str, default="",
                        help="Override cfg.RAW_DATA_FILE if provided.")
    parser.add_argument("--exp_root", type=str, default="",
                        help="Override cfg.EXP_ROOT if provided (not used for outputs when results_root is set).")
    parser.add_argument("--results_root", type=str, default="",
                        help="All ablation outputs will be saved under this folder. Default: ./ablation_results.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max_epochs", type=int, default=-1,
                        help="Override cfg.MAX_EPOCHS if >0.")
    parser.add_argument("--top_k_lakes", type=int, default=-1, help="Override cfg.TOP_K_LAKES if >0.")
    parser.add_argument("--min_effective_steps", type=int, default=-1, help="Override cfg.MIN_EFFECTIVE_STEPS if >0.")
    parser.add_argument("--seq_len", type=int, default=-1, help="Override cfg.SEQ_LEN if >0.")
    parser.add_argument("--pred_len", type=int, default=-1, help="Override cfg.PRED_LEN if >0.")
    parser.add_argument("--batch_size", type=int, default=-1, help="Override cfg.BATCH_SIZE if >0.")
    args = parser.parse_args()

    set_seed(args.seed)
    _set_torch_deterministic()

    base_cfg = Config()
    if args.raw_data_file:
        base_cfg.RAW_DATA_FILE = args.raw_data_file
    if args.exp_root:
        base_cfg.EXP_ROOT = args.exp_root
    if args.max_epochs and args.max_epochs > 0:
        base_cfg.MAX_EPOCHS = args.max_epochs
    if args.top_k_lakes and args.top_k_lakes > 0:
        base_cfg.TOP_K_LAKES = int(args.top_k_lakes)
    if args.min_effective_steps and args.min_effective_steps > 0:
        base_cfg.MIN_EFFECTIVE_STEPS = int(args.min_effective_steps)
    if args.seq_len and args.seq_len > 0:
        base_cfg.SEQ_LEN = int(args.seq_len)
    if args.pred_len and args.pred_len > 0:
        base_cfg.PRED_LEN = int(args.pred_len)
    if args.batch_size and args.batch_size > 0:
        base_cfg.BATCH_SIZE = int(args.batch_size)

    # Ablations target the main ST model
    base_cfg.MODEL_NAME = "stgcn_fusion"

    variants_def = build_variants(base_cfg)
    if args.variants.strip().lower() == "all":
        chosen = list(variants_def.keys())
    else:
        chosen = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in chosen if v not in variants_def]
        if unknown:
            raise ValueError(f"Unknown variants: {unknown}. Available: {list(variants_def.keys())}")

    # Decide output folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = args.results_root.strip() if args.results_root else ""
    if not results_root:
        results_root = os.path.join(script_dir, "ablation_results")
    ensure_dir(results_root)

    exp_dir = os.path.join(results_root, f"ablation_{now_str()}")
    ensure_dir(exp_dir)
    save_json(collect_runtime_env(), os.path.join(exp_dir, "runtime_env.json"))
    save_json({"chosen_variants": chosen, "base_config": asdict(base_cfg), "results_root": results_root},
              os.path.join(exp_dir, "exp_plan.json"))

    results: List[Dict] = []

    for vname in chosen:
        vdef = variants_def[vname]
        cfg = _clone_cfg(base_cfg)
        for k, val in vdef.get("cfg_updates", {}).items():
            setattr(cfg, k, val)

        input_features = _no_cyclic_input_features(cfg) if vdef.get("input_mode") == "no_cyclic" else _default_input_features(cfg)

        run_dir = os.path.join(exp_dir, vname)
        ensure_dir(run_dir)
        save_json({"variant": vname, "cfg_updates": vdef.get("cfg_updates", {}), "input_features": input_features},
                  os.path.join(run_dir, "variant_spec.json"))

        train_loader, val_loader, test_loader, graph_dict, mask_feat_indices, scaler_Y = prepare_data_gnn(cfg, input_features, run_dir)

        summary = train_one_variant_gnn(
            cfg=cfg,
            variant_name=vname,
            run_dir=run_dir,
            input_features=input_features,
            graph_dict=graph_dict,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            scaler_Y=scaler_Y,
            model_patches=vdef.get("patches", []),
        )
        summary["cfg_updates"] = vdef.get("cfg_updates", {})
        results.append(summary)

        # Robustness: masked inputs on TEST, compute both normalized & real metrics
        if args.robustness:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = build_model(cfg, len(input_features), len(cfg.TARGET_FEATURES), graph=graph_dict).to(device)
            for p in vdef.get("patches", []):
                if p == "no_graph_spatial":
                    model.spatial = NoGraphSpatial(len(input_features), cfg.GCN_HIDDEN_DIM, cfg.DROPOUT_RATE).to(device)
                elif p == "fusion_avg":
                    force_average_fusion(model)

            ckpt = torch.load(os.path.join(run_dir, "best_model.pth"), map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

            mask_rates = _parse_float_list(args.mask_rates)
            masked_metrics = {}
            for mr in mask_rates:
                y_true_m, y_pred_m = _collect_masked_outputs(
                    model, test_loader, device,
                    mask_ratio=mr,
                    mask_feat_indices=mask_feat_indices,
                    seed=args.seed + int(mr * 1000) + 17,
                )
                masked_metrics[str(mr)] = {
                    "normalized": _metrics_np(y_true_m, y_pred_m),
                    "real": _metrics_real(scaler_Y, y_true_m, y_pred_m),
                }

            save_json(masked_metrics, os.path.join(run_dir, "robustness_mask_metrics.json"))
            summary["robustness_mask"] = masked_metrics

    save_json(results, os.path.join(exp_dir, "ablation_results.json"))

    # Export a compact table + plot for paper/rebuttal use
    table_rows: List[Dict] = []
    for r in results:
        t = r.get("test", {})
        tr = r.get("test_real", {})
        skill = r.get("skill_scores", {})
        table_rows.append(
            {
                "variant": r.get("variant", ""),
                "test_nse": float(t.get("nse", float("nan"))),
                "test_rmse": float(t.get("rmse", float("nan"))),
                "test_mae": float(t.get("mae", float("nan"))),
                "test_mape": float(t.get("mape", float("nan"))),
                "test_real_rmse": float(tr.get("rmse", float("nan"))),
                "test_real_mae": float(tr.get("mae", float("nan"))),
                "test_real_mape": float(tr.get("mape", float("nan"))),
                "rmse_skill_real": float(skill.get("rmse_skill_real", float("nan"))),
                "mae_skill_real": float(skill.get("mae_skill_real", float("nan"))),
            }
        )

    if len(table_rows) > 0:
        df_tab = pd.DataFrame(table_rows).sort_values(["test_nse", "test_rmse"], ascending=[False, True])
        tab_csv = os.path.join(exp_dir, "ablation_summary_table.csv")
        df_tab.to_csv(tab_csv, index=False, encoding="utf-8-sig")

        # concise figure: NSE and MAE(real) across variants
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        x = np.arange(len(df_tab))
        axes[0].bar(x, df_tab["test_nse"].values, color="#4C72B0")
        axes[0].set_title("Ablation Test NSE")
        axes[0].set_ylabel("NSE")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df_tab["variant"].tolist(), rotation=35, ha="right")
        axes[0].grid(axis="y", linestyle="--", alpha=0.3)

        axes[1].bar(x, df_tab["test_real_mae"].values, color="#DD8452")
        axes[1].set_title("Ablation Test MAE (real scale)")
        axes[1].set_ylabel("MAE")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df_tab["variant"].tolist(), rotation=35, ha="right")
        axes[1].grid(axis="y", linestyle="--", alpha=0.3)

        fig.tight_layout()
        fig_path = os.path.join(exp_dir, "ablation_summary_plots.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved ablation table: {tab_csv}")
        print(f"Saved ablation plots: {fig_path}")

    print("\n=== Ablation Summary (Test NSE / RMSE) ===")
    for r in results:
        nse = r["test"]["nse"]
        rmse_n = r["test"]["rmse"]
        rmse_r = r["test_real"]["rmse"]
        print(f"{r['variant']:>16s} | NSE={nse:.4f} | RMSE(norm)={rmse_n:.4f} | RMSE(real)={rmse_r:.4f}")

    print("\n=== Persistence Baseline (Test NSE / RMSE) ===")
    for r in results:
        nse = r["baseline_persistence"]["nse"]
        rmse_n = r["baseline_persistence"]["rmse"]
        rmse_r = r["baseline_persistence_real"]["rmse"]
        print(f"{r['variant']:>16s} | NSE={nse:.4f} | RMSE(norm)={rmse_n:.4f} | RMSE(real)={rmse_r:.4f}")

    if args.robustness:
        print("\n=== Robustness Summary (Masked Test NSE / RMSE(real)) ===")
        for r in results:
            if "robustness_mask" not in r:
                continue
            for mr, met in r["robustness_mask"].items():
                print(f"{r['variant']:>16s} | mask={mr:>4s} | NSE={met['normalized']['nse']:.4f} | RMSE(real)={met['real']['rmse']:.4f}")


if __name__ == "__main__":
    main()
