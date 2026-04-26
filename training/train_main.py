# main.py
import os
import copy
import argparse
import shutil
import itertools
import subprocess
from typing import Dict, Any, Optional, List, Tuple
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns  # optional
except Exception:  # pragma: no cover
    sns = None
from scipy import stats
from dataclasses import asdict
from torch.utils.data import TensorDataset, DataLoader

# Configure the project import path.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules.
from config.config_taihu import Config
from utils.util_common import (
    ensure_dir, now_str, save_json, load_json, set_seed,
    configure_stdio_for_server, collect_runtime_env,
)
from models.model_gcn import build_model
from evaluation.eval_metrics import (
    evaluate, plot_timeseries_best_station, plot_log_scatter_per_feature,
    inverse_transform_lastdim, compute_metrics_per_feature, visualize_scatters,
    analyze_feature_importance_shap, analyze_temporal_importance_captum
)


# Avoid repeatedly printing dataset summary lines during grid-search.
_PRINTED_DF_COLS: bool = False
from data.data_pipeline import (
    load_raw_data, choose_target_lakes, physical_cleaning,
    split_by_time_per_lake_train_val_test, impute_strict_per_lake,
    add_time_features, join_optional_meteo, fit_scaler,
    normalize_df, augment_series_per_lake, build_windows_grouped,
    build_graph_windows_from_df
)

# Select the available compute device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def _compact_outputs_enabled(cfg: Config) -> bool:
    return bool(getattr(cfg, "COMPACT_OUTPUTS", True))


def _build_lr_scheduler(
    optimizer: optim.Optimizer,
    cfg: Config,
    max_epochs: int,
):
    """Create scheduler from config: plateau / cosine / warmup_cosine."""
    name = str(getattr(cfg, "LR_SCHEDULER", "plateau")).lower().strip()
    if name == "plateau":
        sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        return sch, "plateau"
    if name == "cosine":
        eta_min = float(getattr(cfg, "LEARNING_RATE", 1e-4)) * float(getattr(cfg, "MIN_LR_RATIO", 0.1))
        sch = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(max_epochs)), eta_min=max(0.0, eta_min))
        return sch, "cosine"
    if name == "warmup_cosine":
        warmup = int(max(1, getattr(cfg, "WARMUP_EPOCHS", 3)))
        total = max(1, int(max_epochs))
        min_ratio = float(getattr(cfg, "MIN_LR_RATIO", 0.1))
        min_ratio = max(0.0, min(1.0, min_ratio))

        def lr_lambda(epoch: int) -> float:
            ep = int(epoch) + 1
            if ep <= warmup:
                return float(ep) / float(warmup)
            remain = max(1, total - warmup)
            prog = float(ep - warmup) / float(remain)
            prog = max(0.0, min(1.0, prog))
            cosine = 0.5 * (1.0 + np.cos(np.pi * prog))
            return float(min_ratio + (1.0 - min_ratio) * cosine)

        sch = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return sch, "warmup_cosine"

    sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    return sch, "plateau"

def run_post_processing(model, test_loader, train_loader, input_features, cfg, run_dir, scaler_Y, X_test, test_times):
    """
    Unified post-processing: panel plots + optional interpretability outputs.
    """
    print("-" * 30)
    print("[POST] Generating analysis figures...")
    
    # Generate predictions on the test loader.
    model.eval()
    criterion = nn.MSELoss()
    val_metrics, y_true, y_pred = evaluate(model, test_loader, device, criterion, feature_names=cfg.TARGET_FEATURES)

    # Save raw arrays for downstream inference visualizations.
    try:
        np.savez_compressed(
            os.path.join(run_dir, "analysis_data.npz"),
            y_true=y_true,
            y_pred=y_pred,
            target_features=np.asarray(cfg.TARGET_FEATURES, dtype=object),
            test_times=np.asarray(test_times if test_times is not None else [], dtype=object),
        )
    except Exception as e:
        print(f"[WARN] Failed to save analysis_data.npz: {e}")
    
    # NOTE:
    # For GNN/ST models we keep y_true/y_pred as [S, N, D] so that
    # visualize_scatters can correctly select a single station (node_idx).

    
    # Generate the thesis figure suite from inference outputs.
    print("1. Generating thesis figure suite from inference outputs...")
    try:
        py = sys.executable or "python"
        cmd = [
            py,
            "-m",
            "visualization.viz_paper_figures",
            "--out_dir",
            run_dir,
            "--test_metrics",
            os.path.join(run_dir, "test_metrics.json"),
            "--analysis_npz",
            os.path.join(run_dir, "analysis_data.npz"),
        ]
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"[WARN] thesis 7-figure suite generation failed: {e}")

    # Save per-feature metrics in the original physical scale.
    try:
        y_true_real_all = inverse_transform_lastdim(scaler_Y, y_true)
        y_pred_real_all = inverse_transform_lastdim(scaler_Y, y_pred)
        metrics_by_feature_real = compute_metrics_per_feature(y_true_real_all, y_pred_real_all, cfg.TARGET_FEATURES)
        if _compact_outputs_enabled(cfg):
            tm_path = os.path.join(run_dir, "test_metrics.json")
            if os.path.exists(tm_path):
                tm = load_json(tm_path)
                tm["metrics_by_feature_real"] = metrics_by_feature_real
                save_json(tm, tm_path)
            else:
                save_json({"metrics_by_feature_real": metrics_by_feature_real}, tm_path)
        else:
            save_json({"metrics_by_feature_real": metrics_by_feature_real}, os.path.join(run_dir, "metrics_by_feature_real.json"))
    except Exception as e:
        print(f"[WARN] Failed to save original-scale per-feature metrics: {e}")
    # Error-distribution plots are disabled by default.
    print("3. Skipping error-distribution plots (disabled).")

    # 4) Optional explainability figures (supplementary by default)
    if bool(getattr(cfg, "ENABLE_XAI_PLOTS", False)):
        print("4. Running explainability plots (SHAP + Captum)...")
        try:
            analyze_feature_importance_shap(model, train_loader, input_features, run_dir, device)
        except Exception as e:
            print(f"[WARN] SHAP skipped: {e}")
        try:
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            last_sample = X_test_tensor[-1:].to(device)
            analyze_temporal_importance_captum(model, last_sample, input_features, run_dir, device, sample_id="last")
        except Exception as e:
            print(f"[WARN] Captum skipped: {e}")
    else:
        print("4. Skipping explainability plots (ENABLE_XAI_PLOTS=False).")
    print("[OK] Analysis figures generated.")

def train_run(
    cfg: Config,
    run_dir: str,
    *,
    objective: Optional[str] = None,
    max_epochs: Optional[int] = None,
    early_stop_patience: Optional[int] = None,
    do_test: bool = True,
    do_post: bool = True,
    save_checkpoint: bool = True,
    save_artifacts: bool = True,
    plot_loss: bool = True,
) -> Dict[str, Any]:
    ensure_dir(run_dir)
    configure_stdio_for_server()
    compact_outputs = _compact_outputs_enabled(cfg)
    run_bundle: Dict[str, Any] = {}
    if save_artifacts:
        runtime_env = collect_runtime_env()
        if compact_outputs:
            run_bundle["runtime_env"] = runtime_env
        else:
            save_json(runtime_env, os.path.join(run_dir, "runtime_env.json"))

    # objective: "val_nse"(max), "val_rmse"(min) or "val_mse"(min)
    obj = (objective or getattr(cfg, "TUNE_OBJECTIVE", "val_nse")).lower()
    if obj not in {"val_nse", "val_rmse", "val_mse"}:
        raise ValueError(f"Unsupported objective: {obj}")
    obj_mode = "max" if obj == "val_nse" else "min"

    max_epochs_i = int(max_epochs) if max_epochs is not None else int(cfg.MAX_EPOCHS)
    patience_i = int(early_stop_patience) if early_stop_patience is not None else int(cfg.EARLY_STOP_PATIENCE)

    test_times = None
    df_raw = load_raw_data(cfg)
    df_selected = choose_target_lakes(df_raw, cfg)
    df_clean = physical_cleaning(df_selected, cfg)
    # Strict Train/Val/Test split (per station, time-ordered). Avoid test leakage.
    df_train, df_val, df_test, split_meta = split_by_time_per_lake_train_val_test(
        df_clean, cfg,
        train_ratio=getattr(cfg, "TRAIN_RATIO", 0.7),
        val_ratio=getattr(cfg, "VAL_RATIO", 0.1),
        overlap=getattr(cfg, "SPLIT_OVERLAP", cfg.SEQ_LEN),
    )
    if df_train.empty:
        raise RuntimeError("训练集为空")
    if df_val.empty:
        raise RuntimeError("验证集为空，请检查 TRAIN_RATIO/VAL_RATIO 或数据长度")
    if df_test.empty:
        raise RuntimeError("测试集为空，请检查 TRAIN_RATIO/VAL_RATIO 或数据长度")

    # Save split meta for reproducibility
    if save_artifacts:
        if compact_outputs:
            run_bundle["splits"] = split_meta
        else:
            save_json(split_meta, os.path.join(run_dir, "splits.json"))

    df_train_imp, train_means = impute_strict_per_lake(df_train, cfg, return_train_means=True)
    train_means: Dict[str, float] = train_means
    df_val_imp   = impute_strict_per_lake(df_val, cfg, train_means=train_means)
    df_test_imp  = impute_strict_per_lake(df_test, cfg, train_means=train_means)
    # Save train-only means used for NaN fallback (leakage-safe)
    if save_artifacts:
        if compact_outputs:
            run_bundle["train_feature_means"] = train_means
        else:
            save_json(train_means, os.path.join(run_dir, "train_feature_means.json"))
    df_train_fe = add_time_features(df_train_imp)
    df_val_fe   = add_time_features(df_val_imp)
    df_test_fe  = add_time_features(df_test_imp)
    
    test_timestamps = None

    input_features = cfg.FEATURE_COLS + ["month_sin", "month_cos", "hour_sin", "hour_cos", "t_index"]

    # IMPORTANT:
    # Do NOT min-max scale the periodic time features (sin/cos) and t_index.
    # They are already bounded / meaningful and scaling them can destroy the
    # diurnal pattern, leading to mean-regression (overly smooth predictions).
    scale_cols = list(cfg.FEATURE_COLS)

    scaler_X = fit_scaler(df_train_fe, scale_cols)
    scaler_Y = fit_scaler(df_train_fe, cfg.TARGET_FEATURES)

    df_train_norm = normalize_df(df_train_fe, scaler_X, scale_cols)
    df_val_norm   = normalize_df(df_val_fe, scaler_X, scale_cols)
    df_test_norm  = normalize_df(df_test_fe, scaler_X, scale_cols)
    df_train_norm[cfg.TARGET_FEATURES] = scaler_Y.transform(df_train_fe[cfg.TARGET_FEATURES].values)
    df_val_norm[cfg.TARGET_FEATURES]   = scaler_Y.transform(df_val_fe[cfg.TARGET_FEATURES].values)
    df_test_norm[cfg.TARGET_FEATURES]  = scaler_Y.transform(df_test_fe[cfg.TARGET_FEATURES].values)

    # Train-only means on the *normalized* scale, used as a safe NaN fallback
    # when building shared graph windows. This avoids mixing raw-scale means
    # into standardized data.
    train_fill_means_norm: Dict[str, float] = {}
    for _c in list(cfg.FEATURE_COLS) + list(cfg.TARGET_FEATURES):
        if _c in df_train_norm.columns:
            v = pd.to_numeric(df_train_norm[_c], errors="coerce").mean(skipna=True)
            train_fill_means_norm[_c] = float(v) if pd.notna(v) else 0.0

    is_gnn = cfg.MODEL_NAME.lower() in {
        "stgcn",
        "dcrnn",
        "stgcn_fusion",
    }
    
    global _PRINTED_DF_COLS
    if not _PRINTED_DF_COLS:
        print("df_train_norm columns:", df_train_norm.columns.tolist())
        _PRINTED_DF_COLS = True

    if is_gnn:
        # Optional augmentation for GNN branch (previously only baseline branch used augmentation).
        if bool(getattr(cfg, "AUG_ENABLE_GNN", True)) and int(getattr(cfg, "AUG_TIMES", 0)) > 0:
            df_train_for_windows = augment_series_per_lake(df_train_norm, cfg, cfg.FEATURE_COLS)
            print(f"[AUG] GNN train augmentation enabled: AUG_TIMES={cfg.AUG_TIMES}, NOISE_SCALE={cfg.NOISE_SCALE}")
        else:
            df_train_for_windows = df_train_norm
            print("[AUG] GNN train augmentation disabled.")

        X_train, y_train, train_times, adj_hat, node_ids = build_graph_windows_from_df(
            df_train_for_windows, cfg, input_features, cfg.TARGET_FEATURES,
            train_fill_means=train_fill_means_norm
        )
        X_val, y_val, val_times, _, _ = build_graph_windows_from_df(
            df_val_norm, cfg, input_features, cfg.TARGET_FEATURES,
            node_ids=node_ids, adj_hat=adj_hat, train_fill_means=train_fill_means_norm)
        X_test, y_test, test_times, _, _ = build_graph_windows_from_df(
            df_test_norm, cfg, input_features, cfg.TARGET_FEATURES,
            node_ids=node_ids, adj_hat=adj_hat,  # [OK] 关键：保持节点顺序与图一致
            train_fill_means=train_fill_means_norm
        )

        if X_train.size == 0 or y_train.size == 0:
            raise RuntimeError("窗口化后训练样本为空（GNN）。请检查站点数据长度、SEQ_LEN/PRED_LEN 或 split 比例。")
        print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
        print("y_train min/max:", y_train.min(), y_train.max())
        print("y_train zero ratio:", float((y_train == 0).mean()))
        
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.float32)),
            batch_size=cfg.BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                          torch.tensor(y_val, dtype=torch.float32)),
            batch_size=cfg.BATCH_SIZE, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                          torch.tensor(y_test, dtype=torch.float32)),
            batch_size=cfg.BATCH_SIZE, shuffle=False
        )
    else:
        # Non-GNN baselines: build windows and carry station indices for grouped NSE.
        # Use a shared station vocabulary for consistent station indexing across splits.
        station_vocab = sorted(df_clean["ID"].astype(str).unique().tolist())

        if bool(getattr(cfg, "AUG_ENABLE_BASELINE", True)) and int(getattr(cfg, "AUG_TIMES", 0)) > 0:
            df_train_for_windows = augment_series_per_lake(df_train_norm, cfg, cfg.FEATURE_COLS)
            print(f"[AUG] Baseline train augmentation enabled: AUG_TIMES={cfg.AUG_TIMES}, NOISE_SCALE={cfg.NOISE_SCALE}")
        else:
            df_train_for_windows = df_train_norm
            print("[AUG] Baseline train augmentation disabled.")

        X_train, y_train, sid_train, station_names, _ = build_windows_grouped(
            df_train_for_windows,
            cfg, input_features, cfg.TARGET_FEATURES,
            return_meta=True,
            station_vocab=station_vocab,
        )
        X_val, y_val, sid_val, _, _ = build_windows_grouped(
            df_val_norm, cfg, input_features, cfg.TARGET_FEATURES,
            return_meta=True,
            station_vocab=station_vocab,
        )
        X_test, y_test, sid_test, _, test_times = build_windows_grouped(
            df_test_norm, cfg, input_features, cfg.TARGET_FEATURES,
            return_meta=True,
            station_vocab=station_vocab,
        )

        if X_train.size == 0 or y_train.size == 0:
            raise RuntimeError("窗口化后训练样本为空。请检查所选站点的数据长度、SEQ_LEN/PRED_LEN 或 split 比例。")

        if save_artifacts:
            save_json({"station_names": station_names}, os.path.join(run_dir, "station_names.json"))

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
                torch.tensor(sid_train, dtype=torch.long),
            ),
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
                torch.tensor(sid_val, dtype=torch.long),
            ),
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32),
                torch.tensor(sid_test, dtype=torch.long),
            ),
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
        )

    # Graph/meta info:
    # - For GNN/ST models: provide adj_hat & num_nodes.
    # - For ALL models (including CNN/TCN/LSTM/iTransformer): provide target_indices for residual forecasting.
    target_indices = [input_features.index(t) for t in cfg.TARGET_FEATURES if t in input_features]
    graph: Dict[str, Any] = {"target_indices": target_indices}
    if is_gnn:
        graph.update({
            "adj_hat": torch.tensor(adj_hat, dtype=torch.float32).to(device),
            "num_nodes": int(adj_hat.shape[0]),
        })

    model = build_model(cfg, len(input_features), len(cfg.TARGET_FEATURES), graph=graph).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler, scheduler_name = _build_lr_scheduler(optimizer, cfg, max_epochs=max_epochs_i)
    criterion = nn.MSELoss()
    
    # --- Early-stopping/checkpoint is decided by Val RMSE (paper-style) ---
    best_epoch_rmse = -1
    best_val_rmse = float('inf')
    best_val_mse_at_best = float('inf')
    best_val_nse_at_best = float('-inf')

    # Track best metrics for reporting / tuning objective
    best_epoch_nse = -1
    best_val_nse_max = float('-inf')
    best_epoch_mse = -1
    best_val_mse_min = float('inf')

    best_score = float('-inf') if obj_mode == "max" else float('inf')
    best_score_epoch = -1

    best_state_dict = None
    need_best_state = (not save_checkpoint) and (do_test or do_post)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_mse": [],
        "train_rmse": [],
        "train_nse": [],
        "train_nse_std": [],
        "val_mse": [],
        "val_rmse": [],
        "val_nse": [],
        "val_nse_std": [],
        "val_r2": [],
        "objective": obj,
    }
    best_model_path = os.path.join(run_dir, "best_model.pt")

    print(f"[TRAIN] 启动 {cfg.MODEL_NAME} 训练...")
    for epoch in range(1, max_epochs_i + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            bx = batch[0].to(device)
            by = batch[1].to(device)
            optimizer.zero_grad()
            out = model(bx)

            # --- dynamic-aware loss (reduce mean-regression) ---
            loss = criterion(out, by)

            # Encourage matching the 1-step change (delta) safely.
            delta_w = float(getattr(cfg, "DELTA_LOSS_WEIGHT", 0.35))
            if delta_w > 0 and cfg.TARGET_FEATURES:
                target_idxs = None
                if graph is not None and "target_indices" in graph and len(graph["target_indices"]) > 0:
                    target_idxs = list(graph["target_indices"])
                else:
                    tmp = []
                    for tname in cfg.TARGET_FEATURES:
                        if tname in input_features:
                            tmp.append(input_features.index(tname))
                    if tmp:
                        target_idxs = tmp

                if target_idxs:
                    last = None
                    idxs_safe = [int(i) for i in target_idxs if int(i) < int(bx.shape[-1])]
                    if idxs_safe:
                        out_d = int(out.shape[-1])
                        if out_d == len(idxs_safe):
                            use_idxs = idxs_safe
                        elif out_d == 1 and len(idxs_safe) >= 1:
                            use_idxs = [idxs_safe[0]]
                        else:
                            use_idxs = []

                        if use_idxs:
                            if out.ndim == 4:
                                base = bx[:, -1, :, use_idxs]  # [B,N,D]
                                last = base.unsqueeze(1).expand(-1, out.shape[1], -1, -1)
                            elif out.ndim == 3:
                                base = bx[:, -1, use_idxs]  # [B,D]
                                last = base.unsqueeze(1).expand(-1, out.shape[1], -1)
                            elif out.ndim == 2:
                                last = bx[:, -1, use_idxs]  # [B,D]

                    if last is not None and tuple(last.shape) == tuple(out.shape) and tuple(last.shape) == tuple(by.shape):
                        loss = loss + delta_w * criterion(out - last, by - last)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # IMPORTANT: evaluate Train/Val each epoch for RMSE/NSE logging.
        train_metrics, _, _ = evaluate(model, train_loader, device, criterion, feature_names=cfg.TARGET_FEATURES)
        val_metrics, _, _ = evaluate(model, val_loader, device, criterion, feature_names=cfg.TARGET_FEATURES)

        train_mse = float(train_metrics['mse'])
        train_rmse = float(train_metrics['rmse'])
        train_nse = float(train_metrics['nse'])
        train_nse_std = float(train_metrics.get('nse_std', float('nan')))

        val_mse = float(val_metrics['mse'])
        val_rmse = float(val_metrics['rmse'])
        val_nse = float(val_metrics['nse'])
        val_nse_std = float(val_metrics.get('nse_std', float('nan')))
        val_nse_filt = float(val_metrics.get('nse_filt', val_nse))
        val_nse_filt_std = float(val_metrics.get('nse_filt_std', val_nse_std))
        val_nse_filt_kept = int(val_metrics.get('nse_filt_kept', 0))
        val_nse_filt_total = int(val_metrics.get('nse_filt_total', 0))


        # LR scheduling uses Val RMSE (min)
        if scheduler_name == "plateau":
            scheduler.step(val_rmse)
        else:
            scheduler.step()
        
        history['epoch'].append(int(epoch))
        history['train_loss'].append(float(np.mean(train_losses)))
        history['train_mse'].append(float(train_mse))
        history['train_rmse'].append(float(train_rmse))
        history['train_nse'].append(float(train_nse))
        history['train_nse_std'].append(float(train_nse_std))
        history['val_mse'].append(float(val_mse))
        history['val_rmse'].append(float(val_rmse))
        history['val_nse'].append(float(val_nse))
        history['val_nse_std'].append(float(val_nse_std))
        history['val_r2'].append(float(val_metrics['r2']))

        # Track best NSE/MSE achieved (for reporting / tuning objective)
        if val_nse > best_val_nse_max:
            best_val_nse_max = val_nse
            best_epoch_nse = epoch
        if val_mse < best_val_mse_min:
            best_val_mse_min = val_mse
            best_epoch_mse = epoch

        # Tuning objective score (separate from early stop)
        score = val_nse if obj == "val_nse" else (val_rmse if obj == "val_rmse" else val_mse)
        is_better_obj = (score > best_score) if obj_mode == "max" else (score < best_score)
        if is_better_obj:
            best_score = float(score)
            best_score_epoch = epoch

        # Early-stopping / checkpoint: decided by Val RMSE (min)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch_rmse = epoch
            best_val_mse_at_best = val_mse
            best_val_nse_at_best = val_nse
            if save_checkpoint:
                torch.save(
                    {"model_state_dict": model.state_dict(), "cfg": asdict(cfg), "input_features": input_features},
                    best_model_path,
                )
            if need_best_state:
                best_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        
        curr_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:03d} | Train RMSE: {train_rmse:.5f} | Train NSE: {train_nse:.4f}±{train_nse_std:.4f} "
            f"| Val RMSE: {val_rmse:.5f} | Val NSE: {val_nse:.4f}±{val_nse_std:.4f} | Val NSE(filt): {val_nse_filt:.4f}±{val_nse_filt_std:.4f} [{val_nse_filt_kept}/{val_nse_filt_total}] | LR: {curr_lr:.6f}"
        )

        if (epoch - best_epoch_rmse) > patience_i:
            break

    result: Dict[str, Any] = {
        "run_dir": run_dir,
        "objective": obj,
        "best_score": float(best_score),
        "best_score_epoch": int(best_score_epoch),
        # checkpoint / early-stop
        "best_epoch": int(best_epoch_rmse),
        "best_val_rmse": float(best_val_rmse),
        "best_val_mse": float(best_val_mse_at_best),
        "best_val_nse_at_best_rmse": float(best_val_nse_at_best),
        # reporting
        "best_val_nse": float(best_val_nse_max),
        "best_epoch_nse": int(best_epoch_nse),
        "best_val_mse_min": float(best_val_mse_min),
        "best_epoch_mse": int(best_epoch_mse),
    }

    if save_artifacts:
        mean_x = scaler_X.mean_
        scale_x = scaler_X.scale_
        assert mean_x is not None and scale_x is not None
        scaler_x_payload = {"mean_": [float(v) for v in mean_x], "scale_": [float(v) for v in scale_x], "features": scale_cols}

        mean_y = scaler_Y.mean_
        scale_y = scaler_Y.scale_
        assert mean_y is not None and scale_y is not None
        scaler_y_payload = {"mean_": [float(v) for v in mean_y], "scale_": [float(v) for v in scale_y], "features": cfg.TARGET_FEATURES}

        if compact_outputs:
            run_bundle["history"] = history
            run_bundle["scaler_X"] = scaler_x_payload
            run_bundle["scaler_Y"] = scaler_y_payload
        else:
            save_json(history, os.path.join(run_dir, "metrics_log.json"))
            save_json(scaler_x_payload, os.path.join(run_dir, "scaler_X.json"))
            save_json(scaler_y_payload, os.path.join(run_dir, "scaler_Y.json"))

    if save_checkpoint and (do_test or do_post) and (not os.path.exists(best_model_path)):
        torch.save(
            {"model_state_dict": model.state_dict(), "cfg": asdict(cfg), "input_features": input_features},
            best_model_path,
        )

    if do_test or do_post:
        if save_checkpoint:
            best_ckpt = torch.load(best_model_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
        else:
            if best_state_dict is None:
                raise RuntimeError("best_state_dict is None: save_checkpoint=False 但 do_test/do_post=True")
            model.load_state_dict(best_state_dict)

    if do_test:
        test_metrics, y_true_test, y_pred_test = evaluate(model, test_loader, device, criterion, feature_names=cfg.TARGET_FEATURES)
        # test_metrics now includes structured NSE summaries (by station/horizon/feature) -- keep as-is.
        result["test_metrics"] = test_metrics
        if save_artifacts:
            run_bundle["test_metrics"] = test_metrics
            save_json(test_metrics, os.path.join(run_dir, "test_metrics.json"))
            if not compact_outputs:
                try:
                    np.savez_compressed(
                        os.path.join(run_dir, "test_outputs.npz"),
                        y_true=y_true_test,
                        y_pred=y_pred_test,
                        target_features=np.asarray(cfg.TARGET_FEATURES, dtype=object),
                    )
                except Exception as e:
                    print(f"[WARN] failed to save test_outputs.npz: {e}")

    if do_post:
        print("DEBUG test_times:", None if test_times is None else (type(test_times), len(test_times)))
        run_post_processing(model, test_loader, train_loader, input_features, cfg, run_dir, scaler_Y, X_test, test_times=test_times)

    if plot_loss and save_artifacts:
        plt.figure(figsize=(8, 4))
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_rmse"], label="Val RMSE")
        plt.legend(); plt.savefig(os.path.join(run_dir, "loss_curve.png")); plt.close()

    if save_artifacts and compact_outputs:
        run_bundle["result"] = result
        save_json(run_bundle, os.path.join(run_dir, "run_bundle.json"))

    return result

def eval_run(cfg: Config):
    """
    加载已有的模型进行评估和绘图模式
    """
    if not cfg.LOAD_RUN_ID:
        raise ValueError("使用 eval 模式必须在 config 中指定 LOAD_RUN_ID")
    
    target_run_dir = os.path.join(cfg.EXP_ROOT, cfg.LOAD_RUN_ID)
    print(f"📂 正在从以下路径加载模型: {target_run_dir}")
    
    df_raw = load_raw_data(cfg)
    df_selected = choose_target_lakes(df_raw, cfg)
    df_clean = physical_cleaning(df_selected, cfg)
    df_train, df_val, df_test, _ = split_by_time_per_lake_train_val_test(
        df_clean, cfg,
        train_ratio=getattr(cfg, "TRAIN_RATIO", 0.7),
        val_ratio=getattr(cfg, "VAL_RATIO", 0.1),
        overlap=getattr(cfg, "SPLIT_OVERLAP", cfg.SEQ_LEN),
    )

    df_train_imp, train_means = impute_strict_per_lake(df_train, cfg, return_train_means=True)
    train_means: Dict[str, float] = train_means
    df_test_imp  = impute_strict_per_lake(df_test, cfg, train_means=train_means)
    df_train_fe = add_time_features(df_train_imp)
    df_test_fe  = add_time_features(df_test_imp)

    input_features = cfg.FEATURE_COLS + ["month_sin", "month_cos", "hour_sin", "hour_cos", "t_index"]

    from sklearn.preprocessing import StandardScaler
    scaler_x_path = os.path.join(target_run_dir, "scaler_X.json")
    scaler_y_path = os.path.join(target_run_dir, "scaler_Y.json")
    if os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path):
        scaler_X_json = load_json(scaler_x_path)
        scaler_Y_json = load_json(scaler_y_path)
    else:
        bundle_path = os.path.join(target_run_dir, "run_bundle.json")
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(
                f"Missing scaler files in {target_run_dir}: expected scaler_X.json/scaler_Y.json or run_bundle.json"
            )
        bundle = load_json(bundle_path)
        scaler_X_json = dict(bundle.get("scaler_X", {}))
        scaler_Y_json = dict(bundle.get("scaler_Y", {}))
        if not scaler_X_json or not scaler_Y_json:
            raise KeyError("run_bundle.json missing scaler_X/scaler_Y payloads")

    scaler_X = StandardScaler()
    scaler_X.mean_ = np.array(scaler_X_json['mean_'], dtype=float)
    scaler_X.scale_ = np.array(scaler_X_json['scale_'], dtype=float)
    scaler_X.var_ = scaler_X.scale_ ** 2
    scaler_X.n_features_in_ = int(scaler_X.mean_.shape[0])

    scaler_Y = StandardScaler()
    scaler_Y.mean_ = np.array(scaler_Y_json['mean_'], dtype=float)
    scaler_Y.scale_ = np.array(scaler_Y_json['scale_'], dtype=float)
    scaler_Y.var_ = scaler_Y.scale_ ** 2
    scaler_Y.n_features_in_ = int(scaler_Y.mean_.shape[0])

    scale_cols = list(scaler_X_json['features']) if 'features' in scaler_X_json else cfg.FEATURE_COLS

    df_train_norm = normalize_df(df_train_fe, scaler_X, scale_cols)
    df_test_norm  = normalize_df(df_test_fe, scaler_X, scale_cols)
    df_train_norm[cfg.TARGET_FEATURES] = scaler_Y.transform(df_train_fe[cfg.TARGET_FEATURES].values)
    df_test_norm[cfg.TARGET_FEATURES]  = scaler_Y.transform(df_test_fe[cfg.TARGET_FEATURES].values)

    # Train-only normalized means for NaN fallback (evaluation path)
    train_fill_means_norm: Dict[str, float] = {}
    for _c in list(cfg.FEATURE_COLS) + list(cfg.TARGET_FEATURES):
        if _c in df_train_norm.columns:
            v = pd.to_numeric(df_train_norm[_c], errors="coerce").mean(skipna=True)
            train_fill_means_norm[_c] = float(v) if pd.notna(v) else 0.0

    is_gnn = cfg.MODEL_NAME.lower() in {
        "stgcn",
        "dcrnn",
        "stgcn_fusion",
    }
    graph: Optional[Dict[str, Any]] = None
    test_times = None

    if is_gnn:
        # Rebuild adj/node ordering from Train data (for consistent evaluation)
        X_train, y_train, _, adj_hat, node_ids = build_graph_windows_from_df(
            df_train_norm, cfg, input_features, cfg.TARGET_FEATURES,
            train_fill_means=train_fill_means_norm
        )
        X_test, y_test, test_times, _, _ = build_graph_windows_from_df(
            df_test_norm, cfg, input_features, cfg.TARGET_FEATURES,
            node_ids=node_ids, adj_hat=adj_hat, train_fill_means=train_fill_means_norm)
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=cfg.BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),  torch.tensor(y_test, dtype=torch.float32)),  batch_size=cfg.BATCH_SIZE, shuffle=False)
        target_indices = [input_features.index(t) for t in cfg.TARGET_FEATURES if t in input_features]
        graph = {"adj_hat": torch.tensor(adj_hat, dtype=torch.float32).to(device), "num_nodes": int(adj_hat.shape[0]), "target_indices": target_indices}
        model = build_model(cfg, len(input_features), len(cfg.TARGET_FEATURES), graph=graph).to(device)
    else:
        station_vocab = sorted(df_clean["ID"].astype(str).unique().tolist())
        X_train, y_train, sid_train, _, _ = build_windows_grouped(
            df_train_norm,
            cfg,
            input_features,
            cfg.TARGET_FEATURES,
            return_meta=True,
            station_vocab=station_vocab,
        )
        X_test, y_test, sid_test, _, test_times = build_windows_grouped(
            df_test_norm,
            cfg,
            input_features,
            cfg.TARGET_FEATURES,
            return_meta=True,
            station_vocab=station_vocab,
        )
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
                torch.tensor(sid_train, dtype=torch.long),
            ),
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32),
                torch.tensor(sid_test, dtype=torch.long),
            ),
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
        )
        target_indices = [input_features.index(t) for t in cfg.TARGET_FEATURES if t in input_features]
        graph = {"target_indices": target_indices}
        model = build_model(cfg, len(input_features), len(cfg.TARGET_FEATURES), graph=graph).to(device)

    ckpt_path = os.path.join(target_run_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    run_post_processing(model, test_loader, train_loader, input_features, cfg, target_run_dir, scaler_Y, X_test, test_times=test_times)
    print(f"[DONE] 评估任务完成，图表已存入: {target_run_dir}")


# ==============================
# Auto hyper-parameter search (Random Search) + multi-model runner
# ==============================
def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """Sample from log-uniform(low, high)."""
    low = float(low)
    high = float(high)
    if low <= 0 or high <= 0:
        raise ValueError("log-uniform bounds must be > 0")
    a = np.log(low)
    b = np.log(high)
    return float(np.exp(rng.uniform(a, b)))


def _choice(rng: np.random.Generator, xs: List[Any]) -> Any:
    return xs[int(rng.integers(0, len(xs)))]


def _sample_hparams(model_name: str, rng: np.random.Generator) -> Dict[str, Any]:
    """为不同模型定义“合理”的搜索空间。"""
    m = model_name.lower()
    params: Dict[str, Any] = {}

    # ------------------
    # Common hyperparams
    # ------------------
    if m in {"itransformer", "patchtst"}:
        params["LEARNING_RATE"] = _log_uniform(rng, 5e-5, 5e-4)
        params["DROPOUT_RATE"] = float(rng.uniform(0.0, 0.30))
        # Allow varying attention heads (must divide HIDDEN_DIM).
        params["NUM_HEADS"] = _choice(rng, list(getattr(Config(), "GRID_ITRANSFORMER_HEADS", [1, 2, 4, 8])))
    elif m in {"stgcn", "dcrnn", "stgcn_fusion"}:
        params["LEARNING_RATE"] = _log_uniform(rng, 5e-5, 1e-3)
        params["DROPOUT_RATE"] = float(rng.uniform(0.0, 0.45))
    else:
        # lstm/tcn/cnn
        params["LEARNING_RATE"] = _log_uniform(rng, 1e-4, 3e-3)
        params["DROPOUT_RATE"] = float(rng.uniform(0.0, 0.50))

    params["WEIGHT_DECAY"] = _log_uniform(rng, 1e-6, 1e-2)
    params["BATCH_SIZE"] = int(_choice(rng, [32, 64, 128]))
    params["DELTA_LOSS_WEIGHT"] = float(rng.uniform(0.0, 0.60))

    # ------------------
    # Model-specific
    # ------------------
    if m == "lstm":
        params["HIDDEN_DIM"] = int(_choice(rng, [32, 64, 96, 128, 192, 256]))
        params["NUM_LAYERS"] = int(_choice(rng, [1, 2, 3, 4]))
    elif m == "tcn":
        params["HIDDEN_DIM"] = int(_choice(rng, [32, 64, 96, 128, 192]))
        params["NUM_LAYERS"] = int(_choice(rng, [2, 3, 4, 5, 6]))
        params["TCN_KERNEL_SIZE"] = int(_choice(rng, [2, 3, 5]))
    elif m == "cnn":
        params["HIDDEN_DIM"] = int(_choice(rng, [32, 64, 96, 128, 192]))
        params["NUM_LAYERS"] = int(_choice(rng, [2, 3, 4, 5, 6]))
        params["TEMP_CNN_KERNEL"] = int(_choice(rng, [3, 5, 7]))
        params["DROPOUT_RATE"] = float(min(params.get("DROPOUT_RATE", 0.2), 0.35))
    elif m in {"itransformer", "patchtst"}:
        params["HIDDEN_DIM"] = int(_choice(rng, [32, 64, 96, 128, 192]))
        params["NUM_LAYERS"] = int(_choice(rng, [1, 2, 3, 4]))
        params["NUM_HEADS"] = int(_choice(rng, [2, 4, 8]))
    elif m == "stgcn":
        params["GCN_HIDDEN_DIM"] = int(_choice(rng, [32, 64, 96, 128]))
        params["GCN_LAYERS"] = int(_choice(rng, [1, 2, 3]))
        params["NUM_LAYERS"] = int(_choice(rng, [1, 2, 3]))  # GRU layers
        params["USE_ADAPTIVE_ADJ"] = bool(_choice(rng, [True, False]))
        params["ADAPT_EMB_DIM"] = int(_choice(rng, [8, 16, 32]))
        params["ADJ_ADAPT_WEIGHT"] = float(_choice(rng, [0.1, 0.3, 0.5, 1.0]))
    elif m == "dcrnn":
        params["GCN_HIDDEN_DIM"] = int(_choice(rng, [32, 64, 96, 128]))
        params["NUM_LAYERS"] = int(_choice(rng, [1, 2, 3]))
    elif m == "stgcn_fusion":
        params["GCN_HIDDEN_DIM"] = int(_choice(rng, [32, 64, 96, 128]))
        params["GCN_LAYERS"] = int(_choice(rng, [1, 2, 3]))
        params["FUSION_HIDDEN_DIM"] = int(_choice(rng, [32, 64, 96, 128]))
        params["NUM_LAYERS"] = int(_choice(rng, [2, 3, 4]))
        params["TCN_KERNEL_SIZE"] = int(_choice(rng, [2, 3, 5]))
        params["TEMP_CNN_KERNEL"] = int(_choice(rng, [3, 5]))
        params["USE_ADAPTIVE_ADJ"] = bool(_choice(rng, [True, False]))
        params["ADAPT_EMB_DIM"] = int(_choice(rng, [8, 16, 32]))
        params["ADJ_ADAPT_WEIGHT"] = float(_choice(rng, [0.1, 0.3, 0.5, 1.0]))

    return params


def _apply_hparams(cfg: Config, params: Dict[str, Any]) -> None:
    """把采样到的超参写入 cfg（就地修改）。"""
    for k, v in params.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)


def _is_better(score: float, best: float, mode: str) -> bool:
    return (score > best) if mode == "max" else (score < best)


def _grid_params_for_model(
    model_name: str,
    seq_len: int,
    batch_size: int,
    hidden_size: int,
    num_heads: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    dropout_rate: Optional[float] = None,
    lr_scheduler: Optional[str] = None,
    warmup_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Grid search params required by the paper:

    - observation window (SEQ_LEN)
    - batch size
    - hidden/state size

    For graph models, hidden_size maps to GCN_HIDDEN_DIM (and FUSION_HIDDEN_DIM for fusion model).
    """
    m = str(model_name).lower()
    params: Dict[str, Any] = {
        "SEQ_LEN": int(seq_len),
        # keep split overlap consistent with observation length
        "SPLIT_OVERLAP": int(seq_len),
        "BATCH_SIZE": int(batch_size),
    }
    if m in {"stgcn", "dcrnn"}:
        params["GCN_HIDDEN_DIM"] = int(hidden_size)
    elif m == "stgcn_fusion":
        params["GCN_HIDDEN_DIM"] = int(hidden_size)
        params["FUSION_HIDDEN_DIM"] = int(hidden_size)
    else:
        params["HIDDEN_DIM"] = int(hidden_size)

    # Extra grid for iTransformer attention heads.
    if m in {"itransformer", "i_transformer", "i-transformer"}:
        if num_heads is not None:
            params["NUM_HEADS"] = int(num_heads)
    if learning_rate is not None:
        params["LEARNING_RATE"] = float(learning_rate)
    if weight_decay is not None:
        params["WEIGHT_DECAY"] = float(weight_decay)
    if dropout_rate is not None:
        params["DROPOUT_RATE"] = float(dropout_rate)
    if lr_scheduler is not None:
        params["LR_SCHEDULER"] = str(lr_scheduler)
    if warmup_epochs is not None:
        params["WARMUP_EPOCHS"] = int(warmup_epochs)
    return params


def tune_then_train(
    cfg: Config,
    base_run_id: str,
    model_name: str,
    *,
    do_post: bool = True,
    plot_loss: bool = True,
) -> Dict[str, Any]:
    """Paper-style grid search, then one final full training.

    Grid search dimensions:
      - SEQ_LEN (observation window)
      - BATCH_SIZE
      - Hidden/state size

    Early stopping inside each trial is based on Val RMSE.
    """

    keep_trials = bool(getattr(cfg, "KEEP_TRIAL_DIRS", False))
    tune_limit = int(getattr(cfg, "TUNE_TRIALS", 0))  # <=0 means full grid
    tune_epochs = int(getattr(cfg, "TUNE_MAX_EPOCHS", 30))
    tune_pat = int(getattr(cfg, "TUNE_EARLY_STOP_PATIENCE", 8))

    objective = str(getattr(cfg, "TUNE_OBJECTIVE", "val_nse")).lower()
    obj_mode = "max" if objective == "val_nse" else "min"
    seed0 = int(getattr(cfg, "TUNE_RANDOM_SEED", 2027))

    seq_lens = list(getattr(cfg, "GRID_SEQ_LENS", [cfg.SEQ_LEN]))
    batch_sizes = list(getattr(cfg, "GRID_BATCH_SIZES", [cfg.BATCH_SIZE]))
    hidden_sizes = list(getattr(cfg, "GRID_HIDDEN_SIZES", [cfg.HIDDEN_DIM]))
    lr_grid = list(getattr(cfg, "GRID_LEARNING_RATES", [cfg.LEARNING_RATE]))
    wd_grid = list(getattr(cfg, "GRID_WEIGHT_DECAYS", [cfg.WEIGHT_DECAY]))
    dr_grid = list(getattr(cfg, "GRID_DROPOUT_RATES", [cfg.DROPOUT_RATE]))
    sch_grid = list(getattr(cfg, "GRID_LR_SCHEDULERS", [getattr(cfg, "LR_SCHEDULER", "plateau")]))
    warmup_grid = list(getattr(cfg, "GRID_WARMUP_EPOCHS", [getattr(cfg, "WARMUP_EPOCHS", 3)]))
    search_method = str(getattr(cfg, "TUNE_SEARCH_METHOD", "grid")).lower().strip()
    if search_method in {"optuna", "hyperband"}:
        print(f"[WARN] TUNE_SEARCH_METHOD={search_method} requires optional dependency; fallback to random search over candidate grid.")
        search_method = "random"
    if search_method not in {"grid", "random"}:
        search_method = "grid"
    rng = np.random.default_rng(seed0)

    m_lower = str(model_name).lower()
    if m_lower in {"itransformer", "i_transformer", "i-transformer"}:
        head_grid = list(getattr(cfg, "GRID_ITRANSFORMER_HEADS", [getattr(cfg, "NUM_HEADS", 4)]))
        base_combos = [(s, b, h, nh) for (s, b, h, nh) in itertools.product(seq_lens, batch_sizes, hidden_sizes, head_grid) if int(h) % int(nh) == 0]
    else:
        base_combos = [(s, b, h, None) for (s, b, h, _) in itertools.product(seq_lens, batch_sizes, hidden_sizes, [None])]

    all_params: List[Dict[str, Any]] = []
    for seq_len, bs, h, nh in base_combos:
        for lr, wd, dr, sch, wu in itertools.product(lr_grid, wd_grid, dr_grid, sch_grid, warmup_grid):
            p = _grid_params_for_model(
                model_name,
                int(seq_len),
                int(bs),
                int(h),
                num_heads=(int(nh) if nh is not None else None),
                learning_rate=float(lr),
                weight_decay=float(wd),
                dropout_rate=float(dr),
                lr_scheduler=str(sch),
                warmup_epochs=int(wu),
            )
            all_params.append(p)

    if search_method == "grid":
        trial_params = all_params
    else:
        n_pick = len(all_params) if tune_limit <= 0 else min(int(tune_limit), len(all_params))
        if n_pick <= 0:
            n_pick = len(all_params)
        idx = rng.choice(len(all_params), size=n_pick, replace=False).tolist()
        trial_params = [all_params[int(i)] for i in idx]

    if search_method == "grid" and tune_limit > 0:
        trial_params = trial_params[:tune_limit]

    model_tag = f"{model_name}"
    tune_root = os.path.join(cfg.EXP_ROOT, f"{base_run_id}_{model_tag}_tune")
    ensure_dir(tune_root)

    results: List[Dict[str, Any]] = []
    best_score = float('-inf') if obj_mode == "max" else float('inf')
    best_params: Dict[str, Any] = {}

    print(
        f"\n🔧 开始超参数搜索: model={model_name}, method={search_method}, trials={len(trial_params)}, objective={objective} ({obj_mode})"
    )

    for t, params in enumerate(trial_params):
        trial_cfg = copy.deepcopy(cfg)
        trial_cfg.MODEL_NAME = model_name
        _apply_hparams(trial_cfg, params)

        set_seed(seed0 + t)

        seq_len = int(params.get("SEQ_LEN", cfg.SEQ_LEN))
        bs = int(params.get("BATCH_SIZE", cfg.BATCH_SIZE))
        h = int(params.get("HIDDEN_DIM", params.get("GCN_HIDDEN_DIM", cfg.HIDDEN_DIM)))
        nh = params.get("NUM_HEADS")
        if nh is not None:
            trial_dir = os.path.join(tune_root, f"trial_{t:03d}_S{seq_len}_B{bs}_H{h}_NH{int(nh)}")
        else:
            trial_dir = os.path.join(tune_root, f"trial_{t:03d}_S{seq_len}_B{bs}_H{h}")
        try:
            r = train_run(
                trial_cfg,
                trial_dir,
                objective=objective,
                max_epochs=tune_epochs,
                early_stop_patience=tune_pat,
                do_test=False,
                do_post=False,
                save_checkpoint=False,
                save_artifacts=False,
                plot_loss=False,
            )
            score = float(r.get("best_score", np.nan))
            rec = {
                "trial": int(t),
                "score": score,
                "best_epoch": int(r.get("best_epoch", -1)),
                "best_val_rmse": float(r.get("best_val_rmse", np.nan)),
                "best_val_mse": float(r.get("best_val_mse", np.nan)),
                "best_val_nse": float(r.get("best_val_nse", np.nan)),
                "params": params,
            }
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            rec = {"trial": int(t), "score": None, "error": str(e), "params": params}
        except Exception as e:
            rec = {"trial": int(t), "score": None, "error": str(e), "params": params}

        results.append(rec)

        if rec.get("score") is not None and _is_better(float(rec["score"]), best_score, obj_mode):
            best_score = float(rec["score"])
            best_params = params
            print(f"[OK] trial {t:03d} 刷新最优: score={best_score:.6f}, params={best_params}")

        if (not keep_trials) and os.path.isdir(trial_dir):
            shutil.rmtree(trial_dir, ignore_errors=True)

    grid_meta = {
        "search_method": search_method,
        "GRID_SEQ_LENS": seq_lens,
        "GRID_BATCH_SIZES": batch_sizes,
        "GRID_HIDDEN_SIZES": hidden_sizes,
        "GRID_LEARNING_RATES": lr_grid,
        "GRID_WEIGHT_DECAYS": wd_grid,
        "GRID_DROPOUT_RATES": dr_grid,
        "GRID_LR_SCHEDULERS": sch_grid,
        "GRID_WARMUP_EPOCHS": warmup_grid,
        "num_candidates": len(all_params),
        "num_trials": len(trial_params),
        "limit": tune_limit,
    }
    if m_lower in {"itransformer", "i_transformer", "i-transformer"}:
        grid_meta["GRID_ITRANSFORMER_HEADS"] = list(getattr(cfg, "GRID_ITRANSFORMER_HEADS", []))

    save_json(
        {
            "model": model_name,
            "objective": objective,
            "mode": obj_mode,
            "grid": grid_meta,
            "best_score": best_score,
            "best_params": best_params,
            "trials": results,
        },
        os.path.join(tune_root, "tuning_summary.json"),
    )

    if not best_params:
        # Provide a more actionable message: show the first few errors.
        errs = []
        for r in results:
            e = r.get("error")
            if e:
                errs.append(str(e))
        # Deduplicate while keeping order
        uniq_errs = []
        seen = set()
        for e in errs:
            if e not in seen:
                uniq_errs.append(e)
                seen.add(e)
            if len(uniq_errs) >= 3:
                break
        hint = "" if not uniq_errs else ("\n示例错误（前3条）:\n- " + "\n- ".join(uniq_errs))
        raise RuntimeError(
            f"网格搜索全部失败: model={model_name}，请检查数据/模型或缩小搜索范围" + hint
        )

    # ------------------
    # Final full training
    # ------------------
    final_cfg = copy.deepcopy(cfg)
    final_cfg.MODEL_NAME = model_name
    _apply_hparams(final_cfg, best_params)

    run_id = f"{base_run_id}_{model_tag}"
    if getattr(final_cfg, "RUN_TAG", ""):
        run_id = f"{run_id}_{final_cfg.RUN_TAG}"
    final_dir = os.path.join(final_cfg.EXP_ROOT, run_id)
    ensure_dir(final_dir)

    save_json(asdict(final_cfg), os.path.join(final_dir, "config.json"))
    save_json(best_params, os.path.join(final_dir, "best_hparams.json"))

    print(f"\n🌟 使用最优超参开始最终训练: {run_id}")
    final_res = train_run(
        final_cfg,
        final_dir,
        objective=objective,
        max_epochs=None,
        early_stop_patience=None,
        do_test=True,
        do_post=bool(do_post),
        save_checkpoint=True,
        save_artifacts=True,
        plot_loss=bool(plot_loss),
    )

    final_res["best_hparams"] = best_params
    final_res["tuning_root"] = tune_root
    return final_res


def _parse_models(s: str) -> List[str]:
    raw = [x.strip() for x in s.split(",") if x.strip()]
    out = [_canonical_model_key(x) for x in raw]
    allowed = {
        "stgcn_fusion",
        "stgcn",
        "dcrnn",
        "patchtst",
        "cnn",
        "tcn",
        "lstm",
        "itransformer",
    }
    bad = [m for m in out if m not in allowed]
    if bad:
        raise ValueError(
            f"Unsupported models: {bad}. Allowed models are: {sorted(allowed)}"
        )
    return out


def _canonical_model_key(name: str) -> str:
    """Normalize model name for config lookup."""
    s = str(name).strip().lower()
    alias: Dict[str, str] = {
        "i-transformer": "itransformer",
        "i_transformer": "itransformer",
        "stfusionnet": "stgcn_fusion",
    }
    return alias.get(s, s)


def _is_tunable_model(name: str) -> bool:
    """Only the new model (STFusionNet / STGCN_Fusion) is allowed to run grid search."""
    return _canonical_model_key(name) in {"stgcn_fusion"}


def _apply_model_params(cfg: Config, model_name: str) -> None:
    """Apply per-model parameter overrides from cfg.MODEL_PARAMS if provided."""
    key = _canonical_model_key(model_name)
    mp = getattr(cfg, "MODEL_PARAMS", None)
    if not isinstance(mp, dict):
        return
    overrides = mp.get(key, {})
    if not isinstance(overrides, dict):
        return
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default=None)
    parser.add_argument("--models", type=str, default=None, help="逗号分隔模型名，如 lstm,tcn,cnn")
    parser.add_argument("--tune", action="store_true", help="强制开启自动调参")
    parser.add_argument("--no_tune", action="store_true", help="强制关闭自动调参")
    parser.add_argument(
        "--stf_mode",
        choices=["default", "search"],
        default=None,
        help="仅对 STFusionNet(stgcn_fusion/stfusionnet) 生效：default=不调参直接训练；search=先网格搜索再训练",
    )
    parser.add_argument("--trials", type=int, default=None, help="调参 trials 数")
    parser.add_argument("--objective", choices=["val_nse", "val_rmse", "val_mse"], default=None)
    parser.add_argument("--exp_root", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None, help="给 run_id 加额外后缀")
    parser.add_argument("--load_run_id", type=str, default=None, help="eval 模式下指定 LOAD_RUN_ID")
    parser.add_argument("--top_k_lakes", type=int, default=None, help="仅用前K个站点（加速全流程回归）")
    parser.add_argument("--min_effective_steps", type=int, default=None, help="最小有效步数阈值")
    parser.add_argument("--seq_len", type=int, default=None, help="覆盖配置中的 SEQ_LEN")
    parser.add_argument("--pred_len", type=int, default=None, help="覆盖配置中的 PRED_LEN")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖配置中的 BATCH_SIZE")
    parser.add_argument("--max_epochs", type=int, default=None, help="覆盖配置中的 MAX_EPOCHS")
    parser.add_argument("--no_post", action="store_true", help="禁用训练后处理绘图（timeseries/scatter 等）")
    parser.add_argument("--no_plot_loss", action="store_true", help="禁用 loss_curve.png 绘制")
    return parser.parse_args()

def main():
    configure_stdio_for_server()
    args = parse_args()
    cfg = Config()

    # ----- CLI overrides (optional) -----
    if args.mode is not None:
        cfg.MODE = args.mode
    if args.exp_root is not None:
        cfg.EXP_ROOT = args.exp_root
    if args.tag is not None:
        cfg.RUN_TAG = args.tag
    if args.load_run_id is not None:
        cfg.LOAD_RUN_ID = args.load_run_id
    if args.objective is not None:
        cfg.TUNE_OBJECTIVE = args.objective
    if args.trials is not None:
        cfg.TUNE_TRIALS = int(args.trials)
    if args.tune:
        cfg.AUTO_TUNE = True
    if args.no_tune:
        cfg.AUTO_TUNE = False
    if args.stf_mode is not None:
        # Only affects STFusionNet/stgcn_fusion.
        cfg.STFUSIONNET_TUNE_MODE = str(args.stf_mode)
    if args.top_k_lakes is not None:
        cfg.TOP_K_LAKES = int(args.top_k_lakes)
    if args.min_effective_steps is not None:
        cfg.MIN_EFFECTIVE_STEPS = int(args.min_effective_steps)
    if args.seq_len is not None:
        cfg.SEQ_LEN = int(args.seq_len)
    if args.pred_len is not None:
        cfg.PRED_LEN = int(args.pred_len)
    if args.batch_size is not None:
        cfg.BATCH_SIZE = int(args.batch_size)
    if args.max_epochs is not None:
        cfg.MAX_EPOCHS = int(args.max_epochs)

    ensure_dir(cfg.EXP_ROOT)

    # models to run
    if args.models:
        models = _parse_models(args.models)
    elif getattr(cfg, "RUN_MODELS", None):
        models = [_canonical_model_key(x) for x in list(cfg.RUN_MODELS)]
    else:
        models = [_canonical_model_key(cfg.MODEL_NAME)]

    allowed_models = {
        "stgcn_fusion",
        "stgcn",
        "dcrnn",
        "patchtst",
        "cnn",
        "tcn",
        "lstm",
        "itransformer",
    }
    invalid_models = [m for m in models if m not in allowed_models]
    if invalid_models:
        raise ValueError(
            f"Unsupported models: {invalid_models}. Allowed models are: {sorted(allowed_models)}"
        )

    if cfg.MODE == "train":
        base_run_id = now_str()
        do_post_flag = not bool(args.no_post)
        plot_loss_flag = not bool(args.no_plot_loss)

        all_results: List[Dict[str, Any]] = []
        print(f"\n🧾 本次将顺序运行模型: {models}")
        print(f"🧾 AUTO_TUNE={cfg.AUTO_TUNE}, objective={cfg.TUNE_OBJECTIVE}")

        for m in models:
            cfg_m = copy.deepcopy(cfg)
            cfg_m.MODEL_NAME = m

            # Apply per-model default hyper-parameters (MODEL_PARAMS) if provided.
            # This enables "one-set default params" behavior for each model.
            _apply_model_params(cfg_m, m)
            # Re-apply CLI overrides so smoke/quick settings are not overwritten by MODEL_PARAMS.
            if args.seq_len is not None:
                cfg_m.SEQ_LEN = int(args.seq_len)
            if args.pred_len is not None:
                cfg_m.PRED_LEN = int(args.pred_len)
            if args.batch_size is not None:
                cfg_m.BATCH_SIZE = int(args.batch_size)
            if args.max_epochs is not None:
                cfg_m.MAX_EPOCHS = int(args.max_epochs)

            set_seed(int(getattr(cfg_m, "TUNE_RANDOM_SEED", 2027)))

            # Decide whether to do hyper-parameter search.
            # - For STFusionNet only: cfg_m.STFUSIONNET_TUNE_MODE controls default vs search.
            # - For other models: no tuning (baseline fairness / speed).
            do_tune = bool(cfg_m.AUTO_TUNE) and _is_tunable_model(m)
            if _is_tunable_model(m):
                mode = str(getattr(cfg_m, "STFUSIONNET_TUNE_MODE", "search")).lower()
                if mode == "default":
                    do_tune = False
                elif mode == "search":
                    do_tune = bool(cfg_m.AUTO_TUNE)
                else:
                    raise ValueError(f"Invalid STFUSIONNET_TUNE_MODE: {mode} (expected 'default' or 'search')")

            if do_tune:
                res = tune_then_train(
                    cfg_m,
                    base_run_id,
                    m,
                    do_post=do_post_flag,
                    plot_loss=plot_loss_flag,
                )
            else:
                run_id = f"{base_run_id}_{m}"
                if getattr(cfg_m, "RUN_TAG", ""):
                    run_id = f"{run_id}_{cfg_m.RUN_TAG}"
                run_dir = os.path.join(cfg_m.EXP_ROOT, run_id)
                ensure_dir(run_dir)
                save_json(asdict(cfg_m), os.path.join(run_dir, "config.json"))
                # Explain why no tuning if this is STFusionNet.
                if _is_tunable_model(m):
                    print(f"\n🌟 开始训练(不调参, STFUSIONNET_TUNE_MODE={getattr(cfg_m,'STFUSIONNET_TUNE_MODE','')}): {run_id}")
                else:
                    print(f"\n🌟 开始训练(无调参): {run_id}")
                res = train_run(
                    cfg_m,
                    run_dir,
                    objective=str(getattr(cfg_m, "TUNE_OBJECTIVE", "val_mse")),
                    max_epochs=None,
                    early_stop_patience=None,
                    do_test=True,
                    do_post=do_post_flag,
                    save_checkpoint=True,
                    save_artifacts=True,
                    plot_loss=plot_loss_flag,
                )

            all_results.append({"model": m, **res})

        summary_path = os.path.join(cfg.EXP_ROOT, f"{base_run_id}_summary.json")
        save_json({"base_run_id": base_run_id, "models": models, "results": all_results}, summary_path)
        print(f"\n[OK] 全部模型已运行完毕，汇总已保存: {summary_path}")

    elif cfg.MODE == "eval":
        if models:
            cfg.MODEL_NAME = models[0]
        print(f"🔍 开始评估模式...")
        eval_run(cfg)

if __name__ == "__main__":
    main()
