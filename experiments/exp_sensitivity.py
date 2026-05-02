import argparse
import copy
import os
from dataclasses import asdict
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_taihu import Config
from training.train_main import train_run, tune_then_train, _apply_model_params
from utils.util_common import ensure_dir, now_str, save_json, set_seed, configure_stdio_for_server, collect_runtime_env


def _parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _resample_step_hours(cfg: Config) -> int:
    try:
        hours = pd.Timedelta(str(getattr(cfg, "RESAMPLE_FREQ", "4h"))).total_seconds() / 3600.0
        return max(1, int(round(hours)))
    except Exception:
        return 4


def _parse_horizon_hours(value: str, cfg: Config) -> List[int]:
    raw = [x.strip() for x in str(value).split(",") if x.strip()]
    if not raw:
        raw = [str(x) for x in getattr(cfg, "REPORT_HORIZON_HOURS", [12, 24, 48, 120, 168])]
    step = _resample_step_hours(cfg)
    horizons = [int(float(x)) for x in raw]
    bad = [h for h in horizons if h <= 0 or h % step != 0]
    if bad:
        raise ValueError(f"Horizons must be positive and divisible by {step}h: {bad}")
    return horizons


def _apply_horizon(cfg: Config, horizon_hour: int) -> None:
    pred_len = int(horizon_hour // _resample_step_hours(cfg))
    cfg.PRED_LEN = pred_len
    cfg.HORIZON_MODE = "separate"
    cfg.TARGET_HORIZON_HOURS = int(horizon_hour)
    cfg.REPORT_HORIZON_IDX = int(pred_len - 1)
    cfg.REPORT_HORIZON_HOURS = [int(horizon_hour)]


def _safe_metrics_1d(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return float("nan"), float("nan"), float("nan")
    yt = y_true[m].astype(float)
    yp = y_pred[m].astype(float)
    diff = yt - yp
    mse = float(np.mean(diff * diff))
    rmse = float(np.sqrt(max(mse, 0.0)))
    mae = float(np.mean(np.abs(diff)))
    den = float(np.sum((yt - np.mean(yt)) ** 2))
    nse = float("nan") if den <= 1e-12 else float(1.0 - float(np.sum(diff * diff)) / den)
    return nse, rmse, mae


def _compute_station_feature_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    if yt.shape != yp.shape:
        return rows

    # Graph models: [S, P, N, D]
    if yt.ndim == 4:
        _, _, n_nodes, n_feat = yt.shape
        feat_names = list(feature_names) if len(feature_names) == n_feat else [f"f{i}" for i in range(n_feat)]
        for sid in range(n_nodes):
            for d in range(n_feat):
                nse, rmse, mae = _safe_metrics_1d(yt[:, :, sid, d].reshape(-1), yp[:, :, sid, d].reshape(-1))
                rows.append(
                    {
                        "station_id": int(sid),
                        "feature": str(feat_names[d]),
                        "nse": float(nse),
                        "rmse": float(rmse),
                        "mae": float(mae),
                    }
                )
        return rows

    # Non-graph fallback: [S, P, D] or [S, D] -> treat as single pooled "station".
    if yt.ndim in {2, 3}:
        if yt.ndim == 2:
            yt2 = yt[:, None, :]
            yp2 = yp[:, None, :]
        else:
            yt2 = yt
            yp2 = yp
        n_feat = yt2.shape[-1]
        feat_names = list(feature_names) if len(feature_names) == n_feat else [f"f{i}" for i in range(n_feat)]
        for d in range(n_feat):
            nse, rmse, mae = _safe_metrics_1d(yt2[:, :, d].reshape(-1), yp2[:, :, d].reshape(-1))
            rows.append(
                {
                    "station_id": 0,
                    "feature": str(feat_names[d]),
                    "nse": float(nse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                }
            )
    return rows


def main() -> None:
    configure_stdio_for_server()
    parser = argparse.ArgumentParser(description="KNN graph parameter sensitivity for STFusionNet.")
    parser.add_argument("--k_values", type=str, default="3,6,10,15")
    parser.add_argument("--sigma_values", type=str, default="10,20,30")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--exp_root", type=str, default="")
    parser.add_argument("--max_epochs", type=int, default=-1)
    parser.add_argument("--tag", type=str, default="graph_sens")
    parser.add_argument("--top_k_lakes", type=int, default=-1)
    parser.add_argument("--min_effective_steps", type=int, default=-1)
    parser.add_argument("--seq_len", type=int, default=-1)
    parser.add_argument("--pred_len", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--tune", action="store_true", help="Tune each k/sigma/horizon run before final training.")
    parser.add_argument("--trials", type=int, default=-1, help="Maximum tuning trials per k/sigma/horizon.")
    parser.add_argument("--search_method", choices=["grid", "random"], default="", help="Tuning search method.")
    parser.add_argument("--separate_horizons", action="store_true", help="Run sensitivity independently for each horizon.")
    parser.add_argument("--horizon_hours", type=str, default="12,24,48,120,168", help="Comma-separated horizons in hours.")
    args = parser.parse_args()

    ks = _parse_ints(args.k_values)
    sigmas = _parse_floats(args.sigma_values)

    cfg_base = Config()
    cfg_base.MODEL_NAME = "stgcn_fusion"
    cfg_base.AUTO_TUNE = bool(args.tune)
    cfg_base.STFUSIONNET_TUNE_MODE = "search" if args.tune else "default"
    _apply_model_params(cfg_base, cfg_base.MODEL_NAME)
    if args.exp_root:
        cfg_base.EXP_ROOT = args.exp_root
    if args.max_epochs > 0:
        cfg_base.MAX_EPOCHS = int(args.max_epochs)
    if args.top_k_lakes > 0:
        cfg_base.TOP_K_LAKES = int(args.top_k_lakes)
    if args.min_effective_steps > 0:
        cfg_base.MIN_EFFECTIVE_STEPS = int(args.min_effective_steps)
    if args.seq_len > 0:
        cfg_base.SEQ_LEN = int(args.seq_len)
    if args.pred_len > 0:
        cfg_base.PRED_LEN = int(args.pred_len)
    if args.batch_size > 0:
        cfg_base.BATCH_SIZE = int(args.batch_size)
    if args.trials > 0:
        cfg_base.TUNE_TRIALS = int(args.trials)
    if args.search_method:
        cfg_base.TUNE_SEARCH_METHOD = str(args.search_method)

    root_run_id = f"{now_str()}_{args.tag}"
    root_dir = os.path.join(cfg_base.EXP_ROOT, root_run_id)
    ensure_dir(root_dir)
    save_json(collect_runtime_env(), os.path.join(root_dir, "runtime_env.json"))
    save_json(
        {
            "base_config": asdict(cfg_base),
            "k_values": ks,
            "sigma_values": sigmas,
            "seed": int(args.seed),
            "tune": bool(args.tune),
            "trials": int(getattr(cfg_base, "TUNE_TRIALS", 0)),
            "search_method": str(getattr(cfg_base, "TUNE_SEARCH_METHOD", "grid")),
            "horizons": _parse_horizon_hours(args.horizon_hours, cfg_base) if args.separate_horizons else [],
        },
        os.path.join(root_dir, "plan.json"),
    )

    rows: List[Dict[str, Any]] = []
    rows_station_feature: List[Dict[str, Any]] = []
    rows_feature_agg: List[Dict[str, Any]] = []
    trial_id = 0
    horizons = _parse_horizon_hours(args.horizon_hours, cfg_base) if args.separate_horizons else [None]
    for horizon_hour in horizons:
        for k in ks:
            for sigma in sigmas:
                trial_id += 1
                cfg = copy.deepcopy(cfg_base)
                cfg.KNN_K = int(k)
                cfg.KNN_SIGMA_KM = float(sigma)
                cfg.EXP_ROOT = root_dir
                sens_id = f"k{k}_s{sigma:g}"
                if horizon_hour is not None:
                    _apply_horizon(cfg, int(horizon_hour))
                    sens_id = f"h{int(horizon_hour)}h_k{k}_s{sigma:g}"
                cfg.RUN_TAG = sens_id

                # Keep child run directories compact. Repeating root_run_id in
                # nested Windows paths can exceed the 260-character path limit
                # during torch.save, while adding no useful information.
                run_id = sens_id
                run_dir = os.path.join(cfg.EXP_ROOT, run_id)

                set_seed(args.seed + trial_id)
                if args.tune:
                    cfg.RUN_TAG = ""
                    res = tune_then_train(
                        cfg,
                        sens_id,
                        "stgcn_fusion",
                        do_post=False,
                        plot_loss=False,
                    )
                    run_dir = str(res.get("run_dir", run_dir))
                else:
                    ensure_dir(run_dir)
                    save_json(asdict(cfg), os.path.join(run_dir, "config.json"))
                    res = train_run(
                        cfg,
                        run_dir,
                        objective=str(getattr(cfg, "TUNE_OBJECTIVE", "val_nse")),
                        max_epochs=None,
                        early_stop_patience=None,
                        do_test=True,
                        do_post=False,
                        save_checkpoint=True,
                        save_artifacts=True,
                        plot_loss=False,
                    )

                test_m = res.get("test_metrics", {})
                rows.append(
                    {
                        "k": int(k),
                        "sigma_km": float(sigma),
                        "horizon_hours": (int(horizon_hour) if horizon_hour is not None else None),
                        "pred_len": int(getattr(cfg, "PRED_LEN", -1)),
                        "best_epoch_rmse": int(res.get("best_epoch", -1)),
                        "best_val_rmse": float(res.get("best_val_rmse", float("nan"))),
                        "best_val_nse": float(res.get("best_val_nse", float("nan"))),
                        "test_rmse": float(test_m.get("rmse", float("nan"))),
                        "test_mae": float(test_m.get("mae", float("nan"))),
                        "test_nse": float(test_m.get("nse", float("nan"))),
                    }
                )

                npz_path = os.path.join(run_dir, "test_outputs.npz")
                analysis_npz_path = os.path.join(run_dir, "analysis_data.npz")
                if os.path.exists(npz_path) or os.path.exists(analysis_npz_path):
                    try:
                        pack = np.load(npz_path if os.path.exists(npz_path) else analysis_npz_path, allow_pickle=True)
                        y_true_np = np.asarray(pack["y_true"])
                        y_pred_np = np.asarray(pack["y_pred"])
                        sf_rows = _compute_station_feature_rows(y_true_np, y_pred_np, list(cfg.TARGET_FEATURES))
                        for r in sf_rows:
                            rr = dict(r)
                            rr["k"] = int(k)
                            rr["sigma_km"] = float(sigma)
                            rr["horizon_hours"] = (int(horizon_hour) if horizon_hour is not None else None)
                            rr["pred_len"] = int(getattr(cfg, "PRED_LEN", -1))
                            rows_station_feature.append(rr)
                    except Exception as e:
                        print(f"[WARN] failed to compute station-feature metrics from {run_dir}: {e}")

    if rows_station_feature:
        df_sf = pd.DataFrame(rows_station_feature)
        df_sf.to_csv(
            os.path.join(root_dir, "graph_sensitivity_station_feature_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        group_cols = ["k", "sigma_km", "feature"]
        if "horizon_hours" in df_sf.columns and df_sf["horizon_hours"].notna().any():
            group_cols = ["horizon_hours", "pred_len"] + group_cols
        grp = (
            df_sf.groupby(group_cols, as_index=False)
            .agg(
                nse_mean_across_stations=("nse", "mean"),
                nse_std_across_stations=("nse", "std"),
                rmse_mean_across_stations=("rmse", "mean"),
                rmse_std_across_stations=("rmse", "std"),
                mae_mean_across_stations=("mae", "mean"),
                mae_std_across_stations=("mae", "std"),
                n_stations=("station_id", "nunique"),
            )
            .sort_values(group_cols)
        )
        rows_feature_agg = grp.to_dict(orient="records")
        grp.to_csv(
            os.path.join(root_dir, "graph_sensitivity_feature_station_mean.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        save_json(
            {
                "description": "Per-feature metrics aggregated across stations for each horizon and (k, sigma).",
                "rows": rows_feature_agg,
            },
            os.path.join(root_dir, "graph_sensitivity_feature_station_mean.json"),
        )

    sort_cols = ["test_nse", "test_rmse"]
    sort_asc = [False, True]
    if rows and any(r.get("horizon_hours") is not None for r in rows):
        sort_cols = ["horizon_hours"] + sort_cols
        sort_asc = [True] + sort_asc
    df = pd.DataFrame(rows).sort_values(sort_cols, ascending=sort_asc)
    csv_path = os.path.join(root_dir, "graph_sensitivity_summary.csv")
    json_path = os.path.join(root_dir, "graph_sensitivity_summary.json")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_json(
        {
            "rows": rows,
            "best": (df.iloc[0].to_dict() if len(df) > 0 else None),
            "feature_station_mean_rows": rows_feature_agg,
        },
        json_path,
    )

    # Compact visual evidence for rebuttal: 2D heatmaps over (k, sigma)
    if len(df) > 0:
        try:
            if "horizon_hours" in df.columns and df["horizon_hours"].notna().any():
                plot_df = df[df["horizon_hours"] == df["horizon_hours"].min()].copy()
                heat_suffix = f"_h{int(plot_df['horizon_hours'].iloc[0])}h"
                title_suffix = f" ({int(plot_df['horizon_hours'].iloc[0])}h)"
            else:
                plot_df = df.copy()
                heat_suffix = ""
                title_suffix = ""
            piv_nse = plot_df.pivot(index="k", columns="sigma_km", values="test_nse").sort_index().sort_index(axis=1)
            piv_rmse = plot_df.pivot(index="k", columns="sigma_km", values="test_rmse").sort_index().sort_index(axis=1)

            plt.rcParams.update(
                {
                    "font.family": "Arial",
                    "axes.unicode_minus": False,
                    "figure.facecolor": "#ffffff",
                    "axes.facecolor": "#ffffff",
                    "axes.edgecolor": "#222222",
                    "axes.linewidth": 1.0,
                    "grid.color": "#c9c9c9",
                    "grid.alpha": 0.35,
                    "grid.linestyle": "--",
                }
            )
            fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
            im1 = axes[0].imshow(piv_nse.values, aspect="auto", origin="lower", cmap="YlGnBu")
            axes[0].set_title(f"Test NSE{title_suffix}", fontsize=14, fontweight="bold", pad=8)
            axes[0].set_xlabel("sigma (km)", fontsize=12)
            axes[0].set_ylabel("k", fontsize=12)
            axes[0].set_xticks(range(len(piv_nse.columns)))
            axes[0].set_xticklabels([str(x) for x in piv_nse.columns])
            axes[0].set_yticks(range(len(piv_nse.index)))
            axes[0].set_yticklabels([str(x) for x in piv_nse.index])
            fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

            im2 = axes[1].imshow(piv_rmse.values, aspect="auto", origin="lower", cmap="YlOrRd_r")
            axes[1].set_title(f"Test RMSE{title_suffix}", fontsize=14, fontweight="bold", pad=8)
            axes[1].set_xlabel("sigma (km)", fontsize=12)
            axes[1].set_ylabel("k", fontsize=12)
            axes[1].set_xticks(range(len(piv_rmse.columns)))
            axes[1].set_xticklabels([str(x) for x in piv_rmse.columns])
            axes[1].set_yticks(range(len(piv_rmse.index)))
            axes[1].set_yticklabels([str(x) for x in piv_rmse.index])
            fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

            fig.subplots_adjust(left=0.07, right=0.97, bottom=0.14, top=0.90, wspace=0.24)
            heat_path = os.path.join(root_dir, f"graph_sensitivity_heatmaps{heat_suffix}.png")
            fig.savefig(heat_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"Saved sensitivity heatmaps: {heat_path}")
        except Exception as e:
            print(f"[WARN] failed to draw sensitivity heatmaps: {e}")

    print(f"Saved sensitivity summary: {csv_path}")


if __name__ == "__main__":
    main()
