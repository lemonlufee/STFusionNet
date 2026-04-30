# evaluate.py
import os
import torch
import numpy as np
import math
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
import matplotlib as mpl
import matplotlib.dates as mdates
try:
    import shap  # optional (used only in SHAP analysis functions)
except Exception:  # pragma: no cover
    shap = None
import pandas as pd
import seaborn as sns
from scipy import stats

# --- Matplotlib font fallback for Chinese labels ---
# If your system doesn't have SimHei/Microsoft YaHei, Matplotlib will fall back
# to DejaVu Sans. This prevents Chinese text from becoming garbled in figures.
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
try:
    from captum.attr import IntegratedGradients  # optional
except Exception:  # pragma: no cover
    IntegratedGradients = None
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.util_common import ensure_dir, configure_stdio_for_server
from typing import Any, Dict, List, Optional, Sequence, Tuple

from matplotlib import font_manager as fm

def set_chinese_font():
    candidates = [
        "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC",
        "Source Han Sans SC", "WenQuanYi Zen Hei"
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return True

    # If no Chinese font is available, do not force one to avoid garbled squares.
    plt.rcParams["axes.unicode_minus"] = False
    return False

set_chinese_font()
configure_stdio_for_server()


def apply_paper_plot_style() -> None:
    """Use the same publication-style defaults as the main figure suite."""
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
            "savefig.facecolor": "#ffffff",
        }
    )


def calculate_nse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]
    nse_list = []
    for k in range(y_true.shape[1]):
        yt = y_true[:, k]
        yp = y_pred[:, k]
        num = np.sum((yt - yp) ** 2)
        den = np.sum((yt - np.mean(yt)) ** 2) + 1e-12
        nse_list.append(1 - num / den)
    return float(np.mean(nse_list))


def _nse_1d(yt: np.ndarray, yp: np.ndarray) -> float:
    """NSE for 1-D arrays."""
    yt = np.asarray(yt, dtype=np.float64).reshape(-1)
    yp = np.asarray(yp, dtype=np.float64).reshape(-1)
    if yt.size == 0:
        return float("nan")
    num = np.sum((yt - yp) ** 2)
    den = np.sum((yt - np.mean(yt)) ** 2)
    if not np.isfinite(num) or not np.isfinite(den) or den < 1e-12:
        return float("nan")
    return float(1.0 - num / (den + 1e-12))


def _nse_parts_1d(yt: np.ndarray, yp: np.ndarray) -> tuple[float, float, float]:
    """Return (nse, sse, sst) for 1-D arrays.

    - nse: Nash-Sutcliffe Efficiency
    - sse: sum of squared errors
    - sst: sum of squared deviations from mean (i.e., total variance * n)
    """
    yt = np.asarray(yt, dtype=np.float64).reshape(-1)
    yp = np.asarray(yp, dtype=np.float64).reshape(-1)
    if yt.size == 0:
        return float("nan"), float("nan"), float("nan")
    sse = float(np.sum((yt - yp) ** 2))
    mu = float(np.mean(yt))
    sst = float(np.sum((yt - mu) ** 2))
    if (not np.isfinite(sse)) or (not np.isfinite(sst)) or sst < 1e-12:
        return float("nan"), sse, sst
    nse = float(1.0 - sse / (sst + 1e-12))
    return nse, sse, sst


def grouped_nse_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    station_ids: Optional[np.ndarray] = None,
    feature_names: Optional[list] = None,
    var_eps: float = 1e-4,
) -> dict:
    """Compute NSE per (station, horizon, feature) then report mean/std.

    Supported shapes:
      - Graph/ST:     y = [S, P, N, D]
      - Non-graph:    y = [S, P, D] + station_ids [S]

    Returns a compact summary (mean/std) plus per-horizon/per-feature stats.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    if yt.ndim == 4:
        S, P, N, D = yt.shape
        nse_grid = np.full((N, P, D), np.nan, dtype=np.float64)
        var_grid = np.full((N, P, D), np.nan, dtype=np.float64)
        for n in range(N):
            for p in range(P):
                for d in range(D):
                    nse_v, _sse, _sst = _nse_parts_1d(yt[:, p, n, d], yp[:, p, n, d])
                    nse_grid[n, p, d] = nse_v
                    var_grid[n, p, d] = (float(_sst) / float(max(S, 1))) if np.isfinite(_sst) else float('nan')

        vals = nse_grid[np.isfinite(nse_grid)]
        mean = float(np.nanmean(vals)) if vals.size else float("nan")
        std = float(np.nanstd(vals)) if vals.size else float("nan")

        # Filter out low-variance groups (per-group variance < var_eps)
        finite_mask = np.isfinite(nse_grid) & np.isfinite(var_grid)
        raw_count = int(np.sum(np.isfinite(nse_grid)))
        keep_mask = finite_mask & (var_grid >= float(var_eps))
        vals_f = nse_grid[keep_mask]
        mean_f = float(np.nanmean(vals_f)) if vals_f.size else float('nan')
        std_f = float(np.nanstd(vals_f)) if vals_f.size else float('nan')
        kept = int(vals_f.size)

        by_h_mean = []
        by_h_std = []
        for p in range(P):
            v = nse_grid[:, p, :].reshape(-1)
            v = v[np.isfinite(v)]
            by_h_mean.append(float(np.nanmean(v)) if v.size else float("nan"))
            by_h_std.append(float(np.nanstd(v)) if v.size else float("nan"))

        # per-feature stats
        by_f_mean = {}
        by_f_std = {}
        for d in range(D):
            v = nse_grid[:, :, d].reshape(-1)
            v = v[np.isfinite(v)]
            key = feature_names[d] if feature_names and d < len(feature_names) else f"feat_{d}"
            by_f_mean[key] = float(np.nanmean(v)) if v.size else float("nan")
            by_f_std[key] = float(np.nanstd(v)) if v.size else float("nan")

        # per-station mean
        by_s_mean = {str(n): float(np.nanmean(nse_grid[n].reshape(-1))) for n in range(N)}

        return {
            "nse_mean": mean,
            "nse_std": std,
            "nse_filt_mean": mean_f,
            "nse_filt_std": std_f,
            "nse_filt_kept": kept,
            "nse_filt_total": raw_count,
            "nse_var_eps": float(var_eps),
            # Full grid: [station, horizon, feature]
            "nse_by_station_horizon_feature": nse_grid.tolist(),
            "station_ids_order": [int(i) for i in range(N)],
            "nse_by_horizon_mean": by_h_mean,
            "nse_by_horizon_std": by_h_std,
            "nse_by_feature_mean": by_f_mean,
            "nse_by_feature_std": by_f_std,
            "nse_by_station_mean": by_s_mean,
        }

    if yt.ndim == 3:
        S, P, D = yt.shape
        # if station_ids not provided, treat as one group
        if station_ids is None:
            station_ids = np.zeros((S,), dtype=np.int64)
        station_ids = np.asarray(station_ids).reshape(-1)
        if station_ids.shape[0] != S:
            raise ValueError(f"station_ids length mismatch: {station_ids.shape[0]} vs S={S}")

        uniq = np.unique(station_ids)
        nse_grid = np.full((len(uniq), P, D), np.nan, dtype=np.float64)
        var_grid = np.full((len(uniq), P, D), np.nan, dtype=np.float64)
        for si, sid in enumerate(uniq):
            idx = np.where(station_ids == sid)[0]
            if idx.size == 0:
                continue
            for p in range(P):
                for d in range(D):
                    nse_v, _sse, _sst = _nse_parts_1d(yt[idx, p, d], yp[idx, p, d])
                    nse_grid[si, p, d] = nse_v
                    var_grid[si, p, d] = (float(_sst) / float(max(idx.size, 1))) if np.isfinite(_sst) else float('nan')

        vals = nse_grid[np.isfinite(nse_grid)]
        mean = float(np.nanmean(vals)) if vals.size else float("nan")
        std = float(np.nanstd(vals)) if vals.size else float("nan")


        # Filter out low-variance groups (per-group variance < var_eps)
        finite_mask = np.isfinite(nse_grid) & np.isfinite(var_grid)
        raw_count = int(np.sum(np.isfinite(nse_grid)))
        keep_mask = finite_mask & (var_grid >= float(var_eps))
        vals_f = nse_grid[keep_mask]
        mean_f = float(np.nanmean(vals_f)) if vals_f.size else float("nan")
        std_f = float(np.nanstd(vals_f)) if vals_f.size else float("nan")
        kept = int(vals_f.size)

        by_h_mean = []
        by_h_std = []
        for p in range(P):
            v = nse_grid[:, p, :].reshape(-1)
            v = v[np.isfinite(v)]
            by_h_mean.append(float(np.nanmean(v)) if v.size else float("nan"))
            by_h_std.append(float(np.nanstd(v)) if v.size else float("nan"))

        by_f_mean = {}
        by_f_std = {}
        for d in range(D):
            v = nse_grid[:, :, d].reshape(-1)
            v = v[np.isfinite(v)]
            key = feature_names[d] if feature_names and d < len(feature_names) else f"feat_{d}"
            by_f_mean[key] = float(np.nanmean(v)) if v.size else float("nan")
            by_f_std[key] = float(np.nanstd(v)) if v.size else float("nan")

        by_s_mean = {str(int(sid)): float(np.nanmean(nse_grid[si].reshape(-1))) for si, sid in enumerate(uniq)}

        return {
            "nse_mean": mean,
            "nse_std": std,
            "nse_filt_mean": mean_f,
            "nse_filt_std": std_f,
            "nse_filt_kept": kept,
            "nse_filt_total": raw_count,
            "nse_var_eps": float(var_eps),
            # Full grid: [station, horizon, feature]
            "nse_by_station_horizon_feature": nse_grid.tolist(),
            "station_ids_order": [int(s) for s in uniq.tolist()],
            "nse_by_horizon_mean": by_h_mean,
            "nse_by_horizon_std": by_h_std,
            "nse_by_feature_mean": by_f_mean,
            "nse_by_feature_std": by_f_std,
            "nse_by_station_mean": by_s_mean,
        }

    # Fallback: use flattened NSE
    yt2, yp2 = _flatten_for_metrics(yt, yp)
    return {
        "nse_mean": calculate_nse(yt2, yp2),
        "nse_std": float("nan"),
        "nse_by_horizon_mean": [],
        "nse_by_horizon_std": [],
        "nse_by_feature_mean": {},
        "nse_by_feature_std": {},
        "nse_by_station_mean": {},
    }




def _flatten_for_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Flatten (possibly) multi-step / graph outputs into 2D for metrics.

    Supported shapes:
      - non-graph: [S, P, D] -> [S*P, D]
      - graph:     [S, P, N, D] -> [S*P*N, D]
      - legacy:    [S, D] -> [S, D]
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim == 4:  # [S,P,N,D]
        S, P, N, D = yt.shape
        return yt.reshape(S * P * N, D), yp.reshape(S * P * N, D)
    if yt.ndim == 3:  # [S,P,D]
        S, P, D = yt.shape
        return yt.reshape(S * P, D), yp.reshape(S * P, D)
    if yt.ndim == 2:  # [S,D]
        return yt, yp
    if yt.ndim == 1:
        return yt[:, None], yp[:, None]
    # fallback
    D = yt.shape[-1]
    return yt.reshape(-1, D), yp.reshape(-1, D)



def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute scalar regression metrics on flattened arrays.

    Supports inputs of any shape as long as y_true and y_pred are broadcast-compatible.
    NaN/Inf values are ignored. If the variance of y_true is ~0, NSE is set to np.nan.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    # broadcast then flatten
    try:
        yt_b, yp_b = np.broadcast_arrays(yt, yp)
    except ValueError:
        yt_b = yt
        yp_b = yp
    yt_f = yt_b.reshape(-1)
    yp_f = yp_b.reshape(-1)

    mask = np.isfinite(yt_f) & np.isfinite(yp_f)
    yt_f = yt_f[mask]
    yp_f = yp_f[mask]
    if yt_f.size == 0:
        return {
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "nse": float("nan"),
            "count": 0.0,
        }

    err = yt_f - yp_f
    mse = float(np.mean(err * err))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    # R2: fallback to nan if undefined (less than 2 samples)
    if yt_f.size < 2:
        r2 = float("nan")
    else:
        # r2_score can throw if all yt are constant; we handle that.
        try:
            r2 = float(r2_score(yt_f, yp_f))
        except Exception:
            r2 = float("nan")

    # NSE = 1 - SSE / SST
    yt_mean = float(np.mean(yt_f))
    sse = float(np.sum((yt_f - yp_f) ** 2))
    sst = float(np.sum((yt_f - yt_mean) ** 2))
    nse = float("nan") if sst <= 1e-12 else float(1.0 - sse / sst)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "nse": nse,
        "count": float(yt_f.size),
    }

def inverse_transform_lastdim(scaler: Any, arr: np.ndarray) -> np.ndarray:
    """Inverse-transform an array whose last dimension is the feature dimension.

    sklearn scalers accept 2-D arrays only. This helper reshapes arbitrary
    tensors (e.g., [S,D], [S,P,D], [S,P,N,D]) to 2-D, applies inverse_transform,
    and then reshapes back.
    """
    if scaler is None:
        return np.asarray(arr)
    x = np.asarray(arr)
    if x.ndim == 0:
        return x
    if x.ndim == 1:
        x2 = x.reshape(-1, 1)
        inv = scaler.inverse_transform(x2)
        return np.asarray(inv).reshape(-1)
    d = int(x.shape[-1])
    flat = x.reshape(-1, d)
    inv2 = scaler.inverse_transform(flat)
    return np.asarray(inv2).reshape(x.shape)


def compute_per_feature_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute RMSE/MAE/MSE/NSE/R2 for each feature (water-quality variable)."""
    yt2, yp2 = _flatten_for_metrics(np.asarray(y_true), np.asarray(y_pred))
    d = int(yt2.shape[1])
    if feature_names is None:
        names: List[str] = [f"feat_{i}" for i in range(d)]
    else:
        names = list(feature_names)
        if len(names) != d:
            # Keep it safe (avoid index errors)
            names = names[:d] + [f"feat_{i}" for i in range(len(names), d)]

    out: Dict[str, Dict[str, float]] = {}
    for i, nm in enumerate(names):
        t = yt2[:, i]
        p = yp2[:, i]
        mask = np.isfinite(t) & np.isfinite(p)
        if int(mask.sum()) <= 1:
            out[nm] = {
                "mse": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "nse": float("nan"),
                "r2": float("nan"),
                "count": float(mask.sum()),
            }
            continue
        tt = t[mask]
        pp = p[mask]
        diff = tt - pp
        mse = float(np.mean(diff * diff))
        rmse = float(np.sqrt(max(mse, 0.0)))
        mae = float(np.mean(np.abs(diff)))
        denom = float(np.sum((tt - float(np.mean(tt))) ** 2))
        nse = float("nan") if denom <= 0 else float(1.0 - float(np.sum(diff * diff)) / denom)
        # r2_score expects 1-D
        r2 = float(r2_score(tt, pp))
        out[nm] = {"mse": mse, "rmse": rmse, "mae": mae, "nse": nse, "r2": r2, "count": float(mask.sum())}
    return out



def compute_metrics_per_feature(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Backward-compatible alias for per-feature metric computation.

    This project historically imported `compute_metrics_per_feature` from evaluate.py.
    The implementation lives in `compute_per_feature_metrics` to keep naming explicit.
    """
    return compute_per_feature_metrics(y_true, y_pred, feature_names)

def plot_timeseries_best_station(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    features: Sequence[str],
    out_dir: str,
    scaler_y: Optional[Any] = None,
    test_times: Optional[np.ndarray] = None,
    station_ids: Optional[np.ndarray] = None,
    station_names: Optional[Sequence[str]] = None,
    node_ids: Optional[Sequence[Any]] = None,
    max_points: Optional[int] = None,
    horizon_idx: int = 0,
    save_individual: bool = False,
) -> None:
    """Plot one 2x3 time-series panel for all target features.

    - English labels only
    - Uses full available x-range by default (max_points=None)
    - Always outputs panel figures only (single-figure export disabled)
    """
    os.makedirs(out_dir, exist_ok=True)

    y_true_real = inverse_transform_lastdim(scaler_y, y_true) if scaler_y is not None else y_true
    y_pred_real = inverse_transform_lastdim(scaler_y, y_pred) if scaler_y is not None else y_pred

    if test_times is None:
        test_times = np.arange(y_true_real.shape[0], dtype=float)
    test_times = np.asarray(test_times)

    def _score_series(arr: np.ndarray) -> Tuple[int, float]:
        finite = np.isfinite(arr)
        cnt = int(finite.sum())
        if cnt <= 1:
            return cnt, 0.0
        if arr.ndim == 1:
            return cnt, float(np.nanstd(arr))
        stds = np.nanstd(arr, axis=0)
        std_mean = float(np.nanmean(stds)) if np.isfinite(stds).any() else 0.0
        return cnt, std_mean

    graph_mode = (y_true_real.ndim == 4)  # [S,H,N,D]
    best_node = 0
    best_node_name = "0"
    mask_station: Optional[np.ndarray] = None

    if graph_mode:
        if y_true_real.shape[1] <= horizon_idx:
            raise ValueError(f"horizon_idx={horizon_idx} out of range: y_true.shape[1]={y_true_real.shape[1]}")
        y0_all = y_true_real[:, horizon_idx, :, :]  # [S,N,D]
        n_nodes = y0_all.shape[1]
        best = (-1, -1.0)
        for n in range(n_nodes):
            score = _score_series(y0_all[:, n, :])
            if score[0] > best[0] or (score[0] == best[0] and score[1] > best[1]):
                best = score
                best_node = n
        if node_ids is not None and best_node < len(node_ids):
            best_node_name = str(node_ids[best_node])
        elif station_names is not None and best_node < len(station_names):
            best_node_name = str(station_names[best_node])
        else:
            best_node_name = str(best_node)
    else:
        if station_ids is not None:
            station_ids = np.asarray(station_ids).astype(int)
            uniq = np.unique(station_ids)
            best = (-1, -1.0)
            for sid in uniq:
                m = station_ids == sid
                if y_true_real.ndim == 3:
                    if y_true_real.shape[1] <= horizon_idx:
                        raise ValueError(f"horizon_idx={horizon_idx} out of range: y_true.shape[1]={y_true_real.shape[1]}")
                    yt = y_true_real[m, horizon_idx, :]
                elif y_true_real.ndim == 2:
                    yt = y_true_real[m, :]
                else:
                    yt = y_true_real[m]
                score = _score_series(yt)
                if score[0] > best[0] or (score[0] == best[0] and score[1] > best[1]):
                    best = score
                    best_node = int(sid)
                    mask_station = m
            if station_names is not None and best_node < len(station_names):
                best_node_name = str(station_names[best_node])
            else:
                best_node_name = str(best_node)

    def _sanitize_times_for_mask(times_arr: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        t = np.asarray(times_arr)
        if t.ndim != 1:
            t = t.reshape(-1)
        if t.shape[0] != n:
            return np.arange(n, dtype=float), np.ones(n, dtype=bool)
        if np.issubdtype(t.dtype, np.number):
            return t, np.isfinite(t)
        if np.issubdtype(t.dtype, np.datetime64):
            return t, ~np.isnat(t)
        t_dt = pd.to_datetime(t, errors="coerce").to_numpy()
        m = ~pd.isna(t_dt)
        if int(m.sum()) == 0:
            return np.arange(n, dtype=float), np.ones(n, dtype=bool)
        return t_dt, np.asarray(m, dtype=bool)

    series_payload: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]] = []

    for f_idx, feat in enumerate(features):
        if graph_mode:
            true_vals = y_true_real[:, horizon_idx, best_node, f_idx]
            pred_vals = y_pred_real[:, horizon_idx, best_node, f_idx]
            times_s = test_times
        else:
            if mask_station is not None:
                if y_true_real.ndim == 3:
                    true_vals = y_true_real[mask_station, horizon_idx, f_idx]
                    pred_vals = y_pred_real[mask_station, horizon_idx, f_idx]
                elif y_true_real.ndim == 2:
                    true_vals = y_true_real[mask_station, f_idx]
                    pred_vals = y_pred_real[mask_station, f_idx]
                else:
                    true_vals = y_true_real[mask_station]
                    pred_vals = y_pred_real[mask_station]
                times_s = test_times[mask_station]
            else:
                if y_true_real.ndim == 3:
                    true_vals = y_true_real[:, horizon_idx, f_idx]
                    pred_vals = y_pred_real[:, horizon_idx, f_idx]
                elif y_true_real.ndim == 2:
                    true_vals = y_true_real[:, f_idx]
                    pred_vals = y_pred_real[:, f_idx]
                else:
                    true_vals = y_true_real
                    pred_vals = y_pred_real
                times_s = test_times

        n_now = int(np.asarray(true_vals).shape[0])
        times_plot, time_mask = _sanitize_times_for_mask(times_s, n_now)
        mfin = np.isfinite(true_vals) & np.isfinite(pred_vals) & time_mask
        true_vals = np.asarray(true_vals)[mfin]
        pred_vals = np.asarray(pred_vals)[mfin]
        times_s = np.asarray(times_plot)[mfin]
        if true_vals.size == 0:
            continue

        order = np.argsort(times_s)
        times_s = times_s[order]
        true_vals = true_vals[order]
        pred_vals = pred_vals[order]

        if max_points is not None and int(max_points) > 0:
            l = min(int(max_points), true_vals.shape[0])
            times_s = times_s[:l]
            true_vals = true_vals[:l]
            pred_vals = pred_vals[:l]

        m = compute_metrics(true_vals, pred_vals)
        series_payload.append((str(feat), times_s, true_vals, pred_vals, m))

        _ = save_individual  # kept for backward-compatible signature; no single-figure output

    if len(series_payload) == 0:
        return

    apply_paper_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(13.1, 12.2), facecolor="white")
    flat_axes = axes.ravel()
    for i in range(6):
        ax = flat_axes[i]
        if i < len(series_payload):
            feat, t, y_t, y_p, m = series_payload[i]
            ax.plot(t, y_p, label="Prediction", linewidth=1.35, color="#2d87c8", zorder=2)
            ax.scatter(t, y_t, label="Observed", s=9, alpha=0.56, color="#cf4e62", edgecolors="none", zorder=4)
            ax.text(
                -0.105,
                1.045,
                f"({chr(ord('a') + i)})",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                clip_on=False,
            )
            ax.set_title(feat, fontsize=16, fontweight="bold", loc="left")
            ax.set_xlabel("Date", fontsize=12.5)
            ax.set_ylabel("Value", fontsize=12.5)
            ax.grid(True, linestyle="--", alpha=0.22)
            ax.set_xlim(t.min(), t.max())
            txt = (
                f"NSE={m.get('nse', float('nan')):.3f}\n"
                f"RMSE={m.get('rmse', float('nan')):.3g}\n"
                f"MAE={m.get('mae', float('nan')):.3g}"
            )
            ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
            ax.legend(loc="upper center", bbox_to_anchor=(0.54, 1.14), fontsize=10.5, ncol=2, frameon=False, borderaxespad=0.0)
        else:
            ax.axis("off")

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.07, top=0.94, wspace=0.22, hspace=0.38)
    fig.savefig(os.path.join(out_dir, "timeseries_panel_2x3.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_log_scatter_per_feature(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    features: Sequence[str],
    out_dir: str,
    scaler_y: Optional[Any] = None,
    prefix: str = "test",
    log_axes: bool = True,
    bins: int = 180,
    max_points: int = 200000,
    seed: int = 0,
    save_individual: bool = False,
) -> None:
    """Plot one 2x3 measured-vs-predicted density panel for all target features."""
    os.makedirs(out_dir, exist_ok=True)

    y_true_real = inverse_transform_lastdim(scaler_y, y_true) if scaler_y is not None else y_true
    y_pred_real = inverse_transform_lastdim(scaler_y, y_pred) if scaler_y is not None else y_pred

    if y_true_real.ndim in (3, 4):
        flat_true = y_true_real.reshape(-1, y_true_real.shape[-1])
        flat_pred = y_pred_real.reshape(-1, y_pred_real.shape[-1])
    elif y_true_real.ndim == 2:
        flat_true = y_true_real
        flat_pred = y_pred_real
    else:
        raise ValueError(f"Unsupported y shape: {y_true_real.shape} / {y_pred_real.shape}")

    rng = np.random.default_rng(seed)

    payload = []
    for i, feat in enumerate(features):
        t = flat_true[:, i].astype(float)
        p = flat_pred[:, i].astype(float)
        mask = np.isfinite(t) & np.isfinite(p)
        if log_axes:
            mask &= (t > 0.0) & (p > 0.0)
        t = t[mask]
        p = p[mask]
        if t.size < 5:
            continue
        if t.size > max_points:
            idx = rng.choice(t.size, size=max_points, replace=False)
            t = t[idx]
            p = p[idx]

        xt = np.log10(t) if log_axes else t
        yt = np.log10(p) if log_axes else p

        H, xedges, yedges = np.histogram2d(xt, yt, bins=bins)
        xi = np.searchsorted(xedges, xt, side="right") - 1
        yi = np.searchsorted(yedges, yt, side="right") - 1
        xi = np.clip(xi, 0, H.shape[0] - 1)
        yi = np.clip(yi, 0, H.shape[1] - 1)
        dens = np.maximum(H[xi, yi], 1.0)

        m = compute_metrics(t, p)
        payload.append((str(feat), t, p, dens, m))

        _ = save_individual  # kept for backward-compatible signature; no single-figure output

    if len(payload) == 0:
        return

    apply_paper_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.9), facecolor="white")
    flat_axes = axes.ravel()
    cmap_name = "turbo" if "turbo" in plt.colormaps() else "viridis"

    for i in range(6):
        ax = flat_axes[i]
        if i < len(payload):
            feat, t, p, dens, m = payload[i]
            sc = ax.scatter(t, p, c=dens, s=8, cmap=cmap_name, norm=LogNorm(vmin=1, vmax=float(np.max(dens))), linewidths=0.0)
            lo = float(np.nanpercentile(np.concatenate([t, p]), 0.5))
            hi = float(np.nanpercentile(np.concatenate([t, p]), 99.5))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo = float(np.min(np.concatenate([t, p])))
                hi = float(np.max(np.concatenate([t, p])))
            if log_axes:
                lo = max(lo, 1e-12)
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, zorder=1)
            ax.text(
                -0.12,
                1.06,
                f"({chr(ord('a') + i)})",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                clip_on=False,
            )
            ax.set_title(feat, loc="left", fontsize=13.5, fontweight="bold")
            ax.set_xlabel("Measured", fontsize=11.8)
            ax.set_ylabel("Predicted", fontsize=11.8)
            if log_axes:
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.grid(True, linestyle="--", alpha=0.2)
            txt = f"NSE={m.get('nse', float('nan')):.3f}\nRMSE={m.get('rmse', float('nan')):.3g}\nMAE={m.get('mae', float('nan')):.3g}"
            ax.text(0.98, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
            # lightweight colorbar per subplot for readability
            cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
            cbar.ax.tick_params(labelsize=8)
        else:
            ax.axis("off")

    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.08, top=0.915, wspace=0.24, hspace=0.30)
    fig.savefig(os.path.join(out_dir, "density_scatter_panel_2x3.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
def evaluate(model, data_loader, device, criterion, *, feature_names: Optional[list] = None, nse_var_eps: float = 1e-4):
    model.eval()
    y_true_list, y_pred_list = [], []
    station_id_list = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            # Support loaders yielding (X, y) or (X, y, station_id)
            bx = batch[0]
            by = batch[1]
            sid = batch[2] if (isinstance(batch, (tuple, list)) and len(batch) > 2) else None

            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            loss = criterion(out, by)
            total_loss += loss.item()
            
            y_true_list.append(by.cpu().numpy())
            y_pred_list.append(out.cpu().numpy())
            if sid is not None:
                station_id_list.append(sid.detach().cpu().numpy())
            
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    
    # Flatten for RMSE/R2 computation
    y_true_flat, y_pred_flat = _flatten_for_metrics(y_true, y_pred)

    # Compute metrics
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = float(math.sqrt(max(float(mse), 0.0)))
    mae = float(np.mean(np.abs(y_true_flat - y_pred_flat)))
    try:
        r2 = r2_score(y_true_flat, y_pred_flat)
    except Exception:
        r2 = float("nan")
    # NSE: compute per (station, horizon, feature) then aggregate mean/std
    station_ids_all = None
    if station_id_list:
        try:
            station_ids_all = np.concatenate(station_id_list, axis=0)
        except Exception:
            station_ids_all = None
    nse_stats = grouped_nse_stats(y_true, y_pred, station_ids=station_ids_all, feature_names=feature_names, var_eps=float(nse_var_eps))
    nse = float(nse_stats.get("nse_mean", float("nan")))
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'nse': float(nse),
        'nse_std': float(nse_stats.get("nse_std", float("nan"))),
        'nse_filt': float(nse_stats.get("nse_filt_mean", nse)),
        'nse_filt_std': float(nse_stats.get("nse_filt_std", nse_stats.get("nse_std", float("nan")))),
        'nse_filt_kept': int(nse_stats.get("nse_filt_kept", 0)),
        'nse_filt_total': int(nse_stats.get("nse_filt_total", 0)),
        'nse_var_eps': float(nse_stats.get("nse_var_eps", float(nse_var_eps))),
        'nse_by_horizon_mean': nse_stats.get("nse_by_horizon_mean", []),
        'nse_by_horizon_std': nse_stats.get("nse_by_horizon_std", []),
        'nse_by_feature_mean': nse_stats.get("nse_by_feature_mean", {}),
        'nse_by_feature_std': nse_stats.get("nse_by_feature_std", {}),
        'nse_by_station_mean': nse_stats.get("nse_by_station_mean", {}),
    }

    # Per-feature metrics (computed on flattened arrays in current scaling space)
    try:
        metrics["metrics_by_feature"] = compute_metrics_per_feature(y_true, y_pred, feature_names)
    except Exception:
        metrics["metrics_by_feature"] = {}
    
    return metrics, y_true, y_pred

def visualize_scatters(
    y_true,
    y_pred,
    target_cols,
    run_dir,
    scaler_Y,
    test_times=None,
    node_idx: int = 0,
    node_name: Optional[str] = None,
    num_nodes: Optional[int] = None,
    limit: int = 400,
    pred_step: int = 1,
):
    """Plot single-station time-series comparison for each target feature."""
    if scaler_Y is not None:
        y_true_real = scaler_Y.inverse_transform(y_true)
        y_pred_real = scaler_Y.inverse_transform(y_pred)
    else:
        y_true_real, y_pred_real = y_true, y_pred

    plot_dir = os.path.join(run_dir, "analysis_plots")
    ensure_dir(plot_dir)

    t = np.asarray(test_times) if test_times is not None else None

    def _pick_series(arr: np.ndarray):
        if arr.ndim == 3:
            # [S, N, D]
            s, n, _d = arr.shape
            idx = int(np.clip(node_idx, 0, n - 1))
            series = arr[:, idx, :]
            times_s = t[:s] if (t is not None and len(t) >= s) else np.arange(s)
            return series, np.asarray(times_s)

        if arr.ndim == 2:
            total, _d = arr.shape
            if t is None:
                return arr, np.arange(total)

            if len(t) == total:
                return arr, t

            if num_nodes is not None and int(num_nodes) > 0 and total % int(num_nodes) == 0:
                n = int(num_nodes)
                idx = int(np.clip(node_idx, 0, n - 1))
                series = arr[idx::n, :]
                times_s = t[: len(series)] if len(t) >= len(series) else np.arange(len(series))
                return series, np.asarray(times_s)

            if len(t) > 0 and total % len(t) == 0:
                n = total // len(t)
                idx = int(np.clip(node_idx, 0, n - 1))
                series = arr[idx::n, :]
                return series, t

            return arr, np.arange(total)

        raise ValueError(f"Unsupported y shape for visualize_scatters: {arr.shape}")

    yt_series, ts = _pick_series(np.asarray(y_true_real))
    yp_series, _ = _pick_series(np.asarray(y_pred_real))

    l = int(min(limit, len(ts), yt_series.shape[0], yp_series.shape[0]))
    if l <= 0:
        print("[WARN] visualize_scatters: empty data, skipped")
        return

    yt_series = yt_series[:l]
    yp_series = yp_series[:l]
    ts = np.asarray(ts[:l])

    # Parse timestamps if possible; otherwise keep numeric index.
    try:
        ts_dt = pd.to_datetime(ts)
        if pd.isna(ts_dt).all():
            x = np.arange(l)
        else:
            x = ts_dt
    except Exception:
        x = np.arange(l)

    node_tag = str(node_name) if node_name else str(node_idx)

    for i, feat_name in enumerate(target_cols):
        if i >= yt_series.shape[1]:
            break
        y_t = yt_series[:, i]
        y_p = yp_series[:, i]
        m = np.isfinite(y_t) & np.isfinite(y_p)
        if not np.any(m):
            continue

        apply_paper_plot_style()
        fig, ax = plt.subplots(figsize=(8.8, 4.2))
        ax.plot(np.asarray(x)[m], y_p[m], color="#2d87c8", linewidth=1.35, label="Prediction", zorder=2)
        ax.scatter(np.asarray(x)[m], y_t[m], color="#cf4e62", s=9, alpha=0.56, label="Observed", edgecolors="none", zorder=4)
        ax.set_title(str(feat_name), fontsize=16, fontweight="bold", loc="left")
        ax.set_xlabel("Date", fontsize=12.5)
        ax.set_ylabel("Value", fontsize=12.5)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(loc="upper center", bbox_to_anchor=(0.54, 1.14), fontsize=10.5, ncol=2, frameon=False, borderaxespad=0.0)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"line_{feat_name}.png"), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(f"[OK] single-station plots saved: {plot_dir} (station={node_tag})")
def analyze_feature_importance_shap(model, train_loader, feature_cols, run_dir, device):
    """Optional SHAP analysis (safe no-op when SHAP is unavailable)."""
    print("[XAI] Running SHAP analysis...")
    if shap is None:
        print("[WARN] SHAP is not installed. Skipped.")
        return

    save_dir = os.path.join(run_dir, "explanation_plots")
    os.makedirs(save_dir, exist_ok=True)

    model.train()
    first_batch = next(iter(train_loader))
    batch_x = first_batch[0]
    background = batch_x[:50].to(device)
    test_samples = batch_x[50:60].to(device)
    test_samples.requires_grad = True

    try:
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(test_samples)
        apply_paper_plot_style()
        shap.summary_plot(shap_values, test_samples.cpu().numpy(), feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_summary.png"), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
    except Exception as e:
        print(f"[WARN] SHAP computation failed: {e}")


def analyze_temporal_importance_captum(model, sample_input, feature_cols, run_dir, device, sample_id="last"):
    """Optional Captum attribution plot for one sample."""
    if IntegratedGradients is None:
        print("[WARN] Captum is not installed. Skipped.")
        return

    print(f"[XAI] Running Captum analysis (sample={sample_id})...")
    save_dir = os.path.join(run_dir, "explanation_plots")
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    sample_input = sample_input.to(device)
    sample_input.requires_grad_()

    def _forward_sum(x):
        out = model(x)
        return out.sum()

    try:
        ig = IntegratedGradients(_forward_sum)
        attr = ig.attribute(sample_input)
        attr_np = attr.squeeze(0).detach().cpu().numpy().T

        apply_paper_plot_style()
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            attr_np,
            cmap="coolwarm",
            center=0,
            yticklabels=feature_cols,
            xticklabels=[str(i) for i in range(attr_np.shape[1])],
        )
        plt.title(f"Feature-Time Attribution (sample: {sample_id})", fontsize=14, fontweight="bold", pad=8)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"captum_heatmap_{sample_id}.png"), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
    except Exception as e:
        print(f"[WARN] Captum computation failed: {e}")

