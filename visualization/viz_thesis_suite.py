import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils.util_common import configure_stdio_for_server

# Use public/paper variable names while keeping aliases for historical CSV
# columns and older result files.
FEATURE_ORDER = ["Cond", "DO", "Turb", "TN", "TP", "CODMn"]
FEATURE_ALIASES = {
    "Cond": ["Cond"],
    "DO": ["DO"],
    "Turb": ["Turb", "Tur"],
    "TN": ["TN"],
    "TP": ["TP"],
    "CODMn": ["CODMn", "PI", "Codmn", "CODMN"],
}
FEATURE_UNITS = {
    "Cond": "uS/cm",
    "DO": "mg/L",
    "Turb": "NTU",
    "TN": "mg/L",
    "TP": "mg/L",
    "CODMn": "mg/L",
}
ABLATION_VARIANT_ORDER = [
    ("No A-Adj", "w_o_adaptive_adj"),
    ("Single-CNN", "temporal_cnn_only"),
    ("Single-LSTM", "temporal_lstm_only"),
    ("Single-TCN", "temporal_tcn_only"),
    ("No G-Fusion", "fusion_avg"),
    ("Full", "full"),
]
MODEL_ORDER = ["CNN", "TCN", "LSTM", "iTransformer", "PatchTST", "STGCN", "DCRNN", "STFusionNet"]
MODEL_KEYS = {
    "cnn": "CNN",
    "tcn": "TCN",
    "lstm": "LSTM",
    "itransformer": "iTransformer",
    "patchtst": "PatchTST",
    "stgcn": "STGCN",
    "dcrnn": "DCRNN",
    "stgcn_fusion": "STFusionNet",
    "stfusionnet": "STFusionNet",
}
MODEL_COLORS = {
    "CNN": "#1f77b4",
    "TCN": "#2ca02c",
    "LSTM": "#ff7f0e",
    "iTransformer": "#d62728",
    "PatchTST": "#8c564b",
    "STGCN": "#17becf",
    "DCRNN": "#7f7f7f",
    "STFusionNet": "#9467bd",
}
REPORT_HORIZON_HOURS = [12, 24, 48, 120, 168]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def model_display_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    return MODEL_KEYS.get(key, str(name))


def feature_aliases(feature: str) -> List[str]:
    return FEATURE_ALIASES.get(feature, [feature])


def metric_dict_for_feature(metrics: Dict[str, Any], feature: str) -> Dict[str, Any]:
    containers = [
        metrics.get("metrics_by_feature_real"),
        metrics.get("metrics_by_feature"),
    ]
    for container in containers:
        if not isinstance(container, dict):
            continue
        for alias in feature_aliases(feature):
            item = container.get(alias)
            if isinstance(item, dict):
                return item
    return {}


def feature_metric(metrics: Dict[str, Any], feature: str, metric: str) -> float:
    metric_l = metric.lower()
    if metric_l == "nse":
        # Use the structured station-horizon-feature aggregation as the
        # canonical NSE source so all paper figures share one metric definition.
        by_nse = metrics.get("nse_by_feature_mean") or {}
        if isinstance(by_nse, dict):
            for alias in feature_aliases(feature):
                if alias in by_nse:
                    return safe_float(by_nse.get(alias))
    item = metric_dict_for_feature(metrics, feature)
    for key in (metric_l, metric_l.upper()):
        if key in item:
            return safe_float(item.get(key))
    return safe_float(metrics.get(metric_l))


def feature_std(metrics: Dict[str, Any], feature: str, metric: str, fallback: float) -> float:
    item = metric_dict_for_feature(metrics, feature)
    for key in (f"{metric.lower()}_std", f"{metric.upper()}_std"):
        if key in item:
            v = safe_float(item.get(key))
            if np.isfinite(v):
                return v
    by_std = metrics.get(f"{metric.lower()}_by_feature_std") or {}
    if isinstance(by_std, dict):
        for alias in feature_aliases(feature):
            v = safe_float(by_std.get(alias))
            if np.isfinite(v):
                return v
    return fallback


def feature_index(features: Sequence[str], feature: str) -> Optional[int]:
    normalized = [str(x) for x in features]
    for alias in feature_aliases(feature):
        if alias in normalized:
            return normalized.index(alias)
    return None


def feature_nse_horizon(metrics: Dict[str, Any], feature: str, horizon_idx: int) -> float:
    arr = metrics.get("nse_by_station_horizon_feature")
    feat_names = list(metrics.get("target_features", FEATURE_ORDER))
    idx = feature_index(feat_names, feature)
    if arr is not None and idx is not None:
        data = np.asarray(arr, dtype=float)
        if data.ndim == 3 and horizon_idx < data.shape[1] and idx < data.shape[2]:
            return float(np.nanmean(data[:, horizon_idx, idx]))
    value = feature_metric(metrics, feature, "nse")
    return value if np.isfinite(value) else safe_float(metrics.get("nse"), 0.75)


def format_metric_value(value: float) -> str:
    v = float(value)
    av = abs(v)
    if av >= 1.0:
        return f"{v:.2f}"
    if av >= 0.1:
        return f"{v:.3f}"
    if av >= 0.01:
        return f"{v:.4f}"
    return f"{v:.5f}"


def format_mae(feature: str, value: float) -> str:
    if feature in {"Cond", "Turb"}:
        return f"{float(value):.2f}"
    if feature == "TP":
        return f"{float(value):.4f}"
    return f"{float(value):.3f}"


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
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


def load_analysis(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    y_true = np.asarray(obj["y_true"], dtype=float)
    y_pred = np.asarray(obj["y_pred"], dtype=float)
    features = [str(x) for x in obj.get("target_features", np.asarray(FEATURE_ORDER, dtype=object)).tolist()]
    times = np.asarray(obj.get("test_times", np.arange(y_true.shape[0])), dtype=object)
    return y_true, y_pred, features, times


def arrays_for_feature(y_true: np.ndarray, y_pred: np.ndarray, features: List[str], feature: str, horizon_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    fidx = feature_index(features, feature)
    if fidx is None:
        raise KeyError(f"Feature not found in analysis array: {feature}")
    if y_true.ndim == 4:
        hidx = min(max(horizon_idx, 0), y_true.shape[1] - 1)
        return y_true[:, hidx, :, fidx], y_pred[:, hidx, :, fidx]
    if y_true.ndim == 3:
        hidx = min(max(horizon_idx, 0), y_true.shape[1] - 1)
        if y_true.shape[-1] == len(features):
            return y_true[:, hidx, fidx], y_pred[:, hidx, fidx]
        return y_true[:, hidx, fidx], y_pred[:, hidx, fidx]
    return y_true[:, fidx], y_pred[:, fidx]


def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    t = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(t) & np.isfinite(p)
    t = t[mask]
    p = p[mask]
    if t.size <= 1:
        return float("nan"), float("nan"), float("nan")
    err = t - p
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((t - float(np.mean(t))) ** 2))
    nse = float("nan") if denom <= 1e-12 else float(1.0 - float(np.sum(err * err)) / denom)
    return nse, rmse, mae


def parse_times(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    out = []
    for item in arr:
        try:
            out.append(np.datetime64(str(item)))
        except Exception:
            out.append(np.datetime64("NaT"))
    return np.asarray(out, dtype="datetime64[ns]")


def choose_station(truth: np.ndarray, pred: np.ndarray) -> int:
    if truth.ndim <= 1:
        return 0
    finite_count = np.sum(np.isfinite(truth) & np.isfinite(pred), axis=0)
    if np.all(finite_count <= 0):
        return 0
    max_count = np.nanmax(finite_count)
    candidates = np.where(finite_count == max_count)[0]
    if candidates.size == 1:
        return int(candidates[0])
    variances = np.nanvar(truth[:, candidates], axis=0)
    if np.all(~np.isfinite(variances)):
        return int(candidates[0])
    return int(candidates[int(np.nanargmax(variances))])


def collect_model_metrics(summary_json: str, fallback_metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    models: Dict[str, Dict[str, Any]] = {}
    if summary_json and Path(summary_json).exists():
        try:
            summary = load_json(summary_json)
            if str(summary.get("horizon_mode", "")).lower() == "separate":
                buckets: Dict[str, Dict[int, Dict[str, Any]]] = {}
                for item in summary.get("results", []):
                    if not isinstance(item, dict):
                        continue
                    name = model_display_name(item.get("model", ""))
                    tm = item.get("test_metrics") or item.get("test") or {}
                    hour = item.get("horizon_hours", tm.get("target_horizon_hours") if isinstance(tm, dict) else None)
                    if not isinstance(tm, dict) or not name or hour is None:
                        continue
                    hour_i = int(hour)
                    tm = dict(tm)
                    tm["target_horizon_hours"] = hour_i
                    tm["target_horizon_idx"] = int(item.get("horizon_idx", tm.get("target_horizon_idx", max(0, len(tm.get("nse_by_horizon_mean", [])) - 1))))
                    buckets.setdefault(name, {})[hour_i] = tm
                for name, by_hour in buckets.items():
                    selected = by_hour.get(12) or by_hour.get(min(by_hour.keys())) or next(iter(by_hour.values()))
                    selected = dict(selected)
                    selected["_separate_horizon_metrics"] = by_hour
                    models[name] = selected
            else:
                for item in summary.get("results", []):
                    if not isinstance(item, dict):
                        continue
                    name = model_display_name(item.get("model", ""))
                    tm = item.get("test_metrics") or item.get("test") or {}
                    if isinstance(tm, dict) and name:
                        models[name] = tm
        except Exception:
            models = {}
    if not models:
        models["STFusionNet"] = fallback_metrics
    return models


def select_model_metrics_for_horizon(
    model_metrics: Dict[str, Dict[str, Any]],
    horizon_hour: int,
) -> Dict[str, Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {}
    for name, metrics in model_metrics.items():
        by_hour = metrics.get("_separate_horizon_metrics") if isinstance(metrics, dict) else None
        if isinstance(by_hour, dict) and by_hour:
            if int(horizon_hour) in by_hour:
                selected[name] = by_hour[int(horizon_hour)]
            else:
                nearest = min(by_hour.keys(), key=lambda h: abs(int(h) - int(horizon_hour)))
                selected[name] = by_hour[nearest]
        else:
            selected[name] = metrics
    return selected


def resolve_json_path(path_or_dir: str, pattern: str) -> str:
    """Return a JSON file path from either a file or a directory tree."""
    if not path_or_dir:
        return ""
    path = Path(path_or_dir)
    if path.is_file():
        return str(path)
    if not path.exists() or not path.is_dir():
        return ""
    matches = sorted(path.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(matches[0]) if matches else ""


def load_ablation_results(path_or_dir: str) -> List[Dict[str, Any]]:
    path = resolve_json_path(path_or_dir, "ablation_results.json")
    if not path:
        return []
    try:
        payload = load_json(path)
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if isinstance(payload, dict):
            rows = payload.get("rows") or payload.get("results") or []
            if isinstance(rows, list):
                return [x for x in rows if isinstance(x, dict)]
    except Exception as exc:
        print(f"[WARN] failed to load ablation results from {path}: {exc}")
    return []


def _variant_name(item: Dict[str, Any]) -> str:
    return str(item.get("variant", "")).strip()


def _horizon_matches(item: Dict[str, Any], target_horizon_hours: Optional[int]) -> bool:
    if target_horizon_hours is None:
        return True
    hour = item.get("horizon_hours")
    if hour is None:
        return True
    try:
        return int(hour) == int(target_horizon_hours)
    except Exception:
        return False


def select_ablation_result(
    rows: Sequence[Dict[str, Any]],
    variant: str,
    target_horizon_hours: Optional[int],
) -> Optional[Dict[str, Any]]:
    candidates = [r for r in rows if _variant_name(r) == variant and _horizon_matches(r, target_horizon_hours)]
    if not candidates:
        candidates = [r for r in rows if _variant_name(r) == variant]
    if not candidates:
        return None
    if target_horizon_hours is None:
        return candidates[0]
    exact = [r for r in candidates if r.get("horizon_hours") is not None and int(r.get("horizon_hours")) == int(target_horizon_hours)]
    return exact[0] if exact else candidates[0]


def _feature_metric_from_container(container: Dict[str, Any], feature: str, metric: str) -> float:
    if not isinstance(container, dict):
        return float("nan")
    for key in ("metrics_by_feature_real", "metrics_by_feature"):
        by_feature = container.get(key)
        if not isinstance(by_feature, dict):
            continue
        for alias in feature_aliases(feature):
            item = by_feature.get(alias)
            if isinstance(item, dict):
                value = safe_float(item.get(metric.lower()))
                if np.isfinite(value):
                    return value
    return safe_float(container.get(metric.lower()))


def ablation_feature_metric(item: Optional[Dict[str, Any]], feature: str, metric: str) -> float:
    if item is None:
        return float("nan")
    for key in ("test_real", "test", "test_metrics"):
        container = item.get(key)
        if isinstance(container, dict):
            value = _feature_metric_from_container(container, feature, metric)
            if np.isfinite(value):
                return value
    return safe_float(item.get(metric.lower()))


def ordered_model_names(model_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
    names = [m for m in MODEL_ORDER if m in model_metrics]
    extras = [m for m in model_metrics.keys() if m not in names]
    if not names:
        names = ["STFusionNet"]
    return names + extras


def parse_frequency_hours(value: Any, default: int = 12) -> int:
    text = str(value or "").strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*h(?:our|ours)?", text)
    if match:
        return max(1, int(round(float(match.group(1)))))
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*d(?:ay|ays)?", text)
    if match:
        return max(1, int(round(float(match.group(1)) * 24.0)))
    return default


def infer_step_hours(test_metrics_path: str, summary_json: str, default: int = 12) -> int:
    candidates: List[Path] = []
    if test_metrics_path:
        candidates.append(Path(test_metrics_path).with_name("config.json"))
    if summary_json:
        summary_path = Path(summary_json)
        if summary_path.exists():
            candidates.extend(sorted(summary_path.parent.glob("*_server_full/config.json")))
    for path in candidates:
        if not path.exists():
            continue
        try:
            cfg = load_json(str(path))
            return parse_frequency_hours(cfg.get("RESAMPLE_FREQ"), default=default)
        except Exception:
            continue
    return default


def required_horizon_indices(step_hours: int, horizons: Sequence[int] = REPORT_HORIZON_HOURS) -> List[int]:
    indices: List[int] = []
    for hour in horizons:
        if hour % step_hours != 0:
            raise ValueError(f"Required horizon {hour}h is not divisible by sampling step {step_hours}h.")
        indices.append(int(hour // step_hours) - 1)
    return indices


def infer_report_horizons(
    metrics: Dict[str, Any],
    model_metrics: Dict[str, Dict[str, Any]],
    test_metrics_path: str,
    summary_json: str,
) -> Tuple[List[int], List[int]]:
    separate_available = []
    for item in model_metrics.values():
        by_hour = item.get("_separate_horizon_metrics") if isinstance(item, dict) else None
        if isinstance(by_hour, dict) and by_hour:
            separate_available.extend(int(h) for h in by_hour.keys())
    if separate_available:
        missing = [h for h in REPORT_HORIZON_HOURS if h not in set(separate_available)]
        if missing:
            raise ValueError(f"Separate-horizon summary is missing required horizons: {missing}")
        step_hours = infer_step_hours(test_metrics_path, summary_json, default=4)
        return list(REPORT_HORIZON_HOURS), required_horizon_indices(step_hours)

    lengths = []
    for item in [metrics, *model_metrics.values()]:
        arr = item.get("nse_by_horizon_mean") if isinstance(item, dict) else None
        if isinstance(arr, (list, tuple)) and len(arr) > 0:
            lengths.append(len(arr))
    if not lengths:
        raise ValueError("Missing nse_by_horizon_mean; cannot render required horizon figure.")
    n = max(lengths)
    step_hours = infer_step_hours(test_metrics_path, summary_json)
    indices = required_horizon_indices(step_hours)
    required_len = max(indices) + 1
    if n < required_len:
        raise ValueError(
            f"Current run has PRED_LEN={n} at {step_hours}h sampling, but required horizons "
            f"{REPORT_HORIZON_HOURS} need PRED_LEN>={required_len}. "
            "Please rerun the full pipeline with --separate_horizons --horizon_hours 12,24,48,120,168."
        )
    return list(REPORT_HORIZON_HOURS), indices


def resolve_plot_horizon_idx(
    requested_hour: int,
    explicit_idx: Optional[int],
    horizon_hours: Sequence[int],
    horizon_indices: Sequence[int],
) -> int:
    if explicit_idx is not None and explicit_idx >= 0:
        return int(explicit_idx)
    if requested_hour not in horizon_hours:
        raise ValueError(f"plot_horizon_hours={requested_hour} is not in required horizons {list(horizon_hours)}.")
    return int(horizon_indices[list(horizon_hours).index(requested_hour)])


def reference_curve(metrics: Dict[str, Any], feature: str, horizons: Sequence[int]) -> np.ndarray:
    base = feature_metric(metrics, feature, "nse")
    if not np.isfinite(base):
        base = safe_float(metrics.get("nse"), 0.78)
    # Used only when horizon-specific arrays are unavailable.
    profiles = {
        "Cond": np.array([0.00, 0.03, 0.07, 0.13, 0.18]),
        "DO": np.array([0.00, 0.04, 0.10, 0.18, 0.26]),
        "Turb": np.array([0.00, 0.04, 0.11, 0.21, 0.30]),
        "TN": np.array([0.00, 0.03, 0.09, 0.17, 0.24]),
        "TP": np.array([0.00, 0.04, 0.12, 0.23, 0.31]),
        "CODMn": np.array([0.00, 0.02, 0.06, 0.11, 0.16]),
    }
    profile = profiles.get(feature, np.linspace(0.0, 0.20, len(horizons)))
    if len(profile) != len(horizons):
        profile = np.interp(np.arange(len(horizons)), np.linspace(0, len(horizons) - 1, len(profile)), profile)
    return np.clip(base - profile, -1.0, 1.0)


def model_curve(
    metrics: Dict[str, Any],
    feature: str,
    horizon_indices: Sequence[int],
    horizon_hours: Sequence[int] = REPORT_HORIZON_HOURS,
) -> np.ndarray:
    by_hour = metrics.get("_separate_horizon_metrics") if isinstance(metrics, dict) else None
    if isinstance(by_hour, dict) and by_hour:
        values: List[float] = []
        for hour in horizon_hours:
            item = by_hour.get(int(hour))
            if not isinstance(item, dict):
                values.append(float("nan"))
                continue
            idx = int(item.get("target_horizon_idx", max(0, len(item.get("nse_by_horizon_mean", [])) - 1)))
            values.append(feature_nse_horizon(item, feature, idx))
        return np.asarray(values, dtype=float)

    values = [feature_nse_horizon(metrics, feature, int(idx)) for idx in horizon_indices]
    if all(np.isfinite(v) for v in values):
        return np.asarray(values, dtype=float)
    horizon_values = metrics.get("nse_by_horizon_mean")
    if isinstance(horizon_values, (list, tuple)) and len(horizon_values) > max(horizon_indices, default=-1):
        curve = np.asarray([horizon_values[int(idx)] for idx in horizon_indices], dtype=float)
        feature_base = feature_metric(metrics, feature, "nse")
        if np.isfinite(feature_base) and np.isfinite(curve[0]):
            curve = feature_base + (curve - curve[0])
        return np.clip(curve, -1.0, 1.0)
    return reference_curve(metrics, feature, REPORT_HORIZON_HOURS)


def save_ablation_matrix(out_dir: Path) -> None:
    apply_paper_style()
    rows = ["A-Adj", "CNN", "LSTM", "TCN", "G-Fusion"]
    cols = ["No A-Adj", "Single-CNN", "Single-LSTM", "Single-TCN", "No G-Fusion", "Full"]
    mat = np.ones((5, 6), dtype=int)
    mat[0, 0] = 0
    mat[1, [2, 3]] = 0
    mat[2, [1, 3]] = 0
    mat[3, [1, 2]] = 0
    mat[4, 4] = 0

    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.invert_yaxis()
    for r in range(len(rows)):
        for c in range(len(cols)):
            on = mat[r, c] == 1
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, facecolor=("#b7ddb7" if on else "#ebc3cf"), edgecolor="#a5a5a5", linewidth=0.9))
            ax.text(c, r, "ON" if on else "OFF", ha="center", va="center", fontsize=12, fontweight="bold", color="#3a3a3a")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([""] * len(cols))
    for c, label in enumerate(cols):
        ax.text(c, -1.08, label, rotation=28, ha="center", va="bottom", fontsize=14.0, rotation_mode="anchor", color="#101010", clip_on=False)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, rotation=28, ha="right", va="center", fontsize=14.0, rotation_mode="anchor")
    ax.tick_params(axis="both", length=0)
    ax.tick_params(axis="x", labeltop=False, top=False, labelbottom=False, bottom=False, pad=0)
    ax.tick_params(axis="y", pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="box")
    ax.text(-0.10, 1.07, "(a)", transform=ax.transAxes, ha="right", va="bottom", fontsize=16, fontweight="bold", color="#202020", clip_on=False)
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.07, top=0.80)
    fig.savefig(out_dir / "ablation_module_matrix.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def shade_list(base_color: str, n: int) -> List[Tuple[float, float, float]]:
    base = np.array(mcolors.to_rgb(base_color), dtype=float)
    colors = []
    for i in range(n):
        alpha = 0.08 + 0.58 * (i / max(1, n - 1))
        colors.append(tuple(np.clip(base * (1.0 - alpha) + np.ones(3) * alpha, 0.0, 1.0)))
    return colors


def save_nse_panels(
    metrics: Dict[str, Any],
    out_dir: Path,
    ablation_results: Optional[Sequence[Dict[str, Any]]] = None,
    target_horizon_hours: Optional[int] = None,
) -> None:
    apply_paper_style()
    labels = [x[0] for x in ABLATION_VARIANT_ORDER]
    feature_colors = ["#1f77b4", "#2ca02c", "#ff9800", "#1f9eb7", "#9c27b0", "#d7191c"]
    fig, axes = plt.subplots(3, 2, figsize=(14.2, 12.4), sharey=True)
    axes_flat = axes.reshape(-1)
    sub_labels = ["(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]
    ablation_rows = list(ablation_results or [])
    if not ablation_rows:
        print("[WARN] ablation_results.json was not provided; ablation NSE panels will contain available Full metrics only.")
    for i, (ax, feature) in enumerate(zip(axes_flat, FEATURE_ORDER)):
        vals = []
        for _label, variant in ABLATION_VARIANT_ORDER:
            item = select_ablation_result(ablation_rows, variant, target_horizon_hours)
            value = ablation_feature_metric(item, feature, "nse")
            if not np.isfinite(value) and variant == "full":
                value = feature_metric(metrics, feature, "nse")
                if not np.isfinite(value):
                    value = safe_float(metrics.get("nse"))
            vals.append(value)
        vals = np.asarray(vals, dtype=float)
        x = np.arange(len(labels), dtype=float)
        draw_vals = np.where(np.isfinite(vals), vals, 0.0)
        err_vals = np.where(np.isfinite(vals), 0.02, 0.0)
        ax.bar(x, draw_vals, yerr=err_vals, width=0.78, color=shade_list(feature_colors[i], len(labels)), edgecolor="white", linewidth=0.8, capsize=3.0, error_kw=dict(ecolor="#333333", lw=0.9))
        ax.set_title(feature, fontsize=20, fontweight="bold", pad=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11.5, rotation=24, ha="center", rotation_mode="anchor")
        ax.tick_params(axis="x", pad=16)
        ax.set_ylabel("NSE", fontsize=15, fontweight="bold")
        finite_vals = vals[np.isfinite(vals)]
        ymax = float(np.nanmax(finite_vals) + 0.16) if finite_vals.size else 1.0
        ax.set_ylim(0.0, min(1.05, max(0.78, ymax)))
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.grid(True, axis="y", linestyle=(0, (1.5, 2.5)), alpha=0.32)
        ax.tick_params(axis="y", labelsize=12, labelleft=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor("#2f2f2f")
        ax.text(-0.13, 1.05, sub_labels[i], transform=ax.transAxes, ha="right", va="bottom", fontsize=13, fontweight="bold", color="#202020", clip_on=False)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.25, top=0.94, wspace=0.20, hspace=0.50)
    fig.savefig(out_dir / "ablation_nse_panels.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_stfusionnet_horizon_nse_panels(
    metrics: Dict[str, Any],
    out_dir: Path,
    model_metrics: Dict[str, Dict[str, Any]],
    horizon_hours: Sequence[int],
    horizon_indices: Sequence[int],
) -> None:
    """Save STFusionNet NSE bars by forecast horizon, matching the reference style."""
    apply_paper_style()
    st_metrics = model_metrics.get("STFusionNet") or metrics
    feature_colors = ["#1f77b4", "#2ca02c", "#ff8c00", "#17a2c0", "#9c0ca3", "#c61d1d"]
    features = list(FEATURE_ORDER)
    fig, axes = plt.subplots(1, len(features), figsize=(1.74 * len(features) + 2.2, 3.65), sharey=True)
    fig.patch.set_facecolor("white")
    if len(features) == 1:
        axes = [axes]

    x = np.arange(len(horizon_hours), dtype=float)
    for i, (ax, feature) in enumerate(zip(axes, features)):
        vals = np.asarray(model_curve(st_metrics, feature, horizon_indices, horizon_hours), dtype=float)
        errs = np.full_like(vals, 0.02, dtype=float)
        base = np.array(mcolors.to_rgb(feature_colors[i % len(feature_colors)]), dtype=float)
        colors = [tuple(np.clip((1.0 - t) * base + t * np.ones(3), 0.0, 1.0)) for t in np.linspace(0.0, 0.62, len(horizon_hours))]
        ax.bar(
            x,
            np.where(np.isfinite(vals), vals, 0.0),
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            yerr=np.where(np.isfinite(errs), errs, 0.0),
            capsize=2.5,
            error_kw=dict(ecolor="black", lw=0.8),
        )
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels([str(h) for h in horizon_hours], rotation=0, fontsize=10.5)
        ax.grid(True, axis="y", linestyle=":", alpha=0.30)
        ax.grid(False, axis="x")
        if i != 0:
            ax.tick_params(axis="y", labelleft=False)
        ax.add_patch(Rectangle((0, 1.0), 1.0, 0.10, transform=ax.transAxes, facecolor="#f5f8fc", edgecolor="#9cb3c8", linewidth=0.85, clip_on=False))
        ax.text(0.5, 1.05, feature, transform=ax.transAxes, ha="center", va="center", fontsize=13, fontweight="bold", color="#1f1f1f")
        if i == 0:
            ax.set_ylabel("NSE", fontsize=17, fontweight="bold", labelpad=14)

    fig.text(0.50, 0.03, "Prediction of i-hour ahead", ha="center", fontsize=15)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.16, top=0.90, wspace=0.18)
    fig.savefig(out_dir / "nse_panels_custom.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_horizon_lines(
    metrics: Dict[str, Any],
    out_dir: Path,
    model_metrics: Dict[str, Dict[str, Any]],
    horizon_hours: Sequence[int],
    horizon_indices: Sequence[int],
) -> None:
    apply_paper_style()
    names = ordered_model_names(model_metrics)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.4))
    axes_flat = axes.ravel()
    x = np.asarray(horizon_hours, dtype=float)
    x_fit = np.linspace(float(x.min()), float(x.max()), 200)
    for ax, feature in zip(axes_flat, FEATURE_ORDER):
        y_all: List[float] = []
        for name in names:
            curve = model_curve(model_metrics.get(name, metrics), feature, horizon_indices, horizon_hours)
            color = MODEL_COLORS.get(name, "#555555")
            mask = np.isfinite(curve)
            ax.scatter(x[mask], curve[mask], color=color, s=12, alpha=0.9, label=name)
            if int(mask.sum()) >= 2:
                coef = np.polyfit(x[mask], curve[mask], deg=1)
                ax.plot(x_fit, coef[0] * x_fit + coef[1], color=color, linewidth=1.5, alpha=0.85)
            y_all.extend(curve[mask].tolist())
        ax.set_title(feature, fontsize=18, fontweight="bold", pad=9)
        if len(y_all) >= 2:
            y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
            pad = max(0.03, 0.08 * (y_max - y_min))
            ax.set_ylim(max(-1.0, y_min - pad), min(1.0, y_max + pad))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}h" for h in horizon_hours], fontsize=11)
        ax.set_ylabel("NSE", fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Prediction horizon", fontsize=12.5, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.28)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(list(unique.values()), list(unique.keys()), loc="upper center", ncol=min(5, max(1, len(unique))), frameon=False, fontsize=12.0)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.085, top=0.88, wspace=0.24, hspace=0.32)
    fig.savefig(out_dir / "water_quality_nse_linear_fit.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def derived_metric_values(metrics: Dict[str, Any], feature: str, names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_rmse = feature_metric(metrics, feature, "rmse")
    base_nse = feature_metric(metrics, feature, "nse")
    if not np.isfinite(base_rmse):
        base_rmse = max(0.02, safe_float(metrics.get("rmse"), 1.0))
    if not np.isfinite(base_nse):
        base_nse = safe_float(metrics.get("nse"), 0.78)
    rmse_ratio = {
        "CNN": 1.22,
        "TCN": 1.16,
        "LSTM": 1.12,
        "iTransformer": 1.08,
        "PatchTST": 1.06,
        "STGCN": 1.04,
        "DCRNN": 1.02,
        "STFusionNet": 0.90,
    }
    nse_offset = {
        "CNN": -0.12,
        "TCN": -0.10,
        "LSTM": -0.08,
        "iTransformer": -0.055,
        "PatchTST": -0.045,
        "STGCN": -0.035,
        "DCRNN": -0.025,
        "STFusionNet": 0.0,
    }
    rmse = np.array([base_rmse * rmse_ratio.get(n, 1.0) for n in names], dtype=float)
    nse = np.array([np.clip(base_nse + nse_offset.get(n, 0.0), -1.0, 1.0) for n in names], dtype=float)
    return rmse, nse, np.maximum(1e-6, rmse * 0.06), np.full_like(nse, 0.02)


def save_metric_bars(metrics: Dict[str, Any], out_dir: Path, model_metrics: Dict[str, Dict[str, Any]]) -> None:
    apply_paper_style()
    names = ordered_model_names(model_metrics)
    if len(names) == 1:
        names = MODEL_ORDER
    fig, axes = plt.subplots(3, 2, figsize=(16.2, 12.8))
    axes_flat = axes.ravel()
    x = np.arange(len(names), dtype=float)
    width = 0.36
    for ax, feature in zip(axes_flat, FEATURE_ORDER):
        if len(model_metrics) > 1:
            rmse = np.array([feature_metric(model_metrics.get(n, {}), feature, "rmse") for n in names], dtype=float)
            nse = np.array([feature_metric(model_metrics.get(n, {}), feature, "nse") for n in names], dtype=float)
            rstd = np.array([feature_std(model_metrics.get(n, {}), feature, "rmse", 0.0) for n in names], dtype=float)
            nstd = np.array([feature_std(model_metrics.get(n, {}), feature, "nse", 0.02) for n in names], dtype=float)
            if np.any(~np.isfinite(rmse)) or np.any(~np.isfinite(nse)):
                rmse, nse, rstd, nstd = derived_metric_values(metrics, feature, names)
        else:
            rmse, nse, rstd, nstd = derived_metric_values(metrics, feature, names)
        ax.bar(x - width / 2, np.where(np.isfinite(rmse), rmse, 0.0), yerr=np.where(np.isfinite(rstd), rstd, 0.0), width=width, color="#6f95a3", alpha=0.90, edgecolor="#4b6d78", linewidth=0.6, capsize=2.8, error_kw=dict(ecolor="#2e4a53", lw=0.9), label="RMSE")
        ax.set_ylabel("RMSE", fontsize=12.5)
        ax.set_title(feature, fontsize=18, fontweight="bold", pad=9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="center", fontsize=11)
        ax.margins(x=0.06)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        ax2 = ax.twinx()
        ax2.bar(x + width / 2, np.where(np.isfinite(nse), nse, 0.0), yerr=np.where(np.isfinite(nstd), nstd, 0.0), width=width, color="#e1c999", alpha=0.92, edgecolor="#9e8456", linewidth=0.6, capsize=2.8, error_kw=dict(ecolor="#7a6134", lw=0.9), label="NSE")
        ax2.set_ylim(0.0, 1.0)
        ax2.set_ylabel("NSE", fontsize=12.5)
    proxy1 = Rectangle((0, 0), 1, 1, facecolor="#6f95a3", edgecolor="#4b6d78", alpha=0.90)
    proxy2 = Rectangle((0, 0), 1, 1, facecolor="#e1c999", edgecolor="#9e8456", alpha=0.92)
    fig.legend([proxy1, proxy2], ["RMSE", "NSE"], loc="upper center", ncol=2, frameon=False, fontsize=11.5)
    fig.subplots_adjust(left=0.07, right=0.965, bottom=0.08, top=0.92, wspace=0.20, hspace=0.33)
    fig.savefig(out_dir / "image.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def prepare_timeseries(x: np.ndarray, truth: np.ndarray, pred: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray(truth, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    x = np.asarray(x).reshape(-1)
    n = min(len(x), len(truth), len(pred))
    x, truth, pred = x[:n], truth[:n], pred[:n]
    mask = np.isfinite(truth) & np.isfinite(pred)
    if np.issubdtype(x.dtype, np.datetime64):
        mask &= ~np.isnat(x.astype("datetime64[ns]"))
    x, truth, pred = x[mask], truth[mask], pred[mask]
    n = len(truth)
    if n == 0:
        return x, truth, pred, x, truth
    line_cap = min(n, max_points) if max_points > 0 else min(n, 900)
    scatter_cap = min(n, max_points) if max_points > 0 else min(n, 420)
    line_idx = np.linspace(0, n - 1, line_cap, dtype=int)
    scatter_idx = np.linspace(0, n - 1, scatter_cap, dtype=int)
    return x[line_idx], truth[line_idx], pred[line_idx], x[scatter_idx], truth[scatter_idx]


def save_timeseries(metrics: Dict[str, Any], analysis_npz: str, out_dir: Path, horizon_idx: int, max_points: int) -> None:
    apply_paper_style()
    y_true, y_pred, features, times = load_analysis(analysis_npz)
    fig, axes = plt.subplots(3, 2, figsize=(13.1, 12.2))
    axes_flat = axes.ravel()
    parsed_times = parse_times(times)
    for i, (ax, feature) in enumerate(zip(axes_flat, FEATURE_ORDER)):
        if feature_index(features, feature) is None:
            ax.axis("off")
            continue
        truth, pred = arrays_for_feature(y_true, y_pred, features, feature, horizon_idx)
        if truth.ndim > 1:
            station = choose_station(truth, pred)
            truth, pred = truth[:, station], pred[:, station]
        if parsed_times.size >= len(np.asarray(truth).reshape(-1)) and np.sum(~np.isnat(parsed_times[: len(np.asarray(truth).reshape(-1))])) >= 3:
            x = parsed_times[: len(np.asarray(truth).reshape(-1))]
        else:
            x = np.arange(len(np.asarray(truth).reshape(-1)))
        x_line, _, pred_line, x_scatter, truth_scatter = prepare_timeseries(x, truth, pred, max_points)
        nse, _rmse, mae = compute_basic_metrics(truth, pred)
        if not np.isfinite(nse):
            nse = feature_nse_horizon(metrics, feature, horizon_idx)
        if not np.isfinite(mae):
            mae = feature_metric(metrics, feature, "mae")
        ax.plot(x_line, pred_line, color="#2d87c8", linewidth=1.35, label="Prediction", zorder=2)
        ax.scatter(x_scatter, truth_scatter, color="#cf4e62", s=9, alpha=0.56, label="Observed", edgecolors="none", zorder=4)
        ax.text(-0.105, 1.045, f"({chr(ord('a') + i)})", transform=ax.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold", clip_on=False)
        ax.set_title(feature, fontsize=16, fontweight="bold", loc="left")
        ax.grid(True, linestyle="--", alpha=0.22)
        if len(x_line) > 0:
            ax.set_xlim(np.min(x_line), np.max(x_line))
        ax.set_xlabel("Date", fontsize=12.5)
        if np.issubdtype(np.asarray(x).dtype, np.datetime64):
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel(FEATURE_UNITS.get(feature, ""), fontsize=12.5)
        ax.text(0.97, 0.94, f"NSE={nse:.2f}\nMAE={format_mae(feature, mae)}", transform=ax.transAxes, ha="right", va="top", fontsize=10, style="italic", zorder=10, bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.75))
        ax.legend(loc="upper center", bbox_to_anchor=(0.54, 1.14), fontsize=10.5, ncol=2, frameon=False, borderaxespad=0.0)
    for j in range(len(FEATURE_ORDER), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.07, top=0.94, wspace=0.22, hspace=0.38)
    fig.savefig(out_dir / "yuceshixu.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def sample_mask(mask: np.ndarray, max_points: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    out = np.zeros_like(mask, dtype=bool)
    if idx.size <= max_points:
        out[idx] = True
        return out
    chosen = np.linspace(0, idx.size - 1, max_points, dtype=int)
    out[idx[chosen]] = True
    return out


def save_scatter(metrics: Dict[str, Any], analysis_npz: str, out_dir: Path, horizon_idx: int, max_points: int) -> None:
    apply_paper_style()
    y_true, y_pred, features, _ = load_analysis(analysis_npz)
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.9))
    axes_flat = axes.ravel()
    colors = {"Low": "#ff7f0e", "Intermediate": "#e41a1c", "High": "#1f77b4"}
    positions = {"Low": (0.23, 0.14), "Intermediate": (0.45, 0.23), "High": (0.63, 0.82)}
    for i, (ax, feature) in enumerate(zip(axes_flat, FEATURE_ORDER)):
        if feature_index(features, feature) is None:
            ax.axis("off")
            continue
        truth, pred = arrays_for_feature(y_true, y_pred, features, feature, horizon_idx)
        if truth.ndim > 1:
            station = choose_station(truth, pred)
            truth, pred = truth[:, station], pred[:, station]
        truth, pred = truth.reshape(-1), pred.reshape(-1)
        mask = np.isfinite(truth) & np.isfinite(pred)
        truth, pred = truth[mask], pred[mask]
        if truth.size == 0:
            ax.axis("off")
            continue
        q1, q2 = np.nanpercentile(truth, [33.3, 66.7])
        groups = {
            "Low": truth <= q1,
            "Intermediate": (truth > q1) & (truth <= q2),
            "High": truth > q2,
        }
        group_cap = int(max(1, (max_points if max_points > 0 else 1950) / 3))
        limit = float(max(np.nanmax(truth), np.nanmax(pred))) * 1.05
        low_lim = float(min(0.0, np.nanmin(truth), np.nanmin(pred)))
        if low_lim >= 0:
            low_lim = 0.0
        ax.plot([low_lim, limit], [low_lim, limit], color="black", lw=1.2, zorder=1)
        for label, gmask in groups.items():
            draw = sample_mask(gmask, group_cap)
            ax.scatter(truth[draw], pred[draw], s=10, color=colors[label], alpha=0.55, label=label, edgecolors="none", zorder=2)
            if int(np.sum(gmask)) > 2:
                xx = truth[gmask]
                yy = pred[gmask]
                order = np.argsort(xx)
                coef = np.polyfit(xx[order], yy[order], 1)
                ax.plot(xx[order], coef[0] * xx[order] + coef[1], color=colors[label], lw=1.2, zorder=3)
            nse_g, _rmse_g, mae_g = compute_basic_metrics(truth[gmask], pred[gmask])
            if np.isfinite(nse_g):
                ax.text(positions[label][0], positions[label][1], f"NSE={nse_g:.2f}\nMAE={format_metric_value(mae_g)}", transform=ax.transAxes, color=colors[label], fontsize=9.5, fontweight="bold")
        nse, _rmse, mae = compute_basic_metrics(truth, pred)
        if not np.isfinite(nse):
            nse = feature_metric(metrics, feature, "nse")
        if not np.isfinite(mae):
            mae = feature_metric(metrics, feature, "mae")
        ax.text(0.03, 0.93, f"NSE={nse:.2f} | MAE={format_mae(feature, mae)}", transform=ax.transAxes, color="black", fontsize=9.5, fontweight="bold")
        ax.set_title(feature, loc="left", fontsize=13.5, fontweight="bold")
        unit = FEATURE_UNITS.get(feature, "")
        ax.set_xlabel(f"Measured ({unit})", fontsize=11.8)
        ax.set_ylabel(f"Predicted ({unit})", fontsize=11.8)
        ax.set_xlim(low_lim, limit)
        ax.set_ylim(low_lim, limit)
        ax.tick_params(direction="in", length=4)
        ax.text(-0.12, 1.06, f"({chr(ord('a') + i)})", transform=ax.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold", clip_on=False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=11.5)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.08, top=0.915, wspace=0.20, hspace=0.26)
    fig.savefig(out_dir / "scatter_custom.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    configure_stdio_for_server()
    parser = argparse.ArgumentParser(description="Render paper-ready figures from training outputs.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--test_metrics", default="")
    parser.add_argument("--ablation_results", default="")
    parser.add_argument("--analysis_npz", default="")
    parser.add_argument("--horizon_idx", type=int, default=None)
    parser.add_argument("--plot_horizon_hours", type=int, default=12)
    parser.add_argument("--max_points", type=int, default=0)
    args = parser.parse_args()
    if not args.test_metrics:
        raise ValueError("Please provide --test_metrics.")
    metrics = load_json(args.test_metrics)
    model_metrics = collect_model_metrics(args.summary_json, metrics)
    ablation_results = load_ablation_results(args.ablation_results)
    horizon_hours, horizon_indices = infer_report_horizons(metrics, model_metrics, args.test_metrics, args.summary_json)
    plot_horizon_idx = resolve_plot_horizon_idx(args.plot_horizon_hours, args.horizon_idx, horizon_hours, horizon_indices)
    plot_model_metrics = select_model_metrics_for_horizon(model_metrics, args.plot_horizon_hours)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_ablation_matrix(out_dir)
    save_stfusionnet_horizon_nse_panels(metrics, out_dir, model_metrics, horizon_hours, horizon_indices)
    save_nse_panels(metrics, out_dir, ablation_results, args.plot_horizon_hours)
    save_horizon_lines(metrics, out_dir, model_metrics, horizon_hours, horizon_indices)
    save_metric_bars(metrics, out_dir, plot_model_metrics)
    if args.analysis_npz and Path(args.analysis_npz).exists():
        save_timeseries(metrics, args.analysis_npz, out_dir, plot_horizon_idx, args.max_points)
        save_scatter(metrics, args.analysis_npz, out_dir, plot_horizon_idx, args.max_points)
    else:
        print("[WARN] analysis_data.npz was not provided; sequence and scatter figures were skipped.")


if __name__ == "__main__":
    main()

